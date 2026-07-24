
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from eval.metrics import check_contradicts  # noqa: E402
from baselines.common import derive_nodes  # noqa: E402


def load_predictions(vlm_outputs_dir: Path, model: str, split: str) -> list[dict]:
    root = vlm_outputs_dir / model / split
    return [json.loads(p.read_text()) for p in sorted(root.glob("*/*.json"))]


# Stage A: Viterbi decoding of the legal phase sequence for one video

def phase_evidence_raw(segment: dict, phase_id: str, onto) -> float:
    actions_in_phase = [a for a in onto.action_by_id if onto.phase_for(a) == phase_id]
    return sum(segment["action_probs"].get(a, 0.0) for a in actions_in_phase)


def normalized_phase_evidence(segment: dict, phase_ids: list[str], onto) -> dict[str, float]:
    #Per-segment phase scores that sum to 1 across phases.
    if "phase_probs" in segment:
        return {p: segment["phase_probs"].get(p, 0.0) for p in phase_ids}

    raw = {p: phase_evidence_raw(segment, p, onto) for p in phase_ids}
    total = sum(raw.values())
    if total <= 0:
        return {p: 0.0 for p in phase_ids}
    return {p: v / total for p, v in raw.items()}


def viterbi_decode_phases(segments_ordered: list[dict], onto) -> dict[str, str]:
    """Return {segment_id: decoded_phase} for the subsequence of segments
    that originally had >=1 phase-bearing candidate action."""
    phased = [s for s in segments_ordered if s.get("phase_candidates")]
    if not phased:
        return {}

    phase_ids = sorted(onto.phase_ids)
    evidence = [normalized_phase_evidence(s, phase_ids, onto) for s in phased]

    # dp[i] = {phase: (best_score_ending_here, backpointer_phase_or_None)}
    dp = [{} for _ in phased]
    for p in phase_ids:
        dp[0][p] = (evidence[0][p], None)

    for i in range(1, len(phased)):
        for p in phase_ids:
            best_prev, best_score = None, float("-inf")
            for p_prev in phase_ids:
                if (p_prev, p) not in onto.phase_order:
                    continue
                prev_score = dp[i - 1][p_prev][0]
                if prev_score > best_score:
                    best_score, best_prev = prev_score, p_prev
            if best_prev is None:
                # no legal predecessor reaches this phase; still allow
                # starting fresh here so the DP never dead-ends
                dp[i][p] = (evidence[i][p], None)
            else:
                dp[i][p] = (evidence[i][p] + best_score, best_prev)

    # backtrack from the best final state
    last = max(phase_ids, key=lambda p: dp[-1][p][0])
    decoded = [None] * len(phased)
    cur = last
    for i in range(len(phased) - 1, -1, -1):
        decoded[i] = cur
        cur = dp[i][cur][1] if dp[i][cur][1] is not None else cur

    return {seg["segment_id"]: phase for seg, phase in zip(phased, decoded)}

# Stage B: per-segment action filtering + contradiction resolution

def _actions_producing_node(actions: list[str], node: str, onto) -> list[str]:
    producing = []
    for a in actions:
        if a == node or onto.tool_for(a) == node or onto.tissue_for(a) == node or onto.event_for(a) == node:
            producing.append(a)
        elif onto.phase_for(a) == node:
            producing.append(a)
    return producing


def resolve_contradictions(actions: list[str], action_probs: dict[str, float], onto,
                            fixed_phase: str | None, max_iters: int = 10) -> list[str]:
    actions = list(actions)
    for _ in range(max_iters):
        tools, tissues, events = derive_nodes(actions, onto)
        nodes = set(actions) | set(tools) | set(tissues) | set(events)
        if fixed_phase:
            nodes.add(fixed_phase)
        violated = [pair for pair in onto.contradicts if pair.issubset(nodes)]
        if not violated or len(actions) <= 1:
            break
        pair = violated[0]
        candidates = []
        for node in pair:
            candidates.extend(_actions_producing_node(actions, node, onto))
        candidates = [a for a in set(candidates) if len(actions) > 1]
        if not candidates:
            break
        worst = min(candidates, key=lambda a: action_probs.get(a, 0.0))
        actions.remove(worst)
    return actions


def decode_segment(segment: dict, decoded_phase: str | None, onto) -> dict:
    action_probs = segment["action_probs"]
    actions = list(segment["actions"])

    if decoded_phase is not None:
        actions = [a for a in actions if onto.phase_for(a) is None or onto.phase_for(a) == decoded_phase]
        if not actions:  # safety: never end up with zero predicted actions
            actions = [max(segment["actions"], key=lambda a: action_probs.get(a, 0.0))]

    actions = resolve_contradictions(actions, action_probs, onto, decoded_phase)

    tools, tissues, events = derive_nodes(actions, onto)
    return {
        **{k: segment[k] for k in ("segment_id", "split", "video", "frame_start", "frame_end")},
        "model": segment["model"] + "+decoder",
        "actions": actions,
        "action_probs": action_probs,
        "tools": tools,
        "tissues": tissues,
        "events": events,
        "phase": decoded_phase,
        "phase_ambiguous": False,
        "phase_candidates": [decoded_phase] if decoded_phase else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="source subfolder under vlm_outputs/, e.g. 'clip_linear_probe'")
    parser.add_argument("--split", default="val")
    parser.add_argument("--vlm-outputs-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    segments = load_predictions(args.vlm_outputs_dir, args.model, args.split)
    if not segments:
        raise SystemExit(f"No predictions found under {args.vlm_outputs_dir / args.model / args.split}")

    by_video: dict[str, list[dict]] = {}
    for seg in segments:
        by_video.setdefault(seg["video"], []).append(seg)

    out_root = args.vlm_outputs_dir / f"{args.model}_decoded"
    n_written = 0
    n_contradictions_before = n_contradictions_after = 0

    for video, segs in by_video.items():
        ordered = sorted(segs, key=lambda s: s["frame_start"])
        decoded_phases = viterbi_decode_phases(ordered, onto)

        for seg in ordered:
            n_contradictions_before += len(check_contradicts(seg, onto))
            decoded_phase = decoded_phases.get(seg["segment_id"])
            decoded = decode_segment(seg, decoded_phase, onto)
            n_contradictions_after += len(check_contradicts(decoded, onto))

            out_path = out_root / args.split / video / f"{seg['segment_id']}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(decoded, indent=2))
            n_written += 1

    print(f"Decoded {n_written} segments -> {out_root / args.split}")
    print(f"Segments with contradiction violations: {n_contradictions_before} -> {n_contradictions_after}")


if __name__ == "__main__":
    main()
