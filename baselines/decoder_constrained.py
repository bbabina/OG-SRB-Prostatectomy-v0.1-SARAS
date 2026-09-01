"""Ontology-Grounded Structured Decoder for Surgical Action Recognition.

Corrected v0.2 semantics:
- uses ontology v0.2 by default;
- allows local backward phase transitions with a penalty;
- applies transition costs in log-space Viterbi decoding;
- disallows non-adjacent transitions;
- does NOT apply unsupported global contradiction rules;
- recomputes top-1 after action filtering.

This is structured post-processing / temporal decoding, not a token-level VLM decoder.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.common import derive_nodes  # noqa: E402

EPS = 1e-12
DECODER_NAME = "ontology_grounded_structured_decoder_v0.2"


def load_predictions(vlm_outputs_dir: Path, model: str, split: str) -> list[dict]:
    root = vlm_outputs_dir / model / split
    return [json.loads(p.read_text()) for p in sorted(root.glob("*/*.json"))]


def phase_evidence_raw(segment: dict, phase_id: str, onto) -> float:
    actions_in_phase = [
        a for a in onto.action_by_id if onto.phase_for(a) == phase_id
    ]
    return sum(segment.get("action_probs", {}).get(a, 0.0) for a in actions_in_phase)


def normalized_phase_evidence(
    segment: dict, phase_ids: list[str], onto
) -> tuple[dict[str, float], str]:
    if "phase_probs" in segment:
        raw = {p: max(float(segment["phase_probs"].get(p, 0.0)), 0.0) for p in phase_ids}
        source = "calibrated_phase_probs"
    else:
        raw = {p: max(phase_evidence_raw(segment, p, onto), 0.0) for p in phase_ids}
        source = "action_probability_aggregation_uncalibrated"

    total = sum(raw.values())
    if total <= 0:
        uniform = 1.0 / len(phase_ids)
        return {p: uniform for p in phase_ids}, source
    return {p: v / total for p, v in raw.items()}, source


def viterbi_decode_phases(
    segments_ordered: list[dict], onto
) -> tuple[dict[str, str], dict[str, str]]:
    """Decode proxy phases with visual evidence + ontology transition costs."""
    phased = [s for s in segments_ordered if s.get("phase_candidates")]
    if not phased:
        return {}, {}

    phase_ids = onto.ordered_phase_ids()
    evidence_and_source = [
        normalized_phase_evidence(s, phase_ids, onto) for s in phased
    ]
    evidence = [x[0] for x in evidence_and_source]
    sources = {
        s["segment_id"]: source
        for s, (_, source) in zip(phased, evidence_and_source)
    }

    # dp[i][phase] = (best log-score, previous phase)
    dp: list[dict[str, tuple[float, str | None]]] = [dict() for _ in phased]

    for p in phase_ids:
        dp[0][p] = (math.log(max(evidence[0][p], EPS)), None)

    for i in range(1, len(phased)):
        for p in phase_ids:
            emission = math.log(max(evidence[i][p], EPS))
            best_prev = None
            best_score = -math.inf

            for p_prev in phase_ids:
                transition_cost = onto.transition_cost(p_prev, p)
                if not math.isfinite(transition_cost):
                    continue
                prev_score = dp[i - 1][p_prev][0]
                candidate = prev_score + emission - transition_cost
                if candidate > best_score:
                    best_score = candidate
                    best_prev = p_prev

            dp[i][p] = (best_score, best_prev)

    last = max(phase_ids, key=lambda p: dp[-1][p][0])
    if not math.isfinite(dp[-1][last][0]):
        raise RuntimeError("No legal phase path found under ontology v0.2.")

    decoded = [None] * len(phased)
    cur = last
    for i in range(len(phased) - 1, -1, -1):
        decoded[i] = cur
        prev = dp[i][cur][1]
        if i > 0 and prev is None:
            raise RuntimeError("Broken Viterbi backpointer.")
        cur = prev if prev is not None else cur

    return (
        {seg["segment_id"]: phase for seg, phase in zip(phased, decoded)},
        sources,
    )


def decode_segment(
    segment: dict,
    decoded_phase: str | None,
    onto,
    phase_evidence_source: str | None,
) -> dict:
    action_probs = dict(segment.get("action_probs", {}))
    raw_actions = list(segment.get("actions", []))
    actions = list(raw_actions)

    if decoded_phase is not None:
        actions = [
            a
            for a in actions
            if onto.phase_for(a) is None or onto.phase_for(a) == decoded_phase
        ]

        # Never invent a class that the raw model did not predict.
        # If phase filtering removes everything, keep the highest-scored RAW action.
        if not actions and raw_actions:
            actions = [
                max(raw_actions, key=lambda a: action_probs.get(a, 0.0))
            ]

    # v0.2 has no expert-validated global contradiction rules.
    # No contradiction-removal stage is applied here.

    tools, tissues, events = derive_nodes(actions, onto)
    raw_top1 = (
        max(raw_actions, key=lambda a: action_probs.get(a, 0.0))
        if raw_actions
        else None
    )
    decoded_top1 = (
        max(actions, key=lambda a: action_probs.get(a, 0.0))
        if actions
        else None
    )

    return {
        **{
            k: segment[k]
            for k in ("segment_id", "split", "video", "frame_start", "frame_end")
        },
        "model": segment.get("model", "unknown") + "+ogsd_v0.2",
        "decoder": DECODER_NAME,
        "actions": actions,
        "raw_actions": raw_actions,
        "action_probs": action_probs,
        "raw_top1_prediction": raw_top1,
        "top1_prediction": decoded_top1,
        "tools": tools,
        "tissues": tissues,
        "events": events,
        "phase": decoded_phase,
        "phase_ambiguous": False,
        "phase_candidates": [decoded_phase] if decoded_phase else [],
        "phase_evidence_source": phase_evidence_source,
        "transition_policy": onto.phase_transition_policy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Source subfolder under vlm_outputs/, e.g. clip_linear_probe",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--vlm-outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "vlm_outputs",
    )
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    segments = load_predictions(args.vlm_outputs_dir, args.model, args.split)
    if not segments:
        raise SystemExit(
            f"No predictions found under "
            f"{args.vlm_outputs_dir / args.model / args.split}"
        )

    by_video: dict[str, list[dict]] = {}
    for seg in segments:
        by_video.setdefault(seg["video"], []).append(seg)

    out_root = args.vlm_outputs_dir / f"{args.model}_decoded_v02"
    n_written = 0
    n_uncalibrated = 0

    for video, segs in by_video.items():
        ordered = sorted(segs, key=lambda s: s["frame_start"])
        decoded_phases, evidence_sources = viterbi_decode_phases(ordered, onto)

        for seg in ordered:
            decoded_phase = decoded_phases.get(seg["segment_id"])
            source = evidence_sources.get(seg["segment_id"])
            if source == "action_probability_aggregation_uncalibrated":
                n_uncalibrated += 1

            decoded = decode_segment(seg, decoded_phase, onto, source)
            out_path = (
                out_root / args.split / video / f"{seg['segment_id']}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(decoded, indent=2))
            n_written += 1

    print(f"Decoded {n_written} segments -> {out_root / args.split}")
    print(
        "Decoder: Ontology-Grounded Structured Decoder for Surgical Action Recognition"
    )
    print(
        "Transition policy:",
        json.dumps(onto.phase_transition_policy, indent=2),
    )
    if n_uncalibrated:
        print(
            f"WARNING: {n_uncalibrated} phase-bearing segments used "
            "uncalibrated action-probability aggregation because phase_probs "
            "were unavailable. Report this explicitly for those baselines."
        )
    print(
        "Global contradiction filtering: INACTIVE "
        "(ontology v0.2 has no validated global contradiction rules)."
    )


if __name__ == "__main__":
    main()
