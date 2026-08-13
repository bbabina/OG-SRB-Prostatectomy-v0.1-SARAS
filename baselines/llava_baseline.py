
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.common import build_prediction_record  # noqa: E402

MODEL_NAME = "llava:7b"


def build_prompt(action_ids: list[str]) -> str:
    return (
        "You are analyzing a single frame from a robotic-assisted radical prostatectomy "
        "(RARP) surgery video.\n"
        "Identify which surgical actions are visible in this image. Only choose from this "
        f"exact list:\n{', '.join(action_ids)}\n\n"
        "Respond with ONLY a comma-separated list of 1-3 action names from that list that "
        "best match the image, most likely first. Do not explain. Do not use any words not "
        "in the list."
    )


def parse_response(text: str, action_ids: set[str]) -> tuple[list[str], list[str]]:
    """Returns (valid_actions_in_order, hallucinated_tokens)."""
    tokens = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    valid, invalid = [], []
    seen = set()
    for tok in tokens:
        # tolerate minor formatting noise (trailing periods, stray quotes)
        cleaned = tok.strip(" .\"'`")
        if cleaned in action_ids:
            if cleaned not in seen:
                valid.append(cleaned)
                seen.add(cleaned)
        elif cleaned:
            invalid.append(cleaned)
    return valid, invalid


def representative_frame(segment: dict) -> int:
    frames = segment["frames"]
    return frames[len(frames) // 2]


def build_stratified_sample(segments: list[dict], action_ids: list[str],
                             n_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_action: dict[str, list[dict]] = {a: [] for a in action_ids}
    for seg in segments:
        for a in seg["raw_labels"]:
            if a in by_action:
                by_action[a].append(seg)

    chosen_ids = set()
    chosen = []
    # guarantee >=1 example per action class that actually appears in val
    for a in action_ids:
        candidates = [s for s in by_action[a] if s["segment_id"] not in chosen_ids]
        if candidates:
            pick = rng.choice(candidates)
            chosen.append(pick)
            chosen_ids.add(pick["segment_id"])

    remaining = [s for s in segments if s["segment_id"] not in chosen_ids]
    rng.shuffle(remaining)
    for seg in remaining:
        if len(chosen) >= n_samples:
            break
        chosen.append(seg)
        chosen_ids.add(seg["segment_id"])

    rng.shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesad-root", type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / "mesad-real 2")
    parser.add_argument("--reports-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs" / "llava_baseline")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--n-samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    action_id_set = set(action_ids)
    prompt = build_prompt(action_ids)

    segments = json.loads((args.reports_dir / "segments_val.json").read_text())
    sample = build_stratified_sample(segments, action_ids, args.n_samples, args.seed)
    print(f"Sampled {len(sample)} of {len(segments)} val segments "
          f"(stratified: >=1 per action class present in val, rest random)")

    images_dir = args.mesad_root / "val" / "images"
    n_written = n_hallucinated_tokens = n_empty = 0
    t_start = time.time()

    for i, seg in enumerate(sample, 1):
        frame_num = representative_frame(seg)
        img_path = images_dir / f"{seg['video']}_frame_{frame_num}.jpg"

        resp = ollama.chat(model=MODEL_NAME, messages=[{
            "role": "user", "content": prompt, "images": [str(img_path)],
        }])
        raw_text = resp["message"]["content"]
        valid_actions, invalid_tokens = parse_response(raw_text, action_id_set)
        n_hallucinated_tokens += len(invalid_tokens)

        if not valid_actions:
            n_empty += 1
            valid_actions = [action_ids[0]]  # never predict nothing; arbitrary fallback, logged via n_empty

        # rank-decreasing pseudo-confidence so top1/argmax logic (shared with
        # every other baseline) has something ordered to work with -- LLaVA
        # gives no real calibrated probability, only an ordered guess.
        action_probs = {a: 0.0 for a in action_ids}
        for rank, a in enumerate(valid_actions):
            action_probs[a] = round(1.0 - 0.15 * rank, 4)

        record = build_prediction_record(seg, MODEL_NAME, valid_actions, action_probs, onto)
        record["raw_response"] = raw_text
        record["hallucinated_tokens"] = invalid_tokens

        out_path = args.out_dir / "val" / seg["video"] / f"{seg['segment_id']}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        n_written += 1

        if i % 20 == 0 or i == len(sample):
            elapsed = time.time() - t_start
            print(f"  {i}/{len(sample)} done ({elapsed:.0f}s elapsed, "
                  f"{elapsed / i:.1f}s/segment, ~{elapsed / i * (len(sample) - i):.0f}s remaining)")

    print(f"\nWrote {n_written} predictions -> {args.out_dir / 'val'}")
    print(f"Segments with no valid action parsed (fell back to default): {n_empty}")
    print(f"Out-of-vocabulary tokens across all responses (hallucination signal): {n_hallucinated_tokens}")


if __name__ == "__main__":
    main()
