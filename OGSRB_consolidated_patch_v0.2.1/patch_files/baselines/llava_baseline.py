from __future__ import annotations

import argparse
import json
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


def parse_response(
    text: str, action_ids: set[str]
) -> tuple[list[str], list[str]]:
    tokens = [
        t.strip()
        for t in text.replace("\n", ",").split(",")
        if t.strip()
    ]
    valid, invalid = [], []
    seen = set()
    for tok in tokens:
        cleaned = tok.strip(" .\"'`")
        if cleaned in action_ids:
            if cleaned not in seen:
                valid.append(cleaned)
                seen.add(cleaned)
        elif cleaned:
            invalid.append(cleaned)
    return valid, invalid


def load_manifest_segments(
    manifest_path: Path, reports_dir: Path
) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    ids = manifest["segment_ids"]
    segments = {
        s["segment_id"]: s
        for s in json.loads((reports_dir / "segments_val.json").read_text())
    }
    missing = [sid for sid in ids if sid not in segments]
    if missing:
        raise SystemExit(f"Manifest ids missing from segments_val.json: {missing[:5]}")
    return [segments[sid] for sid in ids]


def representative_frame(segment: dict) -> int:
    frames = segment["frames"]
    return frames[len(frames) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesad-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "mesad-real 2",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reports",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "vlm_outputs"
        / "llava_baseline",
    )
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "test180_v1.json",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(
            f"Missing {args.manifest}. Run: "
            "python benchmarks/create_test180_manifest.py"
        )

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    action_id_set = set(action_ids)
    prompt = build_prompt(action_ids)
    sample = load_manifest_segments(args.manifest, args.reports_dir)

    print(f"Running LLaVA on frozen manifest: {args.manifest}")
    print(f"Segments: {len(sample)}")

    images_dir = args.mesad_root / "val" / "images"
    n_written = n_hallucinated_tokens = n_empty = 0
    t_start = time.time()

    for i, seg in enumerate(sample, 1):
        frame_num = representative_frame(seg)
        img_path = images_dir / f"{seg['video']}_frame_{frame_num}.jpg"

        resp = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(img_path)],
                }
            ],
        )
        raw_text = resp["message"]["content"]
        valid_actions, invalid_tokens = parse_response(
            raw_text, action_id_set
        )
        n_hallucinated_tokens += len(invalid_tokens)

        if not valid_actions:
            n_empty += 1

        action_probs = {a: 0.0 for a in action_ids}
        for rank, action in enumerate(valid_actions):
            action_probs[action] = round(1.0 - 0.15 * rank, 4)

        record = build_prediction_record(
            seg, MODEL_NAME, valid_actions, action_probs, onto
        )
        record["raw_response"] = raw_text
        record["hallucinated_tokens"] = invalid_tokens
        record["parse_failure"] = not bool(valid_actions)
        record["benchmark_manifest"] = args.manifest.name

        out_path = (
            args.out_dir / "val" / seg["video"] / f"{seg['segment_id']}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        n_written += 1

        if i % 20 == 0 or i == len(sample):
            elapsed = time.time() - t_start
            print(
                f"  {i}/{len(sample)} done "
                f"({elapsed:.0f}s, {elapsed / i:.1f}s/segment)"
            )

    print(f"\nWrote {n_written} predictions -> {args.out_dir / 'val'}")
    print(f"Parse failures (counted as wrong, no arbitrary fallback): {n_empty}")
    print(f"Out-of-vocabulary tokens: {n_hallucinated_tokens}")


if __name__ == "__main__":
    main()
