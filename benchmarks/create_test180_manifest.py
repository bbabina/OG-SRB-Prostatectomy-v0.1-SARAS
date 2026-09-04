"""Create the frozen Test-180 v1 manifest.

This deliberately reproduces the sampling logic used by the existing LLaVA baseline:
- seed=0
- at least one validation example for each action class present
- random remainder to 180
- final shuffle

The manifest is for FAIR CROSS-MODEL PERCEPTION COMPARISON.
It is sparse across real3, so temporal decoder metrics MUST NOT be computed directly
on this subset as if the selected segments were consecutive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402


def build_stratified_sample(
    segments: list[dict],
    action_ids: list[str],
    n_samples: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    by_action: dict[str, list[dict]] = {a: [] for a in action_ids}

    for seg in segments:
        for action in seg["raw_labels"]:
            if action in by_action:
                by_action[action].append(seg)

    chosen_ids = set()
    chosen = []

    for action in action_ids:
        candidates = [
            s for s in by_action[action] if s["segment_id"] not in chosen_ids
        ]
        if candidates:
            pick = rng.choice(candidates)
            chosen.append(pick)
            chosen_ids.add(pick["segment_id"])

    remaining = [
        s for s in segments if s["segment_id"] not in chosen_ids
    ]
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
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "reports"
        / "segments_val.json",
    )
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "test180_v1.json",
    )
    parser.add_argument("--n", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(
            f"{args.output} already exists. It is intentionally frozen. "
            "Use --force only if you deliberately create a new benchmark version."
        )

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    segments = json.loads(args.segments.read_text())
    sample = build_stratified_sample(segments, action_ids, args.n, args.seed)

    if len(sample) != args.n:
        raise SystemExit(f"Requested {args.n} samples but obtained {len(sample)}.")

    label_support = Counter()
    for seg in sample:
        label_support.update(seg["raw_labels"])

    items = []
    for seg in sample:
        frames = seg["frames"]
        items.append(
            {
                "segment_id": seg["segment_id"],
                "video": seg["video"],
                "frame_start": seg["frame_start"],
                "frame_end": seg["frame_end"],
                "representative_frame": frames[len(frames) // 2],
                "raw_labels": seg["raw_labels"],
            }
        )

    segment_ids = [x["segment_id"] for x in items]
    checksum = hashlib.sha256(
        "\n".join(segment_ids).encode("utf-8")
    ).hexdigest()

    manifest = {
        "name": "test180_v1",
        "purpose": "fair_cross_model_perception_comparison",
        "n_segments": len(items),
        "split": "val",
        "video": "real3",
        "sampling_strategy": (
            "stratified_class_coverage_then_random_remainder"
        ),
        "seed": args.seed,
        "ontology_version": str(
            onto.raw.get("meta", {}).get("version", "unknown")
        ),
        "temporal_evaluation": False,
        "temporal_note": (
            "Sparse sample. Do not treat selected segments as consecutive "
            "for Viterbi/phase-transition metrics. Decode dense video outputs "
            "first, then optionally report recognition metrics on this manifest."
        ),
        "segment_id_sha256": checksum,
        "class_support": {
            a: int(label_support.get(a, 0)) for a in action_ids
        },
        "segment_ids": segment_ids,
        "items": items,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote frozen manifest: {args.output}")
    print(f"Segments: {len(items)}")
    print(f"SHA256: {checksum}")
    print("Temporal evaluation on this sparse manifest: DISABLED")


if __name__ == "__main__":
    main()
