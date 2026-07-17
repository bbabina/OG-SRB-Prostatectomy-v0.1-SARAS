"""Step 2 QC: flag ontology-illegal segments and compute annotation-level
reasoning-validity metrics as stated) over the ground-truth mapped annotations.

Checks per segment:
  - requires   : every action's required tool is present in `tools`
  - acts_on    : every action's tissue is present in `tissues`
  - contradicts: no forbidden pair present among tools+actions+tissues+
                 phase+events
Checks per video (ordered by frame_start):
  - phase_order: each phase(segment[i]) -> phase(segment[i+1]) transition
                 (skipping segments with no phase) is in the ontology's
                 phase_order set.

Output: reports/qc_report.json + reports/qc_summary.json (RS/AOC/CR/SOC).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
# check_requires/check_acts_on/check_contradicts/check_phase_order now live in
# eval/metrics.py, since eval/evaluator.py needs the exact same checks to
# score VLM predictions later — importing here keeps both call sites in sync.
from eval.metrics import check_requires, check_acts_on, check_contradicts, check_phase_order  # noqa: E402


def load_segments(annotations_dir: Path, split: str) -> list[dict]:
    segments = []
    split_dir = annotations_dir / split
    if not split_dir.exists():
        return segments
    for path in sorted(split_dir.glob("*/*.json")):
        segments.append(json.loads(path.read_text()))
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "annotations")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)

    all_segments = []
    for split in ("train", "val"):
        all_segments.extend(load_segments(args.annotations_dir, split))

    if not all_segments:
        print(f"No mapped annotations found under {args.annotations_dir} "
              f"(run map_annotations.py first)")
        return

    per_segment_report = []
    n_rs_ok = n_aoc_ok = n_cr_violation = 0

    for seg in all_segments:
        req_violations = check_requires(seg, onto)
        acts_violations = check_acts_on(seg, onto)
        contra_violations = check_contradicts(seg, onto)

        if not req_violations:
            n_rs_ok += 1
        if not acts_violations:
            n_aoc_ok += 1
        if contra_violations:
            n_cr_violation += 1

        if req_violations or acts_violations or contra_violations:
            per_segment_report.append({
                "segment_id": seg["segment_id"],
                "video": seg["video"],
                "split": seg["split"],
                "requires_violations": req_violations,
                "acts_on_violations": acts_violations,
                "contradicts_violations": contra_violations,
            })

    # phase_order is checked per-video, across the full (train+val) frame
    # timeline for that video.
    by_video: dict[str, list[dict]] = {}
    for seg in all_segments:
        by_video.setdefault(seg["video"], []).append(seg)

    n_soc_transitions = 0
    n_soc_illegal = 0
    phase_order_report = []
    for video, segs in by_video.items():
        violations, n_transitions = check_phase_order(segs, onto)
        n_soc_transitions += n_transitions
        n_soc_illegal += len(violations)
        phase_order_report.extend(violations)

    n_segments = len(all_segments)
    summary = {
        "n_segments": n_segments,
        "n_unknown_action_segments": sum(1 for s in all_segments if s["unknown_actions"]),
        "requirement_satisfaction_pct": round(100 * n_rs_ok / n_segments, 2),
        "acts_on_consistency_pct": round(100 * n_aoc_ok / n_segments, 2),
        "contradiction_rate_pct": round(100 * n_cr_violation / n_segments, 2),
        "step_ordering_consistency_pct": (
            round(100 * (n_soc_transitions - n_soc_illegal) / n_soc_transitions, 2)
            if n_soc_transitions else None
        ),
        "n_phase_transitions_checked": n_soc_transitions,
        "n_phase_order_violations": n_soc_illegal,
        "n_segments_with_violations": len(per_segment_report),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "qc_report.json").write_text(json.dumps({
        "segment_violations": per_segment_report,
        "phase_order_violations": phase_order_report,
    }, indent=2))
    (args.out_dir / "qc_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nFull violation list -> {args.out_dir / 'qc_report.json'}")


if __name__ == "__main__":
    main()
