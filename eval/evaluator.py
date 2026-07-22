"""Step 5/7: score a model's predictions against ground truth and against
the ontology's own rules, producing the metrics table from the project doc
(Section 5 Evaluation Design / Section 5 Expected Results table).

Usage:
    python3 eval/evaluator.py --model clip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from eval.metrics import (  # noqa: E402
    check_requires, check_acts_on, check_contradicts, check_phase_order,
    ontology_factuality, temporal_coherence, macro_f1, top1_accuracy,
)


def load_segments(root: Path) -> dict[str, dict]:
    """Load every segment JSON under root (any split/video layout) keyed by segment_id."""
    by_id = {}
    for path in root.glob("*/*/*.json"):
        seg = json.loads(path.read_text())
        by_id[seg["segment_id"]] = seg
    return by_id


def evaluate(gt_by_id: dict[str, dict], pred_by_id: dict[str, dict], onto) -> dict:
    matched_ids = sorted(set(gt_by_id) & set(pred_by_id))
    if not matched_ids:
        raise SystemExit("No overlapping segment_ids between ground truth and predictions.")

    n_rs_ok = n_aoc_ok = n_cr_violation = 0
    of_sum = 0.0

    for seg_id in matched_ids:
        pred = pred_by_id[seg_id]
        if not check_requires(pred, onto):
            n_rs_ok += 1
        if not check_acts_on(pred, onto):
            n_aoc_ok += 1
        if check_contradicts(pred, onto):
            n_cr_violation += 1
        of_sum += ontology_factuality(pred, onto)

    n = len(matched_ids)

    # SOC + TC are computed per video, over the predicted segments ordered
    # by frame_start, comparing against that same video's ground-truth
    # phase sequence.
    by_video: dict[str, list[str]] = {}
    for seg_id in matched_ids:
        by_video.setdefault(pred_by_id[seg_id]["video"], []).append(seg_id)

    n_soc_transitions = n_soc_illegal = 0
    tc_scores = []
    for video, seg_ids in by_video.items():
        pred_segs = [pred_by_id[s] for s in seg_ids]
        violations, n_transitions = check_phase_order(pred_segs, onto)
        n_soc_transitions += n_transitions
        n_soc_illegal += len(violations)

        ordered = sorted(seg_ids, key=lambda s: pred_by_id[s]["frame_start"])
        pred_phase_seq = [pred_by_id[s]["phase"] for s in ordered if pred_by_id[s].get("phase")]
        true_phase_seq = [gt_by_id[s]["phase"] for s in ordered if gt_by_id[s].get("phase")]
        tc_scores.append(temporal_coherence(pred_phase_seq, true_phase_seq))

    # Recognition metrics: predicted vs ground-truth actions, per segment.
    action_classes = sorted(onto.action_by_id.keys())
    pred_actions_by_id = {s: pred_by_id[s]["actions"] for s in matched_ids}
    gt_actions_by_id = {s: gt_by_id[s]["actions"] for s in matched_ids}
    f1_result = macro_f1(pred_actions_by_id, gt_actions_by_id, action_classes)

    pred_top1_by_id = {
        s: max(pred_by_id[s]["action_probs"], key=pred_by_id[s]["action_probs"].get)
        for s in matched_ids if "action_probs" in pred_by_id[s]
    }
    top1 = top1_accuracy(pred_top1_by_id, gt_actions_by_id) if pred_top1_by_id else None

    return {
        "n_segments_evaluated": n,
        "ontology_factuality_pct": round(100 * of_sum / n, 2),
        "step_ordering_consistency_pct": (
            round(100 * (n_soc_transitions - n_soc_illegal) / n_soc_transitions, 2)
            if n_soc_transitions else None
        ),
        "temporal_coherence": round(sum(tc_scores) / len(tc_scores), 4) if tc_scores else None,
        "requirement_satisfaction_pct": round(100 * n_rs_ok / n, 2),
        "acts_on_consistency_pct": round(100 * n_aoc_ok / n, 2),
        "contradiction_rate_pct": round(100 * n_cr_violation / n, 2),
        "macro_f1": round(f1_result["macro_f1"], 4),
        "top1_accuracy": round(top1, 4) if top1 is not None else None,
        "per_class_f1": {k: round(v["f1"], 3) for k, v in f1_result["per_class"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="subfolder under vlm_outputs/, e.g. 'clip'")
    parser.add_argument("--annotations-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "annotations")
    parser.add_argument("--vlm-outputs-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--split", choices=["train", "val", "both"], default="val")
    args = parser.parse_args()

    onto = load_ontology(args.ontology)

    gt_by_id = load_segments(args.annotations_dir)
    pred_by_id = load_segments(args.vlm_outputs_dir / args.model)

    if args.split != "both":
        gt_by_id = {k: v for k, v in gt_by_id.items() if v["split"] == args.split}
        pred_by_id = {k: v for k, v in pred_by_id.items() if v["split"] == args.split}

    result = evaluate(gt_by_id, pred_by_id, onto)
    result["model"] = args.model
    result["split"] = args.split

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"metrics_{args.model}_{args.split}.json"
    out_path.write_text(json.dumps(result, indent=2))

    summary = {k: v for k, v in result.items() if k != "per_class_f1"}
    print(json.dumps(summary, indent=2))
    print(f"\nFull report (incl. per-class F1) -> {out_path}")


if __name__ == "__main__":
    main()
