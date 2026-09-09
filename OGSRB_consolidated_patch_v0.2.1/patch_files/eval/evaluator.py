"""Evaluate model predictions against action ground truth and ontology v0.2.

Important protocol distinction:
- Full/dense video predictions may be evaluated with temporal metrics.
- Sparse Test-180 is for fair cross-model perception comparison. Its manifest sets
  temporal_evaluation=false, so phase-transition/temporal metrics are suppressed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from eval.metrics import (  # noqa: E402
    check_requires,
    check_acts_on,
    check_contradicts,
    check_phase_order,
    schema_vocabulary_conformance,
    temporal_coherence,
    macro_f1,
    top1_accuracy,
)


def load_segments(root: Path) -> dict[str, dict]:
    by_id = {}
    for path in root.glob("*/*/*.json"):
        seg = json.loads(path.read_text())
        by_id[seg["segment_id"]] = seg
    return by_id


def load_manifest(path: Path | None) -> tuple[set[str] | None, dict]:
    if path is None:
        return None, {}
    data = json.loads(path.read_text())
    ids = set(data.get("segment_ids", []))
    if not ids and data.get("items"):
        ids = {item["segment_id"] for item in data["items"]}
    if not ids:
        raise SystemExit(f"Manifest has no segment ids: {path}")
    return ids, data


def evaluate(
    gt_by_id: dict[str, dict],
    pred_by_id: dict[str, dict],
    onto,
    temporal_evaluation: bool = True,
) -> dict:
    matched_ids = sorted(set(gt_by_id) & set(pred_by_id))
    if not matched_ids:
        raise SystemExit("No overlapping segment_ids between ground truth and predictions.")

    n_tool_ok = n_target_ok = n_rule_violation = 0
    vocab_sum = 0.0

    for seg_id in matched_ids:
        pred = pred_by_id[seg_id]
        if not check_requires(pred, onto):
            n_tool_ok += 1
        if not check_acts_on(pred, onto):
            n_target_ok += 1
        if onto.contradicts and check_contradicts(pred, onto):
            n_rule_violation += 1
        vocab_sum += schema_vocabulary_conformance(pred, onto)

    n = len(matched_ids)

    proxy_phase_consistency = None
    tc_score = None
    n_phase_transitions = 0

    if temporal_evaluation:
        by_video: dict[str, list[str]] = {}
        for seg_id in matched_ids:
            by_video.setdefault(pred_by_id[seg_id]["video"], []).append(seg_id)

        n_illegal = 0
        tc_scores = []
        for _, seg_ids in by_video.items():
            pred_segs = [pred_by_id[s] for s in seg_ids]
            violations, n_transitions = check_phase_order(pred_segs, onto)
            n_phase_transitions += n_transitions
            n_illegal += len(violations)

            ordered = sorted(seg_ids, key=lambda s: pred_by_id[s]["frame_start"])
            pred_phase_seq = [
                pred_by_id[s]["phase"]
                for s in ordered
                if pred_by_id[s].get("phase")
            ]
            true_phase_seq = [
                gt_by_id[s]["phase"]
                for s in ordered
                if gt_by_id[s].get("phase")
            ]
            tc_scores.append(temporal_coherence(pred_phase_seq, true_phase_seq))

        proxy_phase_consistency = (
            round(
                100 * (n_phase_transitions - n_illegal) / n_phase_transitions,
                2,
            )
            if n_phase_transitions
            else None
        )
        tc_score = (
            round(sum(tc_scores) / len(tc_scores), 4) if tc_scores else None
        )

    action_classes = sorted(onto.action_by_id.keys())
    pred_actions_by_id = {s: pred_by_id[s].get("actions", []) for s in matched_ids}
    gt_actions_by_id = {s: gt_by_id[s].get("actions", []) for s in matched_ids}
    f1_result = macro_f1(pred_actions_by_id, gt_actions_by_id, action_classes)

    pred_top1_by_id = {}
    for s in matched_ids:
        pred = pred_by_id[s]
        if "top1_prediction" in pred:
            pred_top1_by_id[s] = pred["top1_prediction"]
        elif pred.get("actions"):
            probs = pred.get("action_probs", {})
            if probs:
                pred_top1_by_id[s] = max(
                    pred["actions"], key=lambda a: probs.get(a, 0.0)
                )
            else:
                pred_top1_by_id[s] = pred["actions"][0]
        else:
            pred_top1_by_id[s] = None

    top1 = top1_accuracy(pred_top1_by_id, gt_actions_by_id)

    rule_rate = (
        round(100 * n_rule_violation / n, 2) if onto.contradicts else None
    )

    return {
        "n_segments_evaluated": n,
        "schema_vocabulary_conformance_pct": round(100 * vocab_sum / n, 2),
        "canonical_tool_mapping_conformance_pct": round(100 * n_tool_ok / n, 2),
        "canonical_target_mapping_conformance_pct": round(100 * n_target_ok / n, 2),
        "validated_rule_violation_rate_pct": rule_rate,
        "validated_rule_status": (
            "evaluated" if onto.contradicts else "not_applicable_no_validated_rules"
        ),
        "proxy_phase_transition_consistency_pct": proxy_phase_consistency,
        "temporal_coherence": tc_score,
        "temporal_metrics_enabled": temporal_evaluation,
        "n_proxy_phase_transitions": n_phase_transitions if temporal_evaluation else None,
        "macro_f1": round(f1_result["macro_f1"], 4),
        "top1_accuracy": round(top1, 4),
        "per_class_f1": {
            k: round(v["f1"], 3) for k, v in f1_result["per_class"].items()
        },
        "per_class_support": {
            k: int(v["support"]) for k, v in f1_result["per_class"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "annotations",
    )
    parser.add_argument(
        "--vlm-outputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "vlm_outputs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "reports",
    )
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--split", choices=["train", "val", "both"], default="val")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional fixed benchmark manifest (e.g. benchmarks/test180_v1.json).",
    )
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    gt_by_id = load_segments(args.annotations_dir)
    pred_by_id = load_segments(args.vlm_outputs_dir / args.model)

    if args.split != "both":
        gt_by_id = {k: v for k, v in gt_by_id.items() if v["split"] == args.split}
        pred_by_id = {
            k: v for k, v in pred_by_id.items() if v["split"] == args.split
        }

    manifest_ids, manifest = load_manifest(args.manifest)
    if manifest_ids is not None:
        gt_by_id = {k: v for k, v in gt_by_id.items() if k in manifest_ids}
        pred_by_id = {k: v for k, v in pred_by_id.items() if k in manifest_ids}

    temporal_evaluation = True
    if manifest:
        temporal_evaluation = bool(manifest.get("temporal_evaluation", False))

    result = evaluate(
        gt_by_id,
        pred_by_id,
        onto,
        temporal_evaluation=temporal_evaluation,
    )
    result["model"] = args.model
    result["split"] = args.split
    result["ontology_version"] = str(onto.raw.get("meta", {}).get("version", "unknown"))
    result["ontology_path"] = str(args.ontology)
    result["manifest"] = str(args.manifest) if args.manifest else None
    result["evaluation_scope"] = (
        manifest.get("name", "manifest_subset") if manifest else "dense_split"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{manifest.get('name')}" if manifest else ""
    safe_suffix = suffix.replace(" ", "_").replace("/", "_")
    out_path = args.out_dir / f"metrics_{args.model}_{args.split}{safe_suffix}.json"
    out_path.write_text(json.dumps(result, indent=2))

    summary = {
        k: v
        for k, v in result.items()
        if k not in ("per_class_f1", "per_class_support")
    }
    print(json.dumps(summary, indent=2))
    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    main()
