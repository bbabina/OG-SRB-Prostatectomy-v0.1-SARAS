"""Localization experiment: action classifier trained on bounding-box crops
instead of whole frames (see data_prep/extract_bbox_embeddings.py for the
motivation and the ground-truth-box caveat).

Two classifiers, different granularity, for a deliberate reason:

  - Action classifier: trained at the CROP level. Each box has exactly one
    label, so unlike the whole-frame probe (multi-label, one blurry
    embedding representing a whole ~15-frame window that may span more
    than one action 40% of the time), this is clean single-label
    multiclass classification -- a single softmax LogisticRegression over
    all 21 actions, one genuinely comparable distribution per crop.

  - Phase classifier: kept at the SEGMENT level, reusing the existing
    whole-frame multi-frame-pooled embeddings (extract_embeddings.py).
    Phase is a broader "which stage of the surgery" judgment, not tied to
    one small instrument-tissue region -- there's no reason to expect
    zooming into one box to help it, so the part of the pipeline that
    already works (see PROGRESS_REPORT.md decoder section) is left alone.

Segment-level action prediction: every crop in a segment gets scored by
the action classifier, then scores are MAX-pooled across all of a
segment's crops -- if any single crop strongly suggests an action, the
segment gets credit for it. This mirrors how ground truth itself is
defined (a segment's true actions are the UNION over all its frames'
box labels), so max-pooling is the natural aggregation, not an
arbitrary choice.

Output:
  - models/clip_bbox_probe.joblib
  - vlm_outputs/clip_bbox_probe/val/<video>/<segment_id>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.common import build_prediction_record  # noqa: E402
from baselines.clip_linear_probe import load_split_matrix  # noqa: E402
from baselines.clip_baseline import predict_actions  # noqa: E402


def train_phase_head(reports_dir: Path, features_dir: Path, annotations_dir: Path,
                      action_ids: list[str], phase_ids: list[str]):
    """Reuses the existing whole-frame segment embeddings for phase --
    identical approach to clip_linear_probe.py's phase classifier."""
    phase_index = {p: i for i, p in enumerate(phase_ids)}
    _, X_train, _, phases_train = load_split_matrix(
        reports_dir, features_dir, annotations_dir, "train", action_ids,
    )
    mask = [p is not None for p in phases_train]
    X_train_phase = X_train[mask]
    y_train_phase = np.array([phase_index[p] for p in phases_train if p is not None])
    clf_phase = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf_phase.fit(X_train_phase, y_train_phase)
    return clf_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--features-dir", type=Path, default=Path(__file__).resolve().parent.parent / "features")
    parser.add_argument("--features-bbox-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "features_bbox")
    parser.add_argument("--annotations-dir", type=Path, default=Path(__file__).resolve().parent.parent / "annotations")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs" / "clip_bbox_probe")
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--max-actions", type=int, default=3)
    parser.add_argument("--model-tag", type=str, default="clip-bbox-probe",
                         help="Name used for the saved model file and the 'model' field in output "
                              "records -- override so a padded-crop run doesn't overwrite the tight-crop one.")
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    action_index = {a: i for i, a in enumerate(action_ids)}
    phase_ids = sorted(onto.phase_ids)

    print("Loading train crop embeddings ...")
    train_crops = np.load(args.features_bbox_dir / "train_crops.npz", allow_pickle=True)
    X_crops = train_crops["embeddings"]
    y_crops = np.array([action_index[label] for label in train_crops["labels"]])
    print(f"  {X_crops.shape[0]} crop instances, {X_crops.shape[1]}-d embeddings")
    print("  crops per action:", {a: int((y_crops == i).sum()) for a, i in action_index.items()})

    print("Training crop-level action classifier (single softmax over 21 actions) ...")
    action_clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    action_clf.fit(X_crops, y_crops)

    print("Training phase classifier (whole-frame segment embeddings, unchanged approach) ...")
    phase_clf = train_phase_head(args.reports_dir, args.features_dir, args.annotations_dir, action_ids, phase_ids)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / f"{args.model_tag}.joblib"
    joblib.dump({"action_clf": action_clf, "phase_clf": phase_clf,
                 "action_ids": action_ids, "phase_ids": phase_ids}, model_path)
    print(f"Saved -> {model_path}")

    print("Scoring val split (real3, held out) ...")
    val_crops = np.load(args.features_bbox_dir / "val_crops.npz", allow_pickle=True)
    X_val_crops = val_crops["embeddings"]
    val_crop_segment_ids = val_crops["segment_ids"]

    # score every crop, then max-pool per segment
    crop_action_probs = action_clf.predict_proba(X_val_crops)  # [n_crops, 21], classes ordered per action_clf.classes_
    # action_clf.classes_ are the integer indices used in y_crops (0..20 matching action_ids order)
    ordered_probs = np.zeros_like(crop_action_probs)
    for col, cls in enumerate(action_clf.classes_):
        ordered_probs[:, cls] = crop_action_probs[:, col]

    segment_action_probs: dict[str, np.ndarray] = {}
    for seg_id, probs in zip(val_crop_segment_ids, ordered_probs):
        if seg_id not in segment_action_probs:
            segment_action_probs[seg_id] = probs.copy()
        else:
            segment_action_probs[seg_id] = np.maximum(segment_action_probs[seg_id], probs)

    segments_val = {s["segment_id"]: s for s in json.loads((args.reports_dir / "segments_val.json").read_text())}

    # phase_probs still come from the whole-frame segment embeddings
    val_ids_frame, X_val_frame, _, _ = load_split_matrix(
        args.reports_dir, args.features_dir, args.annotations_dir, "val", action_ids,
    )
    phase_probs_matrix = phase_clf.predict_proba(X_val_frame)
    phase_probs_by_seg = {}
    for seg_id, row in zip(val_ids_frame, phase_probs_matrix):
        phase_probs_by_seg[seg_id] = {phase_ids[c]: round(float(p), 4) for c, p in zip(phase_clf.classes_, row)}

    n_written = 0
    n_no_crops = 0
    for seg_id, seg in segments_val.items():
        if seg_id in segment_action_probs:
            probs = segment_action_probs[seg_id]
        else:
            n_no_crops += 1
            probs = np.ones(len(action_ids)) / len(action_ids)  # segment had no boxes at all; fallback uniform

        action_probs = {a: round(float(p), 4) for a, p in zip(action_ids, probs)}
        pred_actions = predict_actions(probs, action_ids, args.threshold, args.max_actions)
        record = build_prediction_record(seg, args.model_tag, pred_actions, action_probs, onto)
        record["phase_probs"] = phase_probs_by_seg.get(seg_id, {p: 0.0 for p in phase_ids})

        out_path = args.out_dir / "val" / seg["video"] / f"{seg_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        n_written += 1

    print(f"Wrote {n_written} val predictions ({n_no_crops} segments had no boxes, used fallback) -> {args.out_dir / 'val'}")


if __name__ == "__main__":
    main()
