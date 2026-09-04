"""Attach a calibrated phase classifier to zero-shot CLIP's existing val predictions.

Controlled ablation, not a new "zero-shot CLIP" baseline: this does NOT touch zero-shot
CLIP's actions/action_probs at all (still pure zero-shot, unchanged accuracy). It only
adds a `phase_probs` field, trained the same way clip_linear_probe.py trains its phase
classifier -- a single softmax LogisticRegression over CLIP embeddings, fit on the train
split's ground-truth phases. That lets the decoder use real calibrated phase evidence
for CLIP zero-shot instead of falling back to summing its raw (unreliable) action
probabilities per phase.

Purpose: isolate whether the decoder harms CLIP zero-shot because of its weak raw
accuracy, or specifically because it lacks calibrated phase evidence -- see
SUPERVISOR_REPLY_v0.2.1.md, "what I'd propose next."

Usage:
    python baselines/clip_calibrated_phase.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.clip_linear_probe import load_split_matrix  # noqa: E402

MODEL_NAME = "clip-zeroshot+calibrated-phase"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--features-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "features")
    parser.add_argument("--annotations-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "annotations")
    parser.add_argument("--clip-preds-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs" / "clip",
                         help="Any existing prediction set to attach calibrated phase_probs to "
                              "(actions/action_probs untouched) -- not necessarily CLIP itself.")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs" / "clip_calibrated_phase")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    phase_ids = sorted(onto.phase_ids)
    phase_index = {p: i for i, p in enumerate(phase_ids)}

    print("Loading train split embeddings + phase labels (action labels unused here) ...")
    _, X_train, _, phases_train = load_split_matrix(
        args.reports_dir, args.features_dir, args.annotations_dir, "train", action_ids,
    )
    phase_mask = [p is not None for p in phases_train]
    X_train_phase = X_train[phase_mask]
    y_train_phase = np.array([phase_index[p] for p in phases_train if p is not None])
    print("  segments per phase:",
          {p: int((y_train_phase == i).sum()) for p, i in phase_index.items()})

    print("Training phase classifier (identical recipe to clip_linear_probe.py) ...")
    clf_phase = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf_phase.fit(X_train_phase, y_train_phase)

    print("Scoring val split embeddings for phase_probs ...")
    val_ids, X_val, _, _ = load_split_matrix(
        args.reports_dir, args.features_dir, args.annotations_dir, "val", action_ids,
    )
    phase_probs_matrix = clf_phase.predict_proba(X_val)
    phase_probs_by_id = {}
    for seg_id, row in zip(val_ids, phase_probs_matrix):
        probs = {phase_ids[c]: round(float(p), 4) for c, p in zip(clf_phase.classes_, row)}
        phase_probs_by_id[seg_id] = {p: probs.get(p, 0.0) for p in phase_ids}

    print("Attaching phase_probs to existing zero-shot CLIP predictions (actions untouched) ...")
    n_written = n_missing = 0
    for pred_path in sorted(args.clip_preds_dir.glob("val/*/*.json")):
        record = json.loads(pred_path.read_text())
        seg_id = record["segment_id"]
        if seg_id not in phase_probs_by_id:
            n_missing += 1
            continue
        record["model"] = args.model_name
        record["phase_probs"] = phase_probs_by_id[seg_id]
        record["phase_probs_source"] = "trained_softmax_on_clip_embeddings_zeroshot_actions_unchanged"

        out_path = args.out_dir / "val" / record["video"] / f"{seg_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        n_written += 1

    print(f"Wrote {n_written} predictions -> {args.out_dir / 'val'}")
    if n_missing:
        print(f"({n_missing} zero-shot CLIP val predictions had no matching embedding, skipped)")


if __name__ == "__main__":
    main()
