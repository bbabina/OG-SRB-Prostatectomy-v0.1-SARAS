from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.common import build_prediction_record  # noqa: E402

MODEL_NAME = "clip-linear-probe"


def load_split_matrix(reports_dir: Path, features_dir: Path, annotations_dir: Path,
                       split: str, action_ids: list[str]):
    """Return (segment_ids, X [n,768], Y [n,21] multi-hot, phases [n] str|None)
    for one split, skipping any segment missing an embedding."""
    segments_path = reports_dir / f"segments_{split}.json"
    segments = json.loads(segments_path.read_text())
    action_index = {a: i for i, a in enumerate(action_ids)}

    seg_ids, X, Y, phases = [], [], [], []
    for seg in segments:
        emb_path = features_dir / split / seg["video"] / f"{seg['segment_id']}.npz"
        ann_path = annotations_dir / split / seg["video"] / f"{seg['segment_id']}.json"
        if not emb_path.exists() or not ann_path.exists():
            continue
        embedding = np.load(emb_path)["embedding"]
        gt = json.loads(ann_path.read_text())

        labels = np.zeros(len(action_ids), dtype=np.float32)
        for a in gt["actions"]:
            if a in action_index:
                labels[action_index[a]] = 1.0

        seg_ids.append(seg["segment_id"])
        X.append(embedding)
        Y.append(labels)
        phases.append(gt["phase"])

    return seg_ids, np.stack(X), np.stack(Y), phases


def predict_actions(probs: np.ndarray, action_ids: list[str], threshold: float) -> list[str]:
    predicted = [action_ids[i] for i, p in enumerate(probs) if p > threshold]
    if not predicted:
        predicted = [action_ids[int(np.argmax(probs))]]  # fallback: always predict something
    return predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--features-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "features")
    parser.add_argument("--annotations-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "annotations")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs" / "clip_linear_probe")
    parser.add_argument("--models-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "models")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    phase_ids = sorted(onto.phase_ids)
    phase_index = {p: i for i, p in enumerate(phase_ids)}

    print("Loading train split (real1, real2, real4) embeddings + labels ...")
    train_ids, X_train, Y_train, phases_train = load_split_matrix(
        args.reports_dir, args.features_dir, args.annotations_dir, "train", action_ids,
    )
    print(f"  {X_train.shape[0]} segments, {X_train.shape[1]}-d embeddings, {Y_train.shape[1]} action classes")
    print("  positive examples per class:",
          {a: int(Y_train[:, i].sum()) for i, a in enumerate(action_ids)})

    print("Training one-vs-rest logistic regression (class_weight=balanced) ...")
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    )
    clf.fit(X_train, Y_train)

    print("Training phase classifier (single softmax model, phase-bearing segments only) ...")
    phase_mask = [p is not None for p in phases_train]
    X_train_phase = X_train[phase_mask]
    y_train_phase = np.array([phase_index[p] for p in phases_train if p is not None])
    print("  segments per phase:",
          {p: int((y_train_phase == i).sum()) for p, i in phase_index.items()})
    clf_phase = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf_phase.fit(X_train_phase, y_train_phase)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "clip_linear_probe.joblib"
    joblib.dump({"clf": clf, "action_ids": action_ids, "clf_phase": clf_phase, "phase_ids": phase_ids}, model_path)
    print(f"Saved trained probe -> {model_path}")

    print("Scoring val split (real3, held out) ...")
    segments_val = {s["segment_id"]: s for s in json.loads((args.reports_dir / "segments_val.json").read_text())}
    val_ids, X_val, Y_val, _phases_val = load_split_matrix(
        args.reports_dir, args.features_dir, args.annotations_dir, "val", action_ids,
    )
    probs = clf.predict_proba(X_val)  # [n_val, 21], independent per-class sigmoid outputs
    phase_probs_matrix = clf_phase.predict_proba(X_val)  # [n_val, <=7], one softmax dist per segment

    n_written = 0
    for seg_id, prob_row, phase_prob_row in zip(val_ids, probs, phase_probs_matrix):
        seg = segments_val[seg_id]
        action_probs = {a: round(float(p), 4) for a, p in zip(action_ids, prob_row)}
        pred_actions = predict_actions(prob_row, action_ids, args.threshold)
        record = build_prediction_record(seg, MODEL_NAME, pred_actions, action_probs, onto)
        # phase_probs is a real softmax distribution over phase_ids (via clf_phase.classes_,
        # since a phase can be absent from y_train_phase if it had zero training examples).
        phase_probs = {phase_ids[c]: round(float(p), 4) for c, p in zip(clf_phase.classes_, phase_prob_row)}
        record["phase_probs"] = {p: phase_probs.get(p, 0.0) for p in phase_ids}
        out_path = args.out_dir / "val" / seg["video"] / f"{seg_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        n_written += 1

    print(f"Wrote {n_written} val predictions -> {args.out_dir / 'val'}")


if __name__ == "__main__":
    main()
