"""Leave-one-video-out (LOVO) evaluation for the linear probe + decoder.

Per supervisor guidance: a single fixed train/val split (real1+real2+real4
train, real3 val) only tells you how the model does on one specific held-out
video. With only 4 procedures total and real videos differing in lighting,
anatomy, and camera angle, that's a thin basis for a robustness claim. LOVO
retrains 4 times, holding out a different one of the 4 videos each time, and
reports the spread across folds -- if performance swings wildly between
folds, that's itself an important finding (the model doesn't generalize
across procedures), not something a single train/val split could reveal.

No new embeddings are needed: features/<split>/<video>/*.npz already covers
every video regardless of which "official" split (train/val folder) it sits
under, so this just re-partitions the existing embeddings by video for each
fold and retrains the same architecture as clip_linear_probe.py (one-vs-rest
action classifier + softmax phase classifier), then runs the same decoder
logic and scores with the same eval/metrics.py functions used everywhere
else in this project.

Output: reports/lovo_summary.json (per-fold metrics + aggregate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from eval.evaluator import evaluate as evaluate_predictions  # noqa: E402
from baselines.common import build_prediction_record  # noqa: E402
from baselines.clip_linear_probe import predict_actions  # noqa: E402
from baselines.decoder_constrained import viterbi_decode_phases, decode_segment  # noqa: E402

ALL_VIDEOS = ["real1", "real2", "real3", "real4"]


def load_all_segments(reports_dir: Path, features_dir: Path, annotations_dir: Path,
                       action_ids: list[str]) -> list[dict]:
    """One record per segment, covering both the train/ and val/ folders,
    each carrying its own embedding and ground truth -- independent of
    which "official" split it originally belonged to."""
    action_index = {a: i for i, a in enumerate(action_ids)}
    records = []
    for split in ("train", "val"):
        segments_path = reports_dir / f"segments_{split}.json"
        if not segments_path.exists():
            continue
        for seg in json.loads(segments_path.read_text()):
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

            records.append({
                "segment_id": seg["segment_id"],
                "video": seg["video"],
                "embedding": embedding,
                "action_labels": labels,
                "phase": gt["phase"],
                "seg_meta": seg,
                "gt": gt,
            })
    return records


def run_fold(held_out_video: str, records: list[dict], action_ids: list[str],
             phase_ids: list[str], onto) -> dict:
    phase_index = {p: i for i, p in enumerate(phase_ids)}

    train_records = [r for r in records if r["video"] != held_out_video]
    val_records = [r for r in records if r["video"] == held_out_video]

    X_train = np.stack([r["embedding"] for r in train_records])
    Y_train = np.stack([r["action_labels"] for r in train_records])

    action_clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
    action_clf.fit(X_train, Y_train)

    phase_mask = [r["phase"] is not None for r in train_records]
    X_train_phase = X_train[phase_mask]
    y_train_phase = np.array([phase_index[r["phase"]] for r in train_records if r["phase"] is not None])
    phase_clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    phase_clf.fit(X_train_phase, y_train_phase)

    X_val = np.stack([r["embedding"] for r in val_records])
    action_probs_matrix = action_clf.predict_proba(X_val)
    phase_probs_matrix = phase_clf.predict_proba(X_val)

    pred_by_id = {}
    for r, prob_row, phase_prob_row in zip(val_records, action_probs_matrix, phase_probs_matrix):
        action_probs = {a: round(float(p), 4) for a, p in zip(action_ids, prob_row)}
        pred_actions = predict_actions(prob_row, action_ids, threshold=0.5)
        record = build_prediction_record(r["seg_meta"], "clip-linear-probe-lovo", pred_actions, action_probs, onto)
        # phase_clf.classes_ may omit a phase with 0 training examples in this fold, so build
        # from clf.classes_ then fill any missing phase with 0.0 rather than assuming all 7 present.
        raw_phase_probs = {phase_ids[c]: round(float(p), 4) for c, p in zip(phase_clf.classes_, phase_prob_row)}
        record["phase_probs"] = {p: raw_phase_probs.get(p, 0.0) for p in phase_ids}
        pred_by_id[r["segment_id"]] = record

    # decode (Stage A + B), same logic decoder_constrained.py uses
    decoded_phases = viterbi_decode_phases(list(pred_by_id.values()), onto)
    decoded_by_id = {}
    for seg_id, pred in pred_by_id.items():
        decoded_phase = decoded_phases.get(seg_id)
        decoded_by_id[seg_id] = decode_segment(pred, decoded_phase, onto)

    gt_by_id = {r["segment_id"]: r["gt"] for r in val_records}

    raw_result = evaluate_predictions(gt_by_id, pred_by_id, onto)
    decoded_result = evaluate_predictions(gt_by_id, decoded_by_id, onto)

    return {
        "held_out_video": held_out_video,
        "n_train_segments": len(train_records),
        "n_val_segments": len(val_records),
        "raw": {k: v for k, v in raw_result.items() if k != "per_class_f1"},
        "decoded": {k: v for k, v in decoded_result.items() if k != "per_class_f1"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--features-dir", type=Path, default=Path(__file__).resolve().parent.parent / "features")
    parser.add_argument("--annotations-dir", type=Path, default=Path(__file__).resolve().parent.parent / "annotations")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    phase_ids = sorted(onto.phase_ids)

    print("Loading all segments (train+val combined) with embeddings + ground truth ...")
    records = load_all_segments(args.reports_dir, args.features_dir, args.annotations_dir, action_ids)
    counts = {v: sum(1 for r in records if r["video"] == v) for v in ALL_VIDEOS}
    print(f"  {len(records)} segments total, per video: {counts}")

    fold_results = []
    for video in ALL_VIDEOS:
        print(f"\n=== Fold: hold out {video} ===")
        result = run_fold(video, records, action_ids, phase_ids, onto)
        fold_results.append(result)
        print(f"  raw     macro_f1={result['raw']['macro_f1']}  top1={result['raw']['top1_accuracy']}  "
              f"SOC={result['raw']['step_ordering_consistency_pct']}")
        print(f"  decoded macro_f1={result['decoded']['macro_f1']}  top1={result['decoded']['top1_accuracy']}  "
              f"SOC={result['decoded']['step_ordering_consistency_pct']}")

    def agg(key_path, stage):
        vals = [r[stage][key_path] for r in fold_results if r[stage][key_path] is not None]
        return {"mean": round(sum(vals) / len(vals), 4), "min": round(min(vals), 4), "max": round(max(vals), 4)} if vals else None

    summary = {
        "folds": fold_results,
        "aggregate": {
            "raw": {
                "macro_f1": agg("macro_f1", "raw"),
                "top1_accuracy": agg("top1_accuracy", "raw"),
                "step_ordering_consistency_pct": agg("step_ordering_consistency_pct", "raw"),
                "temporal_coherence": agg("temporal_coherence", "raw"),
            },
            "decoded": {
                "macro_f1": agg("macro_f1", "decoded"),
                "top1_accuracy": agg("top1_accuracy", "decoded"),
                "step_ordering_consistency_pct": agg("step_ordering_consistency_pct", "decoded"),
                "temporal_coherence": agg("temporal_coherence", "decoded"),
            },
        },
    }

    out_path = args.reports_dir / "lovo_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Aggregate across {len(fold_results)} folds ===")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    main()
