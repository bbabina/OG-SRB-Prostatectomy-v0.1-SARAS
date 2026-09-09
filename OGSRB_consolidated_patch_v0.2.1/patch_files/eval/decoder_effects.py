"""Quantify when the structured decoder helps and when it harms action decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from eval.metrics import macro_f1  # noqa: E402


def load_segments(root: Path) -> dict[str, dict]:
    out = {}
    for path in root.glob("*/*/*.json"):
        seg = json.loads(path.read_text())
        out[seg["segment_id"]] = seg
    return out


def load_manifest_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text())
    return set(data.get("segment_ids", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-model", required=True)
    parser.add_argument("--decoded-model", required=True)
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
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    gt = load_segments(args.annotations_dir)
    raw = load_segments(args.vlm_outputs_dir / args.raw_model)
    dec = load_segments(args.vlm_outputs_dir / args.decoded_model)

    ids = set(gt) & set(raw) & set(dec)
    manifest_ids = load_manifest_ids(args.manifest)
    if manifest_ids is not None:
        ids &= manifest_ids

    # Keep validation split for the primary experiment.
    ids = {sid for sid in ids if gt[sid].get("split") == "val"}
    if not ids:
        raise SystemExit("No matched validation segments.")

    classes = sorted(onto.action_by_id.keys())
    beneficial = harmful = changed = 0
    segments_changed = 0

    for sid in sorted(ids):
        gt_set = set(gt[sid].get("actions", []))
        raw_set = set(raw[sid].get("actions", []))
        dec_set = set(dec[sid].get("actions", []))

        if raw_set != dec_set:
            segments_changed += 1

        for cls in classes:
            raw_has = cls in raw_set
            dec_has = cls in dec_set
            if raw_has == dec_has:
                continue

            changed += 1
            gt_has = cls in gt_set
            raw_correct = raw_has == gt_has
            dec_correct = dec_has == gt_has

            if (not raw_correct) and dec_correct:
                beneficial += 1
            elif raw_correct and (not dec_correct):
                harmful += 1

    pred_raw = {sid: raw[sid].get("actions", []) for sid in ids}
    pred_dec = {sid: dec[sid].get("actions", []) for sid in ids}
    gt_actions = {sid: gt[sid].get("actions", []) for sid in ids}

    raw_f1 = macro_f1(pred_raw, gt_actions, classes)["macro_f1"]
    dec_f1 = macro_f1(pred_dec, gt_actions, classes)["macro_f1"]

    result = {
        "raw_model": args.raw_model,
        "decoded_model": args.decoded_model,
        "n_segments": len(ids),
        "segments_changed": segments_changed,
        "action_decisions_changed": changed,
        "beneficial_action_changes": beneficial,
        "harmful_action_changes": harmful,
        "correction_precision": round(beneficial / changed, 4) if changed else None,
        "correction_harm": round(harmful / changed, 4) if changed else None,
        "raw_macro_f1": round(raw_f1, 4),
        "decoded_macro_f1": round(dec_f1, 4),
        "delta_macro_f1": round(dec_f1 - raw_f1, 4),
        "manifest": str(args.manifest) if args.manifest else None,
        "interpretation": (
            "Correction precision/harm are computed over individual binary "
            "action-decision flips caused by the decoder."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"decoder_effects_{args.raw_model}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
