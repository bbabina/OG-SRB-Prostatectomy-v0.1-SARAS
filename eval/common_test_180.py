# Common Test-180: score every model (raw and decoder-corrected) on the exact
# same 180-segment stratified sample used for LLaVA/Gemini, so the five models
# are compared on identical data rather than CLIP-family models' full 1538-
# segment val set vs. the VLMs' 180-segment subsample.

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from eval.evaluator import load_segments, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

MODELS = ["clip", "clip_linear_probe", "clip_bbox_padded_probe", "llava_baseline", "gemini_baseline"]


def common_180_ids() -> set[str]:
    # llava_baseline and gemini_baseline were both built from the same
    # seed=0 stratified sample (build_stratified_sample in llava_baseline.py),
    # so their segment_ids on disk are already the same 180-id set.
    ids = {json.loads(p.read_text())["segment_id"]
           for p in (ROOT / "vlm_outputs" / "llava_baseline" / "val").glob("*/*.json")}
    gemini_ids = {json.loads(p.read_text())["segment_id"]
                  for p in (ROOT / "vlm_outputs" / "gemini_baseline" / "val").glob("*/*.json")}
    assert ids == gemini_ids, "llava_baseline and gemini_baseline sample sets differ"
    return ids


def main():
    onto = load_ontology(DEFAULT_ONTOLOGY_PATH)
    ids180 = common_180_ids()
    print(f"Common Test-180: {len(ids180)} segments\n")

    gt_by_id_full = load_segments(ROOT / "annotations")
    gt_by_id = {k: v for k, v in gt_by_id_full.items() if k in ids180}

    rows = []
    for model in MODELS:
        for variant, folder in (("raw", model), ("decoded", f"{model}_decoded")):
            pred_dir = ROOT / "vlm_outputs" / folder
            pred_by_id_full = load_segments(pred_dir)
            pred_by_id = {k: v for k, v in pred_by_id_full.items()
                          if k in ids180 and v["split"] == "val"}
            missing = ids180 - set(pred_by_id)
            if missing:
                print(f"WARNING: {folder} missing {len(missing)}/{len(ids180)} of the 180 ids")
            result = evaluate(gt_by_id, pred_by_id, onto)
            result["model"] = model
            result["variant"] = variant
            rows.append(result)

    out_path = ROOT / "reports" / "common_test_180.json"
    out_path.write_text(json.dumps(rows, indent=2))

    header = f"{'model':<24}{'variant':<10}{'top1':>8}{'macroF1':>9}{'SOC':>8}{'TC':>8}{'OF':>8}{'AOC':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        soc = f"{r['step_ordering_consistency_pct']:.1f}" if r['step_ordering_consistency_pct'] is not None else "n/a"
        tc = f"{r['temporal_coherence']:.3f}" if r['temporal_coherence'] is not None else "n/a"
        top1 = f"{r['top1_accuracy']:.3f}" if r['top1_accuracy'] is not None else "n/a"
        print(f"{r['model']:<24}{r['variant']:<10}{top1:>8}{r['macro_f1']:>9.3f}{soc:>8}{tc:>8}"
              f"{r['ontology_factuality_pct']:>8.1f}{r['acts_on_consistency_pct']:>8.1f}")

    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    main()
