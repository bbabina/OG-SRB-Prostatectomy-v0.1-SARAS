from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.common import build_prediction_record  # noqa: E402

DEFAULT_PROBE_WEIGHT = 0.95  # frozen; tuned on the 1358 dense val segments outside Test-180


def load_segments(root: Path) -> dict[str, dict]:
    by_id = {}
    for path in root.glob("*/*/*.json"):
        seg = json.loads(path.read_text())
        by_id[seg["segment_id"]] = seg
    return by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt-model", required=True, help="subfolder under vlm_outputs/")
    parser.add_argument("--probe-model", required=True, help="subfolder under vlm_outputs/")
    parser.add_argument("--vlm-outputs-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--split", default="val")
    parser.add_argument("--manifest", type=Path, default=None,
                         help="Optional: restrict to a manifest's segment_ids (e.g. Test-180).")
    parser.add_argument("--probe-weight", type=float, default=DEFAULT_PROBE_WEIGHT)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())

    gpt = load_segments(args.vlm_outputs_dir / args.gpt_model)
    probe = load_segments(args.vlm_outputs_dir / args.probe_model)

    ids = {k for k, v in gpt.items() if v.get("split") == args.split}
    ids &= {k for k, v in probe.items() if v.get("split") == args.split}
    if args.manifest:
        manifest_ids = set(json.loads(args.manifest.read_text())["segment_ids"])
        ids &= manifest_ids

    w_probe = args.probe_weight
    w_gpt = 1 - w_probe
    n = 0
    for sid in sorted(ids):
        g = gpt[sid]["action_probs"]
        c = probe[sid]["action_probs"]
        g_sum = sum(g.values()) or 1.0
        c_sum = sum(c.values()) or 1.0
        combined = {a: w_gpt * (g.get(a, 0.0) / g_sum) + w_probe * (c.get(a, 0.0) / c_sum)
                    for a in action_ids}
        ranked = sorted(action_ids, key=lambda a: -combined[a])
        actions = [a for a in ranked[:3] if combined[a] > 0.05] or [ranked[0]]

        seg = gpt[sid]
        record = build_prediction_record(seg, "gpt_clip_ensemble", actions, combined, onto)
        record["ensemble_probe_weight"] = w_probe
        record["ensemble_sources"] = {"gpt_model": args.gpt_model, "probe_model": args.probe_model}

        out_path = args.out_dir / args.split / seg["video"] / f"{sid}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        n += 1

    print(f"Wrote {n} ensemble predictions -> {args.out_dir / args.split}")
    print(f"Weight: {w_probe:.2f} probe / {w_gpt:.2f} GPT")


if __name__ == "__main__":
    main()
