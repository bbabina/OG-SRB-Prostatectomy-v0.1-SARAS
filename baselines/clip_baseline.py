"""Step 3 (VLM baselines): CLIP zero-shot open-vocabulary baseline.

For every segment, compares its precomputed CLIP image embedding (from
data_prep/extract_embeddings.py) against a fixed set of text prompts — one
per ontology action node — and predicts which actions are present via
cosine similarity. Tools/tissues/events/phase are then derived from the
predicted actions using the same ontology lookups map_annotations.py uses
for ground truth, so CLIP's predictions live in exactly the same schema as
the ground-truth annotations and can be scored by eval/evaluator.py.

Multi-label prediction rule (documented choice, no ground truth in the loop
at prediction time): take the softmax over the 21 action prompts, always
keep the top-1, and additionally keep any other action with probability
> --threshold, up to --max-actions total. Ground-truth segments average
~1.7 actions each (see project analysis), which is why these defaults
(threshold=0.15, max_actions=3) were picked as a reasonable starting point
— not tuned against ground truth (that would leak the test signal).

Output: vlm_outputs/clip/<split>/<video>/<segment_id>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402

MODEL_NAME = "ViT-L-14-quickgelu"
PRETRAINED = "openai"

TISSUE_DISPLAY = {
    "generic_tissue": "tissue",
    "smoke_plume": "smoke",
}

VERB_TEMPLATE = {
    "pulling": "pulling the {tissue}",
    "cutting": "cutting the {tissue}",
    "clip_application": "applying a clip to the {tissue}",
    "suction": "suctioning {tissue}",
    "dissection": "dissecting the {tissue}",
    "bagging": "placing the {tissue} into a specimen retrieval bag",
    "suturing": "suturing the {tissue}",
}


def tissue_display(tissue_id: str) -> str:
    return TISSUE_DISPLAY.get(tissue_id, tissue_id.replace("_", " "))


def build_prompt(action_node: dict) -> str:
    phrase = VERB_TEMPLATE[action_node["verb"]].format(tissue=tissue_display(action_node["tissue"]))
    return f"a photo from robotic-assisted radical prostatectomy surgery showing {phrase}"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_phase_from_actions(actions: list[str], onto) -> tuple[str | None, bool, list[str]]:
    """Equal-weight majority vote over predicted actions (no per-frame data
    available for predictions, unlike map_annotations.py's frame-weighted
    version for ground truth)."""
    votes = Counter(onto.phase_for(a) for a in actions if onto.phase_for(a))
    if not votes:
        return None, False, []
    phase = votes.most_common(1)[0][0]
    return phase, len(votes) > 1, sorted(votes.keys())


def predict_actions(probs: np.ndarray, action_ids: list[str], threshold: float, max_actions: int) -> list[str]:
    order = np.argsort(-probs)
    predicted = [action_ids[order[0]]]  # always keep top-1
    for idx in order[1:max_actions]:
        if probs[idx] > threshold:
            predicted.append(action_ids[idx])
    return predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "features")
    parser.add_argument("--reports-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs" / "clip")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--max-actions", type=int, default=3)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())

    device = pick_device()
    print(f"Loading {MODEL_NAME} ({PRETRAINED}) on {device} ...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()

    prompts = [build_prompt(onto.action_by_id[a]) for a in action_ids]
    print("Example prompts:")
    for a, p in list(zip(action_ids, prompts))[:3]:
        print(f"  {a}: \"{p}\"")

    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats_np = text_feats.cpu().numpy().astype(np.float32)  # [21, 768]
    logit_scale = model.logit_scale.exp().item()

    total = 0
    for split in ("train", "val"):
        segments_path = args.reports_dir / f"segments_{split}.json"
        if not segments_path.exists():
            print(f"skip {split}: {segments_path} not found")
            continue
        segments = json.loads(segments_path.read_text())

        n_done = 0
        for seg in segments:
            emb_path = args.features_dir / split / seg["video"] / f"{seg['segment_id']}.npz"
            if not emb_path.exists():
                continue
            image_emb = np.load(emb_path)["embedding"]  # [768], already L2-normalized

            sims = text_feats_np @ image_emb  # [21] cosine similarities
            logits = logit_scale * sims
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()  # standard CLIP zero-shot softmax

            pred_actions = predict_actions(probs, action_ids, args.threshold, args.max_actions)
            tools = sorted({onto.tool_for(a) for a in pred_actions if onto.tool_for(a)})
            tissues = sorted({onto.tissue_for(a) for a in pred_actions if onto.tissue_for(a)})
            events = sorted({onto.event_for(a) for a in pred_actions if onto.event_for(a)})
            phase, phase_ambiguous, phase_candidates = resolve_phase_from_actions(pred_actions, onto)

            record = {
                "segment_id": seg["segment_id"],
                "split": seg["split"],
                "video": seg["video"],
                "frame_start": seg["frame_start"],
                "frame_end": seg["frame_end"],
                "model": f"clip-{MODEL_NAME}-{PRETRAINED}",
                "actions": pred_actions,
                "action_probs": {a: round(float(p), 4) for a, p in zip(action_ids, probs)},
                "tools": tools,
                "tissues": tissues,
                "events": events,
                "phase": phase,
                "phase_ambiguous": phase_ambiguous,
                "phase_candidates": phase_candidates,
            }
            out_path = args.out_dir / split / seg["video"] / f"{seg['segment_id']}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(record, indent=2))
            n_done += 1

        print(f"[{split}] predicted {n_done}/{len(segments)} segments (rest missing embeddings)")
        total += n_done

    print(f"Total CLIP predictions written: {total}")


if __name__ == "__main__":
    main()
