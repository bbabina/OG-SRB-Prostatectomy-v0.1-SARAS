from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.common import build_prediction_record  # noqa: E402

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
            action_probs = {a: round(float(p), 4) for a, p in zip(action_ids, probs)}
            record = build_prediction_record(
                seg, f"clip-{MODEL_NAME}-{PRETRAINED}", pred_actions, action_probs, onto,
            )
            out_path = args.out_dir / split / seg["video"] / f"{seg['segment_id']}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(record, indent=2))
            n_done += 1

        print(f"[{split}] predicted {n_done}/{len(segments)} segments (rest missing embeddings)")
        total += n_done

    print(f"Total CLIP predictions written: {total}")


if __name__ == "__main__":
    main()
