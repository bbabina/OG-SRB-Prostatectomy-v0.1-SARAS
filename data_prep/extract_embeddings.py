"""Step 1.3 (preprocessing): compute a CLIP ViT-L/14 image embedding for one
representative frame per segment.

For each segment produced by build_segments.py, picks the middle frame of
its frame list as the "representative frame" (per the project doc's Step 1:
"export a representative frame per segment"), encodes it with CLIP's image
tower, and saves the embedding to disk so Step 3 (clip_baseline.py) doesn't
need to re-run the image encoder for every prompt comparison.

Output: features/<split>/<video>/<segment_id>.npz  (keys: "embedding" [768],
"frame" [int, the representative frame number])
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

MODEL_NAME = "ViT-L-14-quickgelu"  # matches OpenAI's original CLIP activation (avoids a silent accuracy regression)
PRETRAINED = "openai"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


#CLIP only looks at one still image per segment, so this just picks the middle frame of the window as representative
def representative_frame(segment: dict) -> int:
    frames = segment["frames"]
    return frames[len(frames) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesad-root", type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / "mesad-real 2")
    parser.add_argument("--reports-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "features")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = pick_device()
    print(f"Loading {MODEL_NAME} ({PRETRAINED}) on {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    for split in ("train", "val"):
        segments_path = args.reports_dir / f"segments_{split}.json"
        if not segments_path.exists():
            print(f"skip {split}: {segments_path} not found (run build_segments.py first)")
            continue
        segments = json.loads(segments_path.read_text())
        images_dir = args.mesad_root / split / "images"

        todo = []
        for seg in segments:
            out_path = args.out_dir / split / seg["video"] / f"{seg['segment_id']}.npz"
            if out_path.exists():
                continue
            frame_num = representative_frame(seg)
            img_path = images_dir / f"{seg['video']}_frame_{frame_num}.jpg"
            todo.append((seg, frame_num, img_path, out_path))

        print(f"[{split}] {len(segments)} segments, {len(todo)} embeddings to compute")

        for i in tqdm(range(0, len(todo), args.batch_size), desc=split):
            batch = todo[i:i + args.batch_size]
            images = torch.stack([preprocess(Image.open(p).convert("RGB")) for _, _, p, _ in batch]).to(device)
            with torch.no_grad():
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy().astype(np.float32)

            for (seg, frame_num, _, out_path), emb in zip(batch, feats):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out_path, embedding=emb, frame=frame_num)

    print("Done.")


if __name__ == "__main__":
    main()
