"""Localization experiment: compute a CLIP embedding for every individual
bounding-box instance, instead of one embedding for the whole frame/segment.

Motivation (see PROGRESS_REPORT.md): checking what the whole-frame linear
probe actually confuses shows a specific pattern -- e.g. true
PullingBladderNeck segments get predicted as PullingTissue 65% of the time.
The verb ("pulling") is right, the tissue is wrong. That's consistent with
the model not being able to resolve fine anatomical detail from a
720x576 frame (mostly background/other instruments) squeezed down to
CLIP's 224x224 input. The dataset already annotates exactly where the
relevant action is happening -- each labelled instance is one bounding box
(mean ~21% of frame area, so a meaningful zoom-in, not a sliver) -- and
that signal has gone unused so far.

Every bounding box has exactly one action label (verified: box/label line
counts match in all 25,390 frame annotation files), so this produces
clean single-label training data per crop, unlike the whole-frame
approach where one embedding had to represent every action label in a
segment's ~15-frame window at once (40% of segments contain more than one
distinct action across their frames).

Caveat (documented, not hidden): at evaluation time this uses the
dataset's ground-truth box coordinates. A real deployed system without
oracle annotations would need an object/instrument detector to produce
these boxes first -- this experiment measures "how good is action
recognition given known localization," a different and easier question
than the whole-frame model's "recognize blind." Both are reported.

Output: one consolidated file per split (not one file per crop, to avoid
tens of thousands of tiny files):
  features_bbox/<split>_crops.npz  with parallel arrays:
    embeddings   [N, 768] float32, L2-normalized
    labels       [N] str  (the crop's single action label)
    segment_ids  [N] str  (which segment this crop's frame belongs to)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

MODEL_NAME = "ViT-L-14-quickgelu"
PRETRAINED = "openai"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_frame_to_segment(segments: list[dict]) -> dict[tuple[str, int], str]:
    lookup = {}
    for seg in segments:
        for frame_num in seg["frames"]:
            lookup[(seg["video"], frame_num)] = seg["segment_id"]
    return lookup


def read_boxes(ann_dir: Path, video: str, frame_num: int) -> list[tuple[tuple[int, int, int, int], str]]:
    box_path = ann_dir / f"{video}_frame_{frame_num}.bboxes.tsv"
    label_path = ann_dir / f"{video}_frame_{frame_num}.bboxes.labels.tsv"
    if not box_path.exists():
        return []
    box_lines = [l for l in box_path.read_text().splitlines() if l.strip()]
    label_lines = [l for l in label_path.read_text().splitlines() if l.strip()]
    out = []
    for box_line, label in zip(box_lines, label_lines):
        x1, y1, x2, y2 = (int(v) for v in box_line.split())
        if x2 <= x1 or y2 <= y1:
            continue  # skip degenerate boxes
        out.append(((x1, y1, x2, y2), label))
    return out


def pad_box(box: tuple[int, int, int, int], pad_frac: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Expand a box by pad_frac * (width, height) on each side, clipped to
    the image bounds. pad_frac=0.5 means each side grows by half the box's
    own width/height, so the padded box is up to 2x as wide/tall."""
    x1, y1, x2, y2 = box
    pad_w = (x2 - x1) * pad_frac
    pad_h = (y2 - y1) * pad_frac
    return (
        max(0, int(x1 - pad_w)),
        max(0, int(y1 - pad_h)),
        min(img_w, int(x2 + pad_w)),
        min(img_h, int(y2 + pad_h)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesad-root", type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / "mesad-real 2")
    parser.add_argument("--reports-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "features_bbox")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--padding-frac", type=float, default=0.0,
                         help="Expand each box by this fraction of its own width/height per side "
                              "before cropping, to keep some surrounding anatomical context "
                              "(0.0 = tight box, unchanged from the original crop experiment).")
    args = parser.parse_args()

    device = pick_device()
    print(f"Loading {MODEL_NAME} ({PRETRAINED}) on {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        segments_path = args.reports_dir / f"segments_{split}.json"
        if not segments_path.exists():
            print(f"skip {split}: {segments_path} not found")
            continue
        segments = json.loads(segments_path.read_text())
        frame_to_segment = build_frame_to_segment(segments)
        ann_dir = args.mesad_root / split / "annotations"

        # Collect every (crop_source, label, segment_id) first, so encoding
        # can be batched at a fixed size regardless of frame/box boundaries.
        todo = []
        for (video, frame_num), seg_id in frame_to_segment.items():
            for (x1, y1, x2, y2), label in read_boxes(ann_dir, video, frame_num):
                todo.append((video, frame_num, (x1, y1, x2, y2), label, seg_id))

        print(f"[{split}] {len(frame_to_segment)} frames -> {len(todo)} box instances to encode")

        images_dir = args.mesad_root / split / "images"
        embeddings = np.zeros((len(todo), model.visual.output_dim), dtype=np.float32)
        labels = []
        seg_ids = []

        for i in tqdm(range(0, len(todo), args.batch_size), desc=split):
            batch = todo[i:i + args.batch_size]
            crops = []
            for video, frame_num, (x1, y1, x2, y2), label, seg_id in batch:
                img_path = images_dir / f"{video}_frame_{frame_num}.jpg"
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    if args.padding_frac > 0:
                        x1, y1, x2, y2 = pad_box((x1, y1, x2, y2), args.padding_frac, im.width, im.height)
                    crop = im.crop((x1, y1, x2, y2))
                crops.append(preprocess(crop))
            images = torch.stack(crops).to(device)
            with torch.no_grad():
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy().astype(np.float32)

            embeddings[i:i + len(batch)] = feats
            labels.extend(label for *_, label, _ in batch)
            seg_ids.extend(seg_id for *_, seg_id in batch)

        out_path = args.out_dir / f"{split}_crops.npz"
        np.savez(out_path, embeddings=embeddings,
                 labels=np.array(labels, dtype=object),
                 segment_ids=np.array(seg_ids, dtype=object))
        print(f"[{split}] saved {len(todo)} crop embeddings -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
