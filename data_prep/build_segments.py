"""Step 1 (preprocessing): group raw annotated frames into segments.

The MESAD-Real dataset we have on disk contains no source video, only
already-extracted frames with per-frame bounding-box + action-label
annotations. Frame numbers are mostly consecutive but appear in bursts (the surgeons/annotators
only annotated frames during actions of interest), so there is no reliable
fps to derive true "2-second" windows from.


Output: reports/segments_<split>.json — one record per segment listing its
frames and the raw (dataset-native) action labels observed in each frame.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

FRAME_RE = re.compile(r"^(?P<video>real\d+)_frame_(?P<num>\d+)$")

DEFAULT_SEGMENT_LEN = 15   # frames per segment (documented assumption, see module docstring)
DEFAULT_MAX_GAP = 1        # max allowed frame-number gap to stay in the same contiguous run
DEFAULT_MIN_SEGMENT_LEN = 3  # drop trailing partial windows shorter than this


def discover_frames(split_dir: Path) -> dict[str, list[int]]:
    """Return {video_id: sorted [frame_num, ...]} for one split directory."""
    images_dir = split_dir / "images"
    by_video: dict[str, list[int]] = defaultdict(list)
    for img_path in images_dir.glob("*.jpg"):
        m = FRAME_RE.match(img_path.stem)
        if not m:
            continue
        by_video[m.group("video")].append(int(m.group("num")))
    for video in by_video:
        by_video[video].sort()
    return by_video


def contiguous_runs(frame_nums: list[int], max_gap: int) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for n in frame_nums:
        if current and n - current[-1] > max_gap:
            runs.append(current)
            current = []
        current.append(n)
    if current:
        runs.append(current)
    return runs


def chunk(run: list[int], segment_len: int, min_segment_len: int) -> list[list[int]]:
    chunks = [run[i:i + segment_len] for i in range(0, len(run), segment_len)]
    if len(chunks) > 1 and len(chunks[-1]) < min_segment_len:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks


def load_frame_labels(ann_dir: Path, video: str, frame_num: int) -> list[str]:
    labels_path = ann_dir / f"{video}_frame_{frame_num}.bboxes.labels.tsv"
    if not labels_path.exists():
        return []
    text = labels_path.read_text().strip()
    return [line for line in text.splitlines() if line]


def build_segments_for_split(mesad_root: Path, split: str, segment_len: int,
                              max_gap: int, min_segment_len: int) -> list[dict]:
    split_dir = mesad_root / split
    ann_dir = split_dir / "annotations"
    by_video = discover_frames(split_dir)

    segments = []
    for video, frame_nums in sorted(by_video.items()):
        runs = contiguous_runs(frame_nums, max_gap)
        for run in runs:
            for seg_idx, chunk_frames in enumerate(chunk(run, segment_len, min_segment_len)):
                per_frame_labels = {
                    fn: load_frame_labels(ann_dir, video, fn) for fn in chunk_frames
                }
                raw_labels = sorted({lbl for labels in per_frame_labels.values() for lbl in labels})
                segments.append({
                    "segment_id": f"{video}_seg{run[0]:06d}_{seg_idx:03d}",
                    "split": split,
                    "video": video,
                    "frame_start": chunk_frames[0],
                    "frame_end": chunk_frames[-1],
                    "frames": chunk_frames,
                    "frame_labels": per_frame_labels,
                    "raw_labels": raw_labels,
                })
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesad-root", type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / "mesad-real 2")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--segment-len", type=int, default=DEFAULT_SEGMENT_LEN)
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    parser.add_argument("--min-segment-len", type=int, default=DEFAULT_MIN_SEGMENT_LEN)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ("train", "val"):
        segments = build_segments_for_split(
            args.mesad_root, split, args.segment_len, args.max_gap, args.min_segment_len,
        )
        out_path = args.out_dir / f"segments_{split}.json"
        out_path.write_text(json.dumps(segments, indent=2))
        n_frames = sum(len(s["frames"]) for s in segments)
        print(f"[{split}] {len(segments)} segments, {n_frames} frames -> {out_path}")
        total += len(segments)

    print(f"Total segments: {total}")


if __name__ == "__main__":
    main()
