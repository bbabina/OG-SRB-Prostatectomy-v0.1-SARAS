# (ontology mapping): map raw MESAD-Real action labels onto ontology nodes for each segment produced by build_segments.py.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402


def map_segment(segment: dict, onto) -> dict:
    raw_labels = segment["raw_labels"]
    unknown = [a for a in raw_labels if a not in onto.action_by_id]

    tools = sorted({onto.tool_for(a) for a in raw_labels if onto.tool_for(a)})
    tissues = sorted({onto.tissue_for(a) for a in raw_labels if onto.tissue_for(a)})
    events = sorted({onto.event_for(a) for a in raw_labels if onto.event_for(a)})

    # phase resolution: count phase votes weighted by how many frames in the segment carry each action, so a phase supported by more frames wins.
    phase_votes = Counter()
    for frame_labels in segment["frame_labels"].values():
        for action in frame_labels:
            phase = onto.phase_for(action)
            if phase:
                phase_votes[phase] += 1

    if phase_votes:
        phase = phase_votes.most_common(1)[0][0]
        phase_ambiguous = len(phase_votes) > 1
        phase_candidates = sorted(phase_votes.keys())
    else:
        phase = None
        phase_ambiguous = False
        phase_candidates = []

    return {
        "segment_id": segment["segment_id"],
        "split": segment["split"],
        "video": segment["video"],
        "frame_start": segment["frame_start"],
        "frame_end": segment["frame_end"],
        "frames": segment["frames"],
        "actions": raw_labels,
        "tools": tools,
        "tissues": tissues,
        "events": events,
        "phase": phase,
        "phase_ambiguous": phase_ambiguous,
        "phase_candidates": phase_candidates,
        "unknown_actions": unknown,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "annotations")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    args = parser.parse_args()

    onto = load_ontology(args.ontology)

    total = 0
    total_unknown = 0
    for split in ("train", "val"):
        segments_path = args.reports_dir / f"segments_{split}.json"
        if not segments_path.exists():
            print(f"skip {split}: {segments_path} not found (run build_segments.py first)")
            continue
        segments = json.loads(segments_path.read_text())

        for segment in segments:
            mapped = map_segment(segment, onto)
            out_dir = args.out_dir / mapped["split"] / mapped["video"]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{mapped['segment_id']}.json").write_text(json.dumps(mapped, indent=2))
            total += 1
            total_unknown += len(mapped["unknown_actions"])

        print(f"[{split}] mapped {len(segments)} segments -> {args.out_dir / split}")

    print(f"Total segments mapped: {total}; segments with unknown actions: {total_unknown}")


if __name__ == "__main__":
    main()
