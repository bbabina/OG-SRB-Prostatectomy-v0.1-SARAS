"""GPT vision baseline, prompt-variant experiment on the frozen Test-180 manifest.

This is an OPTIONAL EXTENSION on top of baselines/gpt_baseline.py's clean single-image/
21-bare-label baseline (per the instructions doc, extension item 1: "image-only vs
image+ontology prompt"). It does not replace or overwrite that baseline -- it writes to
a separate vlm_outputs/gpt_baseline_<variant>/ folder so the canonical Table 1 result
stays untouched and reproducible.

Motivation: per-segment analysis of the baseline GPT-4o run showed it systematically
confuses PullingTissue (grasper, no cutting) with CuttingTissue (monopolar_scissors) and
ClippingTissue (clip_applier) -- the bare label list gives the model no reason to look at
instrument shape. The ontology already encodes exactly that distinction per action
(verb + tool + target), so the "ontology_grounded" prompt variant surfaces it directly.

Requires:
    pip install openai python-dotenv

Environment:
    OPENAI_API_KEY=... (loaded from SARAS_new/.env if not already set)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.common import build_prediction_record  # noqa: E402
from baselines.gpt_baseline import (  # noqa: E402
    parse_response, image_as_data_url, load_manifest_segments,
)


def build_prompt_baseline(action_ids: list[str], onto, n_frames: int = 1) -> str:
    if n_frames > 1:
        intro = (
            f"You are analyzing {n_frames} frames, in temporal order, sampled evenly across "
            "a single ~2-second clip from a robotic-assisted radical prostatectomy (RARP) "
            "surgery video. Use the motion visible across the frames (not just one still "
            "image) to help judge the action.\n"
        )
    else:
        intro = (
            "You are analyzing a single frame from a robotic-assisted radical prostatectomy "
            "(RARP) surgery video.\n"
        )
    return (
        intro +
        "Identify which surgical actions are visible. Only choose from this "
        f"exact list:\n{', '.join(action_ids)}\n\n"
        "Respond with ONLY a comma-separated list of 1-3 action names from that list that "
        "best match, most likely first. Do not explain. Do not use any words not "
        "in the list."
    )


def build_prompt_ontology_grounded(action_ids: list[str], onto, n_frames: int = 1) -> str:
    lines = []
    for a in action_ids:
        node = onto.action_by_id[a]
        verb = node.get("verb", "?")
        tool = node.get("tool", "?")
        targets = "/".join(onto.targets_for(a)) or "?"
        lines.append(f"- {a}: {verb} on {targets}, using a {tool}")

    return (
        "You are analyzing a single frame from a robotic-assisted radical prostatectomy "
        "(RARP) surgery video.\n"
        "Each possible action below is defined by its verb, its anatomical target, and the "
        "specific instrument used -- identify the instrument visible in the image first, "
        "since that is usually the most reliable cue (a grasper looks and moves differently "
        "from scissors or a clip applier, even when both are touching the same tissue):\n\n"
        + "\n".join(lines) +
        "\n\nRespond with ONLY a comma-separated list of 1-3 action names from the list "
        "above that best match the image, most likely first. Use the exact action name "
        "(e.g. PullingTissue), not the tool or verb alone. Do not explain. Do not use any "
        "words not in the list."
    )


PROMPT_BUILDERS = {
    "baseline": build_prompt_baseline,
    "ontology_grounded": build_prompt_ontology_grounded,
}


def select_frames(segment: dict, n_frames: int) -> list[int]:
    """Evenly-spaced frame numbers spanning the segment, in temporal order."""
    frames = segment["frames"]
    if n_frames <= 1 or len(frames) <= 1:
        return [frames[len(frames) // 2]]
    idxs = [round(i * (len(frames) - 1) / (n_frames - 1)) for i in range(n_frames)]
    seen = []
    for idx in idxs:
        if not seen or frames[idx] != seen[-1]:
            seen.append(frames[idx])
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-variant", choices=list(PROMPT_BUILDERS), required=True)
    parser.add_argument("--n-frames", type=int, default=1,
                         help="Evenly-spaced frames per segment sent to the model (temporal order).")
    parser.add_argument(
        "--mesad-root", type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "mesad-real 2",
    )
    parser.add_argument(
        "--reports-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "reports",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).resolve().parent.parent / "benchmarks" / "test180_v1.json",
    )
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Missing {args.manifest}. Run: python benchmarks/create_test180_manifest.py")

    suffix = f"_{args.n_frames}frames" if args.n_frames > 1 else ""
    out_dir = args.out_dir or (
        Path(__file__).resolve().parent.parent / "vlm_outputs" / f"gpt_baseline_{args.prompt_variant}{suffix}"
    )

    client = OpenAI()
    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    action_id_set = set(action_ids)
    prompt = PROMPT_BUILDERS[args.prompt_variant](action_ids, onto, args.n_frames)
    sample = load_manifest_segments(args.manifest, args.reports_dir)
    images_dir = args.mesad_root / "val" / "images"

    print(f"Prompt variant: {args.prompt_variant}")
    print(f"Frames per segment: {args.n_frames}")
    print(f"Model: {args.model}")
    print(f"Segments: {len(sample)}")

    n_written = n_empty = n_invalid = 0
    started = time.time()

    for i, seg in enumerate(sample, 1):
        frame_nums = select_frames(seg, args.n_frames)
        image_blocks = [
            {"type": "input_image",
             "image_url": image_as_data_url(images_dir / f"{seg['video']}_frame_{fn}.jpg"),
             "detail": args.detail}
            for fn in frame_nums
        ]

        response = client.responses.create(
            model=args.model,
            input=[{
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}] + image_blocks,
            }],
        )
        raw_text = response.output_text.strip()
        valid_actions, invalid_tokens = parse_response(raw_text, action_id_set)
        n_invalid += len(invalid_tokens)
        if not valid_actions:
            n_empty += 1

        action_probs = {a: 0.0 for a in action_ids}
        for rank, action in enumerate(valid_actions):
            action_probs[action] = round(1.0 - 0.15 * rank, 4)

        record = build_prediction_record(seg, args.model, valid_actions, action_probs, onto)
        record["raw_response"] = raw_text
        record["hallucinated_tokens"] = invalid_tokens
        record["parse_failure"] = not bool(valid_actions)
        record["benchmark_manifest"] = args.manifest.name
        record["image_detail"] = args.detail
        record["prompt_variant"] = args.prompt_variant
        record["n_frames"] = args.n_frames
        record["frame_numbers"] = frame_nums
        record["probability_status"] = "rank_pseudo_scores_not_calibrated"

        out_path = out_dir / "val" / seg["video"] / f"{seg['segment_id']}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        n_written += 1

        if args.sleep:
            time.sleep(args.sleep)

        if i % 20 == 0 or i == len(sample):
            elapsed = time.time() - started
            print(f"{i}/{len(sample)} ({elapsed:.0f}s elapsed)")

    print(f"Wrote {n_written} predictions -> {out_dir / 'val'}")
    print(f"Parse failures (counted as wrong): {n_empty}")
    print(f"Out-of-vocabulary tokens: {n_invalid}")
    print(f"Exact model used: {args.model}")


if __name__ == "__main__":
    main()
