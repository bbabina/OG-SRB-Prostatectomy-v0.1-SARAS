"""Step 7b (VLM comparison): Gemini baseline -- the free substitute for the
proprietary vision-capable model comparison the supervisor asked for
(GPT-4V access needs paid billing she hadn't yet confirmed; Gemini's Flash
tier is a genuine, permanent, no-card-required free tier as of Aug 2026).

Reuses the exact same sampling, prompt, and parsing logic as
llava_baseline.py -- same label vocabulary, same stratified sample, same
structured-output format -- so this is a fair, like-for-like comparison
against every other baseline, per her explicit instruction to keep the
evaluation format consistent across models.

Model: gemini-3.5-flash-lite. Two real pilot runs (not assumptions) shaped
this choice: gemini-3.6-flash's free quota turned out to be just 20
requests/day; gemini-2.5-flash-lite turned out to be fully retired for new
API keys, with the API's own error response naming gemini-3.5-flash-lite
as its replacement -- more authoritative than any web search, since it's
live, first-party, and specific to this actual key. Flash-Lite tiers
generally carry more generous free quotas than the newest flagship model.
Check the free-tier model list again if re-running much later, since
Google rotates which versions stay free and what their quotas are.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment -- get one
free, no credit card, at https://aistudio.google.com/apikey.

Output: vlm_outputs/gemini_baseline/val/<video>/<segment_id>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.common import build_prediction_record  # noqa: E402
from baselines.llava_baseline import (  # noqa: E402
    build_prompt, parse_response, representative_frame, load_manifest_segments,
)

MODEL_NAME = "gemini-3.5-flash-lite"


def call_with_retry(client, model: str, prompt: str, image_bytes: bytes, max_retries: int = 4) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
            )
            return resp.text or ""
        except Exception as e:  # rate limiting on the free tier is expected, not exceptional
            wait = 2 ** attempt * 5
            print(f"    (retry {attempt + 1}/{max_retries} after error: {e}; waiting {wait}s)")
            time.sleep(wait)
    return ""  # give up gracefully; caller treats this as "no valid actions parsed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesad-root", type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / "mesad-real 2")
    parser.add_argument("--reports-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs" / "gemini_baseline")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--manifest", type=Path,
                         default=Path(__file__).resolve().parent.parent / "benchmarks" / "test180_v1.json")
    parser.add_argument("--sleep-between-calls", type=float, default=2.0,
                         help="Seconds to wait between requests, to stay under free-tier rate limits.")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Missing {args.manifest}. Run: python benchmarks/create_test180_manifest.py")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment first. "
                          "Get a free key at https://aistudio.google.com/apikey")
    client = genai.Client(api_key=api_key)

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    action_id_set = set(action_ids)
    prompt = build_prompt(action_ids)

    sample = load_manifest_segments(args.manifest, args.reports_dir)
    print(f"Running Gemini on frozen manifest: {args.manifest}")
    print(f"Segments: {len(sample)}")

    images_dir = args.mesad_root / "val" / "images"
    n_written = n_hallucinated_tokens = n_empty = 0
    t_start = time.time()

    for i, seg in enumerate(sample, 1):
        frame_num = representative_frame(seg)
        img_path = images_dir / f"{seg['video']}_frame_{frame_num}.jpg"
        image_bytes = img_path.read_bytes()

        raw_text = call_with_retry(client, MODEL_NAME, prompt, image_bytes)
        valid_actions, invalid_tokens = parse_response(raw_text, action_id_set)
        n_hallucinated_tokens += len(invalid_tokens)

        if not valid_actions:
            n_empty += 1

        action_probs = {a: 0.0 for a in action_ids}
        for rank, a in enumerate(valid_actions):
            action_probs[a] = round(1.0 - 0.15 * rank, 4)

        record = build_prediction_record(seg, MODEL_NAME, valid_actions, action_probs, onto)
        record["raw_response"] = raw_text
        record["hallucinated_tokens"] = invalid_tokens
        record["parse_failure"] = not bool(valid_actions)
        record["benchmark_manifest"] = args.manifest.name

        out_path = args.out_dir / "val" / seg["video"] / f"{seg['segment_id']}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        n_written += 1

        if i % 20 == 0 or i == len(sample):
            elapsed = time.time() - t_start
            print(f"  {i}/{len(sample)} done ({elapsed:.0f}s elapsed, {elapsed / i:.1f}s/segment)")

        time.sleep(args.sleep_between_calls)

    print(f"\nWrote {n_written} predictions -> {args.out_dir / 'val'}")
    print(f"Parse failures (counted as wrong, no arbitrary fallback): {n_empty}")
    print(f"Out-of-vocabulary tokens across all responses (hallucination signal): {n_hallucinated_tokens}")


if __name__ == "__main__":
    main()
