"""Step 7c (VLM comparison): GPT-4V baseline -- the proprietary model
comparison Inna originally asked for, now that funded API access is
available.

Reuses the exact same prompt and parsing logic as llava_baseline.py /
gemini_baseline.py -- same label vocabulary, same structured-output format
-- so this is a fair, like-for-like comparison against every other
baseline.

Unlike LLaVA/Gemini (stratified 180-segment sample, for cost/rate-limit
reasons), this runs on the FULL val split by default, matching the
CLIP-family models. For the apples-to-apples "Common Test-180" comparison,
eval/common_test_180.py subsets these full-val predictions down to the
same 180 segment_ids afterward -- same pattern already used there for the
CLIP models.

Model: gpt-4o -- the direct, documented successor in the GPT-4V lineage
(vision-capable from the base model, not a preview/add-on), chosen over
newer gpt-5.x models in the account's catalog because its chat-completions
image-input format is well-established and verified working with this key,
where the newer models' behavior isn't independently confirmed here.

Requires OPENAI_API_KEY in the environment or in a .env file discoverable
by python-dotenv (this project keeps it at the SARAS_new/.env level, one
directory above this repo, so it's never inside anything git could push).

Output: vlm_outputs/gpt4v_baseline/val/<video>/<segment_id>.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.ontology import load_ontology, DEFAULT_ONTOLOGY_PATH  # noqa: E402
from baselines.common import build_prediction_record  # noqa: E402
from baselines.llava_baseline import build_prompt, parse_response, representative_frame  # noqa: E402

MODEL_NAME = "gpt-4o"


def call_with_retry(client: OpenAI, model: str, prompt: str, image_bytes: bytes,
                     max_retries: int = 4) -> tuple[str, int, int]:
    """Returns (text, prompt_tokens, completion_tokens)."""
    b64 = base64.b64encode(image_bytes).decode()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                max_tokens=60,
            )
            usage = resp.usage
            return (resp.choices[0].message.content or "",
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0)
        except Exception as e:  # rate limiting / transient errors are expected, not exceptional
            wait = 2 ** attempt * 5
            print(f"    (retry {attempt + 1}/{max_retries} after error: {e}; waiting {wait}s)")
            time.sleep(wait)
    return "", 0, 0  # give up gracefully; caller treats this as "no valid actions parsed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesad-root", type=Path,
                         default=Path(__file__).resolve().parent.parent.parent / "mesad-real 2")
    parser.add_argument("--reports-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "reports")
    parser.add_argument("--out-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "vlm_outputs" / "gpt4v_baseline")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY_PATH)
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N val segments (for a cheap pilot run before "
                              "committing to the full val set).")
    parser.add_argument("--sleep-between-calls", type=float, default=0.5,
                         help="Seconds to wait between requests, to stay under rate limits.")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment or in SARAS_new/.env first.")
    client = OpenAI(api_key=api_key)

    onto = load_ontology(args.ontology)
    action_ids = sorted(onto.action_by_id.keys())
    action_id_set = set(action_ids)
    prompt = build_prompt(action_ids)

    segments = json.loads((args.reports_dir / "segments_val.json").read_text())
    if args.limit:
        segments = segments[:args.limit]
    print(f"Running GPT-4V ({MODEL_NAME}) on {len(segments)} val segments"
          f"{' (limited pilot run)' if args.limit else ' (full val set)'}")

    images_dir = args.mesad_root / "val" / "images"
    n_written = n_hallucinated_tokens = n_empty = 0
    total_prompt_tokens = total_completion_tokens = 0
    t_start = time.time()

    for i, seg in enumerate(segments, 1):
        frame_num = representative_frame(seg)
        img_path = images_dir / f"{seg['video']}_frame_{frame_num}.jpg"
        image_bytes = img_path.read_bytes()

        raw_text, p_tok, c_tok = call_with_retry(client, MODEL_NAME, prompt, image_bytes)
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        valid_actions, invalid_tokens = parse_response(raw_text, action_id_set)
        n_hallucinated_tokens += len(invalid_tokens)

        if not valid_actions:
            n_empty += 1
            valid_actions = [action_ids[0]]

        action_probs = {a: 0.0 for a in action_ids}
        for rank, a in enumerate(valid_actions):
            action_probs[a] = round(1.0 - 0.15 * rank, 4)

        record = build_prediction_record(seg, MODEL_NAME, valid_actions, action_probs, onto)
        record["raw_response"] = raw_text
        record["hallucinated_tokens"] = invalid_tokens

        out_path = args.out_dir / "val" / seg["video"] / f"{seg['segment_id']}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2))
        n_written += 1

        if i % 20 == 0 or i == len(segments):
            elapsed = time.time() - t_start
            print(f"  {i}/{len(segments)} done ({elapsed:.0f}s elapsed, "
                  f"{elapsed / i:.1f}s/segment, ~{elapsed / i * (len(segments) - i):.0f}s remaining)")

        time.sleep(args.sleep_between_calls)

    print(f"\nWrote {n_written} predictions -> {args.out_dir / 'val'}")
    print(f"Segments with no valid action parsed (fell back to default): {n_empty}")
    print(f"Out-of-vocabulary tokens across all responses (hallucination signal): {n_hallucinated_tokens}")
    print(f"Total tokens used: {total_prompt_tokens} prompt + {total_completion_tokens} completion "
          f"(check current {MODEL_NAME} pricing on the OpenAI dashboard to convert to cost)")


if __name__ == "__main__":
    main()
