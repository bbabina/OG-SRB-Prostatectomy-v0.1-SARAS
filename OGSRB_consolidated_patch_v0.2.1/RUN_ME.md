# OG-SRB Consolidated Methodology Patch v0.2.1

## Apply only this patch

Do **not** apply any earlier ontology/decoder patch before or after this package.

From the unpacked patch folder:

```bash
python APPLY_PATCH.py --repo "PATH_TO_OG-SRB-Prostatectomy-v0.1-SARAS"
```

The installer backs up every overwritten file under:

`patch_backups/before-v0.2.1-<timestamp>/`

---

## What this patch fixes

1. Makes `ontology/ogsrb_prostatectomy_v0.2.yaml` the canonical default.
2. Renames the implemented method conceptually to:

   **Ontology-Grounded Structured Decoder for Surgical Action Recognition**

3. Implements the v0.2 soft phase semantics:
   - stay: cost 0.0
   - adjacent forward: cost 0.1
   - adjacent backward: cost 0.5
   - non-adjacent transition: prohibited
4. Removes unsupported global contradiction filtering from the active decoder.
5. Supports multi-target mappings such as BladderAnastomosis -> bladder neck + urethra.
6. Recomputes decoded top-1 correctly after action filtering.
   - The old evaluator could keep the raw top-1 even when that action had been removed.
7. Renames ontology metrics so they do not overclaim clinical reasoning.
8. Adds a frozen Test-180 manifest generator.
9. Makes LLaVA use the frozen manifest instead of drawing a new sample.
10. Adds an OpenAI/GPT vision baseline using the exact same Test-180 and label prompt.
11. Adds decoder benefit/harm analysis.

---

# A. Freeze the fair Test-180 benchmark

Run once:

```bash
python benchmarks/create_test180_manifest.py
```

This creates:

`benchmarks/test180_v1.json`

It deliberately reproduces the previous LLaVA sampling logic:
- seed = 0
- at least one example per action class present in validation
- random remainder to 180

Once created, **do not regenerate it**. Commit it to the repository.

### Critical interpretation

Test-180 is a **sparse perception benchmark**.

It is suitable for direct comparison of:

- CLIP
- linear probe
- padded-crop probe
- LLaVA
- Gemini
- GPT

on the same 180 cases.

It is **not** suitable for running Viterbi directly across the selected 180 segments,
because the selected segments are not consecutive in the video.

Therefore:

- use Test-180 for **raw cross-model recognition**;
- use dense/full real3 outputs for **temporal decoder experiments**.

If a model is eventually run on all real3 segments, decode the full sequence first and
only then subset the resulting outputs to Test-180 for a secondary recognition comparison.

---

# B. Re-run the existing dense baselines

Use the existing project commands/configuration, but after applying this patch so they load
ontology v0.2 by default.

Typical commands are:

```bash
python baselines/clip_baseline.py
python baselines/clip_linear_probe.py
```

For bounding-box experiments, use the same tight/padded feature folders and model tags that
were used for the existing experiment. Do not silently mix tight and padded features.

The primary dense validation split remains `real3`.

---

# C. Evaluate every model fairly on Test-180

For models that already have predictions for all real3 segments, evaluation can simply be
filtered by the manifest:

```bash
python eval/evaluator.py --model clip --manifest benchmarks/test180_v1.json
python eval/evaluator.py --model clip_linear_probe --manifest benchmarks/test180_v1.json
python eval/evaluator.py --model clip_bbox_padded_probe --manifest benchmarks/test180_v1.json
```

For LLaVA:

```bash
python baselines/llava_baseline.py
python eval/evaluator.py --model llava_baseline --manifest benchmarks/test180_v1.json
```

For Gemini, make the existing Gemini runner read `benchmarks/test180_v1.json` rather than
sampling its own 180 cases, then evaluate:

```bash
python eval/evaluator.py --model <GEMINI_OUTPUT_FOLDER> --manifest benchmarks/test180_v1.json
```

The repository snapshot used to build this patch did not contain the Gemini runner, so this
package does not overwrite a Gemini file that is not present.

---

# D. Run GPT on exactly the same Test-180

Install the current OpenAI Python SDK if needed:

```bash
pip install openai
```

Set `OPENAI_API_KEY`, then run:

```bash
python baselines/gpt_baseline.py --model <EXACT_VISION_CAPABLE_OPENAI_MODEL_ID>
```

Do not write "GPT-4V" in the thesis unless that is the exact model identifier actually used.
Record:

- exact model id
- date of run
- image detail setting
- prompt
- Test-180 manifest checksum

Then evaluate:

```bash
python eval/evaluator.py --model gpt_baseline --manifest benchmarks/test180_v1.json
```

The GPT and LLaVA runners use the same label vocabulary and output semantics.

---

# E. Re-run the corrected structured decoder on DENSE outputs

Examples:

```bash
python baselines/decoder_constrained.py --model clip_linear_probe
python baselines/decoder_constrained.py --model clip_bbox_padded_probe
```

Outputs are written to:

`vlm_outputs/<MODEL>_decoded_v02/`

Evaluate dense real3:

```bash
python eval/evaluator.py --model clip_linear_probe --split val
python eval/evaluator.py --model clip_linear_probe_decoded_v02 --split val

python eval/evaluator.py --model clip_bbox_padded_probe --split val
python eval/evaluator.py --model clip_bbox_padded_probe_decoded_v02 --split val
```

Do not claim "100% phase ordering" as the goal. Local backward transitions are legitimate
under ontology v0.2.

Do not report "0% contradiction rate" as a scientific achievement. With no expert-validated
global contradiction rules, the corrected evaluator reports this metric as N/A.

---

# F. Quantify when the decoder helps and when it harms

```bash
python eval/decoder_effects.py \
  --raw-model clip_linear_probe \
  --decoded-model clip_linear_probe_decoded_v02

python eval/decoder_effects.py \
  --raw-model clip_bbox_padded_probe \
  --decoded-model clip_bbox_padded_probe_decoded_v02
```

Primary outputs:

- raw Macro-F1
- decoded Macro-F1
- Δ Macro-F1
- Correction Precision
- Correction Harm
- number of segments changed

These are central to the revised research question:

> When does ontology grounding genuinely correct surgical visual predictions, and when
> does it force uncertain visual evidence into a coherent but incorrect interpretation?

---

# G. Final results structure

Use TWO tables, not one mixed table.

## Table 1 — Fair Test-180 perception comparison

| Model | Top-1 | Macro-F1 | Parse/OOV failures |
|---|---:|---:|---:|
| Zero-shot CLIP | | | n/a |
| Linear probe | | | n/a |
| Padded-crop probe (oracle localisation) | | | n/a |
| LLaVA | | | |
| Gemini | | | |
| GPT | | | |

All rows must use exactly `test180_v1.json`.

## Table 2 — Dense ontology-decoder analysis

| Model | Raw F1 | Decoded F1 | ΔF1 | Correction Precision | Correction Harm | Proxy phase transition consistency |
|---|---:|---:|---:|---:|---:|---:|
| Linear probe | | | | | | |
| Padded-crop probe | | | | | | |
| Other dense model, if available | | | | | | |

Do not put sparse GPT/LLaVA/Gemini decoder results into Table 2 unless they have been run
densely enough for legitimate temporal decoding.

---

# H. Final terminology

Use:

**Ontology-Grounded Structured Decoder for Surgical Action Recognition**

Avoid:

**Ontology-Constrained Token Decoder**

unless a genuine token-level constrained generation experiment is separately implemented.

Metric names:

- Schema Vocabulary Conformance
- Canonical Tool Mapping Conformance
- Canonical Target Mapping Conformance
- Validated Rule Violation Rate
- Proxy Phase Transition Consistency
- Temporal Coherence

---

# I. What is still optional after this core run

After the corrected core experiment is complete, the strongest optional extensions are:

1. GPT/Gemini: image-only vs image+ontology prompt.
2. Multi-frame temporal input to GPT/Gemini.
3. Label-efficiency study (1%, 5%, 10%, 25%, 50%, 100% training labels).
4. Confidence-aware/adaptive ontology penalty.
5. Expert review / independent annotation of a small subset.

These are valuable extensions, but they are not required to make the corrected core
experiment internally consistent.
