# Patch Notes — v0.2.1

## Why this consolidated patch exists

The repository already contains ontology v0.2, but the active decoder/evaluator still retained
semantics from the earlier ontology version. This package makes the implementation and the
reported claims consistent with v0.2.

## Methodological changes

### 1. Canonical ontology
`eval/ontology.py` now defaults to `ogsrb_prostatectomy_v0.2.yaml`.

### 2. Soft local backtracking
The decoder now uses transition costs in Viterbi scoring rather than treating all legal
transitions equally.

### 3. No unsupported contradiction filtering
The v0.2 ontology contains no expert-validated global contradiction pairs. The active decoder
therefore does not perform contradiction pruning.

### 4. Correct top-1 after decoding
The previous evaluator inferred top-1 from the original probability vector even after the
decoder removed an action. The patch stores and evaluates explicit decoded top-1.

### 5. Multi-target ontology support
The canonical target set is now read from `targets`, not only the legacy singular `tissue`
field.

### 6. Fair Test-180
The fixed manifest reproduces the previous seed-0 stratified sample logic and is shared by
all cross-model perception baselines.

### 7. Sparse-vs-temporal separation
Test-180 is intentionally marked `temporal_evaluation=false`. A sparse stratified sample must
not be treated as a contiguous surgical sequence.

### 8. Metric terminology
Old names that implied independent clinical reasoning are replaced by ontology v0.2's more
defensible names.

## Files replaced

- eval/ontology.py
- eval/metrics.py
- eval/evaluator.py
- baselines/common.py
- baselines/decoder_constrained.py
- baselines/llava_baseline.py

## Files added

- eval/decoder_effects.py
- benchmarks/create_test180_manifest.py
- baselines/gpt_baseline.py

## Files deliberately NOT changed

- ontology/ogsrb_prostatectomy_v0.2.yaml

The ontology file is already the intended canonical v0.2 research prototype and should remain
subject to expert review as documented in that file.
