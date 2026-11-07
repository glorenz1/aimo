# Reproduction

Inference is independent of the original research repository. `methods/pep/`
and `methods/nrc/` each contain `solution.py`, a fitted classifier and the
Small Models marker. NRC also carries its numerical/regional correction heads.
`scripts/package.py` builds a ZIP with those files at its root. Exact historical
archives are retained as `frozen/PEP.zip` and `frozen/NRC.zip`; their internal
historical names are preserved for checksum traceability. Renamed packages have
new hashes.

## Inference

Use Python 3.11 and `requirements.txt`. Historical qualification used
PyTorch 2.12.1+cu126, CUDA 12.6 and Transformers 5.13.0. Both methods load only
the locally cached DeepSeek reference checkpoint specified in
[data-references.json](data-references.json), using BF16 and SDPA. Metadata
hashes, the system prompt and layer index are embedded in `classifier.json`.
There is no precision fallback or automatic download. Different GPU kernels or
library versions can change numerical outputs.

Supported targets:

- `Qwen/Qwen3.5-4B`
- `Skywork/Skywork-OR1-Math-7B`
- `allenai/Olmo-3-7B-Think`
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`

PEP measures `hidden_states[31]` at the final formatted-prompt token. NRC hooks
decoder block 30 (the same state), pools original-body tokens and input
embeddings, and computes eight correction features. Numerical parsing and masks
preserve original regular expressions and floating-point reduction order. The
inherited PEP logit is both an offset and a standardized correction coordinate.
NRC thresholds at 0.5; PEP thresholds its raw margin at zero. NRC falls back to
PEP for nondefault effort or invalid body/alignment. Nonfinite regional features
use its numerical/alignment fallback head. Empty regions become zero vectors
and do not alone trigger fallback.

Example input (authored illustration, not a scientific observation):

```json
{"id":"example","model_id":"Qwen/Qwen3.5-4B","reasoning_effort":"default","problem":"Find the integer x such that x + 2 = 5."}
```

```bash
HF_HUB_OFFLINE=1 python scripts/predict.py --method pep \
  --inputs authorized-inputs.jsonl --output pep-predictions.jsonl
python scripts/package.py --method pep --output dist/PEP.zip
```

The runner preserves order and emits `id`, `is_robust`, `valid`. Use the
[organizer's ingestion/scoring tools](https://github.com/aimo-interp/getting-started)
for competition-format evaluation. Evaluation labels never enter method features.

## Rebuild training inputs and refit

Training code provides standalone grouped reconstruction, without imports
from the original research harness. Use a separate fitting environment:

```bash
python -m venv refit-env
refit-env/bin/pip install -r training/requirements.txt
```

Obtain authorized JSONL exports of `aimo-interp/train-main-v2` at both pinned
revisions in `docs/data-references.json`. Rows need `problem` (or
`original_problem`), `model_id`, `reasoning_effort` and Boolean/null `is_robust`
(or `model_is_robust`). `training/input-selection.json` freezes the label-free
61-row allowlist, groups, outer folds and feature definitions. Reconstruction
retains 33 Boolean Small Models labels and 21 Boolean auxiliary labels; seven
null labels stay outside fitting. It does not select protected evaluation data.

Provide an authorized historical raw-state cache, or add `--fresh-prefill`
with a nonexistent cache directory to extract it using one bounded GPU
allocation and the pinned local checkpoint:

```bash
refit-env/bin/python training/rebuild_inputs.py \
  --release 23791d9b120132087877b0459f953d5959fc8455=release-1.jsonl \
  --release fa8af758c6519fc97a9d0bd9017feb6f12232b52=release-2.jsonl \
  --bundle frozen/NRC.zip --cache authorized-raw-cache --out runs/bank
refit-env/bin/python training/refit.py --bank runs/bank/training-bank.json \
  --out runs/refits --influence
```

Use a new output directory for reconstruction. A cache needs `cache-index.json`
with encoder revision, layer, cache kind and per-original records containing
text/state hashes, relative FP32 `N x 4096` state paths, token offsets, original
span, body mask and lexical embedding mean. See `certify_cache` for the schema.
Hash checks bind supplied entries but cannot independently certify their
historical encoder execution.

CPU refitting belongs in a bounded CPU allocation. It reconstructs PEP with
fixed C=0.01, balanced logistic regression and auxiliary weight 0.25. NRC uses
group-cross-fitted PEP logits, training-only scaling, group-weighted correction
fits and inner equal-group Brier selection over C in {0.1, 1}. Outputs include
held-group probabilities, fit receipts, final PEP/NRC parameters, the no-white-box
counterpart and optional influential-group refits. This covers the original
comparisons and selected controls; the complete 14-exclusion diagnostic study
is outside this compact repository.

Data, raw states, derived per-case banks and model weights are external inputs.
The provider dataset card specifies `license: other`; obtain data under the
provider's terms. Tests use authored synthetic fixtures and archived code,
without loading models or creating robustness labels.
