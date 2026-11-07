# PEP and NRC for AIMO Interpretability

Code and fitted classifiers for **Prompt Endpoint Probe (PEP)** and **Numerical and
Regional Correction (NRC)** in the AIMO Interpretability Challenge.

The [AIMO Interpretability Challenge](https://aimo-interp.github.io/) asks whether
a model's mathematical accuracy survives perturbations that preserve the
problem's deductive chain. These Small Models Track methods predict robustness
from the original problem, without generating solutions or changing the target.

**PEP** combines a frozen reference encoder's final prompt-token representation
with problem, target-model and reasoning-effort metadata in a regularized
logistic probe. **NRC** adjusts PEP's logit using numerical statistics,
lexical-to-contextual alignment and four relationships between number, relation
and identifier token pools. These masks describe lexical regions, not identified
reasoning components. Both use one shared DeepSeek reference encoder across four
prediction targets.

| Method | Public validation | Coverage | Invalid predictions | Local accuracy | Local balanced accuracy |
|---|---:|---:|---:|---:|---:|
| PEP | 0.64 | 100% | 0 | 59.60% | 66.67% |
| NRC | 0.64 | 100% | 0 | 88.89% | 88.64% |

Public scores are recorded Competition-phase displays (submissions **994986**
and **1023896**); separately reported counts are **9/14 (64.29%) for each**.
They are not final private-test scores. Local evaluation uses 33 labeled
problem–model pairs, 14 reused inferred groups and three grouped procedures:
99 dependent prediction occurrences, not 99 independent examples.

Matched refits attribute +9.09 accuracy points to PEP's internal inputs, +8.08 to
NRC's complete white-box input group, and +5.05 to NRC's regional additions.
These overlapping effects are not additive. NRC's complete white-box accuracy
benefit reverses in 12/14 group-exclusion refits, or 7/14 when original partitions
and regularization choices are retained. Local gains therefore do not establish
independent generalization or an external advantage over PEP.
See [recorded results](docs/results.md).

## Use

Use Python 3.11 and the pinned [inference dependencies](requirements.txt).
Before offline execution, cache
`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` at revision
`6e8885a6ff5c1dc5201574c8fd700323f23c25fa`. Inference requires one visible,
BF16-capable CUDA GPU with **at least 24 GiB free**. Run GPU work inside your HPC
allocation. Checkpoint weights and benchmark statements are external inputs.

```bash
pip install -r requirements.txt
HF_HUB_OFFLINE=1 python scripts/predict.py --method nrc \
  --inputs authorized-inputs.jsonl --output predictions.jsonl
python scripts/package.py --method nrc --output dist/NRC.zip
```

Use `--method pep` for PEP. Input JSONL rows contain `id`, `model_id`, `problem`
and optional `reasoning_effort` (default: `"default"`). Each method exposes the
[official interface](https://github.com/aimo-interp/getting-started):

```python
are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]
```

`True` means robust. Evaluation is offline with a 3,600-second total inference
limit. Historical timing on a small authorized cohort does not certify an
unknown private workload.

- `methods/pep/`, `methods/nrc/`: standalone inference code and fitted artifacts.
- `training/`: authorized-input reconstruction and nested grouped CPU fitting.
- `frozen/PEP.zip`, `frozen/NRC.zip`: exact historical submission archives.
- `docs/`: [reproduction instructions](docs/reproduction.md), results and provenance.
- `tests/`: archive parity, feature, fallback and input-isolation checks.

Run `python -m unittest discover -s tests -v` in the separate fitting environment
described in the reproduction instructions (the tests also use SciPy).
