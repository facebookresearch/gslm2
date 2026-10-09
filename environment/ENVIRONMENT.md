# Environment reproduction

Standard env: **`metasr_t213`** (torch 2.13) — used for unit tests, ABX/MAP/
codebook evals, SLM evals, and SLURM training launches (`gslm2-train`).

The `metasr-*` files document the previous `metasr` env (torch 2.9), kept for
reference: the first training launches and the MAP/codebook evals ran there.

## Files

- `metasr_t213-environment.yml` — `conda env export` output, runnable with
  `conda env create` except for the commented-out lines (CUDA-index-only and
  editable installs, handled in step 2–3 below).
- `metasr_t213-pip-freeze.txt` — informational record.
- `metasr-environment.yml` / `metasr-pip-freeze.txt` — same for the legacy
  torch-2.9 env.

## Recipe

```bash
# 1. Base env (conda deps + all pip deps except torch/editables)
conda env create -f environment/metasr_t213-environment.yml -n gslm2-t213
conda activate gslm2-t213

# 2. CUDA torch stack (local versions, not on PyPI)
pip install --index-url https://download.pytorch.org/whl/cu130 \
  torch==2.13.0+cu130 torchvision==0.28.0+cu130

# 3. The repo itself (editable, no dep changes)
pip install -e /path/to/gslm2 --no-deps

# 4. Verify
python -c "import torch, torchcodec, fastabx, submitit, gslm2; \
  print(torch.__version__, torchcodec.__version__)"
gslm2-train --help          # console scripts on PATH
python -m pytest tests/test_spidr_batch_inference.py -q  # smoke
```

No system FFmpeg needed: `torchcodec` ships its own decoder libs.

## Key versions (`metasr_t213`)

| Package | Version | Notes |
|---|---|---|
| python | 3.12.15 | |
| torch / torchvision | 2.13.0+cu130 / 0.28.0+cu130 | CUDA index only |
| torchcodec | 0.14.0 | audio loading, bundled FFmpeg |
| fastabx | 0.9.0 | satisfies `pyproject>=0.8.0`; grouped ABX scoring |
| numpy / submitit | 2.3.5 / 1.5.4 | |
| speech-map / discophon | (see yml) | MAP + DiscoPhon evals |
| wandb | (see yml) | `wandb_mode: online` training runs |
| torchaudio | — (absent) | only the unit vocoder + HuBERT-baseline branches need it |
| fairseq2 | — (absent) | `test_slm_loss_parity` skips (it needs the deleted fs2 bridge) |

`gslm2` launches need `GSLM2_CLUSTER=FAIR_SC_3` set at submit time.

## Notes

- fastabx 0.8+ changed triplet subsampling order, so ABX scores shift by
  sampling noise vs 0.6.0-computed numbers — not exactly comparable to older
  internal runs.
- `test_slm_loss_parity` fails in envs where `fairseq2` is installed (it
  imports the `gslm2.fs2` bridge deleted in the carve-out). Pre-existing,
  unrelated to any model code; it skips in this env.
