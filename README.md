# GSLM2: Multi-Stream Spoken Language Modeling on SpidR 2.0 Units

This repo contains the code to train **SpidR 2.0** speech encoders, train
**single-stream SLMs** and **multi-stream GSLM2** spoken language models on their
discrete units, and run the paper's evaluations.

Pipeline: `SpidR 2.0 encoder` → discrete units (one stream per encoder layer) →
`SLM` (single stream) or `GSLM2` (six streams, layers 5–10) → optional
`duration predictor` + `unit vocoder` for speech synthesis.

## Installation

Requires Python 3.12+ and FFmpeg 8 (for `torchcodec` audio loading).

```bash
# with uv (recommended)
uv sync

# with conda
conda create -n gslm2 python=3.12 ffmpeg=8 libsndfile -c conda-forge
conda activate gslm2
uv pip install -e . --group dev
```

`torchaudio` must come from the same PyTorch index as `torch` (see
[PyTorch install](https://pytorch.org/get-started/locally/)).

## Usage

Training launches via `gslm2-train` (SLURM/submitit):

```bash
gslm2-train -A <account> -N 2 -G 8 -Q <qos> --dump <run-dir> configs/encoder/spidr2.0-ls960.yaml
```

See [`REPRODUCING.md`](REPRODUCING.md) for the per-table command map,
[`DATA.md`](DATA.md) for corpora and manifest formats, and
[`MODEL_ZOO.md`](MODEL_ZOO.md) for released checkpoints.

## Layout

- `src/gslm2/models/` — SpidR 2.0 encoder, OPT / multi-stream OPT LMs
- `src/gslm2/training/` — encoder, single/multi-stream SLM, and text-LM trainers
- `src/gslm2/eval/` — encoder evals (ABX/MAP/codebook/DiscoPhon) and SLM evals
- `src/gslm2/vocoder/` — duration predictor and unit HiFi-GAN vocoder
- `configs/` — canonical paper configs
- `scripts/` — analysis and figure scripts
