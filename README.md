# GSLM2: Multi-Stream Spoken Language Modeling on SpidR 2.0 Units

This repository trains **SpidR 2.0 encoders**, **single-stream SLMs**,
**multi-stream GSLM2**, **text language models**, **duration predictors**, and
**unit vocoders**.

The encoder uses native PyTorch training. Speech-unit and text language models
use fairseq2. The encoder-to-SLM boundary is a pre-extracted unit manifest.

## Layout

- `src/gslm2/encoder/` — encoder models, configs, audio datasets, training,
  representation evaluation, and unit extraction.
- `src/gslm2/lm/` — fairseq2 models, token datasets, recipes, benchmark
  evaluation, tokenization, and pipeline orchestration.
- `src/gslm2/duration/`, `src/gslm2/vocoder/` — native PyTorch duration and
  unit-HiFi-GAN training and inference.
- `src/gslm2/synthesis.py` — duration expansion followed by unit vocoding.
- `src/gslm2/shared/` — audio I/O, logging, distributed launch helpers, and
  Slurm submission utilities.
- `configs/encoder/` — native encoder training configs.
- `configs/lm/` — public SLM, text LM, and multi-stream experiments with
  editable paths; standalone fairseq2 recipes are under `configs/lm/recipes/`.
- `configs/misc/` — full-rate unit extraction, duration, and vocoder configs.
- `scripts/` — launchers, analysis, and figure scripts.

## Installation

Requires Python 3.12+. Install the repository into two separate environments:

- **`gslm2` / `native` extra:** encoder training and evaluation, standalone speech-unit
  extraction, duration prediction, vocoding, and synthesis. Uses PyTorch 2.13.0,
  TorchCodec 0.14.0, torchaudio >=2.11.0, and NumPy 2.x.
- **`gslm2_lm` / `lm` extra:** LM data preparation (speech-unit extraction and
  text tokenization), fairseq2 training, and evaluation. Uses PyTorch/torchaudio
  2.9.0, TorchCodec 0.9.0,
  fairseq2 0.7, and NumPy 1.x according to fairseq2's declared requirements.

### Conda environments

Run from the repository root. Skip `conda create` for an existing environment.

```bash
# Native training, extraction, and synthesis
conda create -n gslm2 python=3.12 ffmpeg=8 libsndfile -c conda-forge
conda activate gslm2
uv pip install -e '.[native]'

# fairseq2 language models and text tokenization
conda create -n gslm2_lm python=3.12 ffmpeg=8 libsndfile -c conda-forge
conda activate gslm2_lm
uv pip install -e '.[lm]'
```

Append `--group dev` to either install command to add Ruff and mypy. A bare
`uv pip install -e .` installs shared dependencies; select an extra to install
the runtime for your task. The `native` extra includes `torchdata` for vocoder
training and the native evaluation dependencies.

Use matching CUDA builds of PyTorch and torchaudio for your machine. Audio
decoding requires an FFmpeg version supported by the installed TorchCodec.
See [environment guidance](environment/ENVIRONMENT.md) for CUDA library paths
and verification commands.

### uv environments

Alternatively, keep two project environments, each with its own runtime extra:

```bash
UV_PROJECT_ENVIRONMENT=.venv-native uv sync --python 3.12 --extra native
UV_PROJECT_ENVIRONMENT=.venv-lm uv sync --python 3.12 --extra lm
```

Activate `.venv-native/bin/activate` or `.venv-lm/bin/activate` before running
commands directly. `uv sync` includes the `dev` group by default; add `--no-dev`
for a runtime-only environment. The mutually exclusive extras allow a shared
uv lockfile while keeping the two Torch stacks in separate environments.

The Slurm launchers use Conda: `gslm2` for native work and `gslm2_lm` for LM
work. The LM pipeline runs preparation (speech or text), training, and evaluation entirely
in the LM environment. Extracted unit/token files connect the stages.

## Usage

```bash
# Encoder: submitit / Slurm (activate gslm2)
gslm2-train-encoder -A <account> -N 2 -G 8 -q <qos> --dump <run-dir> \
  configs/encoder/spidr2.0-ls960.yaml

# LM: prepare data, train with fairseq2, and evaluate
sbatch --account=ACCOUNT --qos=QOS --nodes=4 \
  scripts/launchers/slurm_lm_steps.sbatch \
  configs/lm/gslm2/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m.yaml
```

`gslm2-train` remains an alias for encoder training. `gslm2-train-lm`
invokes the underlying fairseq2 recipe CLI. The supported LM launcher is
`scripts/launchers/slurm_lm_steps.sbatch`; it accepts one
[end-to-end experiment config](configs/lm/README.md). Replace the template
paths and set your Slurm account and QoS before submitting. See
[LM configs](configs/lm/README.md) for launcher arguments and recipe config usage.

See [REPRODUCING.md](REPRODUCING.md) for commands, [DATA.md](DATA.md) for
manifest formats, [SYNTHESIS.md](SYNTHESIS.md) for duration/vocoder training
and speech synthesis, and [MODEL_ZOO.md](MODEL_ZOO.md) for checkpoints.

## License

Source code and model checkpoints are provided under CC BY-NC 4.0; see
[LICENSE](LICENSE) and [MODEL_ZOO.md](MODEL_ZOO.md).

The unit HiFi-GAN implementation includes its upstream MIT license at
`src/gslm2/vocoder/LICENSE`.
