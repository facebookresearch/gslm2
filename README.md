# GSLM2: Multi-Stream Spoken Language Modeling on SpidR 2.0 Units

This repository provides research code for **GSLM2**, including training and
evaluation pipelines for:

- **Speech encoders:** SpidR 2.0, including representation evaluation and
  speech-unit extraction.
- **Language models:** single-stream speech language models (SLMs),
  multi-stream GSLM2, and text language models.
- **Speech synthesis:** duration predictors and unit vocoders.

See [MODEL_ZOO.md](MODEL_ZOO.md) for checkpoint information.

## Installation

Requires **Python 3.12+**. Choose an environment for your task:

| Conda environment | Install extra | Tasks | Training stack |
|---|---|---|---|
| `gslm2` | `native` | Encoders, unit extraction, and speech synthesis | PyTorch 2.13.0 |
| `gslm2_lm` | `lm` | LM preparation, training, and evaluation | fairseq2 0.7 with PyTorch 2.9.0 |

Both environments use NumPy 2.x. Separate environments allow each training
stack to use its required PyTorch version.

Native PyTorch support for LM training and evaluation is planned for a
future release.

### Conda setup

The Slurm launchers use the Conda environment names above. Run these
commands from the repository root in **Bash**. Skip `conda create` for an
environment that already exists.

**Encoder training and speech synthesis:**

```bash
conda create -n gslm2 python=3.12 ffmpeg=8 libsndfile -c conda-forge
conda activate gslm2
uv pip install -e '.[native]'
```

**LM preparation, training, and evaluation:**

```bash
conda create -n gslm2_lm python=3.12 ffmpeg=8 libsndfile -c conda-forge
conda activate gslm2_lm
uv pip install -e '.[lm]' \
  --overrides <(printf '%s\n' 'numpy>=2.1.3,<2.4')
```

The LM command overrides fairseq2's outdated NumPy 1.x requirement to use
the supported NumPy 2.x stack. Bash process substitution supplies the
override directly.

Add `--group dev` to either install command for Ruff and mypy. Select an
extra when installing: `uv pip install -e .` installs only shared dependencies.
The `native` extra includes `torchdata` and the encoder evaluation packages.

Use matching CUDA builds of PyTorch and torchaudio for your machine. Audio
decoding requires an FFmpeg version supported by the installed TorchCodec.
See [environment guidance](environment/ENVIRONMENT.md) for CUDA library paths
and verification commands.

### uv environments

You can also create two uv environments for running commands directly:

```bash
UV_PROJECT_ENVIRONMENT=.venv-native uv sync --python 3.12 --extra native
UV_PROJECT_ENVIRONMENT=.venv-lm uv sync --python 3.12 --extra lm
```

Activate the environment you need with `source .venv-native/bin/activate`
or `source .venv-lm/bin/activate`. Make a supported FFmpeg installation
available separately for audio decoding.

`uv sync` includes the `dev` group by default; add `--no-dev` to install only
runtime dependencies. NumPy compatibility settings are in `pyproject.toml`.

## Training

Replace the `/path/to/...` values in your config before launching. In the
examples below, replace `ACCOUNT` and `QOS` with your Slurm settings. See
[configs/README.md](configs/README.md) for the available templates.

### SpidR 2.0 encoder

```bash
conda activate gslm2
gslm2-train-encoder -A ACCOUNT -N 2 -G 8 -q QOS \
  --dump /path/to/runs/spidr2.0-ls960 \
  configs/encoder/spidr2.0-ls960.yaml
```

`gslm2-train` is an alias for `gslm2-train-encoder`.

### Speech and text language models

```bash
sbatch --account=ACCOUNT --qos=QOS --nodes=4 \
  scripts/launchers/slurm_lm_steps.sbatch \
  configs/lm/gslm2/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m.yaml
```

The launcher activates `gslm2_lm` and runs data preparation, training with
validation, and benchmark evaluation using one experiment config. See
[configs/lm/README.md](configs/lm/README.md) for path settings, stage selection,
and resuming a prepared run.

`gslm2-train-lm` exposes the underlying fairseq2 recipe CLI. Direct pipeline
commands are documented in [REPRODUCING.md](REPRODUCING.md).

### Duration prediction and vocoding

Use `gslm2` for duration and vocoder training. See [SYNTHESIS.md](SYNTHESIS.md)
for preparing full-rate units, training both components, and generating audio.

## Repository layout

| Path | Contents |
|---|---|
| `src/gslm2/encoder/` | SpidR models, audio datasets, training, evaluation, and unit extraction |
| `src/gslm2/lm/` | fairseq2 models, token datasets, training recipes, preparation, and evaluation |
| `src/gslm2/duration/`, `src/gslm2/vocoder/` | Native PyTorch duration and unit HiFi-GAN training and inference |
| `src/gslm2/synthesis.py` | Duration expansion followed by unit vocoding |
| `src/gslm2/shared/` | Audio I/O, logging, distributed processing, and Slurm utilities |
| `configs/` | Encoder, LM, extraction, duration, and vocoder templates |
| `scripts/` | Launchers, analysis, and figure scripts |

## Further reading

- [DATA.md](DATA.md) — corpora and manifest formats.
- [REPRODUCING.md](REPRODUCING.md) — experiment commands and evaluation.
- [SYNTHESIS.md](SYNTHESIS.md) — duration prediction, vocoding, and audio generation.
- [MODEL_ZOO.md](MODEL_ZOO.md) — checkpoint information.
- [environment/ENVIRONMENT.md](environment/ENVIRONMENT.md) — dependency versions,
  CUDA/FFmpeg setup, and verification.

## Acknowledgements

We thank the authors and maintainers of the following codebases:

- [torchaudio](https://github.com/pytorch/audio) — encoder components
  and speech dataset code.
- [fairseq](https://github.com/facebookresearch/fairseq) — HuBERT
  k-means training code.
- [fairseq2](https://github.com/facebookresearch/fairseq2) — language
  modeling infrastructure.
- [textlesslib](https://github.com/facebookresearch/textlesslib) —
  distributed processing utilities.
- [unit-hifigan](https://github.com/bootphon/unit-hifigan) —
  unit vocoder architectures.

## License

Source code and model checkpoints are provided under CC BY-NC 4.0; see
[LICENSE](LICENSE) and [MODEL_ZOO.md](MODEL_ZOO.md).
