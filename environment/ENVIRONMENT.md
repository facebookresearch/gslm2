# Environment reproduction

Use two environments installed from the same repository:

- `gslm2` with `.[native]`: encoder training/evaluation, standalone speech-unit extraction,
  duration prediction, vocoding, and synthesis. Pins PyTorch 2.13.0 and
  TorchCodec 0.14.0, requires torchaudio >=2.11.0, and uses NumPy 2.x.
- `gslm2_lm` with `.[lm]`: speech/text preparation and fairseq2 training/evaluation.
  Pins PyTorch/torchaudio 2.9.0 and TorchCodec 0.9.0, with
  fairseq2 0.7 and its declared NumPy 1.x dependency.

## Installation

```bash
# Create only if the environment does not already exist.
conda create -n gslm2 python=3.12 ffmpeg=8 libsndfile -c conda-forge
conda activate gslm2
uv pip install -e '.[native]' --group dev
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

conda create -n gslm2_lm python=3.12 ffmpeg=8 libsndfile -c conda-forge
conda activate gslm2_lm
uv pip install -e '.[lm]' --group dev
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The base package contains shared dependencies; select `native` or `lm` for a
runtime. The `dev` group contains Ruff and mypy. Vocoder training's `torchdata`
dependency is included in `native`.

Alternatively, use the two uv project environments documented in
[README.md](../README.md#uv-environments). `tool.uv.conflicts` declares the
runtime extras mutually exclusive so uv can resolve a shared lockfile.

The PyTorch 2.9.0 Linux wheels from PyPI use CUDA 12.8. To select that LM
build explicitly, add `--torch-backend cu128` to its install command. The
native environment export records a PyTorch 2.13.0 CUDA 13.0 build; use the
wheel index that supplies your confirmed native stack if needed.

Use a CUDA build supported by the machine's driver. Audio decoding requires
an FFmpeg version supported by TorchCodec. Keep CUDA/cuDNN libraries
consistent with the installed PyTorch stack; inherited library paths can
cause incompatible system libraries to load.

NumPy constraints are separate: native uses >=2.1.3,<2.4; LM uses >=1.26,<2.
The earlier NumPy metadata overrides have been removed. The LM extra also
keeps Polars below 2 and W&B below 0.28.2 for fairseq2 compatibility; native
uses the original broader constraints and `fastabx>=0.8.0`.

## Launchers

The LM steps launcher uses `gslm2_lm` for all three stages: speech/text
preparation, training, and evaluation. Speech preparation loads the encoder
for inference in the LM stack. Duration and vocoder launchers use `gslm2`.

Pre-extracted unit/token files connect the LM stages.

## Verification

Run these checks in the corresponding installed environment:

```bash
conda activate gslm2
python -c "import torch, torchaudio, torchcodec; print(torch.__version__, torchaudio.__version__, torchcodec.__version__)"
gslm2-train-encoder --help

conda activate gslm2_lm
python -c "import torch, fairseq2, numpy; print(torch.__version__, fairseq2.__version__, numpy.__version__)"
gslm2-train-lm --help
```

## Historical exports

The `metasr_t213-environment.yml`, `metasr_t213-pip-freeze.txt`,
`metasr-environment.yml`, and `metasr-pip-freeze.txt` files record earlier
PyTorch 2.13 and 2.9 installations. The historical LM export used NumPy 2.3.5;
new LM installations follow fairseq2's declared NumPy 1.x requirement.
Current dependency pins are in `pyproject.toml`; use the installation commands
above for new environments.
