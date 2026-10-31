# Environment setup

Use separate environments for native PyTorch and fairseq2. Current dependency
pins are in [pyproject.toml](../pyproject.toml).

## Install

Requires Python 3.12+.

| Environment | Extra | Tasks |
|---|---|---|
| `gslm2` | `native` | Encoders, standalone unit extraction, duration prediction, vocoding, and synthesis |
| `gslm2_lm` | `lm` | Speech/text LM preparation, training, and evaluation |

### Native environment

On a submit node without an NVIDIA driver, prefix `conda create` with
`CONDA_OVERRIDE_CUDA=13` to select the CUDA 13 cuDNN variant.

```bash
conda create -n gslm2 python=3.12 ffmpeg=8 libsndfile cudnn=9.25.1.1 -c conda-forge
conda activate gslm2
uv pip install -e '.[native]' --group dev
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### LM environment

Keep the NumPy override for fairseq2 compatibility.

```bash
conda create -n gslm2_lm python=3.12 ffmpeg=8 libsndfile cudnn=9.25.1.1 -c conda-forge
conda activate gslm2_lm
uv pip install -e '.[lm]' --group dev \
  --overrides <(printf '%s\n' 'numpy>=2.1.3,<2.4')
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The base package installs shared dependencies. `--group dev` adds Ruff and mypy.

The LM launcher uses `gslm2_lm` for all stages, including encoder inference during
preparation. Duration and vocoder launchers use `gslm2`.

## Verify

### Imports and commands

Run each check in its corresponding environment:

```bash
conda activate gslm2
python -c "import torch, torchaudio, torchcodec; print(torch.__version__, torchaudio.__version__, torchcodec.__version__)"
gslm2-train-encoder --help

conda activate gslm2_lm
python -c "import torch, fairseq2, numpy; print(torch.__version__, fairseq2.__version__, numpy.__version__)"
gslm2-train-lm --help
```

### GPU and cuDNN

Run on a compute node with a GPU:

```bash
conda activate gslm2
python -c "import torch; x = torch.randn(2, 16, 1, 400, device='cuda'); \
w = torch.randn(32, 16, 1, 3, device='cuda'); \
print(torch.backends.cudnn.version(), tuple(torch.nn.functional.conv2d(x, w, stride=(1, 2)).shape))"
# native env expect: 92501 (2, 32, 1, 199)
```

## Compatibility and troubleshooting

### CUDA and audio decoding

- **Native:** PyTorch 2.13.0, TorchCodec 0.14.0, and torchaudio >=2.11.0.
  The reference export uses CUDA 13.0; select its wheel index if needed.
- **LM:** PyTorch/torchaudio 2.9.0, TorchCodec 0.9.0, and fairseq2 0.7.
  Linux PyPI wheels use CUDA 12.8. Add `--torch-backend cu128` to the LM install
  command to select that build explicitly.

Use a CUDA build supported by the machine's driver and an FFmpeg version
supported by TorchCodec. For `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` or a crash
during the GPU check, check the conda cuDNN installation and `LD_LIBRARY_PATH`.
Inherited library paths can load incompatible system libraries.

### NumPy and fairseq2

Both environments use NumPy >=2.1.3,<2.4. fairseq2 0.7 runs with NumPy 2,
although its metadata declares `numpy~=1.23`:

- `uv pip install` needs the explicit `--overrides` argument shown above.
- `uv lock` and `uv sync` use `tool.uv.override-dependencies` in `pyproject.toml`.
- `uv pip check` still reports the metadata mismatch in `gslm2_lm`; use the
  [runtime checks](#verify) to verify the installed stack.

The LM extra caps Polars below 2 and W&B below 0.28.2 for fairseq2 compatibility.
Native retains broader constraints and `fastabx>=0.8.0`.

### Historical exports

Reference installations under `environment/`:

- **PyTorch 2.13:** [Conda export](../environment/gslm2-environment.yml)
  and [pip snapshot](../environment/gslm2-pip-freeze.txt).
- **PyTorch 2.9:** [Conda export](../environment/gslm2_lm-environment.yml)
  and [pip snapshot](../environment/gslm2_lm-pip-freeze.txt).

Both snapshots used NumPy 2.3.5. Use the install commands above for new environments.
