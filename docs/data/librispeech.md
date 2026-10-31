# LibriSpeech preparation

See [shared setup](../data.md#before-you-start) for environment and path conventions.

## Encoder training

### 1. Download and extract audio

Download the `train-clean-100`, `train-clean-360`, and `train-other-500` archives
from [LibriSpeech (OpenSLR 12)](https://www.openslr.org/12/).
Extract them into the same download directory. The extracted
`LibriSpeech/` directory should contain all three training subsets, with mono
16 kHz FLAC audio.

### 2. Write and combine manifests

```bash
conda activate gslm2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRISPEECH_DATA_DIR=/path/to/LibriSpeech
export LIBRISPEECH_TRAIN_MANIFEST_DIR="$PWD/build/manifests/librispeech"
mkdir -p "$LIBRISPEECH_TRAIN_MANIFEST_DIR"

for split in train-clean-100 train-clean-360 train-other-500; do
  python -m gslm2.encoder.data.write_manifest \
    "$LIBRISPEECH_DATA_DIR/$split" \
    "$LIBRISPEECH_TRAIN_MANIFEST_DIR/$split.csv" --ext .flac || exit 1
done

python - <<'PY'
import os
from pathlib import Path

import pandas as pd

manifest_dir = Path(os.environ["LIBRISPEECH_TRAIN_MANIFEST_DIR"])
splits = ("train-clean-100", "train-clean-360", "train-other-500")
manifests = [pd.read_csv(manifest_dir / f"{split}.csv") for split in splits]
pd.concat(manifests, ignore_index=True).to_csv(
    manifest_dir / "librispeech-train-960.csv", index=False
)
PY
```

This writes one CSV per subset and a combined `librispeech-train-960.csv`.
Set `data.manifest` in the encoder YAML to this combined file's absolute path.
Continue with the [encoder config guide](../../configs/encoder/README.md#prepare-data-and-paths).

## Generative evaluation

Use the [bundled selections and prompt boundaries](../../manifests/README.md#librispeech-generative-evaluation)
with the framewise extraction configs.

### 1. Prepare your local audio paths

Download the dev-clean and test-clean archives from
[LibriSpeech (OpenSLR 12)](https://www.openslr.org/12/), then extract them into
the same download directory. Point `LIBRISPEECH_DATA_DIR` to the extracted
`LibriSpeech/` root.

```bash
conda activate gslm2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRISPEECH_DATA_DIR=/path/to/LibriSpeech
export LIBRISPEECH_EVAL_MANIFEST_DIR="$PWD/build/eval_manifests/librispeech"
python scripts/prepare_librispeech_eval_manifests.py \
  --data-dir "$LIBRISPEECH_DATA_DIR" \
  --output-dir "$LIBRISPEECH_EVAL_MANIFEST_DIR"
```

The helper writes CSVs with absolute audio paths and copies the bundled prompt
JSONLs unchanged. Use `--splits dev-clean` to prepare one split.

Prompt fields and encoding behavior are described in the
[prompt boundary reference](formats.md#prompt-boundaries).

### 2. Set the extraction config paths

Use the generated local files in `targets.generative_eval`:

```yaml
targets:
  generative_eval:
    manifests:
    - audio_manifest: /path/to/local-manifests/librispeech_dev-clean_1k_min6s.csv
      prompt_boundary_manifest: /path/to/local-manifests/librispeech_dev-clean_1k_min6s_prompt_boundaries_3s.jsonl
    - audio_manifest: /path/to/local-manifests/librispeech_test-clean_1k_min6s.csv
      prompt_boundary_manifest: /path/to/local-manifests/librispeech_test-clean_1k_min6s_prompt_boundaries_3s.jsonl
```

Keep the paired filenames, fixed selections, and boundary records when comparing
results. Continue with [full-rate extraction](../synthesis.md#prepare-training-data).
