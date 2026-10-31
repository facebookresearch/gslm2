# Libri-Light training manifests

See [shared setup](../data.md#before-you-start) for environment and path conventions.

The [official Libri-Light instructions](https://github.com/facebookresearch/libri-light/tree/main/data_preparation#1-getting-and-preparing-the-unlabeled-audio)
provide the archives, checksums, and `cut_by_vad.py`. The archives contain
16 kHz mono FLAC audio and JSON `voice_activity` intervals for segmentation.

## 1. Download and extract audio

Download the audio archives from the
[Libri-Light release](https://github.com/facebookresearch/libri-light/tree/main/data_preparation#1-getting-and-preparing-the-unlabeled-audio).

Choose the subsets needed for training:

| Training scale | Subsets |
|---|---|
| 600 h | `small` |
| 6Kh | `small medium` |
| 60Kh | `small medium large` |

The example downloads all three subsets. Smaller training runs still use
`large` for the validation split described below. Save the archives under
`LL_DATA_DIR/raw/`:

```bash
export LL_DATA_DIR=/path/to/librilight
mkdir -p "$LL_DATA_DIR/raw" "$LL_DATA_DIR/manifests"
for subset in small medium large; do
  curl -fL --continue-at - \
    "https://dl.fbaipublicfiles.com/librilight/data/$subset.tar" \
    -o "$LL_DATA_DIR/raw/$subset.tar" || exit 1
done
```

### Verify downloads

Verify the downloaded archives against the checksums in the official
Libri-Light instructions, then extract them:

```bash
for subset in small medium large; do
  tar -xf "$LL_DATA_DIR/raw/$subset.tar" -C "$LL_DATA_DIR/raw" || exit 1
done
```

## 2. Segment audio and write manifests

Clone the upstream repository to get `cut_by_vad.py`, then segment the audio
and write per-subset manifests:

```bash
git clone https://github.com/facebookresearch/libri-light.git "$LL_DATA_DIR/libri-light"

for subset in small medium large; do
  python "$LL_DATA_DIR/libri-light/data_preparation/cut_by_vad.py" \
    --input_dir "$LL_DATA_DIR/raw/$subset" \
    --output_dir "$LL_DATA_DIR/${subset}_segmented" \
    --target_len_sec 60 || exit 1

  # Increase --num-workers for faster parallel processing.
  python -m gslm2.encoder.data.write_manifest \
    "$LL_DATA_DIR/${subset}_segmented" \
    "$LL_DATA_DIR/manifests/${subset}_segmented_60s.csv" \
    --ext .flac --num-workers 8 || exit 1
done
```

## 3. Prepare the validation split

The experiments use a random sample of 6,000 clips from `large` for validation,
including the 600 h and 6Kh runs. Download and segment `large` for this validation
setup even when training uses only `small` or `small medium`.

The original validation file IDs are not recorded in this repo. Choose a
reproducible sample of 6,000 clips, record the selected file IDs, and save the
manifest as `large_segmented_60s_dev.csv`.

For 60Kh training, exclude those validation clips from the full large manifest
and save the remaining rows as `large_segmented_60s_train.csv`.

## 4. Set the LM config paths

A 6Kh experiment uses the small and medium training manifests:

```yaml
slm:
  feature_extraction:
    manifests_train:
      - /path/to/librilight/manifests/small_segmented_60s.csv
      - /path/to/librilight/manifests/medium_segmented_60s.csv
    manifests_valid:
      - /path/to/librilight/manifests/large_segmented_60s_dev.csv
```

For 60Kh training, also include `large_segmented_60s_train.csv` under `manifests_train`.

Once the manifests are ready, follow
[LM preparation and training](../../configs/lm/README.md#launch)
to extract speech units and train.
