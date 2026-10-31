# LT-Swap evaluation manifests

**LT-Swap (LongTail-Swap)** evaluates speech and text LMs using correct/incorrect
sentence pairs grouped by frequency.

See [shared setup](../data.md#before-you-start) for environment and path conventions.

The versioned LT-Swap release contains audio and pair metadata for each dataset
variant.

## 1. Download and choose a variant

Download the archive matching your dataset size from the
[LT-Swap release directory](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/ltswap/):

- [ltswap_lh_small-v1.tar.zst](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/ltswap/ltswap_lh_small-v1.tar.zst)
- [ltswap_lh_small+medium-v1.tar.zst](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/ltswap/ltswap_lh_small%2Bmedium-v1.tar.zst)
- [ltswap_lh_small+medium+large-v1.tar.zst](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/ltswap/ltswap_lh_small%2Bmedium%2Blarge-v1.tar.zst)

Each archive includes all three tasks, README, license, and release metadata.
Download the accompanying
[SHA256SUMS](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/ltswap/SHA256SUMS)
to verify the archive.

For example, save the small+medium archive under `LTSWAP_DOWNLOAD_DIR`:

```bash
export LTSWAP_DOWNLOAD_DIR=/path/to/ltswap
export LTSWAP_RELEASE_URL=https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/ltswap
export LTSWAP_VARIANT='ltswap_lh_small+medium'
LTSWAP_ARCHIVE="${LTSWAP_VARIANT}-v1.tar.zst"
mkdir -p "$LTSWAP_DOWNLOAD_DIR"
curl -fL --continue-at - \
  "$LTSWAP_RELEASE_URL/${LTSWAP_ARCHIVE//+/%2B}" \
  -o "$LTSWAP_DOWNLOAD_DIR/$LTSWAP_ARCHIVE" || exit 1
curl -fL "$LTSWAP_RELEASE_URL/SHA256SUMS" \
  -o "$LTSWAP_DOWNLOAD_DIR/SHA256SUMS" || exit 1
```

### Verify downloads

Verify the downloaded archive against the release checksums, then extract it:

```bash
(cd "$LTSWAP_DOWNLOAD_DIR" && sha256sum --ignore-missing -c SHA256SUMS) || exit 1
tar --zstd -xf "$LTSWAP_DOWNLOAD_DIR/$LTSWAP_ARCHIVE" -C "$LTSWAP_DOWNLOAD_DIR"
```

For another variant, set `LTSWAP_VARIANT` to `ltswap_lh_small` or
`ltswap_lh_small+medium+large` before downloading. The expected extracted layout
under `LTSWAP_DOWNLOAD_DIR` is:

```text
ltswap-v1/
├── README.md
├── LICENSE
├── RELEASE.json
└── ltswap_lh_small+medium/
    ├── pairs/
    │   ├── agrswap.jsonl
    │   ├── inflswap.jsonl
    │   └── wordswap.jsonl
    └── audio/
        ├── agrswap/
        ├── inflswap/
        └── wordswap/
```

Use the extracted variant directory and activate the LM environment:

```bash
conda activate gslm2_lm

export LTSWAP_DATA_DIR="$LTSWAP_DOWNLOAD_DIR/ltswap-v1/$LTSWAP_VARIANT"
export LTSWAP_MANIFEST_DIR="$PWD/build/eval_manifests/$LTSWAP_VARIANT"
```

## 2. Generate local manifests

Run the [manifest converter](../../scripts/create_ltswap_manifests.py):

```bash
python scripts/create_ltswap_manifests.py \
  --data-dir "$LTSWAP_DATA_DIR" \
  --output-dir "$LTSWAP_MANIFEST_DIR"
```

It writes one file per released task and frequency bin:

| Output | Format | Used by |
|---|---|---|
| `raw/<task>_freq_<bin>.csv` | Audio paths and sample counts | Speech LMs |
| `text/<task>_freq_<bin>.jsonl` | Positive/negative pairs with inline text | Text LMs |

See [LT-Swap release metadata](formats.md#lt-swap-release-metadata) for the input schema.

To convert selected tasks, add `--tasks agrswap inflswap` or another selection.
Download the metadata and audio for those tasks. Use a fresh output directory
when switching release versions.

## 3. Set the LM config paths

Replace the LT-Swap entries in `slm.feature_extraction.manifests_test` with
the generated paths:

- **Speech configs:** use the CSVs under `raw/`.
- **Text configs:** use the corresponding JSONLs under `text/`.

For example, the 6Kh GSLM2 config's `agrswap_freq_0-1.csv` entry becomes:

```yaml
- /path/to/eval_manifests/ltswap_lh_small+medium/raw/agrswap_freq_0-1.csv
```

Preserve benchmark filenames and use the variant specified by the experiment.

For `ltswap_lh_small+medium`, agrswap and inflswap use bins `0-1`, `1-2`,
`2-4`, `4-8`, `8-16`, `16-32`, `32-64`, `64-128`, `128-256`,
`256-512`, and `512-inf`. Wordswap starts at `1-2`. A release containing
every configured bin produces 32 speech CSVs and 32 text JSONLs.
