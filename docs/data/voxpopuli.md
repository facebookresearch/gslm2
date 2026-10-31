# VoxPopuli segmentation metadata

This release provides fixed English, French, and German speech-segment
selections from VoxPopuli as compressed RTTM metadata. Users obtain the original
recordings from
[VoxPopuli](https://github.com/facebookresearch/voxpopuli) and reconstruct
the audio locally with the helper in this repository.

## Released selections

The original RTTM filenames and their contents are preserved, with a `.zst`
compression suffix. Filenames use ISO 639-3 language codes; the original audio
directory names use ISO 639-1 codes:

| Language | RTTM filename code | Audio directory |
|---|---|---|
| English | `eng` | `en` |
| French | `fra` | `fr` |
| German | `deu` | `de` |

Two independent segmentation variants are provided:

- `subsampled_30s/`: used for **encoder training**; contains `train-6000h`,
  `dev-30min`, and `dev-10h` for each language.
- `subsampled_60s/`: used for **SLM training**; contains `train-600h` and
  `train-6000h` for each language.

Keep reconstructed audio and manifests for the two variants in separate
directories: their training filenames are the same, but their segment boundaries
differ.

There are 15 RTTM files. The duration budgets in their filenames are nominal;
`RELEASE.json` reports each file's measured total duration and segment counts.

Segments were selected using VAD followed by language identification filtering.
The minimum speech duration is 0.5 s, with these silence and segment duration
settings:

| Variant | Minimum silence duration | Maximum segment duration |
|---|---|---|
| 30 s | 0.2 s | 30 s |
| 60 s | 5.0 s | 60 s |

The distributed RTTMs contain the resulting selected boundaries. Reconstruction
uses these boundaries directly and does not require rerunning VAD or LID.

## Download segmentation metadata

Download the compressed RTTMs from the
[VoxPopuli segmentation release directory](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/voxpopuli/segmentation/),
together with the
[release README](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/voxpopuli/segmentation/README.md),
[RELEASE.json](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/voxpopuli/segmentation/RELEASE.json),
[SHA256SUMS](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/voxpopuli/segmentation/SHA256SUMS),
and [LICENSE](https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/voxpopuli/segmentation/LICENSE).
RTTM paths are relative to the same release directory, under
`subsampled_60s/` or `subsampled_30s/` as listed above.

For example, download the English 6Kh selection with 30-second segmentation
for encoder training. Save the release files under `VOXPOPULI_SEGMENTATION_DIR`,
preserving the variant subdirectory:

```bash
export VOXPOPULI_SEGMENTATION_DIR=/path/to/segmentation-release
export VOXPOPULI_SEGMENTATION_URL=https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/shared/devai/assets/gslm2/datasets/voxpopuli/segmentation
mkdir -p "$VOXPOPULI_SEGMENTATION_DIR/subsampled_30s"
for filename in README.md RELEASE.json SHA256SUMS LICENSE \
  subsampled_30s/VoxPopuli-eng-train-6000h.rttm.zst; do
  curl -fL "$VOXPOPULI_SEGMENTATION_URL/$filename" \
    -o "$VOXPOPULI_SEGMENTATION_DIR/$filename" || exit 1
done
```

For French or German, replace `eng` with `fra` or `deu`. For other selections,
use the corresponding variant directory and split filename from `RELEASE.json`,
and create that local variant directory before downloading.

### Verify downloads

Verify the downloaded files against the release checksums before reconstruction:

```bash
(cd "$VOXPOPULI_SEGMENTATION_DIR" && sha256sum --ignore-missing -c SHA256SUMS) || exit 1
```

The checksums verify compressed files. `RELEASE.json` additionally records the
SHA-256 digest of each original, uncompressed RTTM.

## RTTM format

Each line has the standard 10-column RTTM structure:

```text
SPEAKER 20090112-0900-PLENARY-10_en 1 0.031 4.606 <NA> <NA> SPEECH <NA> <NA>
```

The second column identifies the source recording. Columns four and five give
the onset and duration in seconds. This row selects the interval
`[0.031, 4.637)` in `20090112-0900-PLENARY-10_en`. `SPEECH` is a speech-region
label; it does not identify an individual speaker.

The original millisecond rounding of onsets and durations can create a 1 ms
overlap between adjacent intervals. These original boundaries are preserved;
`RELEASE.json` records the number of such overlaps in each file.

Source recordings resolve to:

```text
raw_audios/<language>/<year>/<recording_id>.ogg
```

The language is the recording ID's final suffix, and the year is its first four
characters. Local absolute paths are not stored in the released RTTMs.

## Download the original recordings

Follow the upstream VoxPopuli setup instructions. Run its downloader from a
clone of the upstream repository, using the expanded language subsets:

```bash
python -m voxpopuli.download_audios --root /path/to/voxpopuli --subset en_v2
python -m voxpopuli.download_audios --root /path/to/voxpopuli --subset fr_v2
python -m voxpopuli.download_audios --root /path/to/voxpopuli --subset de_v2
```

Download only the languages you need. Recordings are stored under
`/path/to/voxpopuli/raw_audios/`. Keep the original recording IDs and layout.

## Reconstruct FLAC segments and manifests

Use Python 3.12+ with NumPy and SoundFile, available in the GSLM2 native
environment. Reading the compressed RTTMs also requires `zstd` on `PATH`.

Reconstruct the 30-second segments for encoder training with:

```bash
python scripts/reconstruct_voxpopuli_segments.py \
  --rttm "$VOXPOPULI_SEGMENTATION_DIR/subsampled_30s/VoxPopuli-eng-train-6000h.rttm.zst" \
  --audio-root /path/to/voxpopuli/raw_audios \
  --output-dir /path/to/segments/30s/en \
  --manifest /path/to/manifests/30s/VoxPopuli-eng-train-6000h.csv
```

The helper accepts compressed or plain RTTMs. Source audio must be mono at
16 kHz; output segments are 16-bit PCM FLAC.

The CSV contains `fileid`, absolute local `path`, `num_frames`, and `language`.
IDs use `<recording_id>_<start_sample>_<stop_sample>` with the requested boundaries;
`num_frames` counts the actual extracted samples. Segments extending past the
recording's end are clipped; empty intervals are skipped.

Clipped and skipped intervals are recorded in `<manifest_stem>.reconstruction.json`.

Use the generated CSV in the encoder configuration:

```yaml
data:
  manifest: /path/to/manifests/30s/VoxPopuli-eng-train-6000h.csv
```

For SLM training, download and reconstruct the `subsampled_60s/` selection,
write its audio and manifest to separate `60s/` directories, and set
`slm.feature_extraction.manifests_train` to that CSV.

## License and attribution

Copyright (c) 2026 Meta Platforms, Inc. and affiliates.

The segmentation metadata and repository reconstruction helper are distributed
under CC-BY-NC-4.0; see `LICENSE`. The original VoxPopuli audio is obtained
separately and remains subject to the upstream data terms. See the
[VoxPopuli license information](https://github.com/facebookresearch/voxpopuli#license)
and the [European Parliament legal notice](https://www.europarl.europa.eu/legal-notice/en/).

Please cite the GSLM2 work and the original
[VoxPopuli paper](https://aclanthology.org/2021.acl-long.80).
