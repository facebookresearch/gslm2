# Encoder evaluation labels

See [shared setup](../data.md#before-you-start) for environment and path conventions.

The [encoder eval commands](../reproducing.md#encoder-evaluation-tabencoder-eval-figfrontend-eval)
need label inputs over the [LibriSpeech dev/test audio](librispeech.md):

| Input | Used by | Source |
|---|---|---|
| Audio manifest (CSV or TSV) | ABX, MAP, codebook | `write_manifest` (§1) |
| Triphone `.item` file | ABX | fastabx ZeroSpeech 2021 download (§2) |
| `words.jsonl` word segments | MAP | speech-map download (§3) |
| Frame-level phone JSONL | Codebook quality | TextGrid conversion (§4) |

## 1. Write audio manifests

Feature files are keyed by audio stem, which matches the label file IDs.
Fairseq-style TSV manifests ([format](formats.md#audio-manifests)) are accepted
too; add `--fairseq` to the writer to generate that format.

```bash
conda activate gslm2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export EVAL_MANIFEST_DIR="$PWD/build/eval_manifests"
mkdir -p "$EVAL_MANIFEST_DIR"
python -m gslm2.encoder.data.write_manifest \
  /path/to/LibriSpeech/dev-clean \
  "$EVAL_MANIFEST_DIR/dev-clean.csv" --ext .flac
```

Repeat for `dev-other` (and the test splits for test-only evals).

## 2. Download ABX item files

Download the ZeroSpeech 2021 triphone item archive from
[fastabx](https://docs.cognitive-ml.fr/fastabx/items.html). It contains labels for
LibriSpeech dev-clean, dev-other, test-clean, and test-other.

Save the archive under `EVAL_MANIFEST_DIR/items/`, then extract it:

```bash
mkdir -p "$EVAL_MANIFEST_DIR/items"
curl -fL --continue-at - \
  https://cognitive-ml.fr/downloads/phoneme-discovery/zerospeech2021-triphone.tar.gz \
  -o "$EVAL_MANIFEST_DIR/items/zerospeech2021-triphone.tar.gz" || exit 1
tar -xzf "$EVAL_MANIFEST_DIR/items/zerospeech2021-triphone.tar.gz" \
  -C "$EVAL_MANIFEST_DIR/items" --strip-components=1
```

The archive holds per-split triphone `.item` files for LibriSpeech
dev-clean, dev-other, test-clean, and test-other, extracted here as
`triphone-<split>.item`; use the dev files for the standard evals.
(A phoneme variant is available at the same location as
`zerospeech2021-phoneme.tar.gz`.)

## 3. Download word alignments

MAP needs word segments with phone transcriptions (`words.jsonl`).
Download the reference word alignments from
[speech-map](https://github.com/bootphon/speech-map/tree/main/data).
Save the compressed JSONLs under `EVAL_MANIFEST_DIR`, then decompress them:

```bash
export EVAL_MANIFEST_DIR="$PWD/build/eval_manifests"
mkdir -p "$EVAL_MANIFEST_DIR"
for split in dev-clean dev-other test-clean test-other; do
  curl -fL --continue-at - \
    "https://raw.githubusercontent.com/bootphon/speech-map/main/data/$split-words.jsonl.zst" \
    -o "$EVAL_MANIFEST_DIR/$split-words.jsonl.zst" || exit 1
  unzstd -f "$EVAL_MANIFEST_DIR/$split-words.jsonl.zst" || exit 1
done
```

(`unzstd` ships with zstd; `conda install -c conda-forge zstd` if missing.)

Format: one record per word —
`{"fileid", "speaker", "onset", "offset", "transcription"}` with
`transcription` as comma-joined ARPABET phones (stress kept). Words
exclude utterances with out-of-vocabulary items, keep durations within
0.07–0.96 s, and drop singleton spellings, then singleton
transcriptions (MAP divides by count − 1, so singletons would
score NaN).

To rebuild from the underlying TextGrids instead — the LibriSpeech
alignments ([Zenodo 2619474](https://zenodo.org/records/2619474)) or
Robin Algayres's re-alignment (`librispeech` in
[abx_sem_syn](https://gitlab.cognitive-ml.fr/ralgayres/abx_sem_syn)) —
run speech-map's `build.py` :

```bash
git clone https://github.com/bootphon/speech-map.git
conda activate gslm2
python speech-map/data/build.py /path/to/textgrids "$EVAL_MANIFEST_DIR" \
  --splits dev-clean dev-other
```

Do not re-align from scratch: different aligner versions shift word
boundaries and transcriptions, so a fresh alignment will not reproduce
these numbers. For reference, labels regenerated end to end with MFA
3.4.3 match the distributed word files within ~2% of records, but 
run MAP at ~+8% higher scores (identical layer ranking).

## 4. Generate frame-level phone labels

Codebook quality needs per-utterance 10 ms frame phone labels, one
record per utterance —
`{"audio": <absolute path>, "phones": <space-joined labels>}`
with stress stripped and `SIL` for silence. There is no distributed
equivalent: generate it from TextGrids (either aligned source above)
with the `labels` converter, which writes frames for every utterance
in the manifest:

```bash
conda activate gslm2
python scripts/prepare_librispeech_eval_labels.py labels \
  --textgrid-dir /path/to/textgrids/dev-clean \
  --manifest "$EVAL_MANIFEST_DIR/dev-clean.csv" \
  --out-words /tmp/words-tmp.jsonl \
  --out-frames "$EVAL_MANIFEST_DIR/dev-clean.jsonl"
```

(The words output is still written; point it at a scratch path when
only frames are needed.)

## 5. Use the labels in the eval commands

Pass the generated files to the encoder evals:

| Eval | Manifest | Labels |
|---|---|---|
| ABX | `$EVAL_MANIFEST_DIR/dev-clean.csv` | `items/triphone-dev-clean.item` |
| MAP | `$EVAL_MANIFEST_DIR/dev-clean.csv` | `dev-clean-words.jsonl` |
| Codebook | `$EVAL_MANIFEST_DIR/dev-clean.csv` | `dev-clean.jsonl` |

Continue with [encoder evaluation](../reproducing.md#encoder-evaluation-tabencoder-eval-figfrontend-eval).

## DiscoPhon cross-lingual inputs

`gslm2-eval-discophon` takes an audio manifest CSV (via `write_manifest`),
a gold phone alignment `.txt`, and an ABX `.item` file per language:

```text
# alignment .txt rows: #file onset offset #phone
F01_N_000 0.000000 0.170000 a
```

```text
# item rows: #file onset offset #phone prev-phone next-phone speaker
1993 2.1025 2.3225 e m d M10_N
```

The per-language pair files follow the DiscoPhon benchmark release; the
public release location is still to be announced.

## Expected outputs

For the full dev splits at the paths above:

| Labels | dev-clean | dev-other |
|---|---|---:|
| `words.jsonl` word records (distributed) | 44,954 | 41,388 |
| Frame JSONL utterances | 2,703 | 2,864 |
