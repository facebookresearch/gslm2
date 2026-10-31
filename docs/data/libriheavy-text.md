# LibriHeavy-long transcripts for text LMs

See [shared setup](../data.md#before-you-start) for environment and path conventions.
Use the `gslm2_lm` environment.

The text LM configs train on LibriHeavy-long transcripts grouped by Libri-Light
subset (small, medium, large). The helper,
[prepare_libriheavy_text.py](../../scripts/prepare_libriheavy_text.py), builds
collection text files, CSVs, and custom BPE tokenizers from the released
manifests. Audio is not needed for these steps.

## 1. Download the LibriHeavy-long manifests

Download the small, medium, and large manifest archives from the
**long-segment release** at
[pkufool/libriheavy_long](https://huggingface.co/datasets/pkufool/libriheavy_long).
Its segments are approximately 20–100 seconds long; this segmentation determines
the transcript content and collection files produced below.

Save the compressed JSONLs under `LH_DATA_DIR`:

```bash
export LH_DATA_DIR=/path/to/libriheavy/libriheavy-long
export LH_TEXT_DIR="$LH_DATA_DIR/processed_text/cases_and_punc"
mkdir -p "$LH_DATA_DIR"

for subset in small medium large; do
  curl -fL --continue-at - \
    "https://huggingface.co/datasets/pkufool/libriheavy_long/resolve/main/libriheavy_cuts_${subset}.jsonl.gz" \
    -o "$LH_DATA_DIR/libriheavy_cuts_${subset}.jsonl.gz" || exit 1
done
```

### Verify downloads

Verify the downloaded archives against these SHA-256 checksums before extraction.
The extraction helper reads the compressed JSONLs directly.

```bash
(
  cd "$LH_DATA_DIR" || exit 1
  sha256sum --check <<'EOF'
2de8c66d223c13d4a0e6a6e9a551192491a35a31f0cbc46e4f356138732d280d  libriheavy_cuts_small.jsonl.gz
85ca001eb0fda7234c33e56694d930b0d67afbde57f7256a49a9a4d8145f3b1c  libriheavy_cuts_medium.jsonl.gz
6e032c197051eacf282f0e8d05c8fd952f57b8e6a1d660f4b80e387b2a3bc5f5  libriheavy_cuts_large.jsonl.gz
EOF
) || exit 1
```

## 2. Extract collection text files and CSVs

```bash
python scripts/prepare_libriheavy_text.py extract \
  --manifest-dir "$LH_DATA_DIR" \
  --output-dir "$LH_TEXT_DIR" \
  --subsets small medium large
```

The helper reads `supervisions[0].custom.texts[0]`, the aligned original book
transcript. It does not use the ASR transcript at `texts[1]`, preceding context
in `pre_texts`, or the complete source book referenced by `text_path`.

It reproduces the original `run.sh --stage 2 --stop_stage 2` followed by
`extract_text.py`, reading the manifests directly:

- Preserve case and accents. Convert curly quotes to straight quotes and selected
  full-width punctuation to its ASCII equivalent. Replace each ASCII hyphen
  (`-`) with a space, and strip surrounding whitespace. Internal whitespace and
  other Unicode punctuation, including em dashes, are retained.
- Group by the collection directory, the third component of IDs such as
  `small/103/shortpoetry_039_librivox_64kb_mp3/dirge_hemans_krs_64kb_0`.
  Speaker IDs do not separate collections; multiple readers can contribute to
  the same file.
- Sort recording names alphabetically after removing the final numeric segment
  suffix. Within each recording, sort by the segment start time truncated to
  milliseconds. Equal start times retain manifest order.
- Join the cleaned transcripts with a single space between segments. Write one
  UTF-8 `text.txt` per collection, without a trailing newline, then a CSV with
  one absolute `file` path per collection in first-encounter order.

This step does not deduplicate repeated readings or overlapping text, remove
book licenses, or restore gaps between aligned segments. Empty transcripts are
skipped, as in the original text extraction. Duplicate segment IDs keep their
last transcript and start time at their original position.

The resulting layout is:

```text
cases_and_punc/
  libriheavy_text_small.csv
  libriheavy_text_small/<collection>/text.txt
  libriheavy_text_medium.csv
  libriheavy_text_medium/<collection>/text.txt
  libriheavy_text_large.csv
  libriheavy_text_large/<collection>/text.txt
  tokenizers/                              # created in step 3
```

Each CSV has a single `file` column. For example:

```text
file
/path/to/libriheavy/libriheavy-long/processed_text/cases_and_punc/libriheavy_text_small/shortpoetry_039_librivox_64kb_mp3/text.txt
/path/to/libriheavy/libriheavy-long/processed_text/cases_and_punc/libriheavy_text_small/im_nobody_librivox_64kb_mp3/text.txt
```

These manifests produce 172 small, 862 medium, and 7,842 large collection files.
The extraction uses a temporary SQLite database under the output directory to
sort transcripts without loading the full corpus into RAM. Allow several GB of
temporary disk space in addition to the final text files. Existing subset
directories or CSVs are rejected; use a fresh output directory for a new run.

## 3. Train the custom BPE tokenizer

The text LM configs all use `lh-large_bpe4096`, trained on **small + medium + large**:

```bash
python scripts/prepare_libriheavy_text.py train-tokenizer \
  --text-dir "$LH_TEXT_DIR" \
  --train-data lh-large \
  --vocab-size 4096
```

This trains `tokenizers.models.BPE()` from scratch with `ByteLevel()`
pre-tokenization and an empty normalizer sequence. It preserves case, accents,
and spaces, and adds these special tokens in order:
`<s>` (0), `<pad>` (1), `</s>` (2), `<unk>` (3). Its post-processor prepends
`</s>` to each encoded sequence (`</s> $A`). The saved tokenizer has no decoder.
It writes:

```text
$LH_TEXT_DIR/tokenizers/lh-large_bpe4096/tokenizer.json
```

To build all six subset tokenizers, run these commands **instead of**
the single command above:

```bash
for train_data in lh-small lh-medium lh-large; do
  for vocab_size in 4096 1; do
    python scripts/prepare_libriheavy_text.py train-tokenizer \
      --text-dir "$LH_TEXT_DIR" \
      --train-data "$train_data" \
      --vocab-size "$vocab_size" || exit 1
  done
done
```

`lh-small` uses small, `lh-medium` uses small + medium, and `lh-large` uses
all three subsets, reading whole collection texts in CSV order. A vocabulary
target of 1 retains the observed byte alphabet plus special tokens and learns
no merges. These outputs are named by their actual vocabulary sizes:
`lh-small_bpe174`, `lh-medium_bpe178`, and `lh-large_bpe191`. They remain
byte-level BPE tokenizers rather than Unicode character tokenizers.

Existing tokenizer output directories are rejected. Tokenizer training needs
the `tokenizers` dependency installed by the LM environment; these commands were
verified with `tokenizers==0.22.2`. Record its version with the generated
artifacts; rebuilding a vocabulary for an existing checkpoint requires matching
its token IDs.

## 4. Set the text LM config paths

Set the training CSVs and tokenizer path to the generated outputs. For example,
the small text LM config uses:

```yaml
slm:
  feature_extraction:
    text_tokenizer_path: /path/to/libriheavy/libriheavy-long/processed_text/cases_and_punc/tokenizers/lh-large_bpe4096/tokenizer.json
    manifests_train:
    - /path/to/libriheavy/libriheavy-long/processed_text/cases_and_punc/libriheavy_text_small.csv
    manifests_valid:
    - /path/to/validation_text.csv
```

The medium config trains on small + medium, and the large config trains on
small + medium + large.

Validation is a separate collection-text CSV and is not generated by the helper.
Provide a validation CSV with the same single `file` column pointing to your
validation text files, and set `manifests_valid` to that CSV's absolute path.

Continue with [LM preparation and training](../../configs/lm/README.md#launch).
The LM text preparation stage subsequently applies its configured license
filtering and sentence splitting before encoding these collection texts.
