# Data formats

See [dataset preparation](../data.md#prepare-datasets) for commands that create these files.

## Audio manifests

Source audio manifests are CSVs with a header row:

| Column | Meaning |
|---|---|
| `fileid` | Audio filename without its extension |
| `path` | Absolute path to the audio file |
| `num_frames` | Number of waveform samples |
| `language` | Optional language label |

Audio must be mono at 16 kHz. The LM pipeline reads these CSVs to produce unit
shards and evaluation pairs.

Encoder audio readers also accept Fairseq-style `.tsv` manifests: the first line
is the absolute audio root, followed by tab-separated relative paths and sample counts.
Add `--fairseq` to the manifest writer to generate this format.

## Unit manifests (SLM / GSLM2)

The encoder produces raw code IDs through `gslm2-extract-units`. The fairseq2
LM reader adds the reserved-token offset when loading them.

- **Training and validation:** `train/` and `valid/` contain
  `<dataset>_<rank>.txt` shards, one utterance per line. Single-stream lines
  contain space-separated code IDs. Multi-stream lines concatenate aligned
  streams as equal-length blocks.
- **Test utterances:** `test/` contains JSONL records with file paths.
  Multi-stream records use `|` separators between streams.
- **Evaluation pairs:** JSONL records contain `positive` and `negative`
  objects, each with `units`. Speech pairing is handled by
  `gslm2.lm.eval.create_pairs`.

For a recipe-only run, set `dataset.path` to the layer directory containing
`train/` and `valid/`, with at least one `<dataset>_<rank>.txt` shard per rank.

## Text tokens

Text LMs read pretokenized integer `.txt` shards under `train/` and `valid/`.
Set `unit_offset=0`; the vocabulary includes special tokens.

Packed BPE records begin with `</s>` (ID 2). The reader uses that delimiter to
separate documents. Padding uses ID 1, so choose a tokenizer with those IDs.

The combined pipeline uses `gslm2.lm.tokenize` for preprocessing and benchmark
tokenization. Training CSVs have a `file` column naming source text files.
Set `text_tokenizer_path` to a tokenizer JSON and `vocab_size` to its
vocabulary size. Tokenized benchmark pairs use the same positive/negative
JSONL schema as speech LMs.

## Duration and vocoder inputs

Duration and vocoder inputs contain full-rate raw encoder code IDs. See
[full-rate extraction and output routing](../synthesis.md#prepare-training-data)
for extraction settings, source manifests, and the files each trainer reads.

For other full-rate JSONL inputs, use
[`gslm2-prepare-synthesis-data`](../synthesis.md#prepare-other-full-rate-jsonl-inputs)
to write canonical vocoder manifests and duration TXT shards.

## Speech benchmark pairing

The [lexical splitter](../../scripts/split_lexical_manifests.py) classifies each
pair using the **correct word's** `frequency` in `gold.csv`:

- `frequency > 0`: invocab.
- `frequency == 0`: OOV.

Both members of each `(id, voice)` pair stay in the same category. Incorrect
rows have blank frequencies, so filtering individual rows by frequency would
discard the incorrect examples.

The lexical split rule reproduces the existing experiment manifests' file IDs.
CSV row order and the legacy unnamed index column do not affect pairing.

## LT-Swap release metadata

The converter writes speech CSVs with absolute local audio paths and text JSONLs
with stable file IDs in the `positive` / `negative` schema. Sample counts and
frequency labels come from the release metadata.

Each line in `pairs/<task>.jsonl` contains:

- `pair_id`, `task`, and `freq_bin`.
- `correct` and `incorrect` objects with `fileid`, `text`, `audio`, and
  `num_frames`.

For example, a record in `pairs/agrswap.jsonl` has this shape:

```json
{
  "pair_id": "example_pair",
  "task": "agrswap",
  "freq_bin": "0-1",
  "correct": {
    "fileid": "example_pair_correct",
    "text": "The dog runs.",
    "audio": "audio/agrswap/example_pair_correct.wav",
    "num_frames": 16000
  },
  "incorrect": {
    "fileid": "example_pair_incorrect",
    "text": "The dog run.",
    "audio": "audio/agrswap/example_pair_incorrect.wav",
    "num_frames": 16000
  }
}
```

Audio paths are relative to the dataset variant directory.
File IDs must match the WAV stems and use
`<pair_id>_correct` / `<pair_id>_incorrect` for speech pairing.
`num_frames` is the sample count of the released mono 16 kHz WAV.
The `freq_bin` label determines benchmark membership.
Records may also include provenance fields.

## Prompt boundaries

Each prompt record provides `fileid`, `prompt_secs`, and `prompt_frames`, along
with the original prompt text and word-alignment metadata. The supplied
word-boundary cutoffs are near three seconds. Extraction crops the waveform
before encoding and pads it as needed for the supplied SpidR frame count.
