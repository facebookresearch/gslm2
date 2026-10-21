# Data

## Corpora

| Corpus | Used for | Scale |
|---|---|---|
| LibriSpeech (train-960) | SpidR 2.0 encoder | 960h |
| Libri-Light (`unlab-600/6k/60k`, official 60 s segmentation) | SLM / GSLM2 | 600h / 6Kh / 60Kh |
| Libriheavy-long (multi-sentence segments) | Text LM | transcripts of Libri-Light |
| VoxPopuli (En/Fr/De, VAD+LID segmented, 0.5–60 s) | multilingual encoder + SLM | 6Kh per language |

## Manifest format

Audio manifests are CSVs with columns `fileid,path,num_frames`
(`num_frames` = waveform samples at 16kHz), plus optional `language`.
Build one per corpus directory:

```bash
python -m gslm2.encoder.data.write_manifest <audio-dir> <out.csv> --ext .flac
```

Scales that combine Libri-Light subsets need a single merged CSV, e.g.:

```bash
# 6Kh = small + medium (keep one header row)
head -n 1 small_segmented_60s.csv > librilight-6kh.csv
tail -n +2 -q small_segmented_60s.csv medium_segmented_60s.csv >> librilight-6kh.csv
```

## Unit manifests (SLM / GSLM2)

The encoder produces raw code IDs through `gslm2-extract-units`. The fairseq2
LM dataset adds the reserved-token offset when reading these IDs.

- `train/` and `valid/`: `<dataset>_<rank>.txt` shards, one utterance per line. Single-stream
  lines contain space-separated code IDs. Multi-stream lines concatenate the
  aligned streams as equal-length space-separated blocks.
- `test/`: `.jsonl` utterance records with file paths. Multi-stream records
  represent their streams with `|` separators.
- Benchmark pairs: `.jsonl` records with `positive` and `negative` objects,
  each containing `units`. Pairing belongs to `gslm2.lm.eval.create_pairs`.

For a recipe-only run, point `dataset.path` at the layer directory containing
`train/` and `valid/`. Provide at least one `<dataset>_<rank>.txt` shard per distributed rank.
The combined LM pipeline can extract units and prepare benchmark pairs first.

## Text tokens

Text LMs use pretokenized integer `.txt` shards under `train/` and `valid/`.
Their dataset uses `unit_offset=0`; vocabulary size includes special tokens.
Packed BPE records begin with `</s>` (ID 2), and the reader uses that delimiter
to isolate documents. Padding uses ID 1. Use a tokenizer with those IDs.

`gslm2.lm.tokenize` contains text preprocessing and benchmark tokenization for
the combined pipeline. Its input training manifests are CSVs with a `file`
column naming source text files. Set `text_tokenizer_path` to a tokenizer JSON
and `vocab_size` to its vocabulary size in the pipeline preparation config.
Tokenized benchmark pairs use the same positive/negative JSONL schema as SLMs.

## Duration and vocoder inputs

Duration and vocoder training use full-rate raw encoder codes. Extract without
deduplication or chunking, and use `output_format: jsonl` to retain audio paths
for every split. `gslm2-prepare-synthesis-data` writes audio/unit JSONL manifests
for the vocoder and text shards for duration target construction. See
[SYNTHESIS.md](SYNTHESIS.md) for formats, stream order, and commands.
