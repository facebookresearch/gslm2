# Data

## Corpora

| Corpus | Used for | Scale |
|---|---|---|
| LibriSpeech (train-960) | SpidR 2.0 encoder | 960h |
| Libri-Light (`unlab-600/6k/60k`, official 60 s segmentation) | SLM / GSLM2 | 600h / 6Kh / 60Kh |
| Libriheavy-long (multi-sentence segments) | Text LM | transcripts of Libri-Light |
| VoxPopuli (En/Fr/De, VAD+LID segmented, 0.5–60 s) | multilingual encoder + SLM | 6Kh per language |
| LJ Speech (250 held-out utts) | unit vocoder | 24h |

## Manifest format

Audio manifests are CSVs with columns `fileid,path,num_frames`
(`num_frames` = waveform samples at 16kHz), plus optional `language`.
Build one per corpus directory:

```bash
python -m gslm2.data.speech.write_manifest <audio-dir> <out.csv> --ext .flac
```

Scales that combine Libri-Light subsets need a single merged CSV, e.g.:

```bash
# 6Kh = small + medium (keep one header row)
head -n 1 small_segmented_60s.csv > librilight-6kh.csv
tail -n +2 -q small_segmented_60s.csv medium_segmented_60s.csv >> librilight-6kh.csv
```

## Unit manifests (GSLM2 / vocoder / duration)

Multi-stream training, the vocoder, and the duration predictor consume
pre-extracted units. Extract them from a trained encoder (single stream,
layer 7, deduped; six streams, layers 5–10, cross-stream deduped; or
non-deduped vocoder features) with the feature-extraction CLI (see
`gslm2.eval.slm`), then point the configs at the outputs:

- GSLM2: one `.txt` per split, one line per utterance, the 6 streams as
  equal-length space-separated blocks (see `PreExtractedUnitSource`).
- Vocoder: per-split `.jsonl` unit manifests over LJ Speech (see
  `gslm2.vocoder.unit.data`).
- Duration: `.txt` unit/duration files over LL-600h (see
  `gslm2.vocoder.duration.dataset`).

## Text

- Per-scale BPE-4096 over the matching transcripts:
  `python scripts/train_tokenizer.py --type bpe --vocab-size 4096
  --train-file <scale>.txt --output-dir tokenizers/<scale>-bpe4096`
- Text-LM manifests are `.txt` (one document per line) or `.json` lists.
