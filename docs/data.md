# Data

Prepare source manifests for your dataset, then extract speech units or tokenize
text with the training or evaluation pipeline.

## Prepare datasets

- [Libri-Light](data/librilight.md): segment audio and prepare LM training and validation manifests.
- [LJSpeech](data/ljspeech.md): resample audio and prepare fixed vocoder splits.
- [ZeroSpeech and StoryCloze](data/lm-benchmarks.md): create speech benchmark manifests.
- [French and German tSC and sBLIMP](data/lm-benchmarks.md#french-and-german-tsc-and-sblimp):
  download text pairs and synthesize audio for SLM evaluation.
- [LT-Swap (LongTail-Swap)](data/ltswap.md): convert a release into speech and text benchmark inputs.
- [LibriSpeech preparation](data/librispeech.md): prepare training manifests and fixed generative evaluation inputs.
- [VoxPopuli segmentation](data/voxpopuli.md): reconstruct audio and manifests from released RTTM selections.
- [Encoder evaluation labels](data/encoder-eval-labels.md): ABX items, MAP words, and codebook phone alignments.
- [LibriHeavy transcripts](data/libriheavy-text.md): text LM training data.

## Format reference

[Data formats](data/formats.md) describes audio CSVs, unit shards, text tokens,
duration and vocoder inputs, LT-Swap metadata, and prompt boundaries.
The separate [audio pair CSV format](../scripts/PAIR_MANIFEST_FORMAT.md) is used
for optional pair exports.

## Before you start

- Use the [environment and command conventions](../README.md#installation) for your task.
- Speech pipelines expect mono audio at 16 kHz. Keep the benchmark filenames
  shown in the guides: the LM pipeline uses their stems to select pairing logic.
- Direct-download examples use `curl -fL` with an explicit output path;
  `--continue-at -` resumes large downloads. Verify published checksums before
  extracting archives or preparing manifests.

For other audio directories, create a source CSV with:

```bash
python -m gslm2.encoder.data.write_manifest /path/to/audio /path/to/manifest.csv --ext .flac
```

## Corpora

| Corpus | Used for | Scale |
|---|---|---|
| LibriSpeech | SpidR 2.0 encoder and generative evaluation | 960 h training; fixed dev/test selections |
| Libri-Light (`unlab-600/6k/60k`, VAD targeting 60 s) | SLM / GSLM2 | 600 h / 6Kh / 60Kh |
| Libriheavy-long (multi-sentence segments) | Text LM | Transcripts of Libri-Light |
| LJSpeech-1.1 | Unit vocoder | 13,100 utterances |
| VoxPopuli (En/Fr/De, VAD+LID segmented, 0.5–60 s) | Multilingual encoder + SLM | 6Kh per language |
