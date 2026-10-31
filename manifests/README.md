# Fixed dataset selections

These bundled files preserve the dataset membership used by the experiments.
Paths below are relative to `manifests/`.

## LJSpeech

Each TXT file contains one utterance ID per line. The disjoint splits cover
all 13,100 LJSpeech-1.1 utterances.

| File | Split | Utterances |
|---|---|---:|
| [ljspeech_train.txt](ljspeech/ljspeech_train.txt) | Training | 12,600 |
| [ljspeech_valid.txt](ljspeech/ljspeech_valid.txt) | Validation | 250 |
| [ljspeech_test.txt](ljspeech/ljspeech_test.txt) | Test | 250 |

Preparation: [LJSpeech guide](../docs/data/ljspeech.md).
Source: [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/).

## LibriSpeech generative evaluation

Each split contains 1,024 utterances at least six seconds long.

| Asset | dev-clean | test-clean |
|---|---|---|
| Audio selection | [CSV](librispeech/librispeech_dev-clean_1k_min6s.csv) | [CSV](librispeech/librispeech_test-clean_1k_min6s.csv) |
| Prompt boundaries | [JSONL](librispeech/librispeech_dev-clean_1k_min6s_prompt_boundaries_3s.jsonl) | [JSONL](librispeech/librispeech_test-clean_1k_min6s_prompt_boundaries_3s.jsonl) |

CSV audio paths are relative to the extracted `LibriSpeech/` root:
`<split>/<speaker>/<chapter>/<fileid>.flac`.

Prompt JSONLs pair with the CSVs by `fileid`. They store word-boundary cutoffs
near three seconds, encoder frame counts, prompt text, and word-alignment metadata.

Preparation: [LibriSpeech evaluation guide](../docs/data/librispeech.md#generative-evaluation).
Source: [LibriSpeech, OpenSLR 12](https://www.openslr.org/12/).

## Generated files

Keep the bundled selections and prompt records fixed when comparing results.
Keep raw audio in the download directories; write converted WAVs and local
manifests to a separate output directory such as `build/`. Generated audio CSVs
contain absolute local paths.
