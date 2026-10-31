# SLM Pair Manifest Format

## Overview

Audio pair manifests are CSV files describing SLM (Spoken Language Model)
evaluation pairs. Each row represents a **correct** audio and an **incorrect**
audio. Evaluation compares their LM log-likelihoods; accuracy is the fraction of
pairs where the correct audio scores higher.

To create the per-audio `fileid,path,num_frames` CSVs used by LM experiment
configs, follow [Create speech LM evaluation manifests](../docs/data/lm-benchmarks.md).
The combined LM pipeline extracts units from those inputs and creates
`test/pairs/*.jsonl` files for training validation and evaluation. This document
describes the separate audio pair CSV output of
`scripts/create_slm_pair_manifests.py`.

## CSV Schema

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `fileid_correct` | `str` | yes | Identifier of the correct (positive) audio file |
| `fileid_incorrect` | `str` | yes | Identifier of the incorrect (negative) audio file |
| `path_correct` | `str` | yes | Absolute path to the correct (positive) audio file |
| `path_incorrect` | `str` | yes | Absolute path to the incorrect (negative) audio file |

**Requirements:**
- Standard CSV with a header row.
- Both audio files must be single-channel WAV at 16 kHz.
- The `fileid_*` columns preserve the original benchmark identifiers for downstream analysis.
- Extra columns beyond these four are silently ignored.

## Example

```csv
fileid_correct,fileid_incorrect,path_correct,path_incorrect
2490_1c422263-..._correct,2491_1c422263-..._incorrect,/data/tSC/2490_1c422263-..._correct.wav,/data/tSC/2491_1c422263-..._incorrect.wav
780_f3b7819d-..._correct,781_f3b7819d-..._incorrect,/data/tSC/780_f3b7819d-..._correct.wav,/data/tSC/781_f3b7819d-..._incorrect.wav
```

## File naming convention

One CSV per benchmark, stored in a single directory:

```
<base_dir>/
├── tSC.csv
├── sSC.csv
├── lexical_dev_invocab.csv
├── lexical_dev_oov.csv
├── syntactic_dev.csv
└── ...
```

The script uses the benchmark name to read `<manifests_dir>/{name}.csv` and write
`<output_dir>/{name}.csv`. Keep the original benchmark names. The current LM
recipe resolves validation metrics to
`valid_metrics.base_dir / "test/pairs/{name}.jsonl"`.

## Source benchmarks

| Benchmark | Pairing logic | Source |
|-----------|---------------|--------|
| **tSC** / **sSC** (StoryCloze) | Files share a UUID; suffix is `_correct` or `_incorrect` | Individual-file manifest |
| **lexical_\*** (ZR21) | `gold.csv` groups by `[id, voice]`; `correct=1` vs `0` | Individual-file manifest + `gold.csv` |
| **syntactic_\*** (ZR21) | `gold.csv` groups by `[id, voice, subtype]`; `correct=1` vs `0` | Individual-file manifest + `gold.csv` |

After creating per-audio manifests, generate audio pair CSVs with:

```bash
python scripts/create_slm_pair_manifests.py \
  --manifests-dir /path/to/eval_manifests/raw \
  --audio-root /path/to/zr-data/datasets \
  --output-dir /path/to/eval_manifests/pairs \
  --benchmarks \
    lexical_dev_invocab lexical_dev_oov \
    lexical_test_invocab lexical_test_oov \
    syntactic_dev syntactic_test tSC sSC
```

`--audio-root` is the parent of `sLM21-dataset/`, which contains the lexical and
syntactic `gold.csv` files. The two directories passed to `--manifests-dir` and
`--output-dir` should be different because both formats use the same filenames.
