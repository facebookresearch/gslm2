# SLM Pair Manifest Format

## Overview

Pair manifests are CSV files used for SLM (Spoken Language Model) evaluation
during training. Each row represents one evaluation pair: a **correct** audio
and an **incorrect** audio. At validation time the LM assigns a log-likelihood
to each; accuracy = fraction of pairs where the correct audio scores higher.

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

The file stem must match the metric name used in the recipe YAML config:

```yaml
dataset:
  valid_metrics:
    base_dir: /path/to/pair_manifests/
    names:
      - "tSC"
      - "lexical_dev_invocab"
      - "syntactic_dev"
    max_pairs: 1000
    batch_size: 32
```

The recipe resolves each metric to `base_dir / "{name}.csv"`.

## Source benchmarks

| Benchmark | Pairing logic | Source |
|-----------|---------------|--------|
| **tSC** / **sSC** (StoryCloze) | Files share a UUID; suffix is `_correct` or `_incorrect` | Individual-file manifest |
| **lexical_\*** (ZR21) | `gold.csv` groups by `[id, voice]`; `correct=1` vs `0` | Individual-file manifest + `gold.csv` |
| **syntactic_\*** (ZR21) | `gold.csv` groups by `[id, voice, subtype]`; `correct=1` vs `0` | Individual-file manifest + `gold.csv` |

Use `scripts/create_slm_pair_manifests.py` to generate pair manifests from the
original SLM benchmark data.
