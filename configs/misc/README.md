# Duration, vocoder, and extraction configs

| Component | Single stream (SpidR 2.0 L7) | Six streams (SpidR 2.0 L5–L10) |
|---|---|---|
| Extraction | [L7 codes](extract/spidr2-slm_codes-l7-dup.yaml) | [L5–L10 codes](extract/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10-dup.yaml) |
| Duration prediction | [L7 duration](duration/spidr2-slm_codes-l7.yaml) | [L5–L10 duration](duration/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml) |
| Unit vocoder | [L7 vocoder](vocoder/spidr2-slm_codes-l7.yaml) | [L5–L10 vocoder](vocoder/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml) |

## Paths to edit

| Config | Fields |
|---|---|
| Extraction | `checkpoint`, manifests under `targets`, `output_folder` |
| Duration prediction | `train_dirs`, `valid_dirs`, `output_dir` |
| Unit vocoder | `train_manifest`, `val_manifest`, `workdir` |

Follow [Speech synthesis](../../docs/synthesis.md#prepare-training-data) for source
manifests, extraction, output mapping, training commands, and audio generation.
Keep the selected template's stream and deduplication settings.

## Optional generative evaluation

See the [generative evaluation configs](generative_eval/README.md) for dev/test
templates and the [workflow guide](../../docs/generative-evaluation.md) for setup,
stage commands, and sweeps.
