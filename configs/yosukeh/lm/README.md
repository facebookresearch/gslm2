# Yosukeh LM experiments

These nine configs retain local data, checkpoint, tokenizer, and output paths
from the source experiments:

```text
/storage/home/yosukeh/metasr-internal/metasr-internal-main/configs/yosukeh/gslm2/
```

[Public copies](../../lm/README.md#choose-an-experiment) preserve the same model,
training, and benchmark settings with editable paths. The local configs use
the current fairseq2 recipe schema and seed 0.

## Local settings

- `slm.common_folder` is `/checkpoint/devai/yosukeh/gslm2_test/<experiment_name>`.
- Manifest, encoder, tokenizer, and results paths are prefilled for the local installation.
  To relocate data, follow the [path guide](../../lm/README.md#set-your-paths).
- Each YAML header records its original resources. The six-stream 6kh example
  uses `--qos=h200_devai_high --nodes=4` with:
  `configs/yosukeh/lm/gslm2/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m.yaml`.

## Launch

Follow the [LM launcher guide](../../lm/README.md#launch) with a YAML from this
directory. [Stage selection and resuming](../../lm/README.md#select-stages-or-resume),
[outputs](../../lm/README.md#outputs), and [direct commands](../../lm/README.md#run-stages-directly)
use the same workflow as the public templates.
