# SpidR 2.0 duration predictors

These configs follow the SpidR 2.0 experiments in
`/storage/home/yosukeh/slm-adapt/metasr-internal/configs/yosukeh/gslm2/duration_predictor`.
Model and training settings match the references:

- `spidr2-slm_codes-l7.yaml`: layer 7; 128-dimensional embeddings and hidden states,
  six Conformer layers, and one unit stream.
- `spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml`: layers 5–10; 192-dimensional embeddings and hidden states,
  eight Conformer layers, and six aligned unit streams.

Both use 256 raw code IDs per stream, 50 duration classes, batch size 64,
25,000 training steps, learning rate 0.003, and 1,000 warmup steps.

## Data and outputs

The configs read our extracted, non-deduplicated TXT shards under
`/checkpoint/devai/yosukeh/gslm2_test/framewise_units/<extraction_name>/<layer>/duration_predictor/`.
Train and validation share that directory, with separate patterns:

- Training: `small_segmented*.txt` (Libri-Light small).
- Validation: `large_segmented_60s_dev*.txt` (Libri-Light development).

The duration dataset derives deduplicated units and frame durations from each
full-rate row. Multi-stream rows concatenate layers 5–10 in that order;
joint run-length encoding preserves their alignment. Use these full-rate
shards for duration targets; the deduplicated LM shards do not retain durations.

Checkpoints and metrics are written under
`/checkpoint/devai/yosukeh/gslm2_test/duration_predictor/<experiment_name>/`.

## Launch

Submit from the repository root using the installed `gslm2` environment:

```bash
sbatch scripts/launchers/slurm_duration.sbatch \
  configs/yosukeh/misc/duration/spidr2-slm_codes-l7.yaml

sbatch scripts/launchers/slurm_duration.sbatch \
  configs/yosukeh/misc/duration/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10.yaml
```

The launcher requests one GPU. Override `--qos` and `--time` as needed.
Edit the launcher's `conda activate` command to use another environment.
