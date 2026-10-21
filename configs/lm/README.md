# Language model experiments (fairseq2)

The nine experiments in `slm/`, `gslm2/`, and `textlm/` are public templates
based on [`configs/yosukeh/lm`](../yosukeh/lm/README.md). They include
`slm.feature_extraction`, `slm.train`, and `slm.eval` for preparation, training,
and evaluation with one config.

- `slm/` — SpidR layer-7 units on Libri-Light 600h, 6kh, and 60kh.
- `gslm2/` — six aligned SpidR streams, layers 5–10, on the same subsets.
- `textlm/` — BPE-4096 LibriHeavy small, small+medium, and small+medium+large.

The model presets, batching, optimizer, schedule, validation, and benchmark
lists follow the source experiments. `max_num_tokens` is a per-rank budget;
each YAML header gives the original node count, with eight GPUs per node.

## Set your paths

Replace every `/path/to/...` before launching:

- `slm.common_folder` and `slm.output_results_file`: your run and results paths.
- `slm.feature_extraction.checkpoint`: your SpidR checkpoint for speech units.
- `slm.feature_extraction.manifests_train`, `manifests_valid`, and
  `manifests_test`: your corpus and benchmark manifests. Keep benchmark file
  stems because they identify the scoring task.
- `slm.feature_extraction.slm21_root`: your downloaded ZeroSpeech sLM21 dataset
  directory, containing the lexical/syntactic `gold.csv` labels. This setting
  is used for speech benchmark pairing.
- `slm.feature_extraction.text_tokenizer_path`: your BPE-4096 tokenizer JSON
  for text experiments.

See [DATA.md](../../DATA.md) for audio/text manifest formats. CSV/JSONL records
must also point to your own audio and text files. The text benchmark inputs
are JSONL pairs; speech benchmark inputs are audio CSV manifests.

The templates include LTswap benchmarks. Remove entries from `manifests_test`
for benchmarks you do not have, and adjust
`slm.train.dataset.valid_metrics.names` to refer only to included benchmarks.
W&B uses project `gslm2`; configure or disable it under
`slm.train.common.metric_recorders.wandb`.

## Launch

Install the `lm` extra in `gslm2_lm` using the
[installation guidance](../../README.md#installation). Submit from the repo
root, replacing the account and QoS with your own:

```bash
sbatch --account=ACCOUNT --qos=QOS --nodes=4 \
  scripts/launchers/slurm_lm_steps.sbatch \
  configs/lm/gslm2/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m.yaml
```

The launcher uses `gslm2_lm` for all stages. Arguments are
`CONFIG [START_STEP=1] [END_STEP=3] [RUN_NAME]`: step 1 prepares data, step 2
trains and validates, and step 3 evaluates retained checkpoints. Starting at
step 2 or 3 requires an existing `RUN_NAME`, such as `SLM_12345`.

Outputs live under `slm.common_folder/<RUN_NAME>/`, with speech files under
layer `7/` or `5/` and text files directly under the run root. The data
directory contains `train/`, `valid/`, `test/pairs/`, and model checkpoints in
`ckpt/`. Prepared data and checkpoints are retained after evaluation.

## Standalone recipes

The existing training-only recipe templates, including VoxPopuli, are under
`recipes/{slm,gslm2,textlm}/`. Set `dataset.path` to your pre-extracted
`train/` and `valid/` shards, then run the fairseq2 recipe directly:

```bash
conda activate gslm2_lm
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.lm.recipes.train.run_recipe \
  --config-file configs/lm/recipes/slm/ll-600h-90m.yaml runs/slm
```

Speech IDs use `dataset.unit_offset=2`; text tokenizer IDs use offset 0.
See [REPRODUCING.md](../../REPRODUCING.md) for standalone checkpoint scoring.
