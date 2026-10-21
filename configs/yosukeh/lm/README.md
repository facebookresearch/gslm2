# Yosukeh LM experiments

Public copies with editable paths are under
[`configs/lm`](../../lm/README.md).

These nine experiment configs are derived from the top-level YAMLs in:

```text
/storage/home/yosukeh/metasr-internal/metasr-internal-main/configs/yosukeh/gslm2/
```

- `slm/` — SpidR layer-7 units on Libri-Light 600h, 6kh, and 60kh.
- `gslm2/` — six aligned SpidR streams, layers 5–10, on the same subsets.
- `textlm/` — BPE-4096 LibriHeavy small, small+medium, and small+medium+large.

Each YAML contains `slm.feature_extraction`, `slm.train`, and `slm.eval`.
The manifests, encoder checkpoint or text tokenizer, output paths, training
settings, and evaluation benchmarks come from the source experiment.
`slm.train` uses the current fairseq2 recipe schema. Speech IDs use offset 2;
text IDs use offset 0. The seed is 0.

## Launch

Submit from the repository root with one config to run preparation, training,
and evaluation sequentially:

```bash
sbatch --qos=h200_devai_high --nodes=4 \
  scripts/launchers/slurm_lm_steps.sbatch \
  configs/yosukeh/lm/gslm2/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m.yaml
```

Each YAML header gives the original node count. All runs use eight GPUs per
node, and `max_num_tokens` remains a per-rank budget.

Install the `lm` extra in `gslm2_lm` using the
[installation guidance](../../../README.md#installation). The launcher uses
that environment for all three stages, including speech-unit extraction or
text tokenization. It includes fairseq2, audio decoding, and spaCy for sentence
splitting. The launcher always activates `gslm2_lm`.

## Steps and outputs

Launcher arguments are `CONFIG [START_STEP=1] [END_STEP=3] [RUN_NAME]`.
The default runs all three steps. Select an inclusive range, or use the same
start and end step to run only one stage:

```bash
# Preparation only; uses SLM_<new_job_id>.
sbatch scripts/launchers/slurm_lm_steps.sbatch CONFIG 1 1

# Train and evaluate using data prepared by job 12345.
sbatch scripts/launchers/slurm_lm_steps.sbatch CONFIG 2 3 SLM_12345

# Training only, resuming saved training state when present.
sbatch scripts/launchers/slurm_lm_steps.sbatch CONFIG 2 2 SLM_12345

# Evaluation only, using the retained checkpoints from that run.
sbatch scripts/launchers/slurm_lm_steps.sbatch CONFIG 3 3 SLM_12345
```

Replace `CONFIG` with the experiment YAML and use the required Slurm resources.
Starting at step 2 or 3 requires an existing `RUN_NAME`. All stages use that
name to resolve data and checkpoints under `slm.common_folder`. When configured,
`run_postfix` is appended to the supplied name.

1. Speech extraction writes training and validation unit shards and builds
   tokenized benchmark pairs. Multi-stream extraction preserves alignment and
   applies joint deduplication. Text preparation tokenizes the configured
   text manifests and benchmark pairs using the original BPE tokenizer.
2. Fairseq2 trains, validates, and saves checkpoints using `slm.train`.
3. Evaluation scores the retained checkpoints on every benchmark listed in
   `slm.feature_extraction.manifests_test`, including LTswap suites.

Each config sets `slm.common_folder` to
`/checkpoint/devai/yosukeh/gslm2_test/<experiment_name>`. The run output root
is `slm.common_folder/<RUN_NAME>/`, defaulting to `SLM_<job_id>` (with
`run_postfix` appended when configured). Speech files live under layer `7/` or `5/`;
text files live directly under the output root. That data directory contains
`train/`, `valid/`, `test/pairs/`, and the training directory `ckpt/`.
The launcher resolves all paths automatically. W&B run names use the selected run name.
Results append to `slm.output_results_file`, with run and checkpoint metadata.
Prepared data and model checkpoints are retained after evaluation. A failed
step stops the job before the next step starts.
Slurm writes logs to `slurm-LM_<job_id>.out` and `slurm-LM_<job_id>.err`
in the submission directory.
