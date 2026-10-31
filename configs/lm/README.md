# Language model experiments (fairseq2)

Templates contain `slm.feature_extraction`, `slm.train`, and `slm.eval`
for preparation, training, and evaluation with one config.

## Choose an experiment

| Directory | Inputs | Training subsets |
|---|---|---|
| `slm/` | SpidR 2.0 layer-7 units | Libri-Light 600h, 6kh, or 60kh |
| `gslm2/` | Six aligned streams, layers 5–10 | Libri-Light 600h, 6kh, or 60kh |
| `textlm/` | BPE-4096 text | LibriHeavy small, small+medium, or small+medium+large |

The nine templates specify model presets, batching, optimizer, schedule, validation,
and benchmark settings. Each YAML header gives the original node count, with
eight GPUs per node. `max_num_tokens` is a per-rank budget.

> [!NOTE]
> The labels `125m`, `350m`, and `1b` refer to nominal model sizes based on the OPT
> paper. The actual parameter counts reported in our paper are approximately
> 90M, 300M, and 1.2B, respectively.

## Set your paths

| Field | Value |
|---|---|
| `slm.common_folder`, `slm.output_results_file` | Run directory and results file |
| `slm.feature_extraction.checkpoint` | SpidR 2.0 checkpoint for speech units |
| `slm.feature_extraction.manifests_train`, `manifests_valid`, `manifests_test` | Training, validation, and benchmark manifests |
| `slm.feature_extraction.slm21_root` | ZeroSpeech `sLM21-dataset/` containing `gold.csv` labels |
| `slm.feature_extraction.text_tokenizer_path` | BPE-4096 tokenizer JSON for text experiments |

Manifest records must point to your local audio or text. Speech benchmarks use
audio CSVs; text benchmarks use JSONL pairs. Keep benchmark filenames because
their stems select the pairing logic. See [Data formats](../../docs/data/formats.md).

Preparation guides: [Libri-Light](../../docs/data/librilight.md),
[ZeroSpeech and StoryCloze](../../docs/data/lm-benchmarks.md), and
[LT-Swap](../../docs/data/ltswap.md). For LT-Swap, use the release variant named in
your config.

## Launch

Install `.[lm]` in `gslm2_lm` using the [environment guide](../../docs/environment.md).
Submit with your Slurm account, QoS, and the node count in your YAML header:

```bash
sbatch --account=ACCOUNT --qos=QOS --nodes=4 \
  scripts/launchers/slurm_lm_steps.sbatch \
  configs/lm/gslm2/spidr2-ms-slm_codes-l5-l6-l7-l8-l9-l10_ll-6kh_350m.yaml
```

The launcher activates `gslm2_lm` for every stage and stops if a stage fails.

| Step | Config section | Action |
|---|---|---|
| 1 | `slm.feature_extraction` | Extract speech units or tokenize text, then prepare benchmark pairs |
| 2 | `slm.train` | Train, validate, and select checkpoints with fairseq2 |
| 3 | `slm.eval` | Score retained checkpoints on the benchmarks in `manifests_test` |

Speech preparation preserves stream alignment and applies the config's
deduplication settings. Benchmark pairing uses the ZeroSpeech labels at
`slm.feature_extraction.slm21_root`.

## Select stages or resume

Launcher arguments are `CONFIG [START_STEP=1] [END_STEP=3] [RUN_NAME]`.
Stage ranges are inclusive.

Starting at step 2 or 3 requires the existing run's base `RUN_NAME`. Replace
`CONFIG` below with your YAML and add the account, QoS, and node count used for
the original launch:

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

Keep the same config paths and run name to reuse prepared data and checkpoints.

## Outputs

`RUN_NAME` defaults to `SLM_<job_id>`. If `slm.run_postfix` is set,
`_<run_postfix>` is appended; pass the base name when resuming.

The run root is `slm.common_folder/<RUN_NAME>/`, including any postfix. Speech
data lives under `7/` for one stream or `5/` for six streams; text data lives
directly under the run root.

| Path within the data directory | Contents |
|---|---|
| `train/`, `valid/` | Prepared unit or text token shards |
| `test/pairs/` | Tokenized benchmark pairs; multi-stream units use `|` separators |
| `ckpt/` | Training state and retained model checkpoints |

Evaluation appends run and checkpoint metadata to `slm.output_results_file`.
Data and checkpoints are retained by the supplied configs. W&B uses the selected
run name. Default Slurm logs are `slurm/LM_<job_id>.out` and `slurm/LM_<job_id>.err`.

## Run stages directly

On a GPU allocation, use the same experiment YAML and `SLM_RUN_NAME` for every stage:

```bash
conda activate gslm2_lm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
LM_CONFIG=configs/lm/slm/spidr2-slm_codes-l7_ll-6kh_350m.yaml
export SLM_RUN_NAME=SLM_example
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.lm.prepare --config_path "$LM_CONFIG"
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.lm.train --config_path "$LM_CONFIG"
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.lm.eval.benchmarks --config_path "$LM_CONFIG"
```

To reuse a prepared run, set its base `SLM_RUN_NAME` and invoke the needed stages.
`gslm2-train-lm` exposes the underlying fairseq2 recipe CLI.
