# SpidR 2.0 encoder experiments

Train speech encoders with native PyTorch in `gslm2`. Install `.[native]` using
the [environment guide](../../docs/environment.md).

## Choose a config

| Config | Training data |
|---|---|
| [LibriSpeech](spidr2.0-ls960.yaml) | LibriSpeech train-960 |
| [VoxPopuli English](spidr2.0-voxpopuli-en.yaml) | English 6kh |
| [VoxPopuli French](spidr2.0-voxpopuli-fr.yaml) | French 6kh |
| [VoxPopuli German](spidr2.0-voxpopuli-de.yaml) | German 6kh |

The templates use 400,000 training steps on two nodes with eight GPUs each.
Fields omitted from YAML use the [configuration defaults](../../src/gslm2/encoder/config/common.py).

## Prepare data and paths

Follow [LibriSpeech training preparation](../../docs/data/librispeech.md#encoder-training).
For other corpora, prepare mono 16 kHz audio and a training CSV in the
[audio manifest format](../../docs/data/formats.md#audio-manifests). See
[shared setup](../../docs/data.md#before-you-start) for the manifest writer and path conventions.

| Field | Value |
|---|---|
| `run.workdir` | Absolute run directory for checkpoints and metrics |
| `data.manifest` | Training manifest |
| `validation[].config.data.manifest` | Loss validation manifest (CSV or [Fairseq TSV](../../docs/data/formats.md#audio-manifests)) |
| `validation[].config.subset.manifest`, `item_file` | ABX audio manifest and triphone labels |
| `run.slurm_validation` | Account, QoS, and resources for separate validation jobs |
| `run.wandb_project`, `wandb_name`, `wandb_mode` | Logging settings; templates default to offline |

Replace `ACCOUNT` in `run.slurm_validation` as well as in the launch command.
ABX manifests must match their `.item` triphone labels.
Remove unavailable validation entries, or set `validation: []` to train without validation.

## Launch

Launch with your Slurm account and QoS:

```bash
conda activate gslm2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
gslm2-train-encoder -A ACCOUNT -N 2 -G 8 -q QOS -t 4320 \
  --dump /path/to/runs/spidr2.0-ls960 \
  configs/encoder/spidr2.0-ls960.yaml
```

Set `--dump` to the same absolute path as `run.workdir` to keep submission files
and trainer outputs together. Adjust `-t MINUTES` for your job's time limit and
add `-p PARTITION` if required by your cluster. Run `gslm2-train-encoder --help` for
all resource options. `gslm2-train` is an alias for `gslm2-train-encoder`.

If you launch through `srun`, request explicit CPU and memory resources
(e.g. `--cpus-per-task=16 --mem=200G`) so the data loaders have sufficient resources.

Validation launches separate Slurm jobs at saved checkpoints. If you set a
validation entry's `eval_interval`, use a positive multiple of `run.save_interval`.

## Outputs and resuming

Training writes these files under `run.workdir`:

| File | Contents |
|---|---|
| `config.json` | Resolved run and model settings |
| `step_N.pt` | Model, optimizer, scheduler, and training progress |
| `final.pt` | Checkpoint at the end of training |
| `metrics.jsonl` | Training and validation metrics |
| `validation/` | Validation job submissions and logs |

To resume, rerun training with the same config and `run.workdir`. The trainer
restores the saved model and training state automatically. For Slurm launches,
also reuse the same `--dump` directory.
Use a new run directory for a fresh experiment.

Use a saved checkpoint for [encoder evaluation](../../docs/reproducing.md#encoder-evaluation-tabencoder-eval-figfrontend-eval)
or [speech-unit extraction](../../docs/reproducing.md#unit-extraction-slm--gslm2-inputs).
Keep `config.json` alongside the checkpoint when loading it for inference.

## Run directly

On a machine with GPUs, use the same experiment YAML. Remove `run.slurm` and
`run.slurm_validation` from the config, or set both to `null`:

```bash
conda activate gslm2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENCODER_CONFIG=configs/encoder/spidr2.0-ls960.yaml
torchrun --standalone --nproc_per_node=8 \
  -m gslm2.encoder.training.trainer "$ENCODER_CONFIG"
```

Set `--nproc_per_node` to the number of GPUs on the machine.
For multiple nodes, replace `--standalone` with torchrun rendezvous arguments
(`--nnodes`, `--node-rank`, `--rdzv-backend=c10d`, and
`--rdzv-endpoint=<host>:<port>`). Start `torchrun` on each node, using shared paths
for the data and `run.workdir`.

Automatic encoder validation requires Slurm. Outside a Slurm job, due evals
are skipped; run the [encoder evaluation commands](../../docs/reproducing.md#encoder-evaluation-tabencoder-eval-figfrontend-eval)
manually when you need scores.
