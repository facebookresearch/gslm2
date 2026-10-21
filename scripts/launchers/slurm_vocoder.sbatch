#!/bin/bash

# Train a unit HiFi-GAN vocoder on non-deduplicated extracted features.
#
# Usage:
#   sbatch scripts/launchers/slurm_vocoder.sbatch \
#     configs/misc/vocoder/single-stream.yaml
#
# batch_size in the config is per rank, so the effective batch is 8x it here.

#SBATCH --job-name=unit-vocoder
#SBATCH --output=slurm/%x_%j/output.log
#SBATCH --error=slurm/%x_%j/error.log
#SBATCH --account=devai
#SBATCH --qos=h100_devai_high
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=720
#SBATCH --mem=400G

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 CONFIG_PATH" >&2
    exit 2
fi

CONFIG_PATH=$1
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file does not exist: $CONFIG_PATH" >&2
    exit 2
fi
echo "---- Start date: $(date)"
echo "Running on ${SLURM_NNODES} node(s)"
echo "Job id: ${SLURM_JOB_ID}"
echo "Config path: ${CONFIG_PATH}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gslm2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
conda info --envs | grep '*'

PYTHONPATH=src srun --kill-on-bad-exit=1 python -m gslm2.vocoder "${CONFIG_PATH}"

echo "---- End date: $(date)"
