#!/bin/bash

# Train a unit HiFi-GAN vocoder on non-deduplicated extracted features.
#
# Usage:
#   sbatch src/metasr_misc/unit_vocoder/slurm_train.sbatch \
#     configs/yosukeh/gslm2/unit_vocoder/spidr2-slm_codes-l7-dup.yaml
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
echo "---- Start date: $(date)"
echo "Running on ${SLURM_NNODES} node(s)"
echo "Job id: ${SLURM_JOB_ID}"
echo "Config path: ${CONFIG_PATH}"

source /home/yosukeh/miniconda3/etc/profile.d/conda.sh
conda activate metasr_fs2
conda info --envs | grep '*'

PYTHONPATH=src srun python -m metasr_misc.unit_vocoder "${CONFIG_PATH}"

echo "---- End date: $(date)"
