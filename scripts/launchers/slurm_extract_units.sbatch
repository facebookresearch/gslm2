#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Extract discrete speech units with one process per GPU.
# Submit from the repository root:
#   sbatch scripts/launchers/slurm_extract_units.sbatch \
#     configs/misc/extract/spidr2-slm_codes-l7-dup.yaml

#SBATCH --job-name=unit-extract
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --account=devai
#SBATCH --qos=h200_devai_high
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

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
CONFIG_PATH=$1
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config file does not exist: $CONFIG_PATH" >&2
    exit 2
fi

# Replace this Conda path and environment name with your own if needed.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gslm2_lm
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Job: $SLURM_JOB_ID; config: $CONFIG_PATH"
srun --kill-on-bad-exit=1 python -m gslm2.encoder.units --config "$CONFIG_PATH"
