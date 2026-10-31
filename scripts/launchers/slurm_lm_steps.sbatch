#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --job-name=LM
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --account=devai
#SBATCH --qos=h100_lowest
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=4320
#SBATCH --mem=400G

# Usage: sbatch ... slurm_lm_steps.sbatch CONFIG [START_STEP=1] [END_STEP=3] [RUN_NAME]
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "Usage: sbatch $0 CONFIG [START_STEP=1] [END_STEP=3] [RUN_NAME]" >&2
    exit 2
fi
CONFIG_PATH=$1
START_STEP=${2:-1}
END_STEP=${3:-3}
case "$START_STEP:$END_STEP" in
    [123]:[123]) ;;
    *) echo "Steps must be 1 (prepare), 2 (train), or 3 (evaluate)" >&2; exit 2 ;;
esac
if [ "$START_STEP" -gt "$END_STEP" ]; then
    echo "START_STEP must be <= END_STEP" >&2
    exit 2
fi
if [ "$START_STEP" -gt 1 ] && [ -z "${4:-}" ]; then
    echo "Provide RUN_NAME (e.g. SLM_12345) to reuse an existing run" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
export SLM_RUN_NAME=${4:-SLM_${SLURM_JOB_ID}}
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file does not exist: $CONFIG_PATH" >&2
    exit 2
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Replace this Conda path and environment name with your own if needed.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gslm2_lm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "Run: $SLM_RUN_NAME; steps: $START_STEP-$END_STEP"
if [ "$START_STEP" -eq 1 ]; then
    echo "Step 1: prepare data and benchmark pairs"
    srun --kill-on-bad-exit=1 python -m gslm2.lm.prepare --config_path "$CONFIG_PATH"
fi

if [ "$START_STEP" -le 2 ] && [ "$END_STEP" -ge 2 ]; then
    echo "Step 2: train and validate with fairseq2"
    srun --kill-on-bad-exit=1 python -m gslm2.lm.train --config_path "$CONFIG_PATH"
fi

if [ "$END_STEP" -eq 3 ]; then
    echo "Step 3: evaluate saved checkpoints"
    srun --kill-on-bad-exit=1 python -m gslm2.lm.eval.benchmarks --config_path "$CONFIG_PATH"
fi
