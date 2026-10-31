#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Usage: sbatch --account=ACCOUNT --qos=QOS --array=0-LAST slurm_generative_array.sbatch TASKS.jsonl STAGE
# TASKS.jsonl is written by python -m gslm2.eval.generative.sweep.
# Override judge resources as described in slurm_generative.sbatch.
#SBATCH --job-name=generative-grid
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=240
#SBATCH --mem=400G

set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch --array=0-LAST $0 TASKS.jsonl STAGE" >&2
    exit 2
fi
TASK_MANIFEST=$1
STAGE=$2
case "${STAGE}" in
    decode|synthesize|score|judge) ;;
    *) echo "Invalid stage: ${STAGE}" >&2; exit 2 ;;
esac
REPOSITORY_ROOT=$(git rev-parse --show-toplevel)
cd "${REPOSITORY_ROOT}"
# Replace this Conda path and the environment names below with your own if needed.
source ~/miniconda3/etc/profile.d/conda.sh
if [[ "$STAGE" == "decode" ]]; then
    conda activate gslm2_lm
else
    conda activate gslm2
fi
# Resolve cuDNN and other runtime libraries from the active environment.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
srun python -m gslm2.eval.generative.sweep --task-manifest "${TASK_MANIFEST}" \
    --task-index "${SLURM_ARRAY_TASK_ID}" --stage "${STAGE}"
