#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Usage: sbatch --account=ACCOUNT --qos=QOS slurm_generative.sbatch CONFIG.yaml STAGE [overrides]
# STAGE: decode | synthesize | score | judge
# Judge: override --nodes=2 --gpus-per-node=2 --ntasks-per-node=2 --mem=256G.
#SBATCH --job-name=generative-eval
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=240
#SBATCH --mem=400G

set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch $0 CONFIG.yaml STAGE [overrides]" >&2
    exit 2
fi
CONFIG_PATH=$1
STAGE=$2
shift 2
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
srun python -m gslm2.eval.generative "${CONFIG_PATH}" --stage "${STAGE}" "$@"
