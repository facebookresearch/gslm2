#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


## Set output and job name
#SBATCH --job-name=SLM-pipeline
#SBATCH --output=slurm/%x_%j/output.log
#SBATCH --error=slurm/%x_%j/error.log
## partition name
#SBATCH --account=metaspeechssl
#SBATCH --qos=metaspeechssl_high

## number of nodes, tasks per nodes, etc
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=4320
#SBATCH --mem 400G

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export CUDA_LAUNCH_BLOCKING=1 # set to 1 for debugging only
#export HYDRA_FULL_ERROR=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "---- Start date: `date`"
echo "Running on $SLURM_NNODES nodes"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
echo "Job id is $SLURM_JOBID"


# Get parameters
CONFIG_PATH=$1
FEATURE_DIR=$2
START_STEP=${3:-1}  # Default to step 1 if not provided

echo "Config path: $CONFIG_PATH"
echo "Feature dir: $FEATURE_DIR"
echo "Starting from step: $START_STEP"

# Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate metasr_lm_code  # Replace with your own conda env
echo "Active Conda Environment:"
conda info --envs | grep '*'

# Run unified SLM pipeline
if [ -z "$FEATURE_DIR" ]; then
    srun python -m metasr_eval.slm.slm_pipeline --config_path $CONFIG_PATH
else
    srun python -m metasr_eval.slm.slm_pipeline --config_path $CONFIG_PATH --feature_dir $FEATURE_DIR --start_step $START_STEP
fi

echo "---- End date: `date`"
