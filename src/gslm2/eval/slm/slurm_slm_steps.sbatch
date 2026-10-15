#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


## Set output and job name
#SBATCH --job-name=SLM-pipeline
#SBATCH --output=slurm/%x_%j/output.log
#SBATCH --error=slurm/%x_%j/error.log
## partition name
#SBATCH --account=devai
#SBATCH --qos=h100_lowest

## number of nodes, tasks per nodes, etc
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=4320
#SBATCH --mem 400G


echo "---- Start date: `date`"
echo "Running on $SLURM_NNODES nodes"
echo "Job id is $SLURM_JOBID"

CUDA_LAUNCH_BLOCKING=0
HYDRA_FULL_ERROR=1

# Get parameters
CONFIG_PATH=$1
FEATURE_DIR=$2
START_STEP=${3:-1}  # Default to step 1 if not provided

echo "Config path: $CONFIG_PATH"
echo "Feature dir: $FEATURE_DIR"
echo "Starting from step: $START_STEP"

# Feat extraction
source ~/miniconda3/etc/profile.d/conda.sh
conda activate metasr_lm3  # Replace with your own conda env
echo "Active Conda Environment:"
conda info --envs | grep '*'

# Step 1: Feature extraction
if [ $START_STEP -le 1 ]; then
    echo "---- Starting Feature extraction step: `date`"
    srun python -m metasr_eval.slm.slm_feat_ext --config_path $CONFIG_PATH
fi

# Step 2: SLM training
if [ $START_STEP -le 2 ]; then
    echo "---- Starting SLM training step: `date`"
    if [ $START_STEP -eq 1 ]; then
        srun python -m metasr_eval.slm.slm_train --config_path $CONFIG_PATH
    else
        srun python -m metasr_eval.slm.slm_train --config_path $CONFIG_PATH --feature_dir $FEATURE_DIR
    fi
fi

# Step 3: SLM evaluation
if [ $START_STEP -le 3 ]; then
    echo "---- Starting SLM evaluation step: `date`"
    if [ $START_STEP -eq 1 ]; then
        srun python -m metasr_eval.slm.slm_eval --config_path $CONFIG_PATH
    else
        srun python -m metasr_eval.slm.slm_eval --config_path $CONFIG_PATH --feature_dir $FEATURE_DIR
    fi
fi

echo "---- End date: `date`"