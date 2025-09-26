#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --account=gts-phentenryck3-ai4opt
#SBATCH --gres=gpu:1
#SBATCH -p gpu-rtx6000,gpu-v100
#SBATCH --gres-flags=enforce-binding # Map CPUs to GPUs
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --array=81-100  # Array IDs: 1-20 MODEL, 21-40 LLM4CP, 41-60 CNN, 61-80 RNN, 81-100 NP, 101-120 STEMGNN
#SBATCH -q embers
#SBATCH --output=scripts/outs/testing/%x_%A_%a.out  # %A is job array ID, %a is task ID


cd /storage/home/hcoda1/1/scheng326/CSI-4CAST

module load mamba/1.4.9
mamba activate csi-4cast-env

echo "Starting array job ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_JOB_ID}"

python3 -m src.testing.prediction_performance.main

echo "Completed array job ${SLURM_ARRAY_TASK_ID}"

mamba deactivate