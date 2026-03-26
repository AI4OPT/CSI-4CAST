#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --account=[account]
#SBATCH --gres=gpu:1
#SBATCH -p [partition]
#SBATCH --gres-flags=enforce-binding # Map CPUs to GPUs
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --array=1-20
#SBATCH -q [queue]
#SBATCH --output=scripts/outs/testing/%x_%A_%a.out  # %A is job array ID, %a is task ID


cd [project_dir]

module load mamba
mamba activate csi-4cast-env

echo "Starting array job ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_JOB_ID}"

python3 -m src.testing.prediction_performance.main

echo "Completed array job ${SLURM_ARRAY_TASK_ID}"

mamba deactivate
