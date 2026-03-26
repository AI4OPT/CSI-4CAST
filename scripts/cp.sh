#!/bin/bash
#SBATCH --job-name=[csi_pred_training]
#SBATCH --account=[account]
#SBATCH -N1 --gres=gpu:1
#SBATCH --gres-flags=enforce-binding # Map CPUs to GPUs
#SBATCH --mem=256G
#SBATCH --time=30:00:00
#SBATCH -p [partition]
#SBATCH --array=1
#SBATCH -q [queue]
#SBATCH --output=scripts/outs/%x_%j.out


cd [project_dir]

module load mamba
mamba activate csi-4cast-env

python3 -m src.cp.main -hcp [config_file]

mamba deactivate