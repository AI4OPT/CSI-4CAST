#!/bin/bash
#SBATCH --job-name=noise_degree
#SBATCH --account=[account]
#SBATCH -N1 --gres=gpu:1
#SBATCH --gres-flags=enforce-binding # Map CPUs to GPUs
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH -p [partition]
#SBATCH --array=1
#SBATCH -q [queue]
#SBATCH --output=scripts/outs/%x_%j.out


cd [project_dir]

module load mamba
mamba activate csi-4cast-env

python3 -m src.noise.noise_degree

mamba deactivate
