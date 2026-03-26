#!/bin/bash
#SBATCH --job-name=comp_overhead
#SBATCH --account=[account]
#SBATCH -N1 --gres=gpu:1
#SBATCH --gres-flags=enforce-binding # Map CPUs to GPUs
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH -p [partition]
#SBATCH -q [queue]
#SBATCH --output=scripts/outs/%x_%j.out


cd [project_dir]

module load mamba
mamba activate csi-4cast-env

echo "Starting computational overhead analysis"

python3 -m src.testing.computational_overhead.main

echo "Completed computational overhead analysis"

mamba deactivate
