#!/bin/bash
#SBATCH --job-name=comp_overhead_h200
#SBATCH --account=gts-phentenryck3-ai4opt
#SBATCH -N1 --gres=gpu:1
#SBATCH --gres-flags=enforce-binding # Map CPUs to GPUs
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH -p gpu-h200
#SBATCH -q embers
#SBATCH --output=scripts/outs/%x_%j.out


cd /storage/home/hcoda1/1/scheng326/CSI-4CAST

module load mamba/1.4.9
mamba activate csi-4cast-env

echo "Starting computational overhead analysis"

python3 -m src.testing.computational_overhead.main

echo "Completed computational overhead analysis"

mamba deactivate 