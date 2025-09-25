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


cd /storage/home/hcoda1/1/scheng326/CSI_main

module load mamba/1.4.9
mamba activate csi-pred

echo "Starting computational overhead analysis"

python3 -m src.cp_testing.computational_overhead

echo "Completed computational overhead analysis"

mamba deactivate 