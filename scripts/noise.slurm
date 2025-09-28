#!/bin/bash
#SBATCH --job-name=noise
#SBATCH --account=gts-phentenryck3-ai4opt
#SBATCH -N1 --gres=gpu:1
#SBATCH --gres-flags=enforce-binding # Map CPUs to GPUs
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-v100,gpu-rtx6000,gpu-l40s
#SBATCH --array=1
#SBATRCH -q inferno
#SBATCH --output=scripts/outs/%x_%j.out


cd /storage/home/hcoda1/1/scheng326/CSI-4CAST

module load mamba/1.4.9
mamba activate csi-4cast-env

python3 -m src.noise.noise_degree

mamba deactivate