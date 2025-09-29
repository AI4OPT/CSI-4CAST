#!/bin/bash
#SBATCH --job-name=data-gen-test-gen
#SBATCH --account=gts-phentenryck3-ai4opt
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH -p gpu-h200,gpu-h100,gpu-a100,gpu-v100,gpu-rtx6000,gpu-l40s
#SBATCH --array=1-20
#SBATRCH -q inferno
#SBATCH --output=scripts/outs/%x_%A_%a.out

cd /storage/home/hcoda1/1/scheng326/CSI-4CAST

module load mamba/1.4.9
mamba activate csi-4cast-env

python3 -m src.data.generator --is_gen

mamba deactivate