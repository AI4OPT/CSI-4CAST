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
mamba activate csi-pred

# Commands:
#   python3 -m src.data.generator.py --is_train              # Generate training data
#   python3 -m src.data.generator.py                         # Generate regular test data
#   python3 -m src.data.generator.py --is_gen                # Generate generalization test data
#   python3 -m src.data.generator.py --debug --is_train      # Debug mode: minimal training data
#   python3 -m src.data.generator.py --debug                 # Debug mode: minimal test data
#   python3 -m src.data.generator.py --debug --is_gen        # Debug mode: minimal generalization data
# array size:
#   1-20 for generalization test
#   1-9 for training and regular test
# job's mem should aligh wit the batch size of generator

python3 -m src.data.generator --is_gen --debug

mamba deactivate