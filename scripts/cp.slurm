#!/bin/bash
#SBATCH --job-name=fdd_rnn
#SBATCH --account=gts-phentenryck3-coda20
#SBATCH -N1 --gres=gpu:1
#SBATCH --gres-flags=enforce-binding # Map CPUs to GPUs
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH -p gpu-rtx6000
#SBATCH --array=1
#SBATRCH -q embers
#SBATCH --output=scripts/outs/%x_%j.out


cd /storage/home/hcoda1/1/scheng326/CSI-4CAST

module load mamba/1.4.9
mamba activate csi-4cast-env

python3 -m src.cp.main -hcp z_artifacts/config/cp/rnn/fdd_rnn.yaml

mamba deactivate