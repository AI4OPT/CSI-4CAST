#!/bin/bash
#SBATCH --job-name=[job_name]
#SBATCH --account=[account]
#SBATCH -N1 --gres=gpu:1
#SBATCH --gres-flags=enforce-binding # Map CPUs to GPUs
#SBATCH --mem=[memory]
#SBATCH --time=[walltime]
#SBATCH -p [partition]
#SBATCH --array=1
#SBATRCH -q [queue]
#SBATCH --output=scripts/outs/%x_%j.out


cd [project_dir]

module load mamba/[mamba_version]
mamba activate csi-4cast-env

python3 -m src.cp.main -hcp [config_file]

mamba deactivate