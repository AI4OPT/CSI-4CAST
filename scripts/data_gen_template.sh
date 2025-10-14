#!/bin/bash
#SBATCH --job-name=data-gen-[gen|train|test]
#SBATCH --account=[account]
#SBATCH --mem=[memory]
#SBATCH --time=[walltime]
#SBATCH -p [partition]
#SBATCH --array=1-[array_size]
#SBATRCH -q [queue]
#SBATCH --output=scripts/outs/%x_%A_%a.out

cd [project_dir]

module load mamba/[mamba_version]
mamba activate [env_name]

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