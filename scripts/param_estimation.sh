#!/bin/bash
#SBATCH --job-name=param_estimation
#SBATCH --account=[account]
#SBATCH --mem=512G
#SBATCH --time=10:00:00
#SBATCH -q [queue]
#SBATCH --output=scripts/outs/%x_%j.out


cd [project_dir]

module load mamba
mamba activate csi-4cast-env

# python -m src.cp.models.baseline.statistical.param_estimation.main --model wiener
python -m src.cp.models.baseline.statistical.param_estimation.main --model wiener --scenario tdd
# python -m src.cp.models.baseline.statistical.param_estimation.main --model ar --order-candidates "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"