"""Entry point for CP ablation hyperparameter tuning via SLURM.

Usage:
    # Submit single config (3 parallel workers):
    python3 -m src.cp.tune.main \
        --config src/cp/models/ablation/denoiser/no_denoiser/tuning.yaml \
        --num_workers 3

    # Submit multiple configs at once:
    python3 -m src.cp.tune.main \
        --config tuning_a.yaml tuning_b.yaml \
        --num_workers 3
"""

import argparse
from pathlib import Path
import sys

from src.cp.tune.submit import CPSlurmSubmitter


def main():
    parser = argparse.ArgumentParser(description="CP ablation hyperparameter tuning")
    parser.add_argument("--config", required=True, nargs="+", help="Tuning YAML(s)")
    parser.add_argument("--num_workers", type=int, default=1, help="Workers per study")
    parser.add_argument("--time", type=str, default=None, help="SLURM wall time (e.g. 08:00:00)")
    parser.add_argument("--mem", type=str, default=None, help="SLURM memory (e.g. 32G)")
    args = parser.parse_args()

    slurm_overrides = {}
    if args.time:
        slurm_overrides["time"] = args.time
    if args.mem:
        slurm_overrides["mem"] = args.mem

    submitter = CPSlurmSubmitter(
        base_dir=Path.cwd(),
        slurm_settings=slurm_overrides or None,
    )
    submitter.submit_job_array(
        list_configs=args.config,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    sys.exit(main())
