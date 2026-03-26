"""Worker entry point — called by SLURM jobs to run tuning.

Usage (called automatically by SLURM scripts):
    python3 -m src.cp.tune.worker --config <tuning_yaml> --worker_id <id>
"""

import argparse
from pathlib import Path
import sys

from ahpt.base.config import TuningConfig

from src.cp.tune.tune_runner import CPTuningRunner


def main():
    """Parse arguments and run a single tuning worker."""
    parser = argparse.ArgumentParser(description="CP tuning worker")
    parser.add_argument("--config", required=True, help="Tuning YAML")
    parser.add_argument("--worker_id", type=int, default=1)
    parser.add_argument("--timestamp", type=str, default=None, help="Shared timestamp for this job array")
    args = parser.parse_args()

    config = TuningConfig.from_yaml(args.config)

    output_dir = Path(config.output_dir)
    if args.timestamp:
        output_dir = output_dir / args.timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = CPTuningRunner(config, output_dir, worker_id=args.worker_id)
    runner.run_tuning()


if __name__ == "__main__":
    sys.exit(main())
