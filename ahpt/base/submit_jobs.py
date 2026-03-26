"""Base abstract class for SLURM job submission.

This provides a reusable framework that can be inherited and customized
by specific implementations.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
import subprocess
import textwrap
from typing import Any

from ahpt.base.utils import get_current_time


class BaseSlurmSubmitter(ABC):
    """Abstract base class for SLURM job submission."""

    def __init__(self, base_dir, scripts_dir=None, timestamp=None, slurm_settings=None):
        """Initialize the SLURM submitter with directories and optional overrides."""
        self.base_dir = Path(base_dir)

        # Use the provided timestamp or generate a new one
        if timestamp is not None:
            self.timestamp = timestamp
        else:
            self.timestamp = get_current_time()

        # Use a custom scripts_dir if provided, otherwise use the default
        if scripts_dir is not None:
            scripts_dir = Path(scripts_dir)
        else:
            scripts_dir = self.base_dir / "scripts" / "tuning" / self.timestamp

        self.logs_dir = scripts_dir / "outs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create job scripts directory with timestamp
        self.job_scripts_dir = scripts_dir / "jobs"
        self.job_scripts_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Job scripts will be saved to: {self.job_scripts_dir}")

        # Default SLURM settings
        self.default_slurm_settings = {
            "account": "[ACCOUNT_NAME]",
            "partition": "[PARTITION_NAME]",
            "qos": "[QOS_NAME]",
            "nodes": 1,
            "gres": "gpu:1",
            "gres_flags": "enforce-binding",
            "mem": "256G",
            "time": "30:00:00",
        }

        if slurm_settings is not None:
            self.default_slurm_settings.update(slurm_settings)

    @abstractmethod
    def _load_config(self, config_path: str) -> Any:
        """Load configuration from file."""
        pass

    @abstractmethod
    def _get_experiment_name(self, config: Any) -> str:
        """Extract experiment name from config."""
        pass

    @abstractmethod
    def _get_main_command_template(self) -> str:
        """Get the main command template for running the job."""
        pass

    def _get_shared_output_dir(self, config):
        """Get shared output directory from config."""
        return config.tuning.output_dir

    def submit_job_array(
        self,
        list_configs: list[str],
        num_workers: int = 5,
        continue_study: bool = False,
        additional_trials: int = 50,
        from_timestamp: str | None = None,
    ):
        """Submit workers as SLURM job arrays.

        Args:
            list_configs: List of configuration file paths
            num_workers: Number of workers per study
            continue_study: Whether to continue existing studies
            additional_trials: Number of additional trials for continuing studies
            from_timestamp: Timestamp of existing study to copy from (required with continue_study)

        Returns:
            List of submitted job information

        """
        # Validate arguments
        if continue_study and not from_timestamp:
            raise ValueError("from_timestamp is required when continue_study=True")

        print("🚀 SUBMITTING HYPERPARAMETER TUNING JOB ARRAYS")
        print("=" * 60)

        if continue_study:
            print("📈 Using SLURM job arrays to CONTINUE existing studies")
            print(f"Each worker will run {additional_trials} additional trials")
            print(f"Continuing from timestamp: {from_timestamp}")
        else:
            print("🎯 Using SLURM job arrays to START new studies")

        submitted_arrays = []

        for config_path in list_configs:
            pipeline_config = self._load_config(config_path)
            experiment_name = self._get_experiment_name(pipeline_config)

            print(f"📊 Config: {config_path}")
            print(f"🎯 Study: {experiment_name}")
            print(f"👥 Workers: {num_workers}")

            # Generate job array script
            array_script_content, array_job_name = self.generate_job_array_script(
                experiment_name=experiment_name,
                config_path=config_path,
                num_workers=num_workers,
                continue_study=continue_study,
                additional_trials=additional_trials,
                from_timestamp=from_timestamp,
            )

            # Submit job array
            result = self.submit_job_array_script(array_script_content, array_job_name, num_workers)
            job_id, script_path = result if result[0] else (None, result[1])

            if job_id:
                array_info = {
                    "job_id": job_id,
                    "job_name": array_job_name,
                    "config": config_path,
                    "num_workers": num_workers,
                    "script_path": script_path,
                }
                submitted_arrays.append(array_info)
                print(f"  ✅ Submitted job array: {job_id}")
            else:
                print(f"  ❌ Failed to submit job array for {experiment_name}")

            print("")

        print("📋 Job Array Summary:")
        for array in submitted_arrays:
            print(f"  Array ID: {array['job_id']}, Config: {array['config']}, Workers: {array['num_workers']}")

        return submitted_arrays

    def generate_job_array_script(
        self,
        experiment_name,
        config_path,
        num_workers,
        continue_study=False,
        additional_trials=50,
        from_timestamp=None,
    ):
        """Generate a SLURM job array script.

        Args:
            experiment_name: Name of the experiment
            config_path: Path to configuration file
            num_workers: Number of workers
            continue_study: Whether to continue existing study
            additional_trials: Number of additional trials if continuing
            from_timestamp: Timestamp to continue from

        Returns:
            Tuple of (script_content, job_name)

        """
        settings = {**self.default_slurm_settings}
        job_name = f"{experiment_name}_array"

        # Load config to get the shared output directory
        pipeline_config = self._load_config(config_path)
        shared_output_dir = self._get_shared_output_dir(pipeline_config)

        # Generate a single timestamp for this job array
        job_array_timestamp = self.timestamp

        # Build the main command
        main_command = self._get_main_command_template().format(
            config_path=config_path, job_array_timestamp=job_array_timestamp
        )

        # Add continuation flags if needed
        if continue_study:
            if not from_timestamp:
                raise ValueError("from_timestamp is required when continuing a study")
            main_command += (
                f" --continue-study --additional-trials {additional_trials} --from-timestamp {from_timestamp}"
            )

        # Prepare continuation info for the script
        if continue_study and from_timestamp:
            continuation_info = (
                f'echo "Continue Study: YES"\n'
                f'echo "From Timestamp: {from_timestamp}"\n'
                f'echo "To New Timestamp: {job_array_timestamp}"\n'
                f'echo "study.db will be copied from old folder to new shared folder"'
            )
        else:
            continuation_info = 'echo "Continue Study: NO (starting fresh)"'

        # Generate script
        script_template = textwrap.dedent("""
            #!/bin/bash
            #SBATCH --job-name={job_name}
            #SBATCH --account={account}
            #SBATCH --array=1-{num_workers}
            #SBATCH -N{nodes} --gres={gres}
            #SBATCH --gres-flags={gres_flags}
            #SBATCH --mem={mem}
            #SBATCH --time={time}
            #SBATCH -p {partition}
            #SBATCH -q {qos}
            #SBATCH --output={logs_dir}/%x_%A_%a.out

            # Job array information
            echo "=== JOB ARRAY INFORMATION ==="
            echo "Experiment: {experiment_name}"
            echo "Worker ID: $SLURM_ARRAY_TASK_ID"
            echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
            echo "Job ID: $SLURM_JOB_ID"
            echo "Node: $SLURMD_NODENAME"
            echo "Started: $(date)"
            echo "Config: {config_path}"
            echo "GPU: $CUDA_VISIBLE_DEVICES"
            echo "Output Dir: {shared_output_dir}/{job_array_timestamp}"
            echo "Job Array Timestamp: {job_array_timestamp}"
            {continuation_info}
            echo "DIRECT STORAGE MODE - No temp directories"
            echo "=============================="
            echo ""

            # Environment setup
            module load mamba
            mamba activate csi-4cast-env

            # Set working directory
            cd {base_dir}

            # GPU memory management
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            export CUDA_LAUNCH_BLOCKING=1

            # Stagger worker starts to avoid race conditions
            sleep_time=$((($SLURM_ARRAY_TASK_ID - 1) * 30))
            echo "⏰ Worker $SLURM_ARRAY_TASK_ID waiting ${{sleep_time}}s to stagger startup..."
            sleep $sleep_time

            # Clear GPU memory
            if command -v nvidia-smi &> /dev/null; then
                echo "=== INITIAL GPU STATUS ==="
                nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
                echo ""
                nvidia-smi --gpu-reset || echo "GPU reset not available"
            fi

            # Main execution
            echo "=== STARTING HYPERPARAMETER TUNING ==="
            echo "📁 Writing directly to target storage: {shared_output_dir}/{job_array_timestamp}"
            {main_command}

            exit_code=$?

            # Final GPU status
            if command -v nvidia-smi &> /dev/null; then
                echo "=== FINAL GPU STATUS ==="
                nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
            fi

            # Cleanup
            mamba deactivate

            echo ""
            echo "=== JOB COMPLETION ==="
            echo "Worker $SLURM_ARRAY_TASK_ID finished at: $(date)"
            echo "Exit code: $exit_code"
            echo "Total runtime: $SECONDS seconds"

            if [ $exit_code -eq 0 ]; then
                echo "✅ Job completed successfully"
            else
                echo "❌ Job failed with exit code $exit_code"
            fi

            exit $exit_code
        """).strip()

        formatted_script = script_template.format(
            job_name=job_name,
            experiment_name=experiment_name,
            config_path=config_path,
            base_dir=self.base_dir,
            logs_dir=self.logs_dir,
            num_workers=num_workers,
            shared_output_dir=shared_output_dir,
            job_array_timestamp=job_array_timestamp,
            continuation_info=continuation_info,
            main_command=main_command,
            **settings,
        )

        return formatted_script, job_name

    def submit_job_array_script(self, script_content, job_name, num_workers):
        """Submit a job array script to SLURM."""
        # Create script file
        script_filename = f"{job_name}.sh"
        script_path = self.job_scripts_dir / script_filename

        # Write script to file
        script_path.write_text(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)

        print(f"  📝 Created job array script: {script_path}")

        # Submit the job array using sbatch
        print(f"  📤 Submitting job array: {job_name} ({num_workers} workers)...")
        cmd = ["sbatch", "--parsable", str(script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            job_id = result.stdout.strip()
            print(f"  ✅ Submitted job array: {job_id}")
            return job_id, str(script_path)
        else:
            print(f"  ❌ Failed to submit job array: {result.stderr}")
            return None, str(script_path)
