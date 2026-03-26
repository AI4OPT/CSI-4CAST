"""SLURM job submitter for CP ablation hyperparameter tuning."""

from ahpt.base.config import TuningConfig
from ahpt.base.submit_jobs import BaseSlurmSubmitter


class CPSlurmSubmitter(BaseSlurmSubmitter):
    """SLURM job submitter for CSI prediction tuning."""

    def _load_config(self, config_path: str) -> TuningConfig:
        return TuningConfig.from_yaml(config_path)

    def _get_experiment_name(self, config: TuningConfig) -> str:
        return config.study_name

    def _get_main_command_template(self) -> str:
        return (
            "python3 -m src.cp.tune.worker"
            " --config {config_path}"
            " --worker_id $SLURM_ARRAY_TASK_ID"
            " --timestamp {job_array_timestamp}"
        )

    def _get_shared_output_dir(self, config: TuningConfig) -> str:
        return config.output_dir
