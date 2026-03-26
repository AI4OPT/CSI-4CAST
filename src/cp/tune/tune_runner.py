"""Concrete CP tuning runner."""

from ahpt.base.tune_runner import BaseTuningRunner

from src.cp.tune.tune_obj import CPTuningObjective


class CPTuningRunner(BaseTuningRunner):
    """Tuning runner for CSI prediction experiments."""

    def _create_objective(self):
        return CPTuningObjective(self.tuning_config, self.temp_dir, self.worker_id)
