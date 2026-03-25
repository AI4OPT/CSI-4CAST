"""Base abstract classes for automated hyperparameter tuning.

This package provides reusable base classes that can be inherited and customized
for different tuning tasks. Key features:

- BaseTuningObjective: Abstract base for objective functions
- BaseTuningRunner: Abstract base for tuning runners
- BaseSlurmSubmitter: Abstract base for SLURM job submission
- BaseHyperparameterSpace: Abstract base for hyperparameter search spaces

Usage:
    from ahpt.base import BaseTuningObjective, BaseHyperparameterSpace

    class MyTuningObjective(BaseTuningObjective):
        def _setup_base_config(self):
            # Implement config loading
            pass

        def _setup_data_module(self, config):
            # Implement data module setup
            pass

        def _load_model(self, config):
            # Implement model loading
            pass

    class MyHyperparameterSpace(BaseHyperparameterSpace):
        @staticmethod
        def suggest_model_params(trial, model_name):
            # Implement model-specific parameter suggestions
            pass

        @staticmethod
        def suggest_training_params(trial):
            # Implement training-specific parameter suggestions
            pass
"""

from ahpt.base.config import StudyResult, TuningConfig
from ahpt.base.submit_jobs import BaseSlurmSubmitter
from ahpt.base.tune_obj import BaseTuningObjective
from ahpt.base.tune_runner import BaseTuningRunner
from ahpt.base.tune_space import BaseHyperparameterSpace


__all__ = [
    "BaseHyperparameterSpace",
    "BaseSlurmSubmitter",
    "BaseTuningObjective",
    "BaseTuningRunner",
    "StudyResult",
    "TuningConfig",
]
