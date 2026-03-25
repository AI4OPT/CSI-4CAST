"""Base abstract class for tuning runners.

This provides a reusable framework that can be inherited and customized
by specific implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import random
import time
from typing import TYPE_CHECKING

import optuna


if TYPE_CHECKING:
    from ahpt.base.tune_obj import BaseTuningObjective


class BaseTuningRunner(ABC):
    """Abstract base class for tuning runners."""

    def __init__(self, tuning_config, output_dir: Path, worker_id: int = 1):
        self.tuning_config = tuning_config
        self.worker_id = worker_id

        # Setup worker-specific temporary directory
        self._setup_worker_directories(output_dir)

        # Log directory setup
        print(f"Worker {self.worker_id} using output directory: {self.temp_dir}")
        print("Direct storage mode - no copying required")

        # Initialize Optuna
        self._setup_optuna()

    def _setup_worker_directories(self, output_dir):
        """Setup worker-specific directories using target storage directly."""
        self.temp_dir = output_dir

    def _setup_optuna(self):
        """Setup Optuna study configuration."""
        if self.worker_id and self.worker_id > 1:
            delay = 60 * (self.worker_id - 1)
            print(f"Worker {self.worker_id} waiting {delay:.2f}s to reduce race conditions...")
            time.sleep(delay)

        # Setup storage with enhanced SQLite parameters for concurrency
        study_db_path = self.temp_dir / "study.db"
        storage_url = f"sqlite:///{study_db_path.absolute()}?cache=shared&timeout=1200"
        print(f"Using dynamic storage URL: {storage_url}")

        storage = optuna.storages.RDBStorage(
            url=storage_url,
            heartbeat_interval=60,
            grace_period=180,
        )

        # Setup pruner
        if self.tuning_config.enable_pruning:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.tuning_config.pruning_warmup_trials,
                n_warmup_steps=self.tuning_config.pruning_warmup_steps,
                interval_steps=self.tuning_config.pruning_interval_steps,
            )
        else:
            pruner = optuna.pruners.NopPruner()

        # Setup sampler with time-based random seed
        time_seed = int(time.time() * 1000) % (2**31)
        worker_seed = time_seed + (self.worker_id * 1000) + time_seed
        random.seed(worker_seed)

        sampler = optuna.samplers.TPESampler(
            seed=worker_seed,
            constant_liar=True,
            n_startup_trials=self.tuning_config.sampler_n_startup_trials,
            multivariate=True,
        )

        print(f"Worker {self.worker_id} using time-based seed: {worker_seed} for maximum exploration")

        # Create or load study
        self.study = optuna.create_study(
            study_name=self.tuning_config.study_name,
            storage=storage,
            direction=self.tuning_config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        # Log study state after connection
        existing_trials = len(self.study.trials)
        print(f"Connected to study. Existing trials: {existing_trials}")
        if existing_trials > 0:
            print(f"Latest trial numbers: {[t.number for t in self.study.trials[-3:]]}")
            print(f"Running trials: {[t.number for t in self.study.trials if t.state.name == 'RUNNING']}")
            print(f"Completed trials: {[t.number for t in self.study.trials if t.state.name == 'COMPLETE']}")

    def run_tuning(self) -> dict:
        """Run hyperparameter tuning experiment."""
        print(f"Starting hyperparameter tuning study: {self.tuning_config.study_name}")
        if self.worker_id:
            print(f"Worker ID: {self.worker_id}")

        # Setup callbacks
        tuning_callbacks = []
        if hasattr(self.tuning_config, "verbose") and self.tuning_config.verbose:
            tuning_callbacks.append(self._trial_callback)

        # Add MaxTrialsCallback
        existing_trials = len(self.study.trials)
        total_target_trials = existing_trials + self.tuning_config.n_trials

        max_trials_callback = optuna.study.MaxTrialsCallback(
            n_trials=total_target_trials,
            states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL),
        )
        tuning_callbacks.append(max_trials_callback)

        print(
            f"Trial planning: {existing_trials} existing + {self.tuning_config.n_trials} new = {total_target_trials} total target"
        )

        start_time = time.time()

        # Create objective function with temporary directory
        objective = self._create_objective()

        try:
            self.study.optimize(
                objective,
                n_trials=self.tuning_config.n_trials,
                timeout=self.tuning_config.timeout,
                callbacks=tuning_callbacks,
            )

            elapsed_time = time.time() - start_time
            print(f"Tuning completed in {elapsed_time:.2f} seconds")

            summary = self.get_study_summary()
            print("🎯 Tuning completed.")
            print(f"📊 Study summary: {summary}")

            return summary

        except Exception as e:
            print(f"❌ Tuning failed: {e!s}")
            summary = self.get_study_summary()
            print("🎯 Tuning failed but data is in study.db.")
            raise

    @abstractmethod
    def _create_objective(self) -> BaseTuningObjective:
        """Create the objective function instance.

        Returns:
            Objective function instance

        """
        pass

    def _trial_callback(self, study: optuna.Study, trial):
        """Callback function called after each trial."""
        if not hasattr(trial, "state"):
            print(f"Trial {trial.number}: State not available yet")
            return

        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Trial {trial.number} finished with value: {trial.value:.6f} and parameters: {trial.params}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"Trial {trial.number} pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"⚠️ Trial {trial.number} failed")
        else:
            print(f"Trial {trial.number} completed with state: {trial.state}")

    def get_study_summary(self) -> dict:
        """Get summary of current study."""
        if not hasattr(self, "study"):
            return {}

        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]

        summary = {
            "study_name": self.study.study_name,
            "total_trials": len(self.study.trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
            "best_value": self.study.best_value if completed_trials else None,
            "best_params": self.study.best_params if completed_trials else None,
        }

        return summary

    def continue_tuning(self, additional_trials: int) -> dict:
        """Continue tuning with additional trials."""
        print(f"Continuing tuning with {additional_trials} additional trials")

        original_n_trials = self.tuning_config.n_trials
        self.tuning_config.n_trials = additional_trials

        try:
            result = self.run_tuning()
            return result
        finally:
            self.tuning_config.n_trials = original_n_trials
