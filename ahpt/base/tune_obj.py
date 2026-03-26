"""Base abstract class for tuning objective functions.

This provides a reusable framework that can be inherited and customized
by specific implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import gc
from pathlib import Path
import time
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch

from ahpt.base.tune_space import BaseHyperparameterSpace


class BaseTuningObjective(ABC):
    """Abstract base class for tuning objective functions."""

    def __init__(self, tuning_config, temp_dir: Path, worker_id: int = 1):
        """Initialize the objective with tuning config and worker directory."""
        self.tuning_config = tuning_config
        self.worker_id = worker_id
        self.temp_dir = temp_dir

        # Initialize hyperparameter space using abstract method
        self.hp_space = self._setup_hyperparameter_space()

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Track trials
        self.trial_count = 0

        # Setup base config
        self.base_config = self._setup_base_config()

    @abstractmethod
    def _setup_base_config(self) -> Any:
        """Set up the base configuration for this tuning objective.

        This method should be implemented by subclasses to:
        - Load the base config from file or create it programmatically
        - Return the config object
        """
        pass

    @abstractmethod
    def _setup_hyperparameter_space(self) -> BaseHyperparameterSpace:
        """Set up the hyperparameter space for this tuning objective.

        This method should be implemented by subclasses to:
        - Import and instantiate the appropriate hyperparameter space class
        - Return the hyperparameter space instance
        """
        pass

    @abstractmethod
    def _setup_data_module(self, data_cfg: Any) -> Any:
        """Set up the data module for training.

        This method should be implemented by subclasses to:
        - Create and initialize the data module with the given config
        - Return the data module
        """
        pass

    @abstractmethod
    def _load_model(self, config: Any, device: torch.device) -> pl.LightningModule:
        """Load model for training.

        This method should be implemented by subclasses to:
        - Create or load the model based on config
        - Move model to the appropriate device
        - Return the model
        """
        pass

    def _safe_set_trial_attr(self, trial: optuna.Trial, key: str, value) -> None:
        """Safely set trial user attribute with error handling for database concurrency issues."""
        try:
            trial.set_user_attr(key, value)
        except Exception as e:
            print(f"⚠️ Warning: Failed to set trial attribute '{key}': {e}")
            # Don't fail the entire trial for attribute setting issues

    def _is_cuda_oom(self, err: Exception) -> bool:
        """Check if the error is a CUDA out of memory error."""
        msg = str(err)
        return isinstance(err, torch.cuda.OutOfMemoryError) or ("CUDA out of memory" in msg)

    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function called by Optuna for each trial."""
        self.trial_count += 1
        start_time = time.time()

        # Log trial start with basic info
        print("=" * 80)
        print(f"🚀 STARTING TRIAL {trial.number} (Local count: {self.trial_count})")
        print(f"Study: {trial.study.study_name}")
        print(f"Worker ID: {self.worker_id if self.worker_id is not None else 'unknown'}")

        # Log study state when this trial starts
        total_trials = len(trial.study.trials)
        running_trials = [t for t in trial.study.trials if hasattr(t, "state") and t.state.name == "RUNNING"]
        print(f"Study state: {total_trials} total trials, {len(running_trials)} running")
        print(f"Running trial numbers: {[t.number for t in running_trials]}")
        print("=" * 80)

        try:
            # Generate hyperparameters for this trial
            config = self._generate_trial_config(trial)

            # Log all suggested trial parameters before training
            self._log_trial_parameters(trial, config)

            # Train model and get validation loss
            val_loss = self._train_and_evaluate(trial, config)

            # Log trial result
            elapsed_time = time.time() - start_time
            print("=" * 80)
            print(f"✅ TRIAL {trial.number} COMPLETED")
            print(f"⏱️  Duration: {elapsed_time:.2f}s")
            print(f"📊 Validation Loss: {val_loss:.6f}")
            if hasattr(trial, "user_attrs") and "tensorboard_log_dir" in trial.user_attrs:
                print(f"📈 TensorBoard: tensorboard --logdir {trial.user_attrs['tensorboard_log_dir']}")
            print("=" * 80)

            return val_loss

        except AssertionError as e:
            elapsed_time = time.time() - start_time
            print("=" * 80)
            print(f"⚠️  TRIAL {trial.number} SKIPPED (Invalid Hyperparameters)")
            print(f"⏱️  Duration: {elapsed_time:.2f}s")
            print(f"🚨 Assertion Error: {e!s}")
            print("📋 This hyperparameter combination is invalid - trial will be pruned")
            print("=" * 80)

            # Store error information in trial attributes for debugging
            self._safe_set_trial_attr(trial, "error_message", str(e))
            self._safe_set_trial_attr(trial, "error_type", "AssertionError")
            self._safe_set_trial_attr(trial, "pruned_reason", "invalid_hyperparameters")

            # Prune this trial instead of failing - tells Optuna this combination is invalid
            raise optuna.TrialPruned(f"Invalid hyperparameter combination: {e!s}")

        except Exception as e:
            elapsed_time = time.time() - start_time
            print("=" * 80)
            print(f"❌ TRIAL {trial.number} FAILED")
            print(f"⏱️  Duration: {elapsed_time:.2f}s")
            print(f"🚨 Error: {e!s}")
            print(f"🚨 Error Type: {type(e).__name__}")
            print("=" * 80)

            # Store error information in trial attributes for debugging (safely)
            self._safe_set_trial_attr(trial, "error_message", str(e))
            self._safe_set_trial_attr(trial, "error_type", type(e).__name__)

            if self._is_cuda_oom(e):
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                self._safe_set_trial_attr(trial, "pruned_reason", "cuda_oom")
                raise optuna.TrialPruned("Pruned due to CUDA OOM")

            # Let Optuna handle the failure naturally without wrapping in TrialPruned
            raise

    # sometimes this function is also needed to be overridden by subclasses
    def _generate_trial_config(self, trial: optuna.Trial):
        """Generate configuration for this trial based on suggested hyperparameters.

        Subclasses can override this to customize how config is generated.
        """
        # Start with base config
        config = copy.deepcopy(self.base_config)

        # Determine which models to consider
        if self.tuning_config.target_models:
            available_models = self.tuning_config.target_models
        else:
            raise ValueError("No target models specified")

        # Suggest model name
        model_name = self.hp_space.suggest_model_name(trial, available_models)
        config.model.name = model_name
        self._safe_set_trial_attr(trial, "model_name", model_name)

        # Suggest model parameters
        if self.tuning_config.tune_model_params:
            model_params = self.hp_space.suggest_model_params(trial, model_name)
            if config.model.params is None:
                config.model.params = {}
            config.model.params = model_params
            self._safe_set_trial_attr(trial, "model_params", model_params)
        else:
            print(f"🔧 Model parameters not tuned for model {model_name}")

        # Suggest optimizer parameters
        if self.tuning_config.tune_optimizer_params:
            optimizer_name = self.hp_space.suggest_optimizer_name(trial)
            optimizer_params = self.hp_space.suggest_optimizer_params(trial, optimizer_name)
            config.optimizer.name = optimizer_name
            config.optimizer.params = optimizer_params
            self._safe_set_trial_attr(trial, "optimizer_name", optimizer_name)
            self._safe_set_trial_attr(trial, "optimizer_params", optimizer_params)
        else:
            print("🔧 Not tuning optimizer parameters")

        # Suggest scheduler parameters
        if self.tuning_config.tune_scheduler_params:
            scheduler_name = self.hp_space.suggest_scheduler_name(trial)
            scheduler_params = self.hp_space.suggest_scheduler_params(trial, scheduler_name)
            config.scheduler.name = scheduler_name
            config.scheduler.params = scheduler_params
            self._safe_set_trial_attr(trial, "scheduler_name", scheduler_name)
            self._safe_set_trial_attr(trial, "scheduler_params", scheduler_params)
        else:
            print("🔧 Not tuning scheduler parameters")

        # Suggest training parameters
        if self.tuning_config.tune_training_params:
            training_params = self.hp_space.suggest_training_params(trial)

            # Update training config
            for key, value in training_params.items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
                    self._safe_set_trial_attr(trial, f"training_{key}", value)

            # Handle batch_size if present in training params
            if "batch_size" in training_params:
                if hasattr(config, "data"):
                    config.data.batch_size = training_params["batch_size"]
                    self._safe_set_trial_attr(trial, "batch_size", training_params["batch_size"])
        else:
            print("🔧 Training parameters not tuned")

        # Suggest loss parameters
        if self.tuning_config.tune_loss_params:
            loss_name = self.hp_space.suggest_loss_name(trial)
            loss_params = self.hp_space.suggest_loss_params(trial, loss_name)
            config.loss.name = loss_name
            config.loss.params = loss_params
            self._safe_set_trial_attr(trial, "loss_name", loss_name)
            self._safe_set_trial_attr(trial, "loss_params", loss_params)
        else:
            print("🔧 Not tuning loss parameters")

        # Override some settings for tuning
        config.training.num_epochs = self.tuning_config.trial_max_epochs
        config.training.enable_progress_bar = False
        config.training.enable_model_summary = False
        config.deterministic = False

        return config

    def _train_and_evaluate(self, trial: optuna.Trial, config) -> float:
        """Train model with given config and return validation loss."""
        torch.cuda.empty_cache()
        gc.collect()

        # Log initial GPU memory state
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            print(f"🔧 Initial GPU memory: {initial_memory / 1e9:.2f}GB allocated, {max_memory / 1e9:.2f}GB max")
            torch.cuda.reset_peak_memory_stats()

        # Setup trial output directory
        temp_output_dir: Path = self.temp_dir / f"trial_{trial.number}"
        temp_ckpts_dir: Path = temp_output_dir / "ckpts"
        temp_ckpts_dir.mkdir(parents=True, exist_ok=True)

        # Set time-based random seed for maximum exploration
        trial_seed = int(time.time() * 1000000) % (2**31) + self.trial_count * 100
        pl.seed_everything(trial_seed, workers=True)

        # Record the seed in Optuna trial attributes for tracking
        self._safe_set_trial_attr(trial, "trial_seed", trial_seed)
        self._safe_set_trial_attr(trial, "worker_id", self.worker_id)

        # Update the config with the trial-specific seed and save it
        if hasattr(config, "seed"):
            config.seed = trial_seed
        else:
            raise ValueError("Seed not found in config")

        # Save the trial-specific config
        if hasattr(config, "save_yaml"):
            trial_config_path = temp_output_dir / "config.yaml"
            config.save_yaml(str(trial_config_path))
            self._safe_set_trial_attr(trial, "config_path", str(trial_config_path))
        else:
            raise ValueError("save_yaml method not found in config")

        print(f"🎲 Trial {self.trial_count} using time-based random seed: {trial_seed}")

        trainer = None
        model = None
        datamodule = None

        try:
            # Setup data module using abstract method
            datamodule = self._setup_data_module(config.data)

            # Load model using abstract method
            model = self._load_model(config, device=self.device)

            # Setup callbacks
            trainer_callbacks = []

            # Early stopping callback
            if self.tuning_config.trial_early_stopping:
                early_stop_callback = EarlyStopping(
                    monitor=self.tuning_config.trial_monitor,
                    patience=self.tuning_config.trial_patience,
                    mode=self.tuning_config.trial_monitor_mode,
                    min_delta=self.tuning_config.trial_min_delta,
                    verbose=False,
                )
                trainer_callbacks.append(early_stop_callback)

            # Pruning callback for Optuna
            if self.tuning_config.enable_pruning:
                pruning_callback = PyTorchLightningPruningCallback(trial, monitor=self.tuning_config.trial_monitor)
                trainer_callbacks.append(pruning_callback)

            # Model checkpoint callback
            checkpoint_callback = ModelCheckpoint(
                dirpath=temp_ckpts_dir,
                filename="{epoch:03d}-{step:06d}-{val_loss:.10f}",
                monitor=self.tuning_config.trial_monitor,
                mode=self.tuning_config.trial_monitor_mode,
                save_top_k=1,
                verbose=False,
            )
            trainer_callbacks.append(checkpoint_callback)

            # Setup TensorBoard logger
            tb_logger = TensorBoardLogger(
                save_dir=temp_output_dir,
                name="tensorboard",
                version=f"trial_{trial.number}",
                log_graph=False,
                default_hp_metric=False,
            )

            tb_log_dir = tb_logger.log_dir
            self._safe_set_trial_attr(trial, "tensorboard_log_dir", str(tb_log_dir))
            print(f"📊 TensorBoard logs: {tb_log_dir}")

            # Setup trainer
            trainer = pl.Trainer(
                max_time={"seconds": self.tuning_config.max_trial_time} if self.tuning_config.max_trial_time else None,
                max_epochs=config.training.num_epochs,
                min_epochs=self.tuning_config.trial_min_epochs,
                accelerator=config.accelerator,
                devices=config.devices,
                precision=config.precision,
                callbacks=trainer_callbacks,
                logger=tb_logger,
                enable_progress_bar=config.training.enable_progress_bar,
                enable_model_summary=config.training.enable_model_summary,
                gradient_clip_val=config.training.gradient_clip_val,
                accumulate_grad_batches=config.training.accumulate_grad_batches,
                check_val_every_n_epoch=config.training.check_val_every_n_epoch,
                log_every_n_steps=50,
                deterministic=False,
            )

            model.train()
            trainer.fit(model, datamodule)

            # Get best validation loss
            if checkpoint_callback.best_model_score is not None:
                best_val_loss = checkpoint_callback.best_model_score.item()
            else:
                best_val_loss = float("inf")

        except Exception as e:
            print(f"❌ Error during training: {e!s}")
            raise

        finally:
            # Aggressive cleanup
            try:
                if trainer is not None:
                    trainer.strategy.teardown()
                    del trainer
                if model is not None:
                    model.cpu()
                    del model
                if datamodule is not None:
                    del datamodule
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    final_memory = torch.cuda.memory_allocated()
                    peak_memory = torch.cuda.max_memory_allocated()
                    print(f"🧹 Final GPU memory: {final_memory / 1e9:.2f}GB allocated, {peak_memory / 1e9:.2f}GB peak")
            except Exception as cleanup_error:
                print(f"⚠️ Error during cleanup: {cleanup_error}")

        return best_val_loss

    def _log_trial_parameters(self, trial, config):
        """Log all trial parameters in a detailed and organized way."""
        print("📋 TRIAL PARAMETERS:")
        print("-" * 60)

        # Basic trial info
        print(f"Trial Number: {trial.number}")
        if hasattr(trial, "state"):
            print(f"Trial State: {trial.state}")
        else:
            print("Trial State: RUNNING (in progress)")

        # All suggested parameters from Optuna
        if hasattr(trial, "params") and trial.params:
            print("🎯 Optuna Suggested Parameters:")
            for param_name, param_value in trial.params.items():
                print(f"  {param_name}: {param_value}")
        else:
            print("🎯 Optuna Suggested Parameters: None yet")

        # Model configuration
        if self.tuning_config.tune_model_params:
            print("🧠 Model Configuration:")
            print(f"  Model Type: {config.model.name}")
            if config.model.params:
                print("  Model Parameters:")
                for param_name, param_value in config.model.params.items():
                    print(f"    {param_name}: {param_value}")
            else:
                print("  Model Parameters: Using defaults")

        # Optimizer configuration
        if self.tuning_config.tune_optimizer_params:
            print("⚡ Optimizer Configuration:")
            print(f"  Optimizer: {config.optimizer.name}")
            if config.optimizer.params:
                print("  Optimizer Parameters:")
                for param_name, param_value in config.optimizer.params.items():
                    print(f"    {param_name}: {param_value}")

        # Training configuration
        if self.tuning_config.tune_training_params:
            print("🏋️ Training Configuration:")
            print(f"  Max Epochs: {config.training.num_epochs}")
            if hasattr(config, "data") and hasattr(config.data, "batch_size"):
                print(f"  Batch Size: {config.data.batch_size}")

        # Loss configuration
        if self.tuning_config.tune_loss_params:
            print("📉 Loss Configuration:")
            print(f"  Loss Function: {config.loss.name}")
            if config.loss.params:
                print("  Loss Parameters:")
                for param_name, param_value in config.loss.params.items():
                    print(f"    {param_name}: {param_value}")

        # Data configuration
        print("📊 Data Configuration:")
        if hasattr(config, "data"):
            if hasattr(config.data, "batch_size"):
                print(f"  Batch Size: {config.data.batch_size}")
            # Print other data config attributes
            for key, value in config.data.__dict__.items():
                if not key.startswith("_") and key != "batch_size":
                    print(f"  {key}: {value}")

        print("-" * 60)
