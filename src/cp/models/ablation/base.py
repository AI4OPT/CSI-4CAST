"""Shared helpers for ablation models and tuning spaces."""

from copy import deepcopy
from typing import Any

from ahpt.base.tune_space import BaseHyperparameterSpace
import optuna
import torch.nn as nn

from src.cp.config.config import ExperimentConfig
from src.cp.models.common.base import BaseCSIModel
from src.cp.models.common.normalizer import batch_denormalize, batch_normalizer
from src.cp.models.proposed.model_fdd import (
    AdaptiveReweightingLayerProcessor as _FDDARLProcessor,
    CSIEmbeddingShuffleNet as _FDDEmbedding,
    Denoiser as _FDDDenoiser,
    MLPProcessor as _FDDMLPProcessor,
    TransformerPredictor as _FDDTransformer,
)
from src.cp.models.proposed.model_tdd import (
    AdaptiveReweightingLayerProcessor as _TDDARLProcessor,
    CSIEmbeddingShuffleNet as _TDDEmbedding,
    Denoiser as _TDDDenoiser,
    MLPProcessor as _TDDMLPProcessor,
    TransformerPredictor as _TDDTransformer,
)


# ── Default parameters ───────────────────────────────────────────────────────

TDD_BASE_MODEL_PARAMS: dict[str, Any] = {
    "hist_len": 16,
    "pred_len": 4,
    "dim_data": 600,
    "dim_model": 768,
    "denoiser_num_filters_2d": 3,
    "denoiser_filter_size_2d": 3,
    "denoiser_filter_size_1d": 3,
    "denoiser_activation": "relu",
    "denoiser_is_post_processor": False,
    "denoiser_is_residual": True,
    "arl_is_U2D": False,
    "arl_temporal_proj_num_layers": 4,
    "arl_temporal_proj_hidden_dim": 512,
    "arl_temporal_proj_is_arl": False,
    "arl_temporal_proj_output_activation_name": "none",
    "arl_temporal_proj_arl_operation": "multiply",
    "arl_subcarrier_proj_num_layers": 2,
    "arl_subcarrier_proj_hidden_dim": 128,
    "arl_subcarrier_proj_is_arl": True,
    "arl_subcarrier_proj_output_activation_name": "tanh",
    "arl_subcarrier_proj_arl_operation": "multiply",
    "embedding_num_res_layers": 4,
    "embedding_res_dim": 64,
    "embedding_res_groups": 4,
    "embedding_ca_ratio": 2,
    "embedding_embed": "timeF",
    "embedding_freq": "h",
    "embedding_dropout": 0.1,
    "transformer_num_layers": 6,
    "transformer_num_heads": 8,
    "transformer_hidden_dim": 512,
    "transformer_dropout_prob": 0.1,
}


FDD_BASE_MODEL_PARAMS: dict[str, Any] = {
    "hist_len": 16,
    "pred_len": 4,
    "dim_data": 600,
    "dim_model": 768,
    "denoiser_num_filters_2d": 5,
    "denoiser_filter_size_2d": 3,
    "denoiser_filter_size_1d": 3,
    "denoiser_activation": "relu",
    "denoiser_is_post_processor": False,
    "denoiser_is_residual": True,
    "arl_is_U2D": True,
    "arl_temporal_proj_num_layers": 4,
    "arl_temporal_proj_hidden_dim": 512,
    "arl_temporal_proj_is_arl": False,
    "arl_temporal_proj_output_activation_name": "none",
    "arl_temporal_proj_arl_operation": "add",
    "arl_subcarrier_proj_num_layers": 3,
    "arl_subcarrier_proj_hidden_dim": 2048,
    "arl_subcarrier_proj_is_arl": True,
    "arl_subcarrier_proj_output_activation_name": "relu",
    "arl_subcarrier_proj_arl_operation": "add",
    "embedding_num_res_layers": 4,
    "embedding_res_dim": 128,
    "embedding_res_groups": 8,
    "embedding_embed": "timeF",
    "embedding_freq": "h",
    "embedding_dropout": 0.1,
    "transformer_num_layers": 4,
    "transformer_num_heads": 4,
    "transformer_hidden_dim": 2048,
    "transformer_dropout_prob": 0.2,
}


def get_tdd_base_model_params() -> dict[str, Any]:
    """Return a mutable copy of default TDD model parameters."""
    return deepcopy(TDD_BASE_MODEL_PARAMS)


def get_fdd_base_model_params() -> dict[str, Any]:
    """Return a mutable copy of default FDD model parameters."""
    return deepcopy(FDD_BASE_MODEL_PARAMS)


# ── Lightning wrapper ─────────────────────────────────────────────────────────


class AblationLightningModel(BaseCSIModel):
    """Reusable Lightning wrapper for ablation model variants."""

    model_class = None
    model_display_name = "ABLATION"

    def __init__(self, config: ExperimentConfig, *args, **kwargs):
        """Initialize the ablation model from an experiment config."""
        super().__init__(
            optimizer_config=config.optimizer,
            scheduler_config=config.scheduler,
            loss_config=config.loss,
        )

        if self.model_class is None:
            raise ValueError("model_class must be defined by subclass")

        self.name = self.model_display_name
        self.is_separate_antennas = config.model.is_separate_antennas
        self.save_hyperparameters({"model": config.model})

        model_params = config.model.params or {}
        self.model = self.model_class(**model_params)

    def __str__(self):
        """Return the model display name."""
        return self.name

    def forward(self, x):
        """Run the forward pass through the underlying model."""
        return self.model(x)


# ── Tuning space base ─────────────────────────────────────────────────────────


class AblationHyperparameterSpace(BaseHyperparameterSpace):
    """Shared training-parameter tuning defaults for ablation studies."""

    @staticmethod
    def suggest_optimizer_params(trial: optuna.Trial, optimizer_name: str) -> dict[str, Any]:
        """Suggest optimizer hyperparameters for the given optimizer."""
        params: dict[str, Any] = {}
        if optimizer_name in ("Adam", "AdamW"):
            params["lr"] = trial.suggest_float("optimizer_lr", 1e-5, 5e-3, log=True)
            params["weight_decay"] = trial.suggest_float("optimizer_weight_decay", 1e-8, 1e-2, log=True)
        return params

    @staticmethod
    def suggest_scheduler_params(trial: optuna.Trial, scheduler_name: str) -> dict[str, Any]:
        """Suggest scheduler hyperparameters for the given scheduler."""
        params: dict[str, Any] = {}
        if scheduler_name == "ReduceLROnPlateau":
            params["mode"] = "min"
            params["factor"] = trial.suggest_categorical("scheduler_factor", [0.5])
            params["patience"] = trial.suggest_categorical("scheduler_patience", [5])
            params["threshold"] = trial.suggest_categorical("scheduler_threshold", [1e-4])
            params["cooldown"] = trial.suggest_categorical("scheduler_cooldown", [0])
            params["min_lr"] = trial.suggest_categorical("scheduler_min_lr", [1e-8])
        return params

    @staticmethod
    def suggest_training_params(trial: optuna.Trial) -> dict[str, Any]:
        """Suggest training hyperparameters such as batch size."""
        return {
            "batch_size": trial.suggest_categorical("batch_size", [16]),
            "accumulate_grad_batches": trial.suggest_categorical("accumulate_grad_batches", [1]),
        }

    @staticmethod
    def suggest_optimizer_name(trial: optuna.Trial) -> str:
        """Suggest an optimizer name from the candidate list."""
        return trial.suggest_categorical("optimizer_name", ["Adam", "AdamW"])

    @staticmethod
    def suggest_scheduler_name(trial: optuna.Trial) -> str:
        """Suggest a scheduler name from the candidate list."""
        return trial.suggest_categorical("scheduler_name", ["ReduceLROnPlateau"])

    @staticmethod
    def suggest_loss_name(trial: optuna.Trial) -> str:
        """Suggest a loss function name from the candidate list."""
        return trial.suggest_categorical("loss_name", ["NMSE"])


# ── Composable TDD model ─────────────────────────────────────────────────────


class AblationTDDModel(nn.Module):
    """Composable TDD model for ablation studies.

    Subclasses override ``_build_*`` methods to replace specific pipeline
    stages without constructing then discarding the originals.
    """

    def __init__(self, **params):
        """Build the TDD pipeline from keyword parameters."""
        super().__init__()
        self._p = params
        self.hist_len = params.get("hist_len", 16)
        self.pred_len = params.get("pred_len", 4)
        self.dim_data = params.get("dim_data", 600)
        self.dim_model = params.get("dim_model", 768)

        self.denoiser = self._build_denoiser()
        self.arl = self._build_arl()
        self.embedding = self._build_embedding()
        self.transformer = self._build_transformer()
        self.mlp = self._build_mlp()

    # -- helpers ---------------------------------------------------------------

    def _arl_kwargs(self) -> dict[str, Any]:
        """Extract ARL constructor arguments from the init params dict."""
        p = self._p
        return dict(
            hist_len=self.hist_len,
            dim_data=self.dim_data,
            is_U2D=p.get("arl_is_U2D", False),
            temporal_proj_num_layers=p.get("arl_temporal_proj_num_layers", 4),
            temporal_proj_hidden_dim=p.get("arl_temporal_proj_hidden_dim", 512),
            temporal_proj_is_arl=p.get("arl_temporal_proj_is_arl", False),
            temporal_proj_output_activation_name=p.get("arl_temporal_proj_output_activation_name", "none"),
            temporal_proj_arl_operation=p.get("arl_temporal_proj_arl_operation", "multiply"),
            subcarrier_proj_num_layers=p.get("arl_subcarrier_proj_num_layers", 2),
            subcarrier_proj_hidden_dim=p.get("arl_subcarrier_proj_hidden_dim", 128),
            subcarrier_proj_is_arl=p.get("arl_subcarrier_proj_is_arl", True),
            subcarrier_proj_output_activation_name=p.get("arl_subcarrier_proj_output_activation_name", "tanh"),
            subcarrier_proj_arl_operation=p.get("arl_subcarrier_proj_arl_operation", "multiply"),
        )

    # -- builder methods (override in subclasses) ------------------------------

    def _build_denoiser(self) -> nn.Module:
        p = self._p
        if p.get("denoiser_num_filters_2d", 3) > 1:
            return _TDDDenoiser(
                num_filters_2d=p.get("denoiser_num_filters_2d", 3),
                filter_size_2d=p.get("denoiser_filter_size_2d", 3),
                filter_size_1d=p.get("denoiser_filter_size_1d", 3),
                activation=p.get("denoiser_activation", "relu"),
                is_post_processor=p.get("denoiser_is_post_processor", False),
                is_residual=p.get("denoiser_is_residual", True),
            )
        return nn.Identity()

    def _build_arl(self) -> nn.Module:
        return _TDDARLProcessor(**self._arl_kwargs())

    def _build_embedding(self) -> nn.Module:
        p = self._p
        return _TDDEmbedding(
            dim_model=self.dim_model,
            num_res_layers=p.get("embedding_num_res_layers", 4),
            res_dim=p.get("embedding_res_dim", 64),
            res_groups=p.get("embedding_res_groups", 4),
            res_ca_ratio=p.get("embedding_ca_ratio", 2),
            hist_len=self.hist_len,
            dim_data=self.dim_data,
            embed=p.get("embedding_embed", "timeF"),
            freq=p.get("embedding_freq", "h"),
            dropout=p.get("embedding_dropout", 0.1),
        )

    def _build_transformer(self) -> nn.Module:
        p = self._p
        return _TDDTransformer(
            dim_model=self.dim_model,
            num_layers=p.get("transformer_num_layers", 6),
            num_heads=p.get("transformer_num_heads", 8),
            hidden_dim=p.get("transformer_hidden_dim", 512),
            dropout_prob=p.get("transformer_dropout_prob", 0.1),
        )

    def _build_mlp(self) -> nn.Module:
        return _TDDMLPProcessor(
            dim_model=self.dim_model,
            dim_data=self.dim_data,
            hist_len=self.hist_len,
            pred_len=self.pred_len,
        )

    # -- forward ---------------------------------------------------------------

    def forward(self, x):
        """Run the full TDD prediction pipeline."""
        x = self.denoiser(x)
        x, mean, std = batch_normalizer(x)
        x_delay, x_freq = self.arl(x)
        x = self.embedding(x_delay, x_freq)
        x = self.transformer(x)
        x = self.mlp(x)
        x = batch_denormalize(x, mean, std)
        return x


# ── Composable FDD model ─────────────────────────────────────────────────────


class AblationFDDModel(nn.Module):
    """Composable FDD model for ablation studies."""

    def __init__(self, **params):
        """Build the FDD pipeline from keyword parameters."""
        super().__init__()
        self._p = params
        self.hist_len = params.get("hist_len", 16)
        self.pred_len = params.get("pred_len", 4)
        self.dim_data = params.get("dim_data", 600)
        self.dim_model = params.get("dim_model", 768)

        self.denoiser = self._build_denoiser()
        self.arl = self._build_arl()
        self.embedding = self._build_embedding()
        self.transformer = self._build_transformer()
        self.mlp = self._build_mlp()

    # -- helpers ---------------------------------------------------------------

    def _arl_kwargs(self) -> dict[str, Any]:
        p = self._p
        return dict(
            hist_len=self.hist_len,
            dim_data=self.dim_data,
            is_U2D=p.get("arl_is_U2D", True),
            temporal_proj_num_layers=p.get("arl_temporal_proj_num_layers", 4),
            temporal_proj_hidden_dim=p.get("arl_temporal_proj_hidden_dim", 512),
            temporal_proj_is_arl=p.get("arl_temporal_proj_is_arl", False),
            temporal_proj_output_activation_name=p.get("arl_temporal_proj_output_activation_name", "none"),
            temporal_proj_arl_operation=p.get("arl_temporal_proj_arl_operation", "add"),
            subcarrier_proj_num_layers=p.get("arl_subcarrier_proj_num_layers", 3),
            subcarrier_proj_hidden_dim=p.get("arl_subcarrier_proj_hidden_dim", 2048),
            subcarrier_proj_is_arl=p.get("arl_subcarrier_proj_is_arl", True),
            subcarrier_proj_output_activation_name=p.get("arl_subcarrier_proj_output_activation_name", "relu"),
            subcarrier_proj_arl_operation=p.get("arl_subcarrier_proj_arl_operation", "add"),
        )

    # -- builder methods -------------------------------------------------------

    def _build_denoiser(self) -> nn.Module:
        p = self._p
        if p.get("denoiser_num_filters_2d", 5) > 1:
            return _FDDDenoiser(
                num_filters_2d=p.get("denoiser_num_filters_2d", 5),
                filter_size_2d=p.get("denoiser_filter_size_2d", 3),
                filter_size_1d=p.get("denoiser_filter_size_1d", 3),
                activation=p.get("denoiser_activation", "relu"),
                is_post_processor=p.get("denoiser_is_post_processor", False),
                is_residual=p.get("denoiser_is_residual", True),
            )
        return nn.Identity()

    def _build_arl(self) -> nn.Module:
        return _FDDARLProcessor(**self._arl_kwargs())

    def _build_embedding(self) -> nn.Module:
        p = self._p
        return _FDDEmbedding(
            dim_model=self.dim_model,
            num_res_layers=p.get("embedding_num_res_layers", 4),
            res_dim=p.get("embedding_res_dim", 128),
            res_groups=p.get("embedding_res_groups", 8),
            hist_len=self.hist_len,
            dim_data=self.dim_data,
            embed=p.get("embedding_embed", "timeF"),
            freq=p.get("embedding_freq", "h"),
            dropout=p.get("embedding_dropout", 0.1),
        )

    def _build_transformer(self) -> nn.Module:
        p = self._p
        return _FDDTransformer(
            dim_model=self.dim_model,
            num_layers=p.get("transformer_num_layers", 4),
            num_heads=p.get("transformer_num_heads", 4),
            hidden_dim=p.get("transformer_hidden_dim", 2048),
            dropout_prob=p.get("transformer_dropout_prob", 0.2),
        )

    def _build_mlp(self) -> nn.Module:
        return _FDDMLPProcessor(
            dim_model=self.dim_model,
            dim_data=self.dim_data,
            hist_len=self.hist_len,
            pred_len=self.pred_len,
        )

    def forward(self, x):
        """Run the full FDD prediction pipeline."""
        x = self.denoiser(x)
        x, mean, std = batch_normalizer(x)
        x_delay, x_freq = self.arl(x)
        x = self.embedding(x_delay, x_freq)
        x = self.transformer(x)
        x = self.mlp(x)
        x = batch_denormalize(x, mean, std)
        return x
