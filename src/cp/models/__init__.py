"""CSI prediction model registry for FDD/TDD scenarios.

Lookup convention used by src.testing.get_models:
    model_class = getattr(PREDICTORS, f"{model_name}_{scenario}")

Usage:
    from src.cp.models import PREDICTORS
    model = getattr(PREDICTORS, "RNN_TDD")(config)
    baseline = getattr(PREDICTORS, "NP_FDD")()
"""

# Baseline -- learning-based, no prediction, statistical
# Ablation
# Proposed
from src.cp.models.ablation.arl.fdd_no_subcarrier.model import NO_SUBCARRIER_ARL_FDD
from src.cp.models.ablation.arl.no_arl_fdd.model import NO_ARL_FDD
from src.cp.models.ablation.arl.no_arl_tdd.model import NO_ARL_TDD
from src.cp.models.ablation.arl.norm_replace.model import NORM_REPLACE_ARL_TDD
from src.cp.models.ablation.arl.tdd_add_subcarrier.model import ADD_SUBCARRIER_ARL_TDD
from src.cp.models.ablation.denoiser.no_denoiser.model import NO_DENOISER_TDD
from src.cp.models.ablation.embedding.mlp_replace.model import MLP_REPLACE_EMBED_TDD
from src.cp.models.ablation.embedding.mobilenet_replace.model import MOBILENET_REPLACE_EMBED_TDD
from src.cp.models.ablation.idft.no_idft.model import NO_IDFT_TDD
from src.cp.models.ablation.predictor.lstm_replace.model import LSTM_REPLACE_PRED_TDD
from src.cp.models.ablation.predictor.mlp_replace.model import MLP_REPLACE_PRED_TDD
from src.cp.models.baseline.learning.cnn import CNN_cp
from src.cp.models.baseline.learning.llm4cp import LLM4CP_pl
from src.cp.models.baseline.learning.rnn import RNN_pl
from src.cp.models.baseline.learning.stemgnn import STEMGNN_pl
from src.cp.models.baseline.np import NOPREDICTMODEL
from src.cp.models.baseline.statistical.ar import ARMODEL
from src.cp.models.baseline.statistical.pad import PADMODEL
from src.cp.models.baseline.statistical.wiener import WIENERMODEL
from src.cp.models.proposed.model_fdd import MODEL_FDD
from src.cp.models.proposed.model_tdd import MODEL_TDD


class PREDICTORS:
    """Registry for CSI prediction models with FDD/TDD variants."""

    # No-prediction baseline (both scenarios)
    NP_FDD = NOPREDICTMODEL
    NP_TDD = NOPREDICTMODEL

    # Learning-based baselines (both scenarios)
    RNN_FDD = RNN_pl
    RNN_TDD = RNN_pl
    CNN_FDD = CNN_cp
    CNN_TDD = CNN_cp
    STEMGNN_FDD = STEMGNN_pl
    STEMGNN_TDD = STEMGNN_pl
    LLM4CP_FDD = LLM4CP_pl
    LLM4CP_TDD = LLM4CP_pl

    # Statistical baselines
    PAD_TDD = PADMODEL
    AR_TDD = ARMODEL
    WIENER_FDD = WIENERMODEL
    WIENER_TDD = WIENERMODEL

    # Proposed models
    MODEL_FDD = MODEL_FDD
    MODEL_TDD = MODEL_TDD

    # Ablation -- denoiser
    ABL_NO_DENOISER_TDD = NO_DENOISER_TDD

    # Ablation -- IDFT
    ABL_NO_IDFT_TDD = NO_IDFT_TDD

    # Ablation -- ARL
    ABL_NO_ARL_TDD = NO_ARL_TDD
    ABL_NO_ARL_FDD = NO_ARL_FDD
    ABL_NORM_REPLACE_ARL_TDD = NORM_REPLACE_ARL_TDD
    ABL_ADD_SUBCARRIER_ARL_TDD = ADD_SUBCARRIER_ARL_TDD
    ABL_NO_SUBCARRIER_ARL_FDD = NO_SUBCARRIER_ARL_FDD

    # Ablation -- embedding
    ABL_MLP_REPLACE_EMBED_TDD = MLP_REPLACE_EMBED_TDD
    ABL_MOBILENET_REPLACE_EMBED_TDD = MOBILENET_REPLACE_EMBED_TDD
    # Ablation -- predictor
    ABL_MLP_REPLACE_PRED_TDD = MLP_REPLACE_PRED_TDD
    ABL_LSTM_REPLACE_PRED_TDD = LSTM_REPLACE_PRED_TDD
