"""CSI Prediction Models Package.

Unified interface for CSI prediction models supporting FDD/TDD scenarios.
Registry pattern for easy model selection and switching.

Available Models:
- No-Prediction (NP): Baseline that repeats last observation
- RNN: Recurrent Neural Network for sequence prediction

Usage:
    from src.cp.models import PREDICTORS
    model = getattr(PREDICTORS, "RNN_TDD")(config)
    baseline = getattr(PREDICTORS, "NP_FDD")()
"""

from src.cp.models.baseline_models.np import NOPREDICTMODEL
from src.cp.models.baseline_models.rnn import RNN_pl


class PREDICTORS:
    """Registry for CSI prediction models with FDD/TDD variants.

    Example:
        >>> model = getattr(PREDICTORS, "RNN_TDD")(config)
        >>> baseline = getattr(PREDICTORS, "NP_FDD")(pred_len=4)

    """

    # Baseline models
    NP_FDD = NOPREDICTMODEL
    NP_TDD = NOPREDICTMODEL

    # Neural network models
    RNN_FDD = RNN_pl
    RNN_TDD = RNN_pl
