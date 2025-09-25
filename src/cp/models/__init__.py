from src.cp.models.baseline_models.cnn import CNN_pl
from src.cp.models.baseline_models.llm4cp import LLM4CP_pl
from src.cp.models.baseline_models.np import NOPREDICTMODEL
from src.cp.models.baseline_models.rnn import RNN_pl
from src.cp.models.baseline_models.stemgnn import STEMGNN_pl
from src.cp.models.model_fdd import MODEL_fdd_pl
from src.cp.models.model_tdd import MODEL_tdd_pl


class PREDICTORS:
    CNN_FDD = CNN_pl
    CNN_TDD = CNN_pl

    LLM4CP_FDD = LLM4CP_pl
    LLM4CP_TDD = LLM4CP_pl

    NP_FDD = NOPREDICTMODEL
    NP_TDD = NOPREDICTMODEL

    RNN_FDD = RNN_pl
    RNN_TDD = RNN_pl

    STEMGNN_FDD = STEMGNN_pl
    STEMGNN_TDD = STEMGNN_pl

    MODEL_FDD = MODEL_fdd_pl
    MODEL_TDD = MODEL_tdd_pl
