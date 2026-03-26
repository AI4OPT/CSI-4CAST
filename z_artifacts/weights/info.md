All model weights used in the experiments are publicly available and can be downloaded from [CSI-4CAST/weights](https://huggingface.co/CSI-4CAST/weights) on Hugging Face. This includes both the proposed models and all baseline models.

## Downloading Weights

Download the full weights repository:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="CSI-4CAST/weights")
```

Download only one scenario (e.g. FDD or TDD):

```python
snapshot_download(repo_id="CSI-4CAST/weights", allow_patterns=["fdd/*"])
snapshot_download(repo_id="CSI-4CAST/weights", allow_patterns=["tdd/*"])
```

Download a single model checkpoint:

```python
snapshot_download(repo_id="CSI-4CAST/weights", allow_patterns=["tdd/rnn/*"])
```

## Expected Directory Structure

After downloading, place the files under this directory so the codebase can find them:

```
z_artifacts/weights/
├── fdd/
│   ├── abl_no_arl/model.ckpt
│   ├── abl_no_subcarrier_arl/model.ckpt
│   ├── cnn/model.ckpt
│   ├── llm4cp/model.ckpt
│   ├── model/model.ckpt
│   ├── rnn/model.ckpt
│   ├── stemgnn/model.ckpt
│   └── wiener/params.npz
└── tdd/
    ├── abl_add_subcarrier_arl/model.ckpt
    ├── abl_lstm_replace_pred/model.ckpt
    ├── abl_mlp_replace_embed/model.ckpt
    ├── abl_mlp_replace_pred/model.ckpt
    ├── abl_mobilenet_replace_embed/model.ckpt
    ├── abl_no_arl/model.ckpt
    ├── abl_no_denoiser/model.ckpt
    ├── abl_no_idft/model.ckpt
    ├── abl_norm_replace_arl/model.ckpt
    ├── ar/params.npz
    ├── cnn/model.ckpt
    ├── llm4cp/model.ckpt
    ├── model/model.ckpt
    ├── rnn/model.ckpt
    ├── stemgnn/model.ckpt
    └── wiener/params.npz
```

Neural models store checkpoints as `model.ckpt`. Statistical baselines (AR, Wiener) store parameters as `params.npz`.
