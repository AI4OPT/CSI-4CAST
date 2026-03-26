All datasets used in the experiments are publicly available on our [Hugging Face organization](https://huggingface.co/CSI-4CAST).

## Quick Start

- **For specific datasets**: Use `snapshot_download` to download individual datasets you need
- **For all datasets with original structure**: Run `download.py` followed by `reconstruction.py`

## Downloading Individual Datasets

```python
from huggingface_hub import snapshot_download

# Download normalization statistics
snapshot_download(repo_id="CSI-4CAST/stats", repo_type="dataset")

# Download a specific dataset
snapshot_download(repo_id="CSI-4CAST/test_regular_cm_A_ds_030_ms_001", repo_type="dataset")
```

### Dataset Naming Convention

Datasets follow the pattern `[train/test]_[regular/generalization]_cm_[A-E]_ds_[delay]_ms_[speed]`:

- `cm_[A/B/C/D/E]`: Channel models CDL-A through CDL-E
- `ds_[030/050/100/200/300/400]`: Delay spread in nanoseconds
- `ms_[001/.../045]`: User speed in m/s

Each dataset folder contains three PyTorch tensor files:
- `H_U_hist.pt`: Uplink historical CSI (model input)
- `H_U_pred.pt`: Uplink prediction target
- `H_D_pred.pt`: Downlink prediction target (for cross-link scenarios)

## Downloading All Datasets

Use the provided `download.py` script in this directory:

```bash
# Download all datasets
python3 z_artifacts/data/download.py

# Download to a custom directory
python3 z_artifacts/data/download.py --output-dir my_datasets

# Dry run to preview without downloading
python3 z_artifacts/data/download.py --dry-run
```

## Reconstructing Original Folder Structure

Use `reconstruction.py` to restore the original directory layout expected by the codebase:

```bash
python3 z_artifacts/data/reconstruction.py --input-dir datasets --output-dir z_artifacts/data
```

This removes the HF naming prefixes and organizes files into the original structure:

```
z_artifacts/data/
├── stats/
│   ├── fdd/normalization_stats.pkl
│   └── tdd/normalization_stats.pkl
├── train/regular/cm_A_ds_030_ms_001/...
├── test/regular/cm_A_ds_030_ms_001/...
└── test/generalization/cm_A_ds_030_ms_001/...
```

Reconstruction is only necessary if you need to replicate the paper's results or if your code expects the original folder structure. If you are working with individual datasets, you can skip this step.
