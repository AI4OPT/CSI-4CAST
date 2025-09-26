import pickle
from pathlib import Path


a = Path("z_refer/data/stats/fdd/normalization_stats.pkl")

with open(a, "rb") as f:
    data = pickle.load(f)

print(data)
