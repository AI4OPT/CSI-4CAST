import torch

from src.testing.get_models import get_eval_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_eval_model(model_name="MODEL", device=DEVICE, scenario="TDD")

input = torch.randn(32, 16, 600).to(DEVICE)
output = model(input)
print(output.shape)
