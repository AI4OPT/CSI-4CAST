import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int, hidden_dim: int, output_activation: nn.Module):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_activation = output_activation

        if self.num_layers > 2:
            self.input_layers = [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            ]

            self.hidden_layers = []
            for _ in range(self.num_layers - 2):
                self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.hidden_layers.append(nn.ReLU())
                self.hidden_layers.append(nn.Dropout(p=0.1))

            self.output_layers = [
                nn.Linear(hidden_dim, out_dim),
                self.output_activation,
            ]

            self.mlp = nn.Sequential(*self.input_layers, *self.hidden_layers, *self.output_layers)

        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), self.output_activation)

    def forward(self, x):
        x = self.mlp(x)
        return x
