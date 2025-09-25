import torch.nn as nn


def get_activation(activation_name: str) -> nn.Module:
    """
    Convert activation name string to PyTorch activation module.

    Args:
        activation_name: String name of the activation (e.g., 'relu', 'sigmoid', 'tanh', 'identity')

    Returns:
        nn.Module: The corresponding PyTorch activation module

    Raises:
        ValueError: If the activation name is not supported
    """
    activation_name = activation_name.lower()
    supported_activations = {
        "relu": "ReLU",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "identity": "Identity",
        "none": "Identity",
        "leaky_relu": "LeakyReLU",
        "elu": "ELU",
        "selu": "SELU",
        "gelu": "GELU",
        "softplus": "Softplus",
    }

    if activation_name not in supported_activations:
        raise ValueError(
            f"Unsupported activation: {activation_name}. "
            f"Supported activations are: {list(supported_activations.keys())}"
        )

    return getattr(nn, supported_activations[activation_name])()
