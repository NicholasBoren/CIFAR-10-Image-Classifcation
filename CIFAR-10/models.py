import torch
import torch.nn as nn


def NeuralNetA(M: int, d: int, device: str) -> nn.Module:
    """Neural Net with Fully connected output, 1 fully connected hidden layer.

    Args:
        M (int): Output dimension space for the model
        d (int): Input dimension space, given from image.
        device (str): Contains information on where a tensor will be located.

    Returns:
        nn.Module: Returns the described model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(d, M),
        nn.ReLU(),
        nn.Linear(M, 10)
    )

    return model.to(device)


def NeuralNetB(M: int, K: int, N: int, device: str) -> nn.Module:
    """Neural Net with Convolutional layer, with max-pooling and fully connected output.

    Args:
        M (int): Output dimension space for the model
        K (int): Filter size for the kernal
        N (int): Input for max pooling
        device (str): Contains information on where a tensor will be located.

    Returns:
        nn.Module: Returns the described model
    """
    model = nn.Sequential(
        nn.Conv2d(3, M, (K, K), (1, 1), 1),
        nn.ReLU(),
        nn.MaxPool2d(N),
        nn.Flatten(),
        nn.Linear(4 * M, 10)
    )

    return model.to(device)
