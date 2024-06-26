import torch
from torch import Tensor, nn


class NormalizedMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()

    # noinspection PyShadowingBuiltins
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.mean((torch.div(prediction, target) - 1) ** 2)
