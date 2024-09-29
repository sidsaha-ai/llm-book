"""
This implements the transformer block.
"""
import torch
from torch import nn


class TransformerBlock(nn.Module):
    """
    Implements a transformer block.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        return x
