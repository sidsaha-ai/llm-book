"""
Implements a Layer Normalization class.
"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Implements a Layer Normalization module.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        return x
