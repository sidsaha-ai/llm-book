"""
Implements a Layer Normalization class.
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Implements a Layer Normalization module.
    """
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        
        self.emb_dim: int = emb_dim
        self.eps: float = 1e-5
        
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm + self.shift
