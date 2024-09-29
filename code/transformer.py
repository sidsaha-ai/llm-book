"""
This implements the transformer block.
"""
import torch
from attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm
from torch import nn


class TransformerBlock(nn.Module):
    """
    Implements a transformer block.
    """
    def __init__(
            self,
            *,
            emb_dim: int,
            num_heads: int,
            dropout_percent: float,
    ) -> None:
        super().__init__()

        self.emb_dim: int = emb_dim
        self.num_heads: int = num_heads
        self.dropout_percent: float = dropout_percent

        # first set of layers
        self.m1 = nn.Sequential(
            LayerNorm(self.emb_dim),
            MultiHeadAttention(self.emb_dim, self.emb_dim, self.num_heads, self.dropout_percent),
            nn.Dropout(self.dropout_percent),
        )
        # second set of layers
        self.m2 = nn.Sequential(
            LayerNorm(self.emb_dim),
            FeedForward(self.emb_dim),
            nn.Dropout(self.dropout_percent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        x = x + self.m1(x)
        x = x + self.m2(x)

        return x

def main():
    """
    Try out the transformer block.
    """
    emb_dim: int = 768
    num_heads: int = 12
    dropout_percent: float = 0.1
    vocab_size: int = 768

    t = TransformerBlock(
        emb_dim=emb_dim,
        num_heads=num_heads,
        dropout_percent=dropout_percent,
    )

    batch_size: int = 2
    context_len: int = 4
    inputs = torch.randn((batch_size, context_len, vocab_size), dtype=torch.float)

    outputs = t(inputs)

    print(outputs)
    print(f'{inputs.shape=}, {outputs.shape=}')

if __name__ == '__main__':
    main()
