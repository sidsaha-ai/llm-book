"""
This implements the feed-forward module that will be used in the LLM.
"""
import torch
from gelu import GELU
from torch import nn


class FeedForward(nn.Module):
    """
    Implements the feed-forward module.
    """
    def __init__(self, emb_dim: int) -> None:
        super().__init__()

        self.emb_dim: int = emb_dim

        self.layers = nn.Sequential(
            nn.Linear(self.emb_dim, 4 * self.emb_dim),
            GELU(),
            nn.Linear(4 * self.emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        return self.layers(x)


def main():
    """
    Main method to test the feed-forward module.
    """
    batch_size: int = 10
    num_tokens: int = 5
    emb_dim: int = 768

    inputs = torch.randn((batch_size, num_tokens, emb_dim), dtype=torch.float)

    ffn = FeedForward(emb_dim)
    outputs = ffn(inputs)

    print(f'{inputs.shape=}, {outputs.shape=}')


if __name__ == '__main__':
    main()
