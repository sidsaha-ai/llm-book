"""
The first version of attention mechanism.
"""

import math

import torch


class MultiHeadAttention(torch.nn.Module):
    """
    A unified multi-head attention module.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout_percent: float = 0) -> None:
        super().__init__()

        self.in_dim: int = in_dim
        self.out_dim: int = out_dim
        self.num_heads: int = num_heads
        self.dropout_percent: float = dropout_percent

        assert self.out_dim % self.num_heads == 0, 'out_dim and num_heads do not work together'

        self.head_dim: int = int(out_dim / num_heads)

        self.w_query = torch.nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.w_key = torch.nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.w_value = torch.nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.dropout = torch.nn.Dropout(self.dropout_percent)
        self.out_proj = torch.nn.Linear(self.out_dim, self.out_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass for the multi-attention.
        """
        query = self.w_query(batch)
        key = self.w_key(batch)
        value = self.w_value(batch)

        # change query, key, and value to have num_heads
        # last dimension (4) split into 2 heads of 2 dimensions each
        query = query.reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)
        key = key.reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        value = value.reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_score = query @ key.transpose(len(key.shape) - 2, len(key.shape) - 1)
        attn_score = attn_score.masked_fill(
            ~torch.tril(torch.ones_like(attn_score)).bool(), -torch.inf,
        )

        attn_weights = torch.nn.functional.softmax(
            attn_score / math.sqrt(key.shape[-1]), dim=-1,
        )
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights @ value
        outputs = outputs.transpose(len(outputs.shape) - 3, len(outputs.shape) - 2)
        outputs = outputs.reshape(batch.shape[0], batch.shape[1], self.out_dim)
        outputs = self.out_proj(outputs)
        return outputs


def main():
    """
    The main method to test the attention module.
    """
    size = (6, 3)
    inputs = [
        torch.randn(size), torch.randn(size), torch.randn(size), torch.randn(size), torch.randn(size),
    ]
    batch = torch.stack(inputs)

    in_dim: int = batch.shape[-1]
    out_dim: int = 4
    num_heads: int = 2
    dropout_percent: float = 0

    attn = MultiHeadAttention(in_dim, out_dim, num_heads, dropout_percent)
    outputs = attn(batch)
    print(outputs)
    print(f'{outputs.shape=}')


if __name__ == '__main__':
    main()
