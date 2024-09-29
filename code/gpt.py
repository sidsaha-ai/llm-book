"""
This implements the GPT model.
"""
import tiktoken
import torch
from inputs import BatchGenerator
from layer_norm import LayerNorm
from torch import nn
from transformer import TransformerBlock


class GPTModel(nn.Module):
    """
    This is the GPT model.
    """
    def __init__(  # pylint: disable=too-many-arguments
            self,
            *,
            vocab_size: int,
            emb_dim: int,
            context_len: int,
            drop_rate: float,
            num_layers: int,
            num_heads: int,
    ) -> None:
        super().__init__()

        self.vocab_size: int = vocab_size
        self.emb_dim: int = emb_dim
        self.context_len: int = context_len
        self.drop_rate: int = drop_rate
        self.num_layers: int = num_layers

        # token embeddings
        self.tok_emb = nn.Embedding(self.vocab_size, self.emb_dim)
        # positional embeddings
        self.pos_emb = nn.Embedding(self.context_len, self.emb_dim)
        # dropout
        self.dropout = nn.Dropout(self.drop_rate)
        # transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout_percent=drop_rate,
            ) for _ in range(self.num_layers)],
        )
        # final layer norm
        self.layer_norm = LayerNorm(self.emb_dim)
        # output layer
        self.out_head = nn.Linear(self.emb_dim, self.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        _, sequence_len = x.shape

        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(sequence_len))
        x = tok_embeds + pos_embeds
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)

        logits = self.out_head(x)
        return logits


def main():
    """
    Main function to try the GPT block.
    """
    tokenizer = tiktoken.get_encoding('gpt2')
    batch = BatchGenerator.generate()

    model = GPTModel(
        vocab_size=tokenizer.n_vocab,
        emb_dim=768,
        context_len=1024,
        drop_rate=0.1,
        num_layers=12,
        num_heads=12,
    )

    outputs = model(batch)
    print(outputs)
    print(f'{batch.shape=}, {outputs.shape=}')


if __name__ == '__main__':
    main()
