"""
This implements the GPT model.
"""
import torch

class GPTModel(torch.nn.Module):
    """
    This is the GPT model.
    """
    def __init__(
            self,
            vocab_size: int,
            emb_dim: int,
            context_len: int,
            drop_rate: float,
            num_layers: int,
    ) -> None:
        super().__init__()