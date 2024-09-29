"""
This initializes GPT models of various sizes to find the number of parameters and memory size.
"""

from gpt import GPTModel
from dataclasses import dataclass

@dataclass
class ModelParams:
    emb_dim: int
    num_layers: int
    num_heads: int

def main():
    vocab_size: int = 50257
    context_len: int = 1024
    drop_rate: float = 0.1

    models = []

    models.append({
        'name': 'GPT2-Small',
        'params': ModelParams(emb_dim=768, num_layers=12, num_heads=12),
    })
    models.append({
        'name': 'GPT2-Medium',
        'params': ModelParams(emb_dim=1024, num_layers=24, num_heads=16),
    })
    models.append({
        'name': 'GPT2-Large',
        'params': ModelParams(emb_dim=1280, num_layers=36, num_heads=20),
    })
    models.append({
        'name': 'GPT2-XL',
        'params': ModelParams(emb_dim=1600, num_layers=48, num_heads=25),
    })

    for m in models:
        gpt_model = GPTModel(
            vocab_size=vocab_size,
            emb_dim=m.get('params').emb_dim,
            context_len=context_len,
            drop_rate=drop_rate,
            num_layers=m.get('params').num_layers,
            num_heads=m.get('params').num_heads,
        )

        num_params: int = sum(p.numel() for p in gpt_model.parameters())
        size_mb: float = (num_params * 4) / (1024 * 1024)
        print(f'{m.get("name")} model has {num_params:,} parameters and will take {size_mb:.2f} MB')


if __name__ == '__main__':
    main()
