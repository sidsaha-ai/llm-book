"""
This contains a class to train the model.
"""
import torch
import tiktoken

from dataset import GPTDataset, GPTDatasetUtils
from model import GPTModel

from torch.utils.data import DataLoader

class GPTTrainer:
    """
    The trainer class to train the GPT model.
    """
    def __init__(self, model: GPTModel, dataset: GPTDataset, batch_size: int):
        super().__init__()

        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size

        self.data_loader = DataLoader(
            self.dataset, self.batch_size, shuffle=True, num_workers=0, drop_last=True,
        )

        for inputs, targets in self.data_loader:
            print(inputs)
            print(targets)
            print(f'{inputs.shape=}')
            print(f'{targets.shape=}')
            break
    

def main() -> None:
    """
    The main function that trains the GPT model.
    """
    tokenizer = tiktoken.get_encoding('gpt2')

    vocab_size: int = tokenizer.n_vocab
    emb_dim: int = 4
    context_len: int = 8
    drop_rate: float = 0.1
    num_layers: int = 4
    num_heads: int = 4
    model = GPTModel(
        vocab_size=vocab_size, emb_dim=emb_dim, context_len=context_len, drop_rate=drop_rate, num_layers=num_layers, num_heads=num_heads,
    )

    batch_size: int = 2

    filepaths: list[str] = GPTDatasetUtils.input_filepaths()
    dataset = GPTDataset(filepaths, context_len, tokenizer)

    trainer = GPTTrainer(model, dataset, batch_size)


if __name__ == '__main__':
    main()
