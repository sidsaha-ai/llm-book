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

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    def train(self, num_epochs: int) -> None:
        """
        Trains the model with the dataset.
        """
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.model = self.model.to(device)
        self.model.train()

        for epoch in range(num_epochs):
            for ix, (inputs, targets) in enumerate(self.data_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                self.optimizer.zero_grad()

                logits = self.model(inputs)
                logits = logits.flatten(0, 1)
                targets = targets.flatten()

                loss = torch.nn.functional.cross_entropy(logits, targets)
                print(f'Epoch: {epoch}, Batch Num: {ix}, Loss: {loss.item():.4f}')

                loss.backward()
                self.optimizer.step()
                

def main() -> None:
    """
    The main function that trains the GPT model.
    """
    tokenizer = tiktoken.get_encoding('gpt2')

    vocab_size: int = tokenizer.n_vocab
    emb_dim: int = 768
    context_len: int = 1024
    drop_rate: float = 0.1
    num_layers: int = 12
    num_heads: int = 12
    model = GPTModel(
        vocab_size=vocab_size, emb_dim=emb_dim, context_len=context_len, drop_rate=drop_rate, num_layers=num_layers, num_heads=num_heads,
    )

    batch_size: int = 2

    filepaths: list[str] = GPTDatasetUtils.input_filepaths()
    dataset = GPTDataset(filepaths, context_len, tokenizer)

    trainer = GPTTrainer(model, dataset, batch_size)
    
    num_epochs: int = 20
    trainer.train(num_epochs)


if __name__ == '__main__':
    main()
