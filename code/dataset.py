"""
This creates the custom dataset to pretrain the LLM.
"""
import os

import tiktoken
import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    """
    This implements the GPT dataset.
    """
    def __init__(self, filepaths: list[str], context_len: int, tokenizer) -> None:
        self.filepaths = filepaths
        self.context_len = context_len
        self.tokenizer = tokenizer

        self.input_ids = []
        self.target_ids = []

        self._build()

    def _read_file(self, filepath: str) -> str:
        """
        Read and return the contents of the file.
        """
        content: str = ''
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    def _build(self) -> None:
        """
        This function builds the dataset.
        """
        for ix, path in enumerate(self.filepaths):
            print(f'Reading file at index: {ix}, Filepath: {path}')
            content = self._read_file(path)
            tokens = self.tokenizer.encode(content)

            for ix in range(0, len(tokens) - self.context_len, self.context_len):
                inputs = torch.tensor(tokens[ix:ix + self.context_len])
                targets = torch.tensor(tokens[ix + 1:ix + 1 + self.context_len])

                self.input_ids.append(inputs)
                self.target_ids.append(targets)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, ix: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the input and target tensor for a given index.
        """
        return self.input_ids[ix], self.target_ids[ix]


class GPTDatasetUtils:
    """
    Utility function to get the input data filepaths.
    """

    @staticmethod
    def input_filepaths() -> list[str]:
        """
        Reads and returns the filepaths.
        """
        filepaths: list[str] = []
        data_dir: str = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
        )
        for f in os.listdir(data_dir):
            if '.txt' in f:
                filepaths.append(os.path.join(data_dir, f))
        return filepaths


def main():
    """
    Main function to test the dataset.
    """
    filepaths: list[str] = GPTDatasetUtils.input_filepaths()
    tokenizer = tiktoken.get_encoding('gpt2')
    context_len: int = 4

    dataset = GPTDataset(filepaths, context_len, tokenizer)
    print(f'Dataset length: {len(dataset)}')

    inputs, targets = dataset[0]
    print(f'Inputs: {inputs}, Shape: {inputs.shape}')
    print(f'Targets: {targets}, Shape: {targets.shape}')

    inputs, targets = dataset[1]
    print(f'Inputs: {inputs}, Shape: {inputs.shape}')
    print(f'Targets: {targets}, Shape: {targets.shape}')


if __name__ == '__main__':
    main()
