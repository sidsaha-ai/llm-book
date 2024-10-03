"""
This creates the custom dataset to pretrain the LLM.
"""
import os

from torch.utils.data import Dataset
import tiktoken


class GPTDataset(Dataset):
    """
    This implements the GPT dataset.
    """
    def __init__(self, filepaths: list[str], tokenizer) -> None:
        self.filepaths = filepaths
        self.tokenizer = tokenizer

        self.input_ids = []
        self.target_ids = []

        self._build()
    
    def _build(self) -> None:
        """
        This function builds the dataset.
        """
        for ix, path in enumerate(self.filepaths):
            print(f'Index: {ix}, Filepath: {path}')


def _read_input_filepaths() -> list[str]:
    """
    Reads the filepaths for all the input files.
    """
    filepaths: list[str] = []
    data_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data'
    )
    for f in os.listdir(data_dir):
        if '.txt' not in f:
            continue
        filepaths.append(os.path.join(data_dir, f))
    return filepaths
        

def main():
    """
    Main function to test the dataset.
    """
    filepaths: list[str] = _read_input_filepaths()
    tokenizer = tiktoken.get_encoding('gpt2')

    dataset = GPTDataset(filepaths, tokenizer)


if __name__ == '__main__':
    main()
