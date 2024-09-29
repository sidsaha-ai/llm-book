"""
This creates a sample input that we will use during development.
"""
import tiktoken
import torch


class BatchGenerator:
    """
    Generatates a batch of input for development purpose.
    """

    @staticmethod
    def generate() -> torch.Tensor:
        """
        This generates the inputs to be used during development.
        """
        tokenizer = tiktoken.get_encoding('gpt2')

        t1: str = 'Every effort moves a'
        t2: str = 'Every day holds a'
        t3: str = 'Every man speaks up'
        inputs = [
            torch.tensor(tokenizer.encode(t1)),
            torch.tensor(tokenizer.encode(t2)),
            torch.tensor(tokenizer.encode(t3)),
        ]
        batch: torch.Tensor = torch.stack(tuple(inputs), dim=0)

        return batch

def main():
    """
    Main method to try out.
    """
    batch = BatchGenerator.generate()

    print(batch)
    print(f'{batch.shape=}')


if __name__ == '__main__':
    main()
