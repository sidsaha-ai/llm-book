"""
Let's try some tokenization in this script.
"""
import tiktoken
import torch


def main():
    """
    The main function to try tokenization.
    """
    tokenizer = tiktoken.get_encoding('gpt2')

    text1 = 'Every effort moves you'
    text2 = 'Every day holds a'
    text3 = 'Every man speaks up'

    inputs = []
    inputs.append(torch.tensor(tokenizer.encode(text1)))
    inputs.append(torch.tensor(tokenizer.encode(text2)))
    inputs.append(torch.tensor(tokenizer.encode(text3)))

    batch: torch.Tensor = torch.stack(tuple(inputs), dim=0)

    print(batch)


if __name__ == '__main__':
    main()
