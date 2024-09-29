"""
This implements a GELU layer from scratch.
"""
import torch
from matplotlib import pyplot as plt
from torch import nn


class GELU(nn.Module):
    """
    Implements the GELU model.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        res = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0, dtype=x.dtype) / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return res


def main():
    """
    Main method to try out GELU.
    """
    cases = []

    inputs = 1
    cases.append(inputs)

    inputs = [10, -4, 5, 1, -100, 100, 1]
    cases.append(inputs)

    for c in cases:
        inputs = torch.tensor(c, dtype=torch.float)

        gelu = GELU()
        print(f'My GELU: {gelu(inputs)}, PyTorch GELU: {nn.functional.gelu(inputs)}')

def plot_gelu_and_relu():
    """
    Plot GELU and ReLU.
    """
    gelu = GELU()
    relu = nn.ReLU()

    x = torch.linspace(-5, 5, 100)
    y_gelu = gelu(x)
    y_relu = relu(x)

    plt.figure(figsize=(8, 3))

    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ['GELU', 'RELU'])):
        plt.subplot(1, 2, i + 1)
        plt.plot(x, y)
        plt.title(label)
        plt.xlabel('x')
        plt.ylabel(f'{label}(x)')
        plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    plot_gelu_and_relu()
