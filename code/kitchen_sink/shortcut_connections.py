"""
This is for playing with shortcut connections.
"""
import torch
from torch import nn

from gelu import GELU


class ShortcutNN(nn.Module):
    """
    A neural network with shortcut connections.
    """
    def __init__(self, layer_sizes: tuple[int, int], shortcut: bool = False) -> None:
        super().__init__()
        
        self.layer_sizes: tuple[int, int] = layer_sizes
        self.shortcut: bool = shortcut

        self.module_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(l[0], l[1]),
                GELU(),
            )
            for l in self.layer_sizes
        ])

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements the forward pass.
        """
        for m in self.module_list:
            output = m(x)
            x = x + output if self.shortcut and x.shape == output.shape else output
        return x
    
def print_gradients(model, inputs, targets):
    """
    Creates a NN without shortcut, does a forward and a backward pass, and prints gradients.
    """
    outputs = model(inputs)
    
    loss = nn.MSELoss()
    loss = loss(outputs, targets)
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f'{name} has gradient mean of {param.grad.abs().mean().item()}')


def main():
    """
    Main function to try out the shortcut NN.
    """
    layer_sizes: tuple[int, int] = [
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 1),
    ]
    inputs = [1, 0, -1]
    inputs = torch.tensor(inputs, dtype=torch.float)
    targets = [0]
    targets = torch.tensor(targets, dtype=torch.float)
    
    shortcut = False
    model = ShortcutNN(layer_sizes, shortcut)
    print('=== Without shortcut connections ===')
    print_gradients(model, inputs, targets)

    shortcut = True
    model = ShortcutNN(layer_sizes, shortcut)
    print('=== With shortcut connections === ')
    print_gradients(model, inputs, targets)



if __name__ == '__main__':
    main()
