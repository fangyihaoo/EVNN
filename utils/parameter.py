import torch
import torch.nn as nn
import torch.nn.init as init


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def weight_init(m):
    r"""Implement different weight initilization method (bias are set to zero)

    Args:
        m: model layer object
            please refer to 'https://pytorch.org/docs/stable/nn.init.html' for details

    Example:
        >>> model.apply(weight_init)
    """
    if isinstance(m, nn.Linear):
        gain = init.calculate_gain('tanh') 
        init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)