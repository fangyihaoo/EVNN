import torch
import torch.nn as nn
from basic_module import BasicModule
from typing import Callable
from torch import Tensor


class FullNet(BasicModule):
    r"""
        Fully connected netwrok
    """

    def __init__(self, 
        FClayer: int = 5,                                                    # number of fully-connected hidden layers
        activation: Callable[..., Tensor] = nn.Tanh(),                       # activation function
        num_input: int = 2,                                                  # dimension of input, in this case is 2 
        num_node: int = 20,                                                  # number of nodes in one fully-connected hidden layer
        num_oupt: int = 1,                                                   # dimension of output
        **kwargs
    ) -> None:

        super(FullNet, self).__init__()
        self.input = nn.Linear(num_input, num_node)
        self.act = activation
        self.output = nn.Linear(num_node, num_oupt)

        'Fully connected blocks'     
        self.linears_list = [nn.Linear(num_node, num_node) for i in range(FClayer)]
        self.acti_list = [self.act for i in range(FClayer)]
        self.block = nn.Sequential(*[item for pair in zip(self.linears_list, self.acti_list)for item in pair])

    def forward(self, x):
        x = self.input(x)
        x = self.block(x)
        x = self.output(x)
        return x
    

if __name__ == "__main__":
    model = FullNet()
    dat = torch.randn(5, 2)
    print(model(dat))