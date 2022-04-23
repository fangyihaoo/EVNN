import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
import torch.nn.init as init
from typing import Callable, Tuple

"""
This part of code comes from "https://github.com/CW-Huang/CP-Flow"
"""

_scaling_min = 0.001

# torch.set_default_dtype(torch.float64)

def unsqueeze(x):
    return x.unsqueeze(0).unsqueeze(-1).detach()

class ActNorm(torch.nn.Module):
    """
    ActNorm layer
    """
    
    def __init__(self,
                 num_features: int, # number of input dimension
                 logscale_factor: float = 1.,
                 scale: float = 1.,
                 learn_scale: bool = True
                 ) -> None:
        super(ActNorm, self).__init__()

        self.initialized = False
        self.num_features = num_features

        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))
            
    def forward_transform(self, x, logdet=0):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True
            
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size)
                logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6)) / self.logscale_factor
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + _scaling_min
            output = output * scale
            dlogdet = torch.sum(torch.log(scale)) * x.size(-1)  # c x h

            return output.view(input_shape), logdet + dlogdet
        else:
            return output.view(input_shape), logdet

class ActNormNoLogdet(ActNorm):

    def forward(self, x):
        return super(ActNormNoLogdet, self).forward_transform(x)[0]

class SequentialFlow(torch.nn.Module):

    def __init__(self, flows):
        super(SequentialFlow, self).__init__()
        self.flows = torch.nn.ModuleList(flows)

    def forward_transform(self, x, logdet=0):
        for flow in self.flows:
            x, logdet = flow.forward_transform(x, logdet)
        return x, logdet



def symm_softplus(x, softplus_=torch.nn.functional.softplus):
    return softplus_(x) - 0.5 * x


def softplus(x):
    return nn.functional.softplus(x)


def gaussian_softplus(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x**2 / 2) + z * x) / (2*z)


def gaussian_softplus2(x):
    z = np.sqrt(np.pi / 2)
    return (z * x * torch.erf(x / np.sqrt(2)) + torch.exp(-x**2 / 2) + z * x) / z


def laplace_softplus(x):
    return torch.relu(x) + torch.exp(-torch.abs(x)) / 2


def cauchy_softplus(x):
    # (Pi y + 2 y ArcTan[y] - Log[1 + y ^ 2]) / (2 Pi)
    pi = np.pi
    return (x * pi - torch.log(x**2 + 1) + 2 * x * torch.atan(x)) / (2*pi)


def activation_shifting(activation):
    def shifted_activation(x):
        return activation(x) - activation(torch.zeros_like(x))
    return shifted_activation


def get_softplus(softplus_type='softplus', zero_softplus=False):
    if softplus_type == 'softplus':
        act = nn.functional.softplus
    elif softplus_type == 'gaussian_softplus':
        act = gaussian_softplus
    elif softplus_type == 'gaussian_softplus2':
        act = gaussian_softplus2
    elif softplus_type == 'laplace_softplus':
        act = gaussian_softplus
    elif softplus_type == 'cauchy_softplus':
        act = cauchy_softplus
    else:
        raise NotImplementedError(f'softplus type {softplus_type} not supported.')
    if zero_softplus:
        act = activation_shifting(act)
    return act


class Softplus(nn.Module):
    def __init__(self, softplus_type='softplus', zero_softplus=False):
        super(Softplus, self).__init__()
        self.softplus_type = softplus_type
        self.zero_softplus = zero_softplus

    def forward(self, x):
        return get_softplus(self.softplus_type, self.zero_softplus)(x)


class SymmSoftplus(torch.nn.Module):

    def forward(self, x):
        return symm_softplus(x)


class PosLinear(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        gain = 1 / x.size(1)
        return nn.functional.linear(x, torch.nn.functional.softplus(self.weight), self.bias) * gain


class PosLinear2(torch.nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, torch.nn.functional.softmax(self.weight, 1), self.bias)


class PosConv2d(torch.nn.Conv2d):

    def reset_parameters(self) -> None:
        super().reset_parameters()
        # noinspection PyProtectedMember,PyAttributeOutsideInit
        self.fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, torch.nn.functional.softplus(self.weight)) / self.fan_in



class ICNN(torch.nn.Module):
    def __init__(self, dim=2, dimh=16, num_hidden_layers=1):
        super(ICNN, self).__init__()

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh, bias=False))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)
        self.act = nn.Softplus()

    def forward(self, x):
        z = self.act(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)





class ICNN3(torch.nn.Module):
    def __init__(self, 
                 dim: int = 2, # input dimension 
                 dimh: int = 16, # hidden layer dimension
                 num_hidden_layers: int = 2, # number of hidden layer 
                 symm_act_first: bool = False, # refer to the paper
                 softplus_type: str = 'softplus', # refer to the paper
                 zero_softplus: bool = False # refer to the paper
                 ) -> None:
        super(ICNN3, self).__init__()

        self.act = Softplus(softplus_type=softplus_type, zero_softplus=zero_softplus)
        self.symm_act_first = symm_act_first

        Wzs = list()
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PosLinear(dimh, dimh // 2, bias=True))
        Wzs.append(PosLinear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = list()
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh // 2))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)

        Wx2s = list()
        for _ in range(num_hidden_layers - 1):
            Wx2s.append(nn.Linear(dim, dimh // 2))
        self.Wx2s = torch.nn.ModuleList(Wx2s)

        actnorms = list()
        for _ in range(num_hidden_layers - 1):
            actnorms.append(ActNormNoLogdet(dimh // 2))
        actnorms.append(ActNormNoLogdet(1))
        actnorms[-1].b.requires_grad_(False)
        self.actnorms = torch.nn.ModuleList(actnorms)

    def forward(self, x):
        if self.symm_act_first:
            z = symm_softplus(self.Wzs[0](x), self.act)
        else:
            z = self.act(self.Wzs[0](x))
        for Wz, Wx, Wx2, actnorm in zip(self.Wzs[1:-1], self.Wxs[:-1], self.Wx2s[:], self.actnorms[:-1]):
            z = self.act(actnorm(Wz(z) + Wx(x)))
            aug = Wx2(x)
            aug = symm_softplus(aug, self.act) if self.symm_act_first else self.act(aug)
            z = torch.cat([z, aug], 1)
        return self.actnorms[-1](self.Wzs[-1](z) + self.Wxs[-1](x))



class DeepConvexFlow(torch.nn.Module):
    def __init__(self, 
                 icnn: Callable[..., Tensor], # ICNN object
                 bias_w1: float = 0.0, # bias parameters in actnorm layer
                 trainable_w0: bool = True # weight parameters in actnorm layer
                 ) -> None:
        super(DeepConvexFlow, self).__init__()

        self.icnn = icnn
        self.w0 = torch.nn.Parameter(torch.log(torch.exp(torch.ones(1)) - 1), requires_grad=trainable_w0)
        self.w1 = torch.nn.Parameter(torch.zeros(1) + bias_w1)

        
    def get_potential(self, 
                      x: Tensor
                      ) -> Tensor:
        """compute the potential F

        Args:
            x (Tensor): coordinates

        Returns:
            Tensor: potential F (N, 1)
        """
        n = x.size(0)
        icnn = self.icnn(x)
        return F.softplus(self.w1) * icnn + F.softplus(self.w0) * (x.view(n, -1) ** 2).sum(1, keepdim=True) / 2
    
    def forward(self, 
                x: Tensor
                ) -> Tensor:
        """forward function, compute the convex potential f, gradient of F

        Args:
            x (Tensor): coordinate

        Returns:
            Tensor: convex potential, f (N, d)
        """
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            F = self.get_potential(x)
            f = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
        return f
    
    def forward_transform(self, 
                          x: Tensor, 
                          logdet: Tensor = 0
                          ) -> Tuple[Tensor, Tensor]:
        """compute the convex potential f and log determinant

        Args:
            x (Tensor): coordinate
            logdet (Tensor, optional): log determinant (N, 1). Defaults to 0.

        Returns:
            Tuple[Tensor, Tensor]: convex potential, log determinant
        """
        
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            F = self.get_potential(x)
            f = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
            H = []
            
            for i in range(f.shape[1]):
                H.append(torch.autograd.grad(f[:, i].sum(), x, create_graph = True, retain_graph = True)[0])
                
            H = torch.stack(H, dim = 1)
        
        return f, logdet + torch.slogdet(H).logabsdet
    

if __name__ == "__main__":
    dimx = 2
    nblocks = 1
    depth = 6
    k = 32
    symm_act_first = True
    zero_softplus = False
    softplus_type = 'gaussian_softplus2'
    icnns = [ICNN3(dimx, k, depth, symm_act_first=symm_act_first, softplus_type=softplus_type, zero_softplus=zero_softplus) for _ in range(nblocks)]
    layers = [None] * (nblocks + 1)
    layers[0] = ActNorm(dimx)
    layers[1:] = [DeepConvexFlow(icnn, bias_w1=-0.0, trainable_w0=False) for _, icnn in zip(range(nblocks), icnns)]
    flow = SequentialFlow(layers)
    print('Done')