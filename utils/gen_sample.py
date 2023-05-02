import torch
from torch import Tensor
from torch.utils.data import Dataset
from pyDOE import lhs
from numpy import pi
import numpy as np


def poisson(num: int = 1000, 
            boundary: bool = False, 
            device: str ='cpu') -> Tensor:
    """
    2d poisson (0, pi)\times(-2/pi, 2/pi)
    Boundary:  uniform on each boundary
    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or gpu

    Returns:
        Tensor: Date coordinates tensor (N \times 2 or 4N \times 2)
    """
    
    if boundary:
        tb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([pi/2]).repeat(num)[:,None]), dim=1)
        bb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([-pi/2.]).repeat(num)[:,None]), dim=1)
        rb = torch.cat((torch.tensor([pi]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
        lb = torch.cat((torch.tensor([0.]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
        data = torch.cat((tb, bb, rb, lb), dim=0)
        data = data.to(device)
        return data
    else:
        lb = np.array([0., -pi/2])
        ran = np.array([pi, pi])
        data = torch.from_numpy(ran*lhs(2, num) + lb).float().to(device)        # generate the interior points
        return data


def allencahn(num: int = 1000, 
              boundary: bool = False, 
              device: str ='cpu') -> Tensor:
    """
    Allen-Cahn type problem (0, \infity)\times(0, 1)^2
    Boundary:  uniform on each boundary
    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or gpu

    Returns:
        Tensor: Date coordinates tensor (N \times 2 or 4N \times 2)
    """
    
    if boundary:
        tb = torch.cat((torch.rand(num, 1)*2 - 1, torch.tensor([1.]).repeat(num)[:,None]), dim=1)
        bb = torch.cat((torch.rand(num, 1)*2 - 1, torch.tensor([-1.]).repeat(num)[:,None]), dim=1)
        rb = torch.cat((torch.tensor([1.]).repeat(num)[:,None], torch.rand(num, 1)*2 - 1), dim=1)
        lb = torch.cat((torch.tensor([-1.]).repeat(num)[:,None], torch.rand(num, 1)*2 - 1), dim=1)
        data = torch.cat((tb, bb, rb, lb), dim=0)
        data = data.to(device) 
        return data
    else:
        data = torch.from_numpy(lhs(2, num)*2 - 1).float().to(device)        # generate the interior points
        return data
    
def meancur(num: int = 1000, 
              boundary: bool = False, 
              device: str ='cpu') -> Tensor:
    """
    Mean Curvature type problem (0, \infity)\times(-5, 5)^2
    Boundary:  no B.C. here.
    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. For interior points, the output would be 4\times number
        boundary (bool): Boundary or Interior
        device (str): cpu or gpu

    Returns:
        Tensor: Date coordinates tensor (N \times 2 or 4N \times 2)
    """
    
    if boundary:
        tb = torch.cat((torch.rand(num, 1)*2 - 1, torch.tensor([1.]).repeat(num)[:,None]), dim=1)
        bb = torch.cat((torch.rand(num, 1)*2 - 1, torch.tensor([-1.]).repeat(num)[:,None]), dim=1)
        rb = torch.cat((torch.tensor([1.]).repeat(num)[:,None], torch.rand(num, 1)*2 - 1), dim=1)
        lb = torch.cat((torch.tensor([-1.]).repeat(num)[:,None], torch.rand(num, 1)*2 - 1), dim=1)
        data = torch.cat((tb, bb, rb, lb), dim=0)
        data = data.to(device) 
        return data
    else:
        # lhs(k,n) generates n*k samples on [0,1], https://github.com/tisimst/pyDOE/blob/master/pyDOE/doe_lhs.py
        data = torch.from_numpy(5*(lhs(2, num)*2 - 1)).float().to(device)        # generate the interior points
        return data
            
            
def heatpinn(num: int = 1000, 
             data_type: str = 'boundary', 
             device: str = 'cpu') ->Tensor:
    """ 
    2d poisson (0, pi)\times(-2/pi, 2/pi)\times(0,2) for PINN
    Boundary:  uniform on each boundary
    Interior: latin square sampling

    Args:
        num (int, optional): number of data points. Defaults to 1000.
        data_type (str, optional): boundary condition. Defaults to boundary.
        device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.

    Returns:
        Tensor: (num, 3) dimension Tensor
    """
    
    if data_type == 'boundary':
        tb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([pi/2]).repeat(num)[:,None]), dim=1)
        bb = torch.cat((torch.rand(num, 1)* pi, torch.tensor([-pi/2.]).repeat(num)[:,None]), dim=1)
        rb = torch.cat((torch.tensor([pi]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
        lb = torch.cat((torch.tensor([0.]).repeat(num)[:,None], torch.rand(num, 1)*pi - pi/2), dim=1)
        data = torch.cat((tb, bb, rb, lb), dim=0)
        data = torch.cat((data, torch.rand(data.shape[0], 1)*2), dim = 1)
        return data.to(device)
    
    elif data_type == 'initial':
        lb = np.array([0., -pi/2])
        ran = np.array([pi, pi])
        data = torch.from_numpy(ran*lhs(2, num) + lb).float()        # generate the interior points
        data = torch.cat((data, torch.tensor([0]).repeat(data.shape[0])[:,None]), dim = 1)
        return data.to(device)
    
    else:
        lb = np.array([0., -pi/2])
        ran = np.array([pi, pi])
        data = torch.from_numpy(ran*lhs(2, num) + lb).float()       # generate the collocation points
        data = torch.cat((data, torch.rand(data.shape[0], 1)*2), dim = 1)
        return data.to(device)


def heat(num: int = 1000,
         boundary: bool = False,
         device: str = 'cpu') -> Tensor:
    """
    2d heat (0, 2)\times(0, 2)\times(0,2) for PINN
    Boundary:  uniform on each boundary
    Interior: latin square sampling

    Args:
        num (int, optional): number of data points. Defaults to 1000.
        data_type (str, optional): boundary condition. Defaults to boundary.
        device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.

    Returns:
        Tensor: (num, 2) dimension Tensor
    """
    if boundary:
        tb = torch.cat((torch.rand(num, 1)*2, torch.tensor([2.]).repeat(num)[:,None]), dim=1)
        bb = torch.cat((torch.rand(num, 1)*2, torch.tensor([0.]).repeat(num)[:,None]), dim=1)
        rb = torch.cat((torch.tensor([2.]).repeat(num)[:,None], torch.rand(num, 1)*2), dim=1)
        lb = torch.cat((torch.tensor([0.]).repeat(num)[:,None], torch.rand(num, 1)*2), dim=1)
        data = torch.cat((tb, bb, rb, lb), dim=0)
        return data.to(device) 
    else:
        data = torch.from_numpy(lhs(2, num)*2).float().to(device)        # generate the interior points
        return data

    
def poissoncycle(num: int = 1000, 
                 boundary: bool = False,
                 device: str = 'cpu') -> Tensor:
    """
    Poisson equation for -\laplacian u = 1
    in (-1,1) \times (-1, 1), x^2 + y^2 <= 1

    Args:
        num (int, optional): number of data points. Defaults to 1000.
        data_type (str, optional): boundary condition. Defaults to boundary.
        device (str, optional): 'cuda' or 'cpu'. Defaults to 'cpu'.

    Returns:
        Tensor: (num, 2) dimension Tensor
    """
    theta = torch.rand(num)*2*pi
    if boundary:
        data = torch.cat((torch.cos(theta).unsqueeze_(1), torch.sin(theta).unsqueeze_(1) ), dim = 1)
        return data.to(device)
    else:
        r = torch.sqrt(torch.rand(num))
        data = torch.cat(((r*torch.cos(theta)).unsqueeze_(1), (r*torch.sin(theta)).unsqueeze_(1)), dim = 1)
        return data.to(device)

def PoiHighGrid(num: int = 1000,
            d: int = 10,
            device: str ='cpu') -> Tensor:
    """
    poisson (0, 1)^d 
    d: dimention
    Interior: latin square sampling

    Args:
        num (int): number of points need to be sample. 
        device (str): cpu or gpu

    Returns:
        Tensor: Date coordinates tensor (N \times 2 or 4N \times 2)
    """
    return torch.from_numpy(2*lhs(d, num) - 1).float().to(device)

def disk_grid_regular (n, r, c, ng ):
    "copy from https://people.sc.fsu.edu/~jburkardt/py_src/disk_grid/disk_grid.py"

    cg = np.zeros ( [ 2, ng ] )

    p = 0

    for j in range ( 0, n + 1 ):

        i = 0
        x = c[0]
        y = c[1] + r * float ( 2 * j ) / float ( 2 * n + 1 )
        cg[0,p] = x
        cg[1,p] = y
        p = p + 1

        if ( 0 < j ):

            cg[0,p] = x
            cg[1,p] = 2.0 * c[1] - y
            p = p + 1

        while ( True ):

            i = i + 1
            x = c[0] + r * float ( 2 * i ) / float ( 2 * n + 1 )
            if ( r * r < ( x - c[0] ) ** 2 + ( y - c[1] ) ** 2 ):
                break

            cg[0,p] = x
            cg[1,p] = y
            p = p + 1
            cg[0,p] = 2.0 * c[0] - x
            cg[1,p] = y
            p = p + 1

            if ( 0 < j ):
                cg[0,p] = x
                cg[1,p] = 2.0 * c[1] - y
                p = p + 1;
                cg[0,p] = 2.0 * c[0] - x
                cg[1,p] = 2.0 * c[1] - y
                p = p + 1
        
    cg = np.transpose(cg)
    return cg[~np.all(cg == 0, axis=1)]
