import torch
from torch import Tensor
from typing import Callable, List, Tuple
from numpy import pi

def L2_Reg(model1: Callable[..., Tensor],
           model2: Callable[..., Tensor]):
    """
    output the L2 regulation on weight parameters between current and previous models
    """
    reg = 0
    for (name1, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if 'weight' in name1:
            reg += torch.sum(torch.square(p1 - p2.detach()))
    return reg

def Willmore3dLoss(model: Callable[...,Tensor], 
         dat_i: Tensor, 
         #dat_b : Tensor, 
         previous: Tuple[Tensor,Tensor]) -> Tuple[Tensor,Tensor]:
    """
    3d Willmore flow equation: u_t = \nabla \cdot (\frac{\nabla u}{|\nabla u|}) * |\nabla u|, x \in [-2,2]^3
    ellipsoid: x**2/1.+ y**2/0.5 + z**2/0.5 -1=0
    I.C. u(x, 0) = signed distance function
    B.C. homogenous \partial u/\partial \n = 0
    Energy: \kappa^2/2, where \kappa = \nabla \cdot \frac{\nabla u}{|\nabla u|}
    Constrain: |\nabla u| =1.


    Args:
        model (Callable[...,Tensor]): [description]
        dat_i (Tensor): [description]
        dat_b (Tensor): [description]
        previous (Tuple[Tensor,Tensor]): [description]
    # return two loss, for debug
    Returns:
        Tuple[Tensor,Tensor]: [description]
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    #print(f'dat_i: {dat_i.shape} , output_i: {output_i.shape}, ux:{ux.shape}')
    #ux_normlized = ux/torch.norm(ux,dim=1,keepdim=True)
    # kappa = divergence of ux_normlized
    s, d = dat_i.shape
    kappa = 0.
    for i in range(d):
        #kappa += torch.autograd.grad(outputs = ux_normlized[:,i], inputs = dat_i, grad_outputs = torch.ones_like(ux_normlized[:,i]), retain_graph=True, create_graph=True)[0][...,i:i+1]
        kappa += torch.autograd.grad(outputs = ux[:,i], inputs = dat_i, grad_outputs = torch.ones_like(ux[:,i]), retain_graph=True, create_graph=True)[0][...,i:i+1]
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(kappa, 2),dim=1,keepdim=True)) * 64
    
    #penalty term
    lam = 500
    loss_i += lam*torch.mean(torch.pow(torch.norm(ux,dim=1,keepdim=True)-1. ,2))*64
    #loss_b = torch.mean(torch.pow(output_b,2))

    loss_p = 50*torch.mean(torch.pow(output_i - previous[0], 2)) * 64
    #print(f'loss_i:{loss_i}, loss_p:{loss_p}')

    return loss_i + loss_p, loss_i

def WillmoreLoss(model: Callable[...,Tensor], 
         dat_i: Tensor, 
         #dat_b : Tensor, 
         previous: Tuple[Tensor,Tensor]) -> Tuple[Tensor,Tensor]:
    """
    2d Willmore flow equation: u_t = \nabla \cdot (\frac{\nabla u}{|\nabla u|}) * |\nabla u|, x \in [-5,5]^2
    shape: any shape on 2D 
    I.C. u(x, 0) = signed distance function
    B.C. homogenous \partial u/\partial \n = 0
    Energy: \kappa^2/2, where \kappa = \nabla \cdot \frac{\nabla u}{|\nabla u|}
    Constrain: |\nabla u| =1.


    Args:
        model (Callable[...,Tensor]): [description]
        dat_i (Tensor): [description]
        dat_b (Tensor): [description]
        previous (Tuple[Tensor,Tensor]): [description]
    # return two loss, for debug
    Returns:
        Tuple[Tensor,Tensor]: [description]
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    #output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    #print(f'dat_i: {dat_i.shape} , output_i: {output_i.shape}, ux:{ux.shape}')
    ux_normlized = ux/torch.norm(ux,dim=1,keepdim=True)
    # kappa = divergence of ux_normlized
    s, d = dat_i.shape
    kappa = 0.
    for i in range(d):
        #kappa += torch.autograd.grad(outputs = ux_normlized[:,i], inputs = dat_i, grad_outputs = torch.ones_like(ux_normlized[:,i]), retain_graph=True, create_graph=True)[0][...,i:i+1]
        kappa += torch.autograd.grad(outputs = ux[:,i], inputs = dat_i, grad_outputs = torch.ones_like(ux[:,i]), retain_graph=True, create_graph=True)[0][...,i:i+1]
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(kappa, 2),dim=1,keepdim=True)) * 100
    
    #penalty term
    lam = 250
    loss_i += lam*torch.mean(torch.pow(torch.norm(ux,dim=1,keepdim=True)-1. ,2))*100
    #loss_b = torch.mean(torch.pow(output_b,2))

    loss_p = 500*torch.mean(torch.pow(output_i - previous[0], 2)) * 100
    #print(f'loss_i:{loss_i}, loss_p:{loss_p}')

    return loss_i + loss_p, loss_i

def MeanCurLoss(model: Callable[...,Tensor], 
         dat_i: Tensor, 
         #dat_b : Tensor, 
         previous: Tuple[Tensor,Tensor]) -> Tuple[Tensor,Tensor]:
    """
    2d Mean curvature flow equation:
    I.C. u(x, 0) = signed distance function
    B.C. homogenous \partial u/\partial \n = 0
    Energy: (\nabla u)^2
    Constrain: |\nabla u| =1.

    Args:
        model (Callable[...,Tensor]): [description]
        dat_i (Tensor): [description]
        dat_b (Tensor): [description]
        previous (Tuple[Tensor,Tensor]): [description]
    # return two loss, for debug
    Returns:
        Tuple[Tensor,Tensor]: [description]
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    #output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    #print(f'dat_i: {dat_i.shape} , output_i: {output_i.shape}, ux:{ux.shape}')
    
    #square of L2 norm
    #loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)) * 100

    #L2 norm
    loss_i =  torch.mean(torch.norm(ux,dim=1,keepdim=True)) * 100
    
    #penalty term
    lam = 1 
    loss_i += lam*torch.mean(torch.pow(torch.norm(ux,dim=1,keepdim=True)-1. ,2))*100
   
    tau = 0.001
    loss_p = torch.mean(torch.pow(output_i - previous[0], 2)) * 100/(2*tau)
    #print(f'loss_i:{loss_i}, loss_p:{loss_p}')

    return loss_i + loss_p, loss_i


def PoiLoss(model: Callable[..., Tensor], 
            dat_i: Tensor, 
            dat_b: Tensor,
            previous: List[Tensor]) -> Tuple[Tensor,Tensor]:
    """
    Loss function for 2d Poisson equation
    -\laplacia u = 2sin(x)cos(y),    u \in \Omega
    u = 0,                           u \in \partial \Omega (0, pi) \times (-pi/2, pi/2)

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior point
        dat_b (Tensor): Boundary point
        previous (Tuple[Tensor, Tensor]): Result from previous time step model. interior point for index 0, boundary for index 1

    Returns:
        Tuple[Tensor,Tensor]: loss
    """

    f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    # f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)- f*output_i)
    loss_b = torch.mean(torch.pow(output_b,2))
    if previous:
        loss_p = 50*torch.mean(torch.pow(output_i - previous[0], 2))
        loss_p += 50*torch.mean(torch.pow(output_b - previous[1], 2))
    else:
        loss_p = 0

    return loss_i + 500*loss_b + loss_p, loss_i


def PoiHighLoss(model: Callable[..., Tensor], 
            dat: Tensor, 
            previous: List[Tensor] = None) -> Tensor:
    """
    Loss function for 2d Poisson equation
    -\laplacia u = pi^2 \sum_k=1^d cos(pi*x_k),    u \in \Omega
    u = 0,                           u \in \partial \Omega (-1, 1) ^ d

    Args:
        model (Callable[..., Tensor]): Network 
        dat (Tensor): Interior point
        previous (Tensor): Result from previous time step model. 

    Returns:
        Tensor: loss
    """

    f = pi*pi*torch.sum(torch.cos(pi*dat), dim = 1, keepdim = True)
    dat.requires_grad = True
    output = model(dat)
    ux = torch.autograd.grad(outputs = output, inputs = dat, grad_outputs = torch.ones_like(output), retain_graph=True, create_graph=True)[0]
    loss =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)- f*output)
    # loss_p = 50*torch.mean(torch.pow(output - previous, 2))
    if previous is not None:
        loss_p = 50*torch.mean(torch.pow(output - previous, 2))
    else:
        loss_p = 0
    return loss +  loss_p + 0.5*torch.mean(output)**2


def AllenCahnW(model: Callable[..., Tensor], 
               dat_i: Tensor, 
               dat_b: Tensor, 
               previous: List[Tensor]) -> Tuple[Tensor,Tensor]:
    """
    \int 0.5*|\nabla \phi|^2 + 0.25*(\phi^2 - 1)^2/epislon^2 dx + W*(\int\phidx - A)^2
    r = 0.25
    A = (4 - pi*(r**2))*(-1) + pi*(r**2)
    W = 1000

    Args:
        model (Callable[..., Tensor]): Network
        dat_i (Tensor): Interior point
        dat_b (Tensor): Boundary point
        previous (Tuple[Tensor, Tensor]): Result from previous time step model. interior point for index 0, boundary for index 1

    Returns:
        Tuple[Tensor,Tensor]: loss
    """
    r = 0.25
    A = (-1 + pi*(r**2)) + pi*(r**2)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]

    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True) + 25*torch.pow(torch.pow(output_i, 2) - 1, 2)) 
    loss_b = torch.mean(torch.pow((output_b + 1), 2))
    loss_w = 1000*torch.pow((torch.mean(output_i) - A), 2)
    loss_p = 50*torch.mean(torch.pow(output_i - previous[0], 2))
    loss_p += 50*torch.mean(torch.pow(output_b - previous[1], 2))

    return loss_i + 100*loss_b + loss_w + loss_p, loss_i + loss_w


def HeatPINN(model: Callable[..., Tensor], 
              dat_i: Tensor, 
              dat_b: Tensor, 
              dat_f: Tensor) -> Tensor:
    """The loss function for heat equation with PINN

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Initial points
        dat_b (Tensor): Boundary points
        dat_f (Tensor): Collocation points

    Returns:
        Tensor: loss
    """
    output_i = model(dat_i)
    output_b = model(dat_b)
    dat_f.requires_grad = True
    f = (2*torch.sin(dat_f[:,0])*torch.cos(dat_f[:,1])).unsqueeze_(1)
    output_f = model(dat_f)
    du = torch.autograd.grad(outputs = output_f, inputs = dat_f, grad_outputs = torch.ones_like(output_f), retain_graph=True, create_graph=True)[0]
    ut = du[:,2].unsqueeze_(1)
    ddu = torch.autograd.grad(outputs = du, inputs = dat_f, grad_outputs = torch.ones_like(du), create_graph=True)[0]
    lu = ddu[:,0:2]
    
    loss = torch.mean(torch.pow(output_i, 2))
    loss += 100*torch.mean(torch.pow(output_b, 2))
    loss += torch.mean(torch.pow(ut - torch.sum(lu, dim=1, keepdim=True) - f, 2))
    
    return loss


# def PoissPINN(model: Callable[..., Tensor], 
#               dat_i: Tensor, 
#               dat_b: Tensor) -> Tensor:
#     """The loss function for poisson equation with PINN

#     Args:
#         model (Callable[..., Tensor]): Network 
#         dat_i (Tensor): Interior points
#         dat_b (Tensor): Boundary points

#     Returns:
#         Tensor: loss
#     """
    
#     bd = lambda x, y : (torch.sin(x)*torch.cos(y)).unsqueeze_(1) # for the exact solution at boundary
    
#     output_b = model(dat_b)
#     f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
#     dat_i.requires_grad = True
#     output_i = model(dat_i)
#     du = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
#     ddu = torch.autograd.grad(outputs = du, inputs = dat_i, grad_outputs = torch.ones_like(du), retain_graph=True, create_graph=True)[0]
#     loss = torch.mean(torch.pow(output_b - bd(dat_b[:,0], dat_b[:,1]), 2))
#     loss += torch.mean(torch.pow(torch.sum(ddu, dim=1, keepdim=True) + f, 2))
    
#     return loss

def PoissPINN(model: Callable[..., Tensor], 
              dat_i: Tensor, 
              dat_b: Tensor) -> Tensor:
    """The loss function for poisson equation with PINN

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior points
        dat_b (Tensor): Boundary points

    Returns:
        Tensor: loss
    """
    
    bd = lambda x, y : (torch.sin(x)*torch.cos(y)).unsqueeze_(1) # for the exact solution at boundary
    
    output_b = model(dat_b)
    f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    du = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    # laplacian operator
    s, d = dat_i.shape
    lap = torch.zero(s)
    for i in range(d):
        ddu = torch.autograd.grad(outputs = du[:,i], inputs = dat_i, grad_outputs = torch.ones_like(du[:,i]), retain_graph=True, create_graph=True)[0]
        lap += ddu[:, i]
    lap.unsqueeze_(1)
    loss = torch.mean(torch.pow(output_b - bd(dat_b[:,0], dat_b[:,1]), 2))
    loss += torch.mean(torch.pow(lap + f, 2))
    
    return loss


# def PoissCyclePINN(model: Callable[..., Tensor], 
#                    dat_i: Tensor, 
#                    dat_b: Tensor) -> Tensor:
#     """The loss function for poisson equation with PINN (cycle)
    
#     -\nabla u = 1,   u = 1 - 0.25(x^2 + y^2)

#     Args:
#         model (Callable[..., Tensor]): Network 
#         dat_i (Tensor): Interior points
#         dat_b (Tensor): Boundary points

#     Returns:
#         Tensor: loss
#     """
    
#     bd = lambda x, y : (1 - 0.25*(x**2 + y**2)).unsqueeze_(1) # for the exact solution at boundary
    
#     output_b = model(dat_b)
#     dat_i.requires_grad = True
#     output_i = model(dat_i)
#     du = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
#     ddu = torch.autograd.grad(outputs = du, inputs = dat_i, grad_outputs = torch.ones_like(du), retain_graph=True, create_graph=True)[0]
#     loss = torch.mean(torch.pow(output_b - bd(dat_b[:,0], dat_b[:,1]), 2))
#     loss += torch.mean(torch.pow(torch.sum(ddu, dim=1, keepdim=True) + 1, 2))
    
#     return loss


def PoissCyclePINN(model: Callable[..., Tensor], 
                   dat_i: Tensor, 
                   dat_b: Tensor) -> Tensor:
    """The loss function for poisson equation with PINN (cycle)
    
    -\nabla u = 1,   u = 1 - 0.25(x^2 + y^2)

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior points
        dat_b (Tensor): Boundary points

    Returns:
        Tensor: loss
    """
    
    bd = lambda x, y : (1 - 0.25*(x**2 + y**2)).unsqueeze_(1) # for the exact solution at boundary
    
    output_b = model(dat_b)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    du = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    # laplacian operator
    s, d = dat_i.shape
    lap = torch.zero(s)
    for i in range(d):
        ddu = torch.autograd.grad(outputs = du[:,i], inputs = dat_i, grad_outputs = torch.ones_like(du[:,i]), retain_graph=True, create_graph=True)[0]
        lap += ddu[:, i]
    lap.unsqueeze_(1)
    loss = torch.mean(torch.pow(output_b - bd(dat_b[:,0], dat_b[:,1]), 2))
    loss += torch.mean(torch.pow(lap + 1, 2))
    
    return loss

    
def PoiCycleLoss(model: Callable[..., Tensor], 
            dat_i: Tensor, 
            dat_b: Tensor,
            previous: List[Tensor]) -> Tuple[Tensor,Tensor]:
    """
    Loss function for 2d Poisson equation
    -\laplacia u = 1,    u \in \Omega
    u = 0,                           u \in \partial \Omega x^2 + y^2 = 1

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior point
        dat_b (Tensor): Boundary point
        previous (Tuple[Tensor, Tensor]): Result from previous time step model. interior point for index 0, boundary for index 1

    Returns:
        Tuple[Tensor,Tensor]: loss
    """
    bd = lambda x, y : (1 - 0.25*(x**2 + y**2)).unsqueeze_(1) # for the exact solution at boundary

    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True) - output_i)
    loss_b = torch.mean(torch.pow(output_b - bd(dat_b[:,0], dat_b[:,1]),2))
    
    if previous:
        loss_p = 50*torch.mean(torch.pow(output_i - previous[0], 2))
        loss_p += 50*torch.mean(torch.pow(output_b - previous[1], 2))
    else:
        loss_p = 0

    return loss_i + 500*loss_b + loss_p, loss_i
    

def Heat(model: Callable[...,Tensor], 
         dat_i: Tensor, 
         dat_b: Tensor, 
         previous: Tuple[Tensor,Tensor]) -> Tuple[Tensor,Tensor]:
    """
    2d Heat equation: u_t = \nabla u, x \in [0,2]^2
    u(x, t) = 0, x \in \delta\Omega
    u(x, t) = 50, if x_2 <= 1. u(x, t) = 0, if x_2 > 1.

    Args:
        model (Callable[...,Tensor]): [description]
        dat_i (Tensor): [description]
        dat_b (Tensor): [description]
        previous (Tuple[Tensor,Tensor]): [description]

    Returns:
        Tuple[Tensor,Tensor]: [description]
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    output_b = model(dat_b)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)) * 4
    loss_b = torch.mean(torch.pow(output_b,2))
    
    loss_p = 50*torch.mean(torch.pow(output_i - previous[0], 2)) * 4
    loss_p += 50*torch.mean(torch.pow(output_b - previous[1], 2)) * 4

    return loss_i + 500*loss_b + loss_p, loss_i

def FokkerPlanck(phi: Tensor,
                 coor: Tensor,
                 rho: Tensor,
                 lgdet: Tensor,
                 V:Tensor
                 ) -> Tensor:
    """Loss function for Fokker Planck equation

    Args:
        phi (Tensor): The convex potential of the normalizing flow [N, d]
        coor (Tensor): Input data [N, d]
        rho (Tensor): Estimated function [N, 1] 
        lgdet (Tensor): Log determinant [1, N]
        V (Tensor): density [N, 1]

    Returns:
        loss
    """
    Cap_Phi = 50*torch.mean(rho*torch.sum((phi - coor)**2, dim = 1, keepdim = True)) # captical phi
    vol = torch.mean(rho * (torch.log(rho) - lgdet.unsqueeze_(-1) + V)) # second part
    
    return Cap_Phi + vol, vol
