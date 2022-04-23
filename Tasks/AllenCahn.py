import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import torch
import torch.nn as nn
from Model import AllenCahnConfig
from Model import ResidualNet
from utils import AllenCahnW
from utils import weight_init
from utils import logger_init

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import pi
from torch import Tensor
from typing import Callable, List, Tuple



def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    config = AllenCahnConfig.DefaultConfig()
    config._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.linspace(-0.99, 0.99, 201)
    y = torch.linspace(-0.99, 0.99, 201)
    X, Y = torch.meshgrid(x, y)
    datI = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    datI = datI.to(device)
    
    tb = torch.cat((torch.linspace(-1, 1, steps=1000)[:,None], torch.tensor([1.]).repeat(1000)[:,None]), dim=1)
    bb = torch.cat((torch.linspace(-1, 1, steps=1000)[:,None], torch.tensor([-1.]).repeat(1000)[:,None]), dim=1)
    rb = torch.cat((torch.tensor([1.]).repeat(1000)[:,None], torch.linspace(-1, 1, steps=1000)[:,None]), dim=1)
    lb = torch.cat((torch.tensor([-1.]).repeat(1000)[:,None], torch.linspace(-1, 1, steps=1000)[:,None]), dim=1)
    data = torch.cat((tb, bb, rb, lb), dim=0)
    datB = data.to(device) 
    
    x = torch.linspace(-0.99, 0.99, 301)
    y = torch.linspace(-0.99, 0.99, 301)
    X, Y = torch.meshgrid(x, y)
    quadture = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    quadture = quadture.to(device)

    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    ACTIVATION_MAP = {'relu' : nn.ReLU(),
                    'tanh' : nn.Tanh(),
                    'sigmoid': nn.Sigmoid(),
                    'leakyrelu': nn.LeakyReLU()}
    keys = {'FClayer':config.FClayer, 
            'num_blocks':config.num_blocks,
            'activation':ACTIVATION_MAP[config.act],
            'num_input':config.num_input,
            'num_output':config.num_oupt, 
            'num_node':config.num_node}
    losfunc = AllenCahnW
    
    model = ResidualNet.ResNet(**keys)
    model.to(device)
    model.apply(weight_init)
    modelold = ResidualNet.ResNet(**keys)
    modelold.to(device)
    
    log_name = 'EVNN_Allen_Cahn' 
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log')
    logger = logger_init(log_file_name = log_name, log_dir = path)
    
    energy = []
    previous = []
    init_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'checkpoints', 'AllenCahn', config.pretrain)
    modelold.load_state_dict(torch.load(init_path))
    with torch.no_grad():
        previous.append(modelold(datI))
        previous.append(modelold(datB))
    # -------------------------------------------------------------------------------------------------------------------------------------





    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(config.max_epoch):
        # ---------------training setup in each time step---------------
        # ---------------------------------------------------------------
        # --------------Optimization Loop at each time step--------------
        with torch.no_grad():
            previous[0] = modelold(datI)
            previous[1] = modelold(datB)
        def closure():
            optimizer.zero_grad()
            loss = losfunc(model, datI, datB, previous)
            loss[0].backward()
            return loss[0]
        optimizer = torch.optim.LBFGS(model.parameters(),
                                history_size=config.max_iter,
                                max_iter=config.max_iter,
                                line_search_fn="strong_wolfe")
        optimizer.step(closure)
        ENERGY = CompEnergy(model, quadture)
        energy.append(ENERGY.item())
        logger.info(f'In epoch {epoch + 1}, the energy is {4 * ENERGY.item()}')
        model.save(f'AllenCahn/PhaseField{epoch+1}.pt')
        modelold.load_state_dict(model.state_dict())
        
    log_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'AllenCahn',  'energyLBFGS.pt')
    torch.save(energy, log_path)

    # -------------------------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def eval(model: Callable[..., Tensor], 
        grid: Tensor, 
        exact: Tensor):
    """
    Compute the relative L2 norm
    """
    model.eval()
    pred = model(grid)
    err  = torch.pow(torch.mean(torch.pow(pred - exact, 2))/torch.mean(torch.pow(exact, 2)), 0.5)
    model.train()
    return err


def CompEnergy(model: Callable[..., Tensor], 
               dat_i: Tensor) -> Tuple[Tensor,Tensor]:
    """
    \int 0.5*|\nabla \phi|^2 + 0.25*(\phi^2 - 1)^2/epislon^2 dx + W*(\int\phidx - A)^2
    r = 0.25
    A = (4 - pi*(r**2))*(-1) + pi*(r**2)
    W = 1000

    Args:
        model (Callable[..., Tensor]): Network
        dat_i (Tensor): Interior point

    Returns:
        Tuple[Tensor,Tensor]: Energy
    """
    r = 0.25
    A = (-1 + pi*(r**2)) + pi*(r**2)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]

    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True) + 25*torch.pow(torch.pow(output_i, 2) - 1, 2)) 
    loss_w = 1000*torch.pow((torch.mean(output_i) - A), 2)
    
    return loss_i + loss_w


if __name__=='__main__':
    import fire
    fire.Fire()