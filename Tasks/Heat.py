import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import torch
import torch.nn as nn
from Model import HeatConfig
from Model import ResidualNet
from utils import Heat
from utils import weight_init
from utils import logger_init
from torch import Tensor
from typing import Callable, Tuple

def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    # load the exact solution if exist
    config = HeatConfig.DefaultConfig()
    config._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gridpath = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Heat', config.grid)
    exactpath = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Heat', config.exact)
    grid = torch.load(gridpath, map_location = device)
    exact = torch.load(exactpath, map_location = device)
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    # Training data
    x = torch.linspace(0.001, 1.999, 301)
    y = torch.linspace(0.001, 1.999, 301)
    X, Y = torch.meshgrid(x, y)
    datI = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    datI = datI.to(device)
    tb = torch.cat((torch.linspace(0, 2, steps=1000)[:,None], torch.tensor([2.]).repeat(1000)[:,None]), dim=1)
    bb = torch.cat((torch.linspace(0, 2, steps=1000)[:,None], torch.tensor([0.]).repeat(1000)[:,None]), dim=1)
    rb = torch.cat((torch.tensor([2.]).repeat(1000)[:,None], torch.linspace(0, 2, steps=1000)[:,None]), dim=1)
    lb = torch.cat((torch.tensor([0.]).repeat(1000)[:,None], torch.linspace(0, 2, steps=1000)[:,None]), dim=1)
    data = torch.cat((tb, bb, rb, lb), dim=0)
    datB = data.to(device) 
    # -------------------------------------------------------------------------------------------------------------------------------------
    # model configuration
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
    losfunc = Heat
    
    log_name = 'EVNN_Heat' 
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log')
    logger = logger_init(log_file_name = log_name, log_dir = path)
    
    # model initialization
    model = ResidualNet.ResNet(**keys)
    model.to(device)
    model.apply(weight_init)
    modelold = ResidualNet.ResNet(**keys)
    modelold.to(device)
    energy_list = []
    timestamp = list(range(1,31)) + list(range(40, 101, 10))
    previous = []
    init_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'checkpoints', 'Heat', config.pretrain)
    modelold.load_state_dict(torch.load(init_path))
    with torch.no_grad():
        previous.append(modelold(datI))
        previous.append(modelold(datB))
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part

    for epoch in range(config.max_epoch):

        with torch.no_grad():
            previous[0] = modelold(datI)
            previous[1] = modelold(datB)

        optimizer = torch.optim.LBFGS(model.parameters(),
                                lr = config.lr,
                                history_size=config.max_iter,
                                max_iter=config.max_iter,
                                line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = losfunc(model, datI, datB, previous)
            loss[0].backward()
            return loss[0]
        optimizer.step(closure)
        energy = CompEnergy(model, grid)
        energy_list.append(energy.item())
        if (epoch + 1) <= 100:
            logger.info(f'The error in epoch {epoch} is: {abserr(model, grid, exact[epoch+1])}')
        if (epoch + 1) in timestamp:         
            model.save(f'Heat/heat{epoch+1}.pt')
        modelold.load_state_dict(model.state_dict())
    log_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Heat', 'heatenergyLBFGS.pt')
    torch.save(energy_list, log_path)



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

@torch.no_grad()
def abserr(model: Callable[..., Tensor], 
        grid: Tensor, 
        exact: Tensor):
    """
    Compute the relative L2 norm
    """
    model.eval()
    pred = model(grid)
    err = torch.mean(abs(pred - exact))
    model.train()
    return err

def CompEnergy(model: Callable[...,Tensor], 
         dat_i: Tensor) -> Tuple[Tensor,Tensor]:
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
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)) * 4
    return loss_i

if __name__=='__main__':
    import fire
    fire.Fire()