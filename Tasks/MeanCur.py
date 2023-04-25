import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import torch
import torch.nn as nn
from Model import MeanCurConfig
from Model import ResidualNet
from utils import MeanCurLoss
from utils import weight_init
from utils import logger_init

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import pi
from torch import Tensor
from typing import Callable, List, Tuple



def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    config = MeanCurConfig.DefaultConfig()
    config._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.linspace(-5, 5, 501)
    y = torch.linspace(-5, 5, 501)
    X, Y = torch.meshgrid(x, y)
    datI = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    datI = datI.to(device)
    

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
    
    losfunc = MeanCurLoss
    
    model = ResidualNet.ResNet(**keys)
    model.to(device)
    model.apply(weight_init)
    modelold = ResidualNet.ResNet(**keys)
    modelold.to(device)
    
    log_name = 'EVNN_circle_Meacur' 
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log')
    logger = logger_init(log_file_name = log_name, log_dir = path)
    
    energy = []
    previous = []
    init_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'checkpoints', 'MeanCur', config.pretrain)
    modelold.load_state_dict(torch.load(init_path))
    with torch.no_grad():
        previous.append(modelold(datI))
    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(int(config.max_epoch)):
        # ---------------training setup in each time step---------------
        # ---------------------------------------------------------------
        # --------------Optimization Loop at each time step--------------
        with torch.no_grad():
            previous[0] = modelold(datI)
            #previous[1] = modelold(datB)
        # when did LBFGS iteration stop for one time step
        def closure():
            optimizer.zero_grad()
            loss = losfunc(model=model,dat_i= datI,previous=previous)
            loss[0].backward()
            return loss[0]
        optimizer = torch.optim.LBFGS(model.parameters(),
                                history_size=config.max_iter,
                                max_iter=config.max_iter,
                                line_search_fn="strong_wolfe")
        optimizer.step(closure)
        #ENERGY, penaltyTerm = CompWillmoreEnergy(model, datI)
        ENERGY, penaltyTerm = CompMeacurEnergy(model, datI)
        energy.append(ENERGY.item())
        logger.info(f'epoch {epoch + 1}, energy {ENERGY.item()}, penalty: {penaltyTerm.item()}')
        #print(model.input.weight.grad)
        model.save(f'MeanCur/circle{epoch+1}.pt')
        #model.save(f'MeanCur/ellipseMeacur{epoch+1}.pt')
        modelold.load_state_dict(model.state_dict())
        
    log_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'MeanCur',  'circleEnergy.pt')
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


def CompWillmoreEnergy(model: Callable[..., Tensor], 
               dat_i: Tensor) -> Tuple[Tensor,Tensor]:
    """
    Energy: 0.5*\int |\kappa|^2

    Args:
        model (Callable[..., Tensor]): Network
        dat_i (Tensor): Interior point

    Returns:
        Tuple[Tensor,Tensor]: Energy
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    ux_normlized = ux/torch.norm(ux,dim=1,keepdim=True)
    s, d = dat_i.shape
    kappa = 0.
    for i in range(d):
        kappa += torch.autograd.grad(outputs = ux_normlized[:,i], inputs = dat_i, grad_outputs = torch.ones_like(ux_normlized[:,i]), retain_graph=True, create_graph=True)[0][...,i:i+1]
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(kappa, 2),dim=1,keepdim=True)) * 100
    
    #penalty term
    lam = 0
    penalty = lam*torch.mean(torch.pow(torch.norm(ux,dim=1,keepdim=True)-1. ,2))*100

    return loss_i, penalty

def CompMeacurEnergy(model: Callable[..., Tensor], 
               dat_i: Tensor) -> Tuple[Tensor,Tensor]:
    """
    Energy: 0.5*\int |\nabla u|^2

    Args:
        model (Callable[..., Tensor]): Network
        dat_i (Tensor): Interior point

    Returns:
        Tuple[Tensor,Tensor]: Energy
    """

    dat_i.requires_grad = True
    output_i = model(dat_i)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)) * 100
    
    #penalty term
    lam = 1
    penalty = lam*torch.mean(torch.pow(torch.norm(ux,dim=1,keepdim=True)-1. ,2))*100

    return loss_i, penalty

if __name__=='__main__':
    import fire
    fire.Fire()
