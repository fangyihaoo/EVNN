import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import torch
import torch.nn as nn
from Model import PoissonConfig
from Model import ResidualNet
from utils import poisson, poissoncycle, weight_init
from utils import PoiCycleLoss, PoiLoss
from utils import logger_init
from torch import Tensor
from typing import Callable, Tuple



def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    # load the exact solution if exist
    config = PoissonConfig.DefaultConfig()
    config._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exactpath = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Poisson', config.exact)
    gridpath = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Poisson', config.grid)
    grid = torch.load(gridpath, map_location = device)
    exact = torch.load(exactpath, map_location = device)
    # exact = exact.double()
    # grid = grid.double()
    # -------------------------------------------------------------------------------------------------------------------------------------

    DATASET_MAP = {'poi': poisson,
                   'poissoncycle': poissoncycle}
    LOSS_MAP = {'poi':PoiLoss,
                'poissoncycle': PoiCycleLoss}
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
    gendat = DATASET_MAP[config.type]
    losfunc = LOSS_MAP[config.type]
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    # model initialization
    model = ResidualNet.ResNet(**keys)
    model.to(device)
    model.apply(weight_init)
    model.to(torch.float)
    modelold = ResidualNet.ResNet(**keys)
    modelold.to(device)
    modelold.to(torch.float)
    previous = [0, 0]
    modelold.load_state_dict(model.state_dict())
     # model recorder
    energy = []
    error = []
    log_name = 'EVNN_' + config.type
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log')
    logger = logger_init(log_file_name = log_name, log_dir = path)
    # -------------------------------------------------------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(config.max_epoch):
        # ---------------training setup in each time step---------------

        datI = gendat(num = 8000, boundary = False, device = device)
        datB = gendat(num = 8000, boundary = True, device = device)  
        datI = datI.float()
        datB = datB.float()
        
        with torch.no_grad():
            previous[0] = modelold(datI)
            previous[1] = modelold(datB)

        optimizer = torch.optim.LBFGS(model.parameters(),
                                lr = config.lr,
                                history_size=100,
                                max_iter=100,
                                line_search_fn="strong_wolfe")
            
        def closure():
            optimizer.zero_grad()
            loss = losfunc(model, datI, datB, previous)
            loss[0].backward()
            return loss[0]
        
        optimizer.step(closure)

        modelold.load_state_dict(model.state_dict())
        err = eval(model, grid, exact)
        ENERGY = CompEnergy(model, grid, config.type)
        energy.append(ENERGY.item())
        error.append(err.item())
        logger.info(f'The epoch is {epoch}, The energy is {ENERGY.item()},  The error is {err.item()}')
    
    torch.save(error, osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Poisson', config.type + 'EVNN_L2.pt'))
    torch.save(energy, osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Poisson', config.type + 'EVNN_ENERGY.pt'))
    

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
            dat_i: Tensor,
            type: str) -> Tuple[Tensor,Tensor]:
    """
    Loss function for 2d Poisson equation
    -\laplacia u = 2sin(x)cos(y),    u \in \Omega
    u = 0,                           u \in \partial \Omega (0, pi) \times (-pi/2, pi/2)

    Args:
        model (Callable[..., Tensor]): Network 
        dat_i (Tensor): Interior point
        type (str): f = 1 or f = 2sinxcosx

    Returns:
        Tuple[Tensor,Tensor]: loss
    """

    if type ==  'poi':
        f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    dat_i.requires_grad = True
    output_i = model(dat_i)
    ux = torch.autograd.grad(outputs = output_i, inputs = dat_i, grad_outputs = torch.ones_like(output_i), retain_graph=True, create_graph=True)[0]
    # f = (2*torch.sin(dat_i[:,0])*torch.cos(dat_i[:,1])).unsqueeze_(1)
    
    if type == 'poissoncycle':
        loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)- output_i)
    else:
        loss_i =  torch.mean(0.5 * torch.sum(torch.pow(ux, 2),dim=1,keepdim=True)- f*output_i)
    
    return loss_i



if __name__=='__main__':
    import fire
    fire.Fire()
    # path = osp.dirname(osp.dirname(osp.realpath(__file__)))
    # print(path)