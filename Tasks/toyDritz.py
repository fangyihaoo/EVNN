import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import torch
import torch.nn as nn
from Model import ResidualNet
from Model import PoissonDritzConfig
from utils import poisson, poissoncycle, weight_init
from utils import PoiCycleLoss, PoiLoss
from utils import logger_init
from torch.optim.lr_scheduler import StepLR
from torch import Tensor
from typing import Callable



def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    # load the exact solution if exist
    config = PoissonDritzConfig.DefaultConfig()
    config._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exactpath = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Poisson', config.exact)
    gridpath = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Poisson', config.grid)
    grid = torch.load(gridpath, map_location = device)
    exact = torch.load(exactpath, map_location = device)
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # model configuration, modified the DATASET_MAP and LOSS_MAP according to your need
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
    model = model.to(torch.float)
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    scheduler = StepLR(optimizer, step_size= config.step_size, gamma = config.lr_decay)
    error = []
    log_name = 'Dritz_' + config.type
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log')
    logger = logger_init(log_file_name = log_name, log_dir = path)
    # -------------------------------------------------------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(config.max_epoch):
        # ---------------training setup in each time step---------------
        # ---------------------------------------------------------------
        # --------------Optimization Loop at each time step--------------
        optimizer.zero_grad()
        datI = gendat(num = 1000, boundary = False, device = device)
        datB = gendat(num = 250, boundary = True, device = device)
        datI = datI.float()
        datB = datB.float()
        loss = losfunc(model, datI, datB, None) 
        loss[0].backward()
        optimizer.step()
        scheduler.step()
        err = eval(model, grid, exact)
        error.append(err)
        if epoch % 1000 == 0:
            logger.info(f'The epoch is {epoch}, The error is {err.item()}')
    error = torch.FloatTensor(error)
    torch.save(error, osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Poisson', config.type + 'Dritz.pt'))
    
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




if __name__=='__main__':
    import fire
    fire.Fire()