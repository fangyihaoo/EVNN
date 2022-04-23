import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import torch
import torch.nn as nn

from Model import HighDimConfig
from Model import ResidualNet
from utils import PoiHighGrid
from utils import PoiHighLoss
from utils import PoiHighExact
from utils import weight_init, count_parameters
from utils import logger_init
from typing import Callable, List
from torch import Tensor


def train(**kwargs):
    # -------------------------------------------------------------------------------------------------------------------------------------
    # load the setting
    config = HighDimConfig.DefaultConfig()
    config._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    gendat = PoiHighGrid
    losfunc = PoiHighLoss
    # -------------------------------------------------------------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------------------------------------------------------------
    # model initialization
    model = ResidualNet.ResNet(**keys)
    model.to(device)
    model.apply(weight_init)
    modelold = ResidualNet.ResNet(**keys)
    modelold.to(device)
    previous = 0
    modelold.load_state_dict(model.state_dict())
    
    log_name = 'EVNN_High_Dimension' 
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log')
    logger = logger_init(log_file_name = log_name, log_dir = path)
    
    prev_par = []
    for p in model.parameters():
        prev_par.append(torch.zeros_like(p))
    timestamp = [20*i  for i in range(1, 10)]
    # MinError = float('inf')
    # -------------------------------------------------------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------------------------------------------------------
    # train part
    for epoch in range(config.max_epoch):
        # ---------------training setup in each time step---------------
        step = 0
        optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
        
        # ---------------------------------------------------------------
        # --------------Optimization Loop at each time step--------------
        while True:
            optimizer.zero_grad()
            datI = gendat(num = config.num_sample, d = config.dimension, device = device)
            with torch.no_grad():
                previous = modelold(datI)
            loss = losfunc(model, datI, previous)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),  1)
            optimizer.step()
            step += 1
            dif, prev_par = WeightDiff(model, prev_par)      
            if dif <= 1e-6 or step == config.step_size:
                break
        err = eval(model, datI, PoiHighExact(datI))
        if epoch in timestamp:
            config.lr = config.lr * config.lr_decay
        logger.info(f'The epoch is {epoch}, The error is {err}', f'step is {step}')
        modelold.load_state_dict(model.state_dict())

    """
    l2 relative error
    """
    datI = gendat(num = 40000, d = config.dimension, device = device)
    logger.info(f'The final error is : {eval(model, datI, PoiHighExact(datI))}')
    logger.info(f'There are {count_parameters(modelold)} parameters')

    

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
def WeightDiff(model: Callable[..., Tensor],
           previous: List[Tensor]):
    """
    Compute the weight diff between current and previous time step
    """
    reg, cnt = 0, 0
    for p1, p2 in zip(model.parameters(), previous):    
        reg += torch.sum(torch.square(p1.detach() - p2))
        # cnt += p1.numel() # compute the number of parameters
    previous = []
    for p1 in model.parameters():
        previous.append(p1.detach().clone())
    return torch.sqrt(reg), previous


if __name__=='__main__':
    import fire
    fire.Fire()
    # path = osp.dirname(osp.dirname(osp.realpath(__file__)))
    # print(path)