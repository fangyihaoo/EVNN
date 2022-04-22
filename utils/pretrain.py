import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
from numpy import pi
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from Model import HeatConfig
from Model import ResidualNet
from utils import heat
from utils import weight_init
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor


def HeatNew(Z: Tensor, boundary: str = False):
    N = Z.shape[0]
    if boundary:
        return torch.tensor([0]).repeat(N).reshape((-1,1))
    else:
        data = (torch.sin(0.5*pi*Z[:,0])*torch.sin(0.5*pi*Z[:,1])).unsqueeze_(1)
        return data

def pretrain_Heat(**kwargs):
    # heat model configuration
    config = HeatConfig.DefaultConfig()
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
    model = ResidualNet.ResNet(**keys)
    model.to(device)
    model.to(torch.float)
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.7)
    path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'data', 'Heat' ,"")
    grid = torch.load(path + 'heatgrid.pt', map_location = device)
    # exact = torch.load(path + 'heatexact.pt', map_location = device)
    
    # train the initialization model
    for _ in range(40000):
        optimizer.zero_grad()
        datI = heat(num = 5000, boundary = False, device = device)
        datB = heat(num = 1000, boundary = True, device = device)
        datI = datI.float()
        datB = datB.float()
        out_i = model(datI)
        out_b = model(datB)
        real_i = HeatNew(datI, boundary = False)
        real_b = HeatNew(datB, boundary = True)
        loss = torch.mean((out_i - real_i)**2)
        loss += 50*torch.mean((out_b - real_b)**2)
        if _ % 500 == 0:
            print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # check the model and save it
    
    plt.figure()
    pred = model(grid)
    pred = pred.detach().numpy()
    pred = pred.reshape(101, 101)
    pred = np.transpose(pred)
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',extent=[0, 2, 0, 2],origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'Plots', "")
    plt.savefig(path + 'heatInitiliationLBFGS.png')
    model.save('Heat/heatInitiliationLBFGS.pt')
    
    













if __name__ == "__main__":
    import fire
    fire.Fire()