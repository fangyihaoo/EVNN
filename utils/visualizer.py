import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from Model import HeatConfig
from Model import ResidualNet


from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D 



def toy():
    
    Evnn = torch.load('../data/Poisson/poiEVNN_L2.pt', map_location=torch.device('cpu'))
    Ritz = torch.load('../data/Poisson/poiDritz.pt', map_location=torch.device('cpu'))
    pinn = torch.load('../data/Poisson/poipinn.pt', map_location=torch.device('cpu'))

    cycEvnn = torch.load('../data/Poisson/poissoncycleEVNN_L2.pt', map_location=torch.device('cpu'))
    cycRitz = torch.load('../data/Poisson/poissoncycleDritz.pt', map_location=torch.device('cpu'))
    cycPinn = torch.load('../data/Poisson/poissoncyclepinn.pt', map_location=torch.device('cpu'))
    epoch = torch.arange(0, 50000)
    
    font_size = 25

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_title(r'$f = 1$', fontsize=font_size)

    lines = []
    ax.set_yscale('log')
    lines = ax.plot(epoch[99:50000:100], cycEvnn,  color= '#EE82EE' )
    lines += ax.plot(epoch[99:50000:100], cycRitz[99:50000:100], color='#5E5A80')
    lines += ax.plot(epoch[99:50000:100], cycPinn[99:50000:100], color='#69ECEB')
    ax.legend(lines[:3], ['EVNN', 'DeepRitz', 'PINN'], loc='upper right', frameon=False, prop={'size': 18})
    ax.set_xlabel('epoch',fontsize=font_size)
    ax.set_ylabel('Relative L2 Norm',fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.xlim(left=-1000)
    plt.savefig('../Plots/l2cycle.png',pad_inches = 0.05, bbox_inches='tight')
    
    energyLBFGS = torch.load('../data/Poisson/poiEVNN_ENERGY.pt', map_location=torch.device('cpu'))
    interval = torch.arange(1, 501)
    interval = [ele * 0.01 for ele in interval]
    color = '#AF7AC5'
    fig, ax = plt.subplots(figsize=(15, 5))
    categories = 'Free Energy'
    ax.plot(interval[1:], energyLBFGS[1:], alpha=0.70, color=color,  label=categories, linewidth=4)
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('Free Energy', fontsize=font_size)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_title(r'$f = 2\sin{x}\cdot\cos{y}$', fontsize=font_size)
    plt.xlim(left=-0.05)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.savefig('../Plots/poi_energy.png', pad_inches = 0.1, bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_title(r'$f = 2\sin{x}\cdot\cos{y}$', fontsize=font_size)

    lines = []
    ax.set_yscale('log')
    lines = ax.plot(epoch[99:50000:100], Evnn,  color= '#EE82EE' )
    lines += ax.plot(epoch[99:50000:100], Ritz[99:50000:100], color='#5E5A80')
    lines += ax.plot(epoch[99:50000:100], pinn[99:50000:100], color='#69ECEB')
    ax.legend(lines[:3], ['EVNN', 'DeepRitz', 'PINN'], loc='upper right', frameon=False, prop={'size': 18})
    ax.set_xlabel('epoch',fontsize=font_size)
    ax.set_ylabel('Relative L2 Norm',fontsize=font_size)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.xlim(left=-1000)
    plt.savefig('../Plots/l2sincos.png',pad_inches = 0.05, bbox_inches='tight')

    energyLBFGS = torch.load('../data/Poisson/poissoncycleEVNN_ENERGY.pt', map_location=torch.device('cpu'))
    interval = torch.arange(1, 501)
    interval = [ele * 0.01 for ele in interval]
    color = '#AF7AC5'
    fig, ax = plt.subplots(figsize=(15, 5))
    categories = 'Free Energy'
    ax.plot(interval, energyLBFGS, alpha=0.70, color=color,  label=categories, linewidth=4)
    ax.set_xlabel('Time', fontsize=font_size)
    ax.set_ylabel('Free Energy', fontsize=font_size)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.set_title(r'$f = 1$', fontsize=font_size)
    plt.xlim(left=-0.05)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.savefig('../Plots/poicyc_energy.png', pad_inches = 0.1, bbox_inches='tight')
    

def heat():
    
    class OOMFormatter(ticker.ScalarFormatter):
        def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
            self.oom = order
            self.fformat = fformat
            ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
        def _set_order_of_magnitude(self):
            self.orderOfMagnitude = self.oom
        def _set_format(self, vmin=None, vmax=None):
            self.format = self.fformat
            if self._useMathText:
                self.format = r'$\mathdefault{%s}$' % self.format
    config = HeatConfig.DefaultConfig()
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
    gridpath = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Heat', config.grid)
    exactpath = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Heat', config.exact)
    energyLBFGS = torch.load(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Heat', 'heatenergyLBFGS.pt'))
    grid = torch.load(gridpath)
    exact = torch.load(exactpath)
    
    for i in [1, 20, 40, 60]:
        model_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'checkpoints', 'Heat', f'heat{i}.pt')
        model.load(model_path)
        model.to(torch.float)
        pred = model(grid)
        error = pred - exact[i]
        error = error.detach().numpy()
        error = error.reshape(101, 101)
        error = np.transpose(error)
        plt.figure(figsize=(6,6))
        ax = plt.subplot(1, 1, 1)
        h = plt.imshow(error, interpolation='nearest', cmap='RdBu',extent=[0, 2, 0, 2],origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(h, cax=cax, format=OOMFormatter(-3, mathText=False)).ax.yaxis.offsetText.set_fontsize(22)
        plt.tick_params(labelsize=22)
        ax.tick_params(axis='both', which='major', labelsize=22)
        plt.savefig(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'Plots', f'heat{i}error.png'), pad_inches = 0.1, bbox_inches='tight') 
        plt.close()
        
        pred = pred.detach().numpy()
        x = grid[:,0]
        y = grid[:,1]
        z = pred.flatten()
        plt.rcParams["figure.figsize"] = 6, 6
        plt.style.context('fivethirtyeight')
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_facecolor('white')
        ax.view_init(30, 140)
        surf = ax.plot_trisurf(x, y, z, linewidth=0, cmap='twilight_shifted',antialiased=False,edgecolor='none')
        ax.set_zlim(0, 1);
        ax.yaxis.set_ticks(np.arange(0,2.2,0.5))
        ax.xaxis.set_ticks(np.arange(0,2.2,0.5))
        ax.zaxis.set_ticks(np.arange(0,1.1,0.25))
        cax = fig.add_axes([0.18, .87, 0.7, 0.03])
        ax.tick_params(axis='both', which='major', labelsize=14)
        fig.colorbar(surf, orientation='horizontal', cax=cax)
        plt.xticks(fontsize= 14)
        plt.savefig(f'../Plots/heat3d{i}.png', pad_inches = 0.25, bbox_inches='tight')

    interval = torch.arange(1, 101)
    interval = [ele * 0.01 for ele in interval]
    color = '#AF7AC5'
    fig, ax = plt.subplots(figsize=(15, 5))
    categories = 'EVNN-LBFGS'
    ax.scatter(interval, energyLBFGS[0:100], alpha=0.70, color=color,  label=categories, marker='^')
    ax.set_xlabel('Time', fontsize=25)
    ax.set_ylabel('Free Energy', fontsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.xlim([0, 1.1])
    # ax.legend(categories, prop={"size":10})
    plt.savefig(osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'Plots', 'energy.png'), pad_inches = 0.1, bbox_inches='tight')



def AllenCahn():
    pass



if __name__ == "__main__":
    # toy()
    # heat()
    AllenCahn()