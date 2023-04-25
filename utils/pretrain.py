import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
from numpy import pi
import skfmm
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from Model import HeatConfig, AllenCahnConfig, MeanCurConfig,Willmore3dConfig
from Model import ResidualNet
from utils import heat, allencahn, meancur
from utils import weight_init
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

def ellipsoid_sdf(x,y,z):
    u = ellipsoid(x,y,z).detach().numpy()
    d = skfmm.distance(u,dx=4./200)
    d = torch.from_numpy(d)
    return d.reshape((-1,1))

def ellipsoid(x,y,z):
    u = x**2/1.+ y**2/0.5 + z**2/0.5 -1
    return u

def circle(x,y):
    u = x**2/4 + y**2/4 -1
    return u

def dumbbell(x,y):
  a = 3
  b = 3.1
  u = (x**2+y**2+a**2)**2-4*a**2*x**2 -b**4
  return u

def heart(x,y):
    u = (x**2+y**2-1)**3 - x**2*y**3
    return u

def ellipse(x,y):
    # I.C. for mean curvature equ
    u = x**2/4 + y**2 -1
    return u

def twoEllipse(x,y):
  u1 = (x-1)**2/2+(y+1)**2/1 - 1
  u2 = (x+1)**2/2+(y-1)**2/1 - 1
  return torch.min(u1,u2)

def sdf_square(X,Y):
    points = torch.stack([X.flatten(), Y.flatten()], axis=1)

    # Define the vertices of the square
    vertices = torch.tensor([[-3, 3], [-3, -3], [3, -3], [3, 3]], dtype=torch.float)

    # Compute the signed distance function to the edges of the square
    distances = torch.zeros_like(X)
    for i in range(points.shape[0]):
        p = points[i]
        if (-3 <= p[0] <= 3) and (-3 <= p[1] <= 3):
            distances[i // 501, i % 501] = -torch.min(torch.tensor([p[0] + 3, 3 - p[0], p[1] + 3, 3 - p[1]], dtype=torch.float))
        elif (p[0] > 3 and p[1] > 3):
            distances[i // 501, i % 501] = torch.norm(p - torch.tensor([3, 3], dtype=torch.float))
        elif (p[0] > 3 and p[1] < -3):
            distances[i // 501, i % 501] = torch.norm(p - torch.tensor([3, -3], dtype=torch.float))
        elif (p[0] < -3 and p[1] < -3):
            distances[i // 501, i % 501] = torch.norm(p - torch.tensor([-3, -3], dtype=torch.float))
        elif (p[0] < -3 and p[1] > 3):
            distances[i // 501, i % 501] = torch.norm(p - torch.tensor([-3, 3], dtype=torch.float))
        else:
            distances[i // 501, i % 501] = torch.min(torch.abs(torch.tensor([p[0] + 3, 3 - p[0], p[1] + 3, 3 - p[1]], dtype=torch.float)))

    return distances.reshape((-1,1))


def sdf(x,y,shape='ellipse'):
    # signed distance function, https://scikit-fmm.readthedocs.io/en/latest/
    if shape == 'ellipse':
        u = ellipse(x,y).detach().numpy()
    elif shape == 'heart':
        u = heart(x,y).detach().numpy()
    elif shape =='twoEllipse':
        u = twoEllipse(x,y).detach().numpy()
    elif shape == 'dumbbell':
        u = dumbbell(x,y).detach().numpy()
    elif shape == 'circle':
        u = circle(x,y).detach().numpy()
    else:
        raise Exception("This shape hasn't implemented yet")
    d = skfmm.distance(u,dx=10./500)
    d = torch.from_numpy(d)

    return d.reshape((-1,1))


def elliptical(x: Tensor, y: Tensor):
    # change this part according to your specific task
    m = nn.Tanh()
    return (-m(10*(torch.sqrt(x**2 + 4*y**2) - 0.5))).reshape((-1, 1))

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
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.lr_decay)
    path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'data', 'Heat' ,"")
    grid = torch.load(path + 'heatgrid.pt', map_location = device)
    # exact = torch.load(path + 'heatexact.pt', map_location = device)
    
    # train the initialization model
    for _ in range(40000):
        optimizer.zero_grad()
        datI = heat(num = 5000, boundary = False, device = device)
        datB = heat(num = 1000, boundary = True, device = device)
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
    
    
def pretrain_Allen(**kwargs):
    
    config = AllenCahnConfig.DefaultConfig()
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
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.lr_decay)
    
    for _ in range(config.max_epoch+1):
        optimizer.zero_grad()
        datI = allencahn(num = 2500, boundary = False, device = device)
        datB = allencahn(num = 500, boundary = True, device = device)
        out_i = model(datI)
        out_b = model(datB)
        real_i = elliptical(datI[:,0], datI[:,1])
        real_b = elliptical(datB[:,0], datB[:,1])
        loss = torch.mean((out_i - real_i)**2)
        loss += 50*torch.mean((out_b - real_b)**2)
        if _ % 500 == 0:
            print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f'The model contains {sum(p.numel() for p in model.parameters())} parameters')
    x = torch.linspace(-1, 1, 101)
    y = torch.linspace(-1, 1, 101)
    X, Y = torch.meshgrid(x, y)
    grid = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    plt.figure()
    pred = model(grid)
    pred = pred.detach().numpy()
    pred = pred.reshape(101, 101)
    # pred = np.transpose(pred)
    ax = plt.subplot(1, 1, 1)
    h = plt.imshow(pred, interpolation='nearest', cmap='rainbow',extent=[-1, 1, -1, 1],origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(h, cax=cax)
    path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'Plots', "")
    plt.savefig(path + 'PhaseFieldInitilization.png')
    model.save('AllenCahn/PhaseFieldInitilizationLFBGS.pt')



def pretrain_Mean(**kwargs):
    
    config = MeanCurConfig.DefaultConfig()
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
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.lr_decay)
    
    x = torch.linspace(-5,5,501)
    y = torch.linspace(-5,5,501)
    X,Y = torch.meshgrid(x,y)
    grid = torch.cat((X.flatten()[:,None],Y.flatten()[:,None]),dim=1)
    real_i = sdf(X,Y,'circle')
    #real_i = sdf(X,Y,'twoEllipse')
    #real_i = sdf(X,Y,'dumbbell')
    #real_i = sdf_square(X,Y)

 
    for _ in range(config.max_epoch+1):
        optimizer.zero_grad()
        out_i = model(grid)
        loss = torch.mean((out_i - real_i)**2)
        if _ % 50 == 0:
            print(loss)
            #print(real_i[:10,])
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f'The model contains {sum(p.numel() for p in model.parameters())} parameters')

    plt.figure()
    pred = model(grid)
    pred = pred.detach().numpy()
    pred = pred.reshape(501, 501)
    # pred = np.transpose(pred)
    ax = plt.subplot(1, 1, 1)
    plt.title('Distance from the boundary')
    #### Plot the true distance function
    # real_i = real_i.detach().numpy()
    # real_i = real_i.reshape(501,501)
    # plt.contour(X, Y, ellipse(X,Y),[0], linewidths=(3), colors='black')
    # plt.contour(X,Y,real_i,15)

    #### Plot the distance function from NN predition
    plt.contour(X, Y, pred, 15)
    plt.colorbar()
    path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'Plots', "")
    # plt.savefig(path + 'MeanCurDistance.png')
    plt.savefig(path + 'circle.png')
    model.save('MeanCur/circleInitilizationLFBGS.pt')


def pretrain_Mean3d(**kwargs):
    
    config = MeanCur3dConfig.DefaultConfig()
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
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.lr_decay)
    
    x = torch.linspace(-2,2,201)
    y = torch.linspace(-2,2,201)
    z = torch.linspace(-2,2,201)
    X,Y,Z = torch.meshgrid(x,y,z)
    grid = torch.cat((X.flatten()[:,None],Y.flatten()[:,None],Z.flatten()[:,None]),dim=1)
    grid = grid.to(device)
    real_i = ellipsoid_sdf(X,Y,Z)
    real_i = real_i.to(device)

 
    for _ in range(config.max_epoch+1):
        optimizer.zero_grad()
        out_i = model(grid)
        loss = torch.mean((out_i - real_i)**2)
        if _ % 50 == 0:
            print(loss)
            #print(real_i[:10,])
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f'The model contains {sum(p.numel() for p in model.parameters())} parameters')

    # plt.figure()
    # pred = model(grid)
    # pred = pred.detach().numpy()
    # pred = pred.reshape(201, 201,201)
    # # pred = np.transpose(pred)
    # ax = plt.subplot(1, 1, 1)
    # plt.title('Distance from the boundary')
    # #### Plot the true distance function
    # # real_i = real_i.detach().numpy()
    # # real_i = real_i.reshape(501,501)
    # # plt.contour(X, Y, ellipse(X,Y),[0], linewidths=(3), colors='black')
    # # plt.contour(X,Y,real_i,15)

    # #### Plot the distance function from NN predition
    # plt.contour(X, Y, pred, 15)
    # plt.colorbar()
    # path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'Plots', "")
    # # plt.savefig(path + 'MeanCurDistance.png')
    # plt.savefig(path + 'MeanCurInitilization.png')
    model.save('MeanCur3d/MeanCu3dInitilizationLFBGS.pt')



if __name__ == "__main__":
    import fire
    fire.Fire()
    
    # python pretrain.py pretrain_Allen --max_epoch=40000 --lr=1e-2 --lr_decay=0.7 --step_size=5000
    
    # python pretrain.py pretrain_Heat --max_epoch=4000 --lr_decay=0.7 --step_size=5000 --lr=1e-2

    # python3 utils/pretrain.py pretrain_Mean --max_epoch=4000 --lr_decay=0.7 --step_size=500 --lr=1e-2

    # python3 utils/pretrain.py pretrain_Mean3d --max_epoch=40000 --lr_decay=0.7 --step_size=5000 --lr=1e-2
