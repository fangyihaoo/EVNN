import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import torch
import torch.nn as nn
from Model import AllenCahnConfig
from Model import ResidualNet
from scipy.io import savemat
from torch import Tensor

def main(**kwargs):
    config = AllenCahnConfig.DefaultConfig()
    config._parse(kwargs)
    keys = {'FClayer':config.FClayer, 
            'num_blocks':config.num_blocks,
            'num_input':config.num_input,
            'num_output':config.num_oupt, 
            'num_node':config.num_node}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResidualNet.ResNet(**keys)
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'AllenCahn', 'PhaseFieldGrid.pt')
    grid = torch.load(path, map_location = device)
    grid = grid.float()
    for i in [1, 5, 3, 10, 30]:
        path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'checkpoints', 'AllenCahn', f'PhaseField{i}.pt')
        model.load_state_dict(torch.load(path,  map_location=torch.device('cpu')))
        sol = model(grid)
        sol = sol.detach().numpy()
        print(sol.shape)
        sol = {"a": sol, "label": "PhaseFieldLBFGS" + str(i)}
        loc = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'AllenCahn','')
        savemat(loc + 'PhSol' + str(i), sol)
        

if __name__ == "__main__":
    
    import fire
    fire.Fire()
    # python save_to_Matlab.py main