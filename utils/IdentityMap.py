import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import torch
from Model import NormalFlow


torch.set_default_dtype(torch.float64)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
setting, same with "https://github.com/CW-Huang/CP-Flow"
"""
dimx = 2
nblocks = 1
depth = 6
k = 32
lr = 0.005
symm_act_first = True
zero_softplus = False
softplus_type = 'gaussian_softplus2'

icnns = [NormalFlow.ICNN3(dimx, k, depth, symm_act_first=symm_act_first, softplus_type=softplus_type, zero_softplus=zero_softplus) for _ in range(nblocks)]
layers = [None] * (nblocks + 1)
layers[0] = NormalFlow.ActNorm(dimx)
layers[1:] = [NormalFlow.DeepConvexFlow(icnn, bias_w1=-0.0, trainable_w0=False) for _, icnn in zip(range(nblocks), icnns)]
IDENTICAL = NormalFlow.SequentialFlow(layers)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


"""
Identical mapping training
"""
def Identity(phi, coor):
    return torch.mean(torch.sum((phi - coor)**2, dim = 1, keepdim = True))

def FokkerPretrain(model):
    x = torch.linspace(-3, 3, 101)
    y = torch.linspace(-3, 3, 101)
    X, Y = torch.meshgrid(x, y)
    coor = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)

    optimizer = torch.optim.LBFGS(model.parameters(),
                            history_size=100,
                            max_iter = 100,
                            line_search_fn= 'strong_wolfe')
    def closure():
        optimizer.zero_grad()
        phi, _ = model.forward_transform(coor)
        loss = Identity(phi, coor)
        loss.backward()
        return loss

    optimizer.step(closure)
    
def main():
    FokkerPretrain(IDENTICAL)
    # path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'checkpoints', 'Fokker', 'IDENTICAL.pt')
    # torch.save(IDENTICAL.state_dict(), path)

if __name__ == "__main__":
    main()