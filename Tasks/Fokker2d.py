import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
import torch
import math
from Model import NormalFlow, FokkerConfig
from utils import V, MulNormal, FokkerPlanck
from utils import logger_init

torch.set_default_dtype(torch.float64)


def train(**kwargs):
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    setting, same with "https://github.com/CW-Huang/CP-Flow"
    """
    config = FokkerConfig.DefaultConfig()
    config._parse(kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    icnns = [NormalFlow.ICNN3(config.num_input, config.num_node, config.FClayer, symm_act_first=config.symm_act_first, softplus_type=config.act, zero_softplus=config.zero_softplus) for _ in range(config.num_blocks)]
    layers = [None] * (config.num_blocks + 1)
    layers[0] = NormalFlow.ActNorm(config.num_input)
    layers[1:] = [NormalFlow.DeepConvexFlow(icnn, bias_w1=-0.0, trainable_w0=False) for _, icnn in zip(range(config.num_blocks), icnns)]

    mu = torch.tensor([1/3, 1/3]).to(device)
    sigma = torch.tensor([[5/8, -3/8], [-3/8, 5/8]]).to(device)
    
    x = torch.linspace(-3, 3, 101)
    y = torch.linspace(-3, 3, 101)
    X, Y = torch.meshgrid(x, y)
    coor = torch.cat((X.flatten()[:, None], Y.flatten()[:, None]), dim=1)
    mu0 = torch.tensor([0, 0])
    sigma0 = torch.tensor([[1 , 0], [0, 1]])
    rho = MulNormal(mu0, sigma0, coor)
    rho = rho.to(device)
    coor = coor.to(device)
    abserror = []
    relativeerror = []
    ENERGY = []
    log_name = 'EVNN_Fokker' 
    log_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'log')
    logger = logger_init(log_file_name = log_name, log_dir = log_path)

    init_path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'checkpoints', 'Fokker', 'IDENTICAL.pt')
    IDENTICAL = NormalFlow.SequentialFlow(layers)
    IDENTICAL.load_state_dict(torch.load(init_path))
    flow = NormalFlow.SequentialFlow(layers)
    IDENTICAL.to(device)
    flow.to(device)
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------





    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------

    """
    Training Part
    """
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'data', 'Fokker', '')
    
    for epoch in range(config.max_epoch):
        t = 0.01*(epoch + 1)
        mt = (1 - math.exp(-4*t))*mu
        mt = mt.to('cpu')
        sigt = torch.tensor([[5/8 + (3/8) * math.exp(-8*t), -3/8 + (3/8) * math.exp(-8*t)], [-3/8 + (3/8) * math.exp(-8*t), 5/8 + (3/8) * math.exp(-8*t)]])
        
        optimizer = torch.optim.LBFGS(flow.parameters(),
                                history_size = config.max_iter,
                                max_iter = config.max_iter,
                                line_search_fn= 'strong_wolfe')
        def closure():
            optimizer.zero_grad()
            phi, lgdet = flow.forward_transform(coor)
            loss = FokkerPlanck(phi, coor, rho, lgdet, V(phi, mu, sigma))[0]
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            phi, lgdet = flow.forward_transform(coor)
            # energy = FokkerPlanck(phi, coor, rho, lgdet, V(phi, mu, sigma))[1]
            # ENERGY.append(energy.item())
        
        coor = phi
        coor = coor.detach()
        lgdet = lgdet.unsqueeze_(-1).detach()
        rho = rho / (torch.exp(lgdet) + 1e-8)

        torch.save(coor, path + f'coor{epoch + 1}.pt')
        torch.save(rho, path + f'rho{epoch + 1}.pt')
        coor =  coor.cpu()
        act_val = MulNormal(mt, sigt, coor)
        act_val = act_val.to(device)
        err = torch.mean((rho - act_val)**2)
        relerr = err/torch.mean(act_val**2)
        
        abserror.append(err.item())
        relativeerror.append(relerr.item())
        logger.info(f'in epoch {epoch}, the L2 absolute error is {err.item()}, the L2 relative error is {relerr.item()}')
        flow.load_state_dict(IDENTICAL.state_dict())
        coor = coor.to(device)
    torch.save(abserror, path + 'l2AbsError.pt')
    torch.save(relativeerror, path + 'l2RelativeError.pt')
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------



if __name__=='__main__':
    import fire
    fire.Fire()