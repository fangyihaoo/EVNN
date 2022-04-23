import torch
from torch import Tensor
from scipy.stats import multivariate_normal


def V(phi: Tensor,
      mu: Tensor,
      sigma: Tensor) -> Tensor:
    """
    Applied bilinear transformation on the each row of the input
    V(x) = 1/2 * (x - \mu)^T \Sigma^{-1} (x - \mu)

    Args:
        phi (Tensor [N, d]): normalizing flow output
        mu (Tensor [1, d]): mean vector
        sigma (Tensor [d, d]): covariance matrix
    
    Return:
        Tensor [N, 1]
    """
    x_cen = phi - mu
    return 0.5 * torch.sum(torch.mm(x_cen, torch.inverse(sigma))*x_cen, 1, keepdim=True)

def MulNormal(mu: Tensor,
              sigma: Tensor,
              pos: Tensor) -> Tensor:
    """Genrate the density for multivariate gaussian distribution

    Args:
        mu (Tensor): mean vector
        sigma (Tensor): sigma matrix
        pos (Tensor): position for density evaluation

    Returns:
        Density (Tensor): density at position
    """
    rv = multivariate_normal(mu, sigma)
    return torch.from_numpy(rv.pdf(pos)).unsqueeze_(1)