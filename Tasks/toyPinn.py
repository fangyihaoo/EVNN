import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import models
from data import poisson, poissoncycle
from utils import Optim
from utils import PoissPINN, PoissCyclePINN
from utils import weight_init
import os.path as osp
from typing import Callable
from torch import Tensor
from config import opt
