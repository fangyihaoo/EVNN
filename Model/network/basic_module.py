import torch
import torch.nn as nn
# import time
import os.path as osp


class BasicModule(nn.Module):
    """
    Encapsulation for 'save' and 'load' method
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self)) # default name

    def load(self, path, dev = torch.device('cpu')):
        """
        load the model at specific path
        """
        self.load_state_dict(torch.load(path, map_location=dev))

    def save(self, name: str = None):
        """
        save the model
        """
        path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), '..', 'checkpoints', '')     
        if name is None:
            path += self.model_name + '.pt'
            # name = time.strftime(prefix + '%m%d_%H:%M:%S.pt')
        else:
             path += name
        torch.save(self.state_dict(), path)
        
        
if __name__ == "__main__":
    path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), '..', 'checkpoints', '')  
    print(path)