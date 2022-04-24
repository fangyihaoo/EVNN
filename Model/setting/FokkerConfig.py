import warnings

class DefaultConfig(object):
    """
    Model setting for Allen-Cahn equation
    """
    def __init__(self) -> None:

        self.FClayer = 6 

        self.num_blocks = 1

        self.num_input = 2

        self.num_node = 32

        self.act = 'gaussian_softplus2'  # tanh,  relu,  sigmoid,  leakyrelu

        self.max_epoch = 100 # number of epoch

        # self.lr = 1e-2 # initial learning rate

        self.max_iter = 20
        
        self.symm_act_first = True
        
        self.zero_softplus = False
    
    def _parse(self, kwargs):
        '''
        update parameters according to user preference
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                if '_parse' in k:
                    continue
                else:
                    print(k,getattr(self,k))


if __name__ == "__main__":
    opt = DefaultConfig()
    print(opt.dimension)