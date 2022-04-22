import warnings

class DefaultConfig(object):
    """
    Model setting for Poisson equation
    """
    def __init__(self) -> None:


        self.FClayer = 2 

        self.num_blocks = 1

        self.num_input = 2

        self.num_oupt = 1

        self.num_node = 20

        self.act = 'tanh'  # tanh,  relu,  sigmoid,  leakyrelu

        self.max_epoch = 200 # number of epoch

        self.lr = 1e-2 # initial learning rate
        
        self.lr_decay= 1e-7
        
        self.step_size = 200
        
        self.dimension = 10

    
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