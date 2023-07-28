import torch as T
import numpy as np

# Noise Generator class. The noise generator in train mode can have different options based on what are the assumptions
# of the knowledge.




class NoiseGenerator:
    def __init__(self, variance=0.2, bound=None, mode=1):
        self.mode = mode
        self.var = variance

        if bound==None:
            self.bound = 0.5
        else:
            self.bound = bound


    def nu(self, x, mode=None):
        if mode==None:
            mode = self.mode

        if mode == 0:
            raise NotImplementedError

        elif mode == 1:
            noise = T.rand_like(x)-0.5
            return T.tensor(x+0.4*self.bound*noise)

        elif mode == 2:
            x = x.cpu().detach().numpy()
            noise = np.zeros_like(x)
            for i,xi in enumerate(x):
                noise[i] = np.clip(np.add(np.random.normal(0, self.var), np.asarray(xi)),-self.bound, self.bound).astype(xi.dtype)
            return T.tensor(noise)
