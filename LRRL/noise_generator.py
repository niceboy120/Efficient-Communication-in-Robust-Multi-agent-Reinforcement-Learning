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

        # self.shape = obs_space.shape()
        
        # self.low = 
        # self.high = 


        # if bound is None:
        #     if self.obs_space.is_bounded():
        #         self.bound = self.high.min()
        #         self.minbound = self.low.max()
        #     else:
        #         self.bound = 10 # random high bound for noise
        #         self.minbound = -10  # random high bound for noise
        # else:
        #     self.bound = min(self.high.min(), bound)
        #     self.minbound = -self.bound
        # self.k = 0.5*(self.bound-self.minbound)

    def nu(self, x, mode=None):
        if mode==None:
            mode = self.mode

        if mode == 0:
            # # Uniform unbounded noise
            # if self.image:
            #     noise = np.zeros_like(x)
            #     for i, xi in enumerate(x):
            #         if np.random.uniform() > self.p_uniform:
            #             noise[i] = np.random.uniform(self.low, self.high, self.shape).astype(xi.dtype)
            #         else:
            #             noise[i] = xi
            # else:
            #     noise = np.zeros_like(x)
            #     for i,xi in enumerate(x):
            #         if np.random.uniform() > self.p_uniform:
            #             noise[i] = np.random.uniform(self.minbound, self.bound, self.shape)
            #         else: noise[i] = xi
            raise NotImplementedError

        elif mode == 1:
            # Uniform bounded noise
            # noise = T.rand_like(x)-0.5
            # return x.detach() + 0.3*self.bound*noise.detach()
            # for i,xi in enumerate(x):
            #     noise[i] = np.clip(np.add(np.random.uniform(self.bound, self.minbound, self.shape),np.asarray(xi)),
            #                        self.low,self.high)if np.random.uniform() > self.p_uniform else xi
            # return torch.tensor(noise)
            noise = T.rand_like(x)-0.5
            return T.tensor(x+0.4*self.bound*noise)

        elif mode == 2:
            # # Gaussian noise
            # x = x.detach().cpu().numpy()
            # noise = np.zeros_like(x)
            # for i,xi in enumerate(x):
            #     noise[i] = np.clip(np.random.normal(0, self.var), -self.bound, self.bound)
            # return T.tensor(x+noise).detach()
            x = x.cpu().detach().numpy()
            noise = np.zeros_like(x)
            for i,xi in enumerate(x):
                noise[i] = np.clip(np.add(np.random.normal(0, self.var), np.asarray(xi)),-self.bound, self.bound).astype(xi.dtype)
            return T.tensor(noise)
