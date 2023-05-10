import torch as T    

class WIP():
    def __init__(self):
        self.n_objectives = 2
        self.n_labdas = self.n_objectives-1
        self.labda = [0.0 for _ in range(self.n_labdas)]
        self.j = [0.0 for _ in range(self.n_labdas)]

        self.beta = list(reversed(range(1, self.n_objectives+1))) 
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, self.n_objectives+1)))]
        self.vareps = 0.01
        

    def update_lagrange(self, losses):
        # Save relevant loss information for updating Lagrange parameters
        for i in range(self.n_labdas):
            self.j[i] = losses.mean()
        # Update Lagrange parameters
        for i in range(self.n_labdas):
            self.labda[i] += self.eta[i] * (self.j[i] - self.vareps*self.j[i] - losses[i][-1])
            self.labda[i] = max(self.labda[i], 0.0)

    def compute_weights(self):
        # Compute weights for different components of the first order objective terms
        first_order = []
        for i in range(self.n_objectives - 1):
            w = self.beta[i] + self.labda[i] * sum([self.beta[j] for j in range(i + 1, self.n_objectives)])
            first_order.append(w)
        first_order.append(self.beta[self.n_objectives - 1])
        first_order_weights = T.tensor(first_order)
        return first_order_weights

    def robust_loss(self,states,actions):
        self.config.state_normalizer.set_read_only()
        disturbed = self.noise.nu(states)
        target = self.network.actor(self.network.feature(self.config.state_normalizer(disturbed)))
        self.config.state_normalizer.unset_read_only()
        loss = self.coeff*self.config.kppo*0.5 * (actions.detach()-target).pow(2).mean()
        self.coeff = min([1,self.coeff*1.0001])
        return T.clip(loss,-self.loss_bound,self.loss_bound)