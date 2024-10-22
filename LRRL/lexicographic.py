import torch as T    
import collections
import numpy as np

class LexicographicWeights():
    def __init__(self, noise):
        self.n_objectives = 2
        self.n_labdas = self.n_objectives-1
        self.labda = [0.0 for _ in range(self.n_labdas)]
        self.j = [0.0 for _ in range(self.n_labdas)]

        self.beta = list(range(1, self.n_objectives+1))
        self.eta = [1e-3 * eta for eta in list(reversed(range(1, self.n_objectives+1)))]
        self.vareps = 0.5

        self.noise = noise

    
    def init_recent_losses(self):
        return [collections.deque(maxlen=25) for i in range(self.n_objectives)]
        

    def update_lagrange(self, recent_losses):
        # Save relevant loss information for updating Lagrange parameters
        for i in range(self.n_labdas):
            self.j[i] = T.tensor(recent_losses[i]).mean()
        # Update Lagrange parameters
        for i in range(self.n_labdas):
            self.labda[i] += self.eta[i] * (self.j[i] - self.vareps - recent_losses[i][-1])
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

    def robust_loss_actor(self, states, actions, agent, device, noise_mode=None):
        agent.actor.eval()
        disturbed = self.noise.nu(states, noise_mode)
        disturbed_actions = agent.actor.forward(disturbed.to(device))

        loss = 0.5 * (actions.detach()-disturbed_actions).pow(2).mean()

        return loss
    
    def robust_loss_critic(self, states, actions, agent, device, noise_mode=None):
        disturbed = self.noise.nu(states, noise_mode)

        agent.critic.eval()
        Q = agent.critic.forward(states.detach(), actions).flatten()
        Q_disturbed = agent.critic.forward(disturbed.to(device).detach(), actions).flatten()
        loss = 0.5 * (Q.detach() - Q_disturbed).pow(2).mean()
            
        return loss