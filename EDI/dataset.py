import numpy as np
import random
import torch as T
from utils import obs_list_to_state_vector



## TO DO HERE: do the Q values sheit, check the structure of sequence to make sure the slicing is done correcly in len and gamma

class DataSet:
    def __init__(self, agents):
        self.agents = agents


    def calculate_IO(self, sequence, cooperating_agents_mask, pos_others_mask, pos_mask):
        I = len(sequence)
        num_samples_max = 3

        io = []
        for i in range(2,I):
            num_samples = min(num_samples_max, i)
            mask = []

            for k in range(len(cooperating_agents_mask)):
                mask.append(random.sample(range(0, i+1), num_samples))

            for j in range(num_samples):
                O = []
                Ohat = []
                for k, agent_idx in enumerate(cooperating_agents_mask):
                    O.append(sequence[i][agent_idx])
                    O[k][pos_others_mask] = sequence[mask[k][j]][agent_idx][pos_others_mask]
                    Ohat.append(O[k])
                    Ohat[k][pos_mask] = sequence[mask[k-1][j]][k][pos_mask]


                o1 = sequence[i][0]
                o1[pos_others_mask] = sequence[mask[0][j]][0][pos_others_mask]
                ohat1 = o1
                ohat1[pos_mask] = sequence[mask[1][j]][0][pos_mask]

                o2 = sequence[i][1]
                o2[pos_others_mask] = sequence[mask[1][j]][1][pos_others_mask]
                ohat2 = o2
                ohat2[pos_mask] = sequence[mask[0][j]][1][pos_mask]
                
                x1 = sequence[i]
                x1[0] = o1
                x2 = sequence[i]
                x2[1] = o2
                x = sequence[i]
                xhat = sequence[i]
                xhat[0] = o1
                xhat[1] = o2
       
                # print(self.get_Q(x, xhat, self.agents.agents[0]) - self.get_Q(x, xhat, self.agents.agents[1])) 
                zeta1 = self.get_Q(x, x1, self.agents.agents[0]) - self.get_Q(x, xhat, self.agents.agents[0])
                zeta2 = self.get_Q(x, x2, self.agents.agents[1]) - self.get_Q(x, xhat, self.agents.agents[1])
                zeta = max(zeta1, zeta2)

                io.append([np.concatenate([o1, ohat1]), zeta])
                io.append([np.concatenate([o2, ohat2]), zeta])
        return io

    def get_Q(self, state, state_action, agent):  
        device = agent.target_critic.device
        Q = agent.target_critic.forward(T.tensor(np.array([obs_list_to_state_vector(state)]), dtype=T.float32).to(device), self.get_mu(state_action)).flatten()

        return Q.detach().cpu().numpy()[0]
    

    def get_mu(self, state_mu):
        actions = []

        # Loop through agents and get optimal actions for state_mu
        for agent_idx, agent in enumerate(self.agents.agents):
            device = self.agents.agents[agent_idx].target_actor.device
            agent_state_mu = T.tensor(np.array([state_mu[agent_idx]]), dtype=T.float32).to(device)

            action = agent.target_actor.forward(agent_state_mu)
            actions.append(action)

        mu = T.cat([acts for acts in actions], dim=1)
        return mu