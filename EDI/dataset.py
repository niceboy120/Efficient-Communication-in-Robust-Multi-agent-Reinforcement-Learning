import numpy as np
import torch as T
from utils import obs_list_to_state_vector
from EDI.network import GammaNet



## TO DO HERE: do the Q values sheit, check the structure of sequence to make sure the slicing is done correcly in len and gamma

class DataSet:
    def __init__(self, agents, alpha):
        self.agents = agents
        self.alpha = alpha


    def calculate_IO(self, sequence, min_length_sequence=20):
        I = len(sequence)-1
        
        io = []
        gamma_cache = {}
        for i in range(I-min_length_sequence):
            # gamma = self.calculate_gamma(sequence[i:])
            gamma = gamma_cache.get(i, self.calculate_gamma(sequence[i:]))
            gamma_cache[i] = gamma
            for j in range(I-i-1):
                for k in range(1,self.agents.n_agents):
                    io.append([sequence[i][k], sequence[i+j+1][k], gamma[k-1]])
                # io_ij = np.column_stack([sequence[i][1:], sequence[i+j+1][1:], gamma])
                # io.append(io_ij)
        return io

    def calculate_gamma(self, sequence):           
        number_of_transitions = len(sequence)-1 # Need to know the number of transitions in the sequence

        # Initialization
        done = False
        i = 1

        mu0 = self.get_mu(sequence[0])

        # Loop
        while not done and i<= number_of_transitions:
            if any(self.get_Q_values(sequence[i], mu0) <= np.array(self.get_Q_values(sequence[i], self.get_mu(sequence[i])))-self.alpha): 
                # If for either agent 1 or agent 2 the Q values differ too much, stop.
                done = True
            else:
                i += 1

        # Calculate gamma as the norm between the states
        # gamma = np.linalg.norm(self.concat_obs(sequence[0])-self.concat_obs(sequence[i-1]))

        # gamma = []
        # for k in range(1, self.agents.n_agents):
        #     gamma.append(np.linalg.norm(sequence[0][k]-sequence[i-1][k]))

        gamma = [np.linalg.norm(sequence[0][k]-sequence[i-1][k]) for k in range(1, self.agents.n_agents)]
        return gamma


    def get_Q_values(self, state, mu):
        Q_all = []

        # Loop through agents and get Q values for state, with optimal actions for state_mu
        for i in range(1, self.agents.n_agents): #agent_idx, agent in enumerate(self.agents.agents):
            agent = self.agents.agents[i]        
            device = self.agents.agents[i].target_critic.device
            Q = agent.target_critic.forward(T.tensor([obs_list_to_state_vector(state)], dtype=T.float).to(device), mu).flatten()
            Q_all.append(Q.detach().cpu().numpy()[0])
        
        # We only ned the Q values for the "good" agents, not the adversaries
        return Q_all
    

    def get_mu(self, state_mu):
        actions = []

        # Loop through agents and get optimal actions for state_mu
        for agent_idx, agent in enumerate(self.agents.agents):
            device = self.agents.agents[agent_idx].target_actor.device
            agent_state_mu = T.tensor([state_mu[agent_idx]], dtype=T.float).to(device)

            action = agent.target_actor.forward(agent_state_mu)
            actions.append(action)

        mu = T.cat([acts for acts in actions], dim=1)
        return mu




    # @ staticmethod
    # def concat_obs(observation):
    #     return np.concatenate((observation[0], observation[1], observation[2]))