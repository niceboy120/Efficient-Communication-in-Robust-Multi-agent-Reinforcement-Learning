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
        for i in range(I-min_length_sequence):
            gamma = self.calculate_gamma(sequence[i:])
            for j in range(I-i-1):
                # io.append([np.concatenate((sequence[i][0], sequence[i][1], sequence[i][2])), np.concatenate((sequence[i+j+1][0], sequence[i+j+1][1], sequence[i+j+1][2])), gamma])
                io.append([sequence[i], sequence[i+j+1], gamma])
        return io

    def calculate_gamma(self, sequence):           
        number_of_transitions = len(sequence)-1 # Need to know the number of transitions in the sequence

        # Initialization
        done = False
        i = 1

        # Loop
        while not done:
            if i>number_of_transitions:
                done = True
            elif any(self.get_Q_values(sequence[i], sequence[0]) <= np.array(self.get_Q_values(sequence[i], sequence[i]))-self.alpha): 
                # If for either agent 1 or agent 2 the Q values differ too much, stop.
                done = True
            else:
                i += 1

        # Calculate gamma as the norm between the states
        seq1 = np.concatenate((sequence[0][0], sequence[0][1], sequence[0][2]))
        seq2 = np.concatenate((sequence[i-1][0], sequence[i-1][1], sequence[i-1][2]))
        gamma = np.linalg.norm(seq1-seq2)

        return gamma


    def get_Q_values(self, state, state_mu):
        actions = []

        # Loop through agents and get optimal actions for state_mu
        for agent_idx, agent in enumerate(self.agents):
            device = self.agents[agent_idx].target_actor.device
            agent_state_mu = T.tensor([state_mu[agent_idx]], dtype=T.float).to(device)

            action = agent.target_actor.forward(agent_state_mu)
            actions.append(action)

        mu = T.cat([acts for acts in actions], dim=1)

        Q_all = []

        # Loop through agents and get Q values for state, with optimal actions for state_mu
        for agent_idx, agent in enumerate(self.agents):
            device = self.agents[agent_idx].target_critic.device
            Q = agent.target_critic.forward(T.tensor([obs_list_to_state_vector(state)], dtype=T.float).to(device), mu).flatten()
            Q_all.append(Q.detach().cpu().numpy()[0])
        
        # We only ned the Q values for the "good" agents, not the adversaries
        return Q_all[1:]
