import os
import torch as T
T.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import sys
# sys.path.insert(0, '../../..')

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                    n_agents, n_actions, name, chkpt_dir, scenario):
        super(CriticNetwork, self).__init__()

        if scenario=='simple_tag_webots':
            chkpt_dir = '../../../'+chkpt_dir

        self.chkpt_dir = chkpt_dir
        self.scenario = scenario
        self.name = name
        self.chkpt_file = os.path.join(chkpt_dir+scenario, name)

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, alternative_location=None):
        if alternative_location == None:
            self.load_state_dict(T.load(self.chkpt_file))
        else:
            chkpt_file = os.path.join(self.chkpt_dir+alternative_location, self.name)
            self.load_state_dict(T.load(chkpt_file))



class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir, scenario):
        super(ActorNetwork, self).__init__()

        if scenario=='simple_tag_webots':
            chkpt_dir = '../../../'+chkpt_dir

        self.chkpt_dir = chkpt_dir
        self.scenario = scenario
        self.name = name
        self.chkpt_file = os.path.join(chkpt_dir+scenario, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        pi = T.softmax(pi, dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self, alternative_location=None):
        if alternative_location == None:
            self.load_state_dict(T.load(self.chkpt_file))
        else:
            chkpt_file = os.path.join(self.chkpt_dir+alternative_location, self.name)
            self.load_state_dict(T.load(chkpt_file))

