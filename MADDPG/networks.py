import os
import torch as T
T.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims,
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims, bias=False)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims, bias=False)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.action_value = nn.Linear(n_agents*n_actions, fc2_dims, bias=False)
        self.fc3 = nn.Linear(fc2_dims + fc2_dims, fc3_dims, bias=False)
        self.q = nn.Linear(fc3_dims, 1, bias=False)

        # self.initialization()
        self.optimizer = optim.RMSprop(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def initialization(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.action_value.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

        nn.init.xavier_uniform_(self.fc3.weight,
                                gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, state, action):
        x = F.leaky_relu(self.fc1(state))
        # x = self.bn1(x)

        x = F.leaky_relu(self.fc2(x))
        # x = self.bn2(x)

        y = F.leaky_relu(self.action_value(action))

        x = F.relu(self.fc3(T.cat((x,y), dim=1)))

        q = (self.q(x))

        # print("Q:",q)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims, bias=False)
        self.bn1 = nn.LayerNorm(fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims, bias=False)
        self.bn2 = nn.LayerNorm(fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims, bias=False)
        self.bn3 = nn.LayerNorm(fc3_dims)

        self.pi = nn.Linear(fc3_dims, n_actions, bias=False)

        # self.initialization()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def initialization(self):
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('tanh'))

        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('tanh'))

        nn.init.xavier_uniform_(self.fc3.weight,
                                gain=nn.init.calculate_gain('tanh'))

        nn.init.xavier_uniform_(self.pi.weight,
                                gain=nn.init.calculate_gain('tanh'))

    def forward(self, state):
        x = T.sigmoid(self.fc1(state))
        # x = self.bn1(x)

        x = T.sigmoid(self.fc2(x))
        # x = self.bn2(x)

        x = (self.fc3(x))
        # x = self.bn3(x)

        pi =F.tanh(self.pi(x))

        # print("pi:", pi)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

