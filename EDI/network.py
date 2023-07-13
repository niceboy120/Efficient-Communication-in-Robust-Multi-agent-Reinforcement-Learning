import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GammaNet(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, name, chkpt_dir):


        super(GammaNet, self).__init__()
        
        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims*2, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.Gam = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state_1, state_2):
        x = F.relu(self.fc1(T.cat([state_1, state_2], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        Gam = F.relu(self.Gam(x))

        return Gam

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
