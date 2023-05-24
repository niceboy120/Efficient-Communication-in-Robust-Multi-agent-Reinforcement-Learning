from EDI.dataset import DataSet
from EDI.network import GammaNet
import torch as T
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset




class NetUtilities():
    def __init__(self, agents, input_dims, alpha=0.0, batch_size = 32, chkpt_dir='/trained_nets/regular'):
        self.alpha = alpha
        self.batch_size = batch_size
        self.dataset = DataSet(agents, alpha=self.alpha)
        self.gammanet = GammaNet(beta=0.01, input_dims=input_dims, fc1_dims=64, fc2_dims=64, fc3_dims=64, name='GammaNet_'+str(self.alpha), chkpt_dir='EDI'+chkpt_dir)

    def get_gamma_from_net(self, x1, x2): # Getting Gamma from the network given two states
        device = self.gammanet.device

        x1 = T.tensor(np.array([x1]), dtype=T.float32).to(device)
        x2 = T.tensor(np.array([x2]), dtype=T.float32).to(device)

        gamma = self.gammanet.forward(x1, x2)
        return gamma.detach().cpu().numpy()[0][0]

    # def check_communication(): # Check, given a gamma, if the state is too much changed compared with last broadcast


    def learn(self, sequence, cooperating_agents_mask): # learning step for the network?
        io = self.dataset.calculate_IO(sequence, cooperating_agents_mask)

        device = self.gammanet.device

        # Create TensorDataset from io
        inputs = [data [0:2]for data in io]
        targets = [data[2] for data in io]

        dataset = TensorDataset(T.tensor(np.array(inputs), dtype = T.float).to(device), T.tensor(targets, dtype = T.float).to(device))

        # Create DataLoader
        batch_size = self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        
        # Train the network using batches
        for inputs, targets in dataloader:
            self.gammanet.optimizer.zero_grad()
            outputs = self.gammanet.forward(inputs[:, 0], inputs[:, 1])
            loss = F.mse_loss(outputs, targets.unsqueeze(dim=1))
            loss.backward()
            self.gammanet.optimizer.step()


    def communication(self, x1, x2):
        gamma = self.get_gamma_from_net(x1, x2)

        if np.linalg.norm(x1-x2) > gamma:
            return True 
        else:
            return False


    def save(self):
        print('... saving gammanet checkpoint ...')
        self.gammanet.save_checkpoint()

        
    def load(self):
        print('... loading gammanet checkpoint ...')
        self.gammanet.load_checkpoint()
            