from EDI.dataset import DataSet
from EDI.network import GammaNet
import torch as T
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset




class NetUtilities():
    def __init__(self, agents, input_dims, scenario, batch_size = 32, chkpt_dir='/trained_nets/regular'):
        self.batch_size = batch_size
        self.dataset = DataSet(agents)
        self.gammanet = GammaNet(beta=0.01, input_dims=input_dims, fc1_dims=64, fc2_dims=64, fc3_dims=64, name='GammaNet', chkpt_dir='EDI'+chkpt_dir+scenario)

    def get_gamma_from_net(self, x1, x2, zeta): # Getting Gamma from the network given two states
        device = self.gammanet.device

        data = np.concatenate((x1, x2, np.array([zeta])))
        inputs = T.tensor(data, dtype=T.float32).to(device)

        gamma = self.gammanet.forward(inputs)
        return gamma.detach().cpu().numpy()[0]

    # def check_communication(): # Check, given a gamma, if the state is too much changed compared with last broadcast


    def learn(self, sequence, cooperating_agents_mask): # learning step for the network?
        zeta = np.random.uniform(0,1)
        io = self.dataset.calculate_IO(sequence, cooperating_agents_mask, zeta)

        device = self.gammanet.device

        # Create TensorDataset from io
        inputs = [T.tensor(data[0], dtype=T.float32) for data in io]
        targets = [T.tensor([data[1]], dtype=T.float32) for data in io]

        # Convert the lists to tensors and stack them
        input_tensor = T.stack(inputs)
        target_tensor = T.stack(targets)

        # Create the TensorDataset
        dataset = TensorDataset(input_tensor.to(device), target_tensor.to(device))

        # Create DataLoader
        batch_size = self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        
        # Train the network using batches
        for inputs, targets in dataloader:
            self.gammanet.optimizer.zero_grad()
            outputs = self.gammanet.forward(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            self.gammanet.optimizer.step()


    def communication(self, x1, x2, zeta):
        gamma = self.get_gamma_from_net(x1, x2, zeta)

        if np.linalg.norm(x1-x2, np.inf) > gamma:
            return True 
        else:
            return False


    def save(self):
        print('... saving gammanet checkpoint ...')
        self.gammanet.save_checkpoint()

        
    def load(self):
        print('... loading gammanet checkpoint ...')
        self.gammanet.load_checkpoint()
            