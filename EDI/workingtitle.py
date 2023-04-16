from EDI.dataset import DataSet
from EDI.network import GammaNet
import torch as T
import torch.nn.functional as F
import numpy as np




class WorkingTitle():
    def __init__(self, agents, input_dims, alpha=0.0):
        self.alpha = alpha
        self.dataset = DataSet(agents, alpha=self.alpha)
        self.gammanet = GammaNet(beta=0.01, input_dims=input_dims, fc1_dims=64, fc2_dims=64, fc3_dims=64, name='GammaNet_'+str(self.alpha), chkpt_dir='EDI/tmp/')

    def get_gamma_from_net(self, state_1, state_2): # Getting Gamma from the network given two states
        x1 = np.concatenate((state_1[0], state_1[1], state_1[2]))
        x2 = np.concatenate((state_2[0], state_2[1], state_2[2]))

        device = self.gammanet.device

        x1 = T.tensor([x1], dtype=T.float).to(device)
        x2 = T.tensor([x2], dtype=T.float).to(device)

        gamma = self.gammanet.forward(x1, x2)
        return gamma

    # def check_communication(): # Check, given a gamma, if the state is too much changed compared with last broadcast


    def learn(self, sequence): # learning step for the network?
        io = self.dataset.calculate_IO(sequence)

        for i in range(len(io)):
            # print(io[i][0], io[i][1])
            output = self.get_gamma_from_net(io[i][0], io[i][1])

            print("test", output, io[i][2])

            device = self.gammanet.device
            outp =  T.tensor(output, dtype = T.float).to(device)
            inp = T.tensor(io[i][2], dtype = T.float).to(device)

            loss = F.mse_loss(inp, outp)
            self.gammanet.zero_grad()
            loss.backward(retain_graph=True)
            self.gammanet.optimizer.step()

            