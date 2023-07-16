import numpy as np

class HyperParameters:
    def __init__(self):
        self.N_games = 50000
        self.N_games_edi = 1001
        self.N_games_test = 1001
        self.lexi_activate_episode_threshold = 15000
        self.print_interval = 250
        self.autosave_interval = 1000
        self.max_steps = 60
        self.eps = 0.9
        self.gamma_batch_size = 32
        self.gamma = 0.99
        self.tau = 0.01
        self.noise_mode = 1

# class Config:
#     def __init__(self):
#         self.training = ConfigTraining
#         self.Testing = ConfigTesting

# class ConfigTraining:
#     def __init__(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=False, alpha=0.0, greedy=False, decreasing_eps=True, N_games=None, reward_mode=4, lexi_mode=False):



# class ConfigTesting:
#     def __init__(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=True, alpha=0.0, greedy=False, decreasing_eps=False, N_games=None, reward_mode=4, lexi_mode=False):




#     def training(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=False, alpha=0.0, greedy=False, decreasing_eps=True, N_games=None, reward_mode=4, lexi_mode=False):
#         if edi_mode=='disabled':
#             edi_load = False

#         is_testing = False
#         if edi_mode!='disabled' and edi_mode!='test' and edi_mode!='train':
#             raise Exception('Invalid mode for edi_mode selected')
#         history = self.run_episodes(is_testing, edi_mode, load, load_adversaries, edi_load, render, alpha, greedy, decreasing_eps, N_games, reward_mode, lexi_mode)
#         return history
    

#     def testing(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=True, alpha=0.0, greedy=False, decreasing_eps=False, N_games=None, reward_mode=4, lexi_mode=False):
#         if edi_mode=='disabled':
#             edi_load = False

#         is_testing = True
#         if edi_mode!='disabled' and edi_mode!='test' and edi_mode!='train':
#             raise Exception('Invalid mode for edi_mode selected')
#         history = self.run_episodes(is_testing, edi_mode, load, load_adversaries, edi_load, render, alpha, greedy, decreasing_eps, N_games, reward_mode, lexi_mode)
#         return history



def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

