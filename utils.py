import numpy as np

class HyperParameters:
    def __init__(self):
        self.N_games = 10000 # Number of episodes for regular training, can be overwritten
        self.N_games_edi = 1001 # Number of episodes for EDI training, can be overwritten
        self.N_games_test = 1001 # Number of episodes for testing, can be overwritten
        self.lexi_activate_episode_threshold = 200 # Episode number to activate the lexi mode
        self.print_interval = 250 # At what episode interval to print updates
        self.autosave_interval = 250 # At what episode interval to autosave
        self.max_steps = 60 # Steps per episode
        self.eps = 0.9 # Greedy epsilon parameter
        self.gamma_batch_size = 32 # Learning batch size
        self.gamma = 0.99 # Discount rate for return
        self.tau = 0.01 # Learning rate target networks

class Config:
    def __init__(self): 

        # - edi_mode: ('disabled'default()/'train'/'test') determines whether or not we want to use EDI
        # - load: (True(default)/False) determines to load pretraines MADDPG nets
        # - load_adversaries: (True(default)/False) determines to load pretrained MADDPG nets for the adversaries, to make separate training of the agents possible
        # - edi_load: (True(default)/False) determines to load pretrained robustness surrogates networks
        # - render: (True(default testing)/False(default training)) determines if the episode is rendered 
        # - zeta: (float, 0.0 is default) sensitivity parameter for EDI
        # - greedy: (True(default)/False) uses epsilon greedy if True, additive noise if False. For exploration during training.
        # - decreasing_eps: (True(default)/False) determines whether the epsilon decreases over episodes
        # - N_games: (integer, None is default) used to overwite the default number of episodes defined in utils
        # - lexi_mode: (True/False(delfault)) determines whether or not the policy is lexicographically robust or vanilla
        # - robust_actor_loss: (True(default)/False) determines whether to use the actor loss or critic loss for robustness objective
        # - log: (True/False(default)) turning logging to tensorboard on/off
        # - noisy: (True/False(default)) toggles additive noise to the observations
        # - load_alt_location: (None is default) used to specify an alternative directory to load the networks from
        # - noise_mode: (1/2/None(default)) 1 is uniform noise, 2 is Gaussian noise
        # - run: (True(default)/False) whether or not the testing/training functions automatically run episodes

        self.is_testing = None # Determined by Train.testing/Train.training
        self.edi_mode = None
        self.load = None
        self.load_adversaries = None
        self.edi_load = None
        self.render = None
        self.zeta = None
        self.greedy = None
        self.decreasing_eps = None
        self.N_games = None
        self.lexi_mode = None
        self.robust_actor_loss = None
        self.log = None
        self.noisy = None
        self.load_alt_location = None
        self.noise_mode=None

    def set(self, is_testing=None, edi_mode=None, load=None, load_adversaries=None, edi_load=None, render=None, zeta=None, greedy=None, decreasing_eps=None, N_games=None, lexi_mode=None, robust_actor_loss=None, log=None, noisy=None, load_alt_location=None, noise_mode=None):
        if is_testing != None:
            self.is_testing = is_testing

        if edi_mode != None:
            self.edi_mode = edi_mode

        if load != None:
            self.load = load

        if load_adversaries != None:
            self.load_adversaries = load_adversaries

        if edi_load != None:
            self.edi_load = edi_load

        if render != None:
            self.render = render

        if zeta != None:
            self.zeta = zeta

        if greedy != None:
            self.greedy = greedy
        
        if decreasing_eps != None:
            self.decreasing_eps = decreasing_eps

        if N_games != None:
            self.N_games = N_games

        if lexi_mode != None:
            self.lexi_mode = lexi_mode

        if robust_actor_loss != None:
            self.robust_actor_loss = robust_actor_loss

        if log != None:
            self.log = log

        if noisy != None:
            self.noisy = noisy

        if load_alt_location != None:
            self.load_alt_location = load_alt_location

        if noise_mode != None:
            self.noise_mode=noise_mode

class Session_parameters:
    def __init__(self, total_steps, history, best, N_games):
        self.total_steps = total_steps
        self.history = history
        self.best = best
        self.N_games = N_games
        self.next_episode = True

class Episode_parameters:
    def __init__(self, lexi_mode_active, score, communications, done, episode_step, episode_sequence):
        self.lexi_mode_active = lexi_mode_active
        self.score = score
        self.communications = communications
        self.done = done
        self.episode_step = episode_step
        self.episode_sequence = episode_sequence

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

