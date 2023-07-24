import numpy as np

class HyperParameters:
    def __init__(self):
        self.N_games = 30000
        self.N_games_edi = 501
        self.N_games_test = 1001
        self.lexi_activate_episode_threshold = 5000
        self.print_interval = 250
        self.autosave_interval = 250
        self.max_steps = 60
        self.eps = 0.9
        self.gamma_batch_size = 32
        self.gamma = 0.99
        self.tau = 0.01

class Config:
    def __init__(self): 
        self.is_testing = None
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

