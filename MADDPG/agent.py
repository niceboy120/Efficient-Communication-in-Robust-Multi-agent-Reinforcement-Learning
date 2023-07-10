import torch as T
import numpy as np
from MADDPG.networks import ActorNetwork, CriticNetwork
import random
from LRRL.lexicographic import LexicographicWeights
from LRRL.noise_generator import NoiseGenerator

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, noise_mode, chkpt_dir, scenario,
                    lr_actor=0.01, lr_critic=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(lr_actor, actor_dims, fc1, fc2, n_actions, 
                                  chkpt_dir=chkpt_dir, scenario=scenario, name=self.agent_name+'_actor')
        self.critic = CriticNetwork(lr_critic, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, scenario=scenario, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(lr_actor, actor_dims, fc1, fc2, n_actions,
                                        chkpt_dir=chkpt_dir, scenario=scenario, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(lr_critic, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir, scenario=scenario,
                                            name=self.agent_name+'_target_critic')
        

        self.noise = NoiseGenerator(mode = noise_mode)
        self.lexicographic_weights = LexicographicWeights(self.noise)
        self.recent_losses = self.lexicographic_weights.init_recent_losses()
        
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, greedy, eps, ratio, decreasing_eps):
        if greedy:
            if decreasing_eps:
                eps = (1-ratio)*eps# + (1-eps)

        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)

        if greedy:
            if random.uniform(0,1)>eps:
                action = actions
            else:
                action = 0*actions+noise
        else:
            action = actions + noise
        return action.detach().cpu().numpy()[0]
    
    def eval_choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.target_actor.device)
        actions = self.target_actor.forward(state)

        return actions.detach().cpu().numpy()[0]
    
    def eval_choose_action_noisy(self, observation, noise_mode=None):
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.target_actor.device)
        disturbed = self.noise.nu(state, noise_mode)
        actions = self.target_actor.forward(disturbed.to(self.target_actor.device))

        return actions.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self, load_alt_location):
        self.actor.load_checkpoint(alternative_location=load_alt_location)
        self.target_actor.load_checkpoint(alternative_location=load_alt_location)
        self.critic.load_checkpoint(alternative_location=load_alt_location)
        self.target_critic.load_checkpoint(alternative_location=load_alt_location)
