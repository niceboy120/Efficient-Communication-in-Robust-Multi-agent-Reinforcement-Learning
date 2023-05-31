import torch as T
# T.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from MADDPG.agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, noise_mode,
                 scenario='simple',  lr_actor=0.01, lr_critic=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='trained_nets/regular/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario 
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, noise_mode, lr_actor=lr_actor, lr_critic=lr_critic,
                            chkpt_dir=chkpt_dir))
        self.scaler = GradScaler()
        

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self, load_mask):
        print('... loading checkpoint ...')
        for i, agent in enumerate(self.agents):
            if i in load_mask:
                agent.load_models()

    def choose_action(self, raw_obs, greedy, eps, ratio, decreasing_eps):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], greedy, eps, ratio, decreasing_eps)
            actions.append(action)
        return actions
    
    def eval_choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.eval_choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions
    
    def clear_cache(self):
        T.cuda.empty_cache()

    def learn(self, memory, lexi_mode, ratio, robust_actor_loss, writer=None, i=None):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(np.array(states), dtype=T.float32).to(device)
        actions = T.tensor(np.array(actions), dtype=T.float32).to(device)
        rewards = T.tensor(np.array(rewards), dtype = T.float32).to(device)
        states_ = T.tensor(np.array(states_), dtype=T.float32).to(device)
        dones = T.tensor(np.array(dones)).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(np.array(actor_new_states[agent_idx]), 
                                 dtype=T.float32).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = T.tensor(np.array(actor_states[agent_idx]), 
                                 dtype=T.float32).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            
            old_agents_actions.append(actions[agent_idx])


        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_ = critic_value_.masked_fill(dones[:,0], 0.0)
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + critic_value_*agent.gamma
            target = target.detach()

            # with autocast():
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()

            # self.scaler.scale(critic_loss).backward(retain_graph=True)
            # self.scaler.unscale_(agent.critic.optimizer)
            # self.scaler.step(agent.critic.optimizer)
            # self.scaler.update()

            actor_loss = agent.critic.forward(states.detach(), mu).flatten()
            actor_loss = -T.mean(actor_loss)

            if lexi_mode:
                robust_loss = agent.lexicographic_weights.robust_loss(T.tensor(actor_states[agent_idx], dtype=T.float32).to(device), all_agents_new_mu_actions[agent_idx], agent, device, robust_actor_loss)
                # robust_loss = T.tensor(robust_loss, dtype=T.float32).to(device) 

                if writer is not None:
                    writer.add_scalar("robustness loss", robust_loss, i)
                    writer.add_scalar("policy loss", actor_loss, i)
                
                agent.recent_losses[0].append(-actor_loss.detach().cpu().numpy())
                agent.recent_losses[1].append(robust_loss.detach().cpu().numpy())
                agent.lexicographic_weights.update_lagrange(agent.recent_losses)
                w = agent.lexicographic_weights.compute_weights()
        
                loss = actor_loss + robust_loss*(w[1]/w[0])
            else:
                loss = actor_loss
            
            agent.actor.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            # self.scaler.scale(loss).backward(retain_graph=True)
            # self.scaler.unscale_(agent.actor.optimizer)
            # self.scaler.step(agent.actor.optimizer)
            # self.scaler.update()

            agent.update_network_parameters()
