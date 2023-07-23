import torch as T
# T.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from MADDPG.agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple',  lr_actor=0.01, lr_critic=0.01, fc1=64, 
                 fc2=64, gamma=0.99, tau=0.01, chkpt_dir='trained_nets/regular/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,  
                            n_actions, n_agents, agent_idx, lr_actor=lr_actor, lr_critic=lr_critic,
                            chkpt_dir=chkpt_dir, scenario=scenario))
        self.scaler = GradScaler()
        self.scenario = scenario
        self.chkpt_dir = chkpt_dir
        

    def save_checkpoint(self):
        print('... saving checkpoint at ', self.chkpt_dir+self.scenario, ' ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self, load_mask, load_alt_location=None):
        if load_alt_location == None:
            print('... loading checkpoint from ', self.chkpt_dir+self.scenario, ' ...')
        else:
            print('... loading checkpoint from ', self.chkpt_dir+load_alt_location, ' ...')
        for i, agent in enumerate(self.agents):
            if i in load_mask:
                agent.load_models(load_alt_location)

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
    
    def eval_choose_action_noisy(self, raw_obs, noise_mode):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.eval_choose_action_noisy(raw_obs[agent_idx], noise_mode)
            actions.append(action)
        return actions
    
    def clear_cache(self):
        T.cuda.empty_cache()

    def learn(self, memory, lexi_mode, robust_actor_loss, writer=None, i=None, noise_mode=None):
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
            agent.target_actor.eval()
            agent.target_critic.eval()
            agent.actor.eval()
            agent.critic.eval()

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
            agent.critic.train()
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward()
            agent.critic.optimizer.step()

            # self.scaler.scale(critic_loss).backward(retain_graph=True)
            # self.scaler.unscale_(agent.critic.optimizer)
            # self.scaler.step(agent.critic.optimizer)
            # self.scaler.update()

            agent.critic.eval()
            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss) # Mean in case of stochastic policy??? or because we sample multiple states at a time maybe

            if lexi_mode:
                w = agent.lexicographic_weights.compute_weights()

                if robust_actor_loss:
                    robust_loss = agent.lexicographic_weights.robust_loss_actor(T.tensor(np.array(actor_states[agent_idx]), dtype=T.float32).to(device), all_agents_new_mu_actions[agent_idx], agent, device, noise_mode)
                else:
                    robust_loss = agent.lexicographic_weights.robust_loss_critic(states, mu, agent, device, noise_mode)                
                # robust_loss = T.tensor(robust_loss, dtype=T.float32).to(device) 
               
                loss = actor_loss + robust_loss*(w[1]/w[0])
                # loss = robust_loss

                agent.recent_losses[0].append(-actor_loss.detach().cpu().numpy())
                agent.recent_losses[1].append(robust_loss.detach().cpu().numpy())
                agent.lexicographic_weights.update_lagrange(agent.recent_losses)

                if writer is not None and agent_idx==0:
                    writer.add_scalar("robustness loss", robust_loss, i)
                    writer.add_scalar("policy loss", actor_loss, i)
                    writer.add_scalar("weight 1", w[0], i)
                    writer.add_scalar("weight 2", w[1], i)
                    writer.add_scalar("lambda", agent.lexicographic_weights.labda[0], i)
                    writer.add_scalar("optimal", agent.lexicographic_weights.j[0], i)
                    writer.add_scalar("last", agent.recent_losses[0][-1], i)
                    
                agent.actor.train()
                agent.actor.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                agent.actor.optimizer.step()

                if writer is not None and agent_idx==0:
                    writer.add_scalar("total policy loss", loss, i)

            else:
                if writer is not None and agent_idx==0:
                    writer.add_scalar("policy loss", actor_loss, i)
                agent.actor.train()
                agent.actor.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                agent.actor.optimizer.step()

            # self.scaler.scale(loss).backward(retain_graph=True)
            # self.scaler.unscale_(agent.actor.optimizer)
            # self.scaler.step(agent.actor.optimizer)
            # self.scaler.update()

            agent.update_network_parameters()
