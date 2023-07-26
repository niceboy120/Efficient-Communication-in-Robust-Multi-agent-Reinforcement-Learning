import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from MADDPG.maddpg import MADDPG
from MADDPG.buffer import MultiAgentReplayBuffer
from EDI.netutilities import NetUtilities
from MPE.make_env import make_env
from utils import obs_list_to_state_vector, HyperParameters, Config, Session_parameters, Episode_parameters
import datetime



class Train:
    def __init__(self, scenario, chkpt_dir='/trained_nets/regular/'):
        self.par = HyperParameters()
        self.chkpt_dir = chkpt_dir
        self.scenario = scenario

        self.env = make_env(scenario)
        self.n_agents = self.env.n

        self.n_adversaries = 0
        self.n_good_agents = 0

        for i in range(self.n_agents):
            if self.env.world.agents[i].adversary:
                self.n_adversaries += 1
            else:
                self.n_good_agents += 1

        if self.n_adversaries>1:
            self.cooperating_agents_mask = list(range(0,self.n_adversaries))
        else:
            self.cooperating_agents_mask = list(range(self.n_adversaries,self.n_adversaries+self.n_good_agents))


        # CHANGE THIS 
        if scenario=='simple_tag' or scenario=='simple_tag_mpc' or scenario=='simple_tag_webots':
            self.pos_mask = [2,3]
            self.pos_others_mask = list(range(4, 4+(self.n_adversaries-1)*2))
            self.pos_others_and_agent_mask = list(range(4, 4+(self.n_adversaries)*2))
        elif scenario=='simple_adversary':
            self.pos_mask = [0,1]
            self.pos_others_mask = list(range(8, 8+(self.n_good_agents-1)*2))
        elif scenario=='simple_tag_elisa':
            self.pos_mask = [2,3,4]
            self.pos_others_mask = list(range(5, 5+(self.n_adversaries-1)*3))
        else:
            print("WARNING: You picked a scenario for which EDI is not implemented.")

        actor_dims = []
        for i in range(self.n_agents):
            actor_dims.append(self.env.observation_space[i].shape[0])
        critic_dims = sum(actor_dims)

        self.n_actions = self.env.action_space[0].n
        self.maddpg_agents = MADDPG(actor_dims, critic_dims, self.n_agents, self.n_actions, 
                            scenario=scenario, lr_actor=0.01, lr_critic=0.01,   
                            fc1=64, fc2=64, gamma=self.par.gamma,
                            tau=self.par.tau, chkpt_dir='MADDPG'+chkpt_dir)

        self.memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                            self.n_actions, self.n_agents, batch_size=1024)
        
        self.gamma_input_dims = self.env.observation_space[self.cooperating_agents_mask[0]].shape[0]
        self.gammanet = NetUtilities(self.maddpg_agents, self.gamma_input_dims, self.scenario, batch_size = self.par.gamma_batch_size, chkpt_dir=self.chkpt_dir)
        

    def run_episodes(self):
        self.session_setup()

        for episode_n in range(self.session_pars.N_games):
            obs, last_comm = self.episode_setup(episode_n)

            while not any(self.episode_pars.done):
                obs_, last_comm, actions, reward, done = self.episode_step_pt1(obs, last_comm, episode_n, self.session_pars.N_games)
                obs = self.episode_step_pt2(obs, obs_, actions, reward, done, episode_n)

            self.episode_wrapup(episode_n)
        return self.session_wrapup()
            
    def session_setup(self):
        self.print_start_message()            
        
        if self.config.log:
            self.writer = SummaryWriter()

        total_steps = 0
        history = [] # Score adversaries, score good agents, communications of cooperating agents when applicable
        best = [-1000,-1000] # Best score adversaries, best score good agents

        if self.config.is_testing:
            load = True
        self.load_nets(self.config.load, self.config.edi_load, self.config.load_adversaries, self.config.load_alt_location)            

        if self.config.N_games == None:
            if self.config.edi_mode=='train':
                    N_games = self.par.N_games_edi
            else:
                if self.config.is_testing:
                    N_games = self.par.N_games_test
                else:
                    N_games = self.par.N_games
        else:
            N_games = self.config.N_games

        self.session_pars = Session_parameters(total_steps, history, best, N_games)


    def episode_setup(self, i, obs_webots=None):
        if self.config.lexi_mode and i>self.par.lexi_activate_episode_threshold:
            lexi_mode_active = True
        elif self.config.lexi_mode and self.config.edi_mode!='disabled':
            lexi_mode_active = True
        else:
            lexi_mode_active = False
            
        obs = self.env.reset()
        if self.scenario=='simple_tag_webots':
            obs = self.replace_obs(obs, obs_webots) # obs_webots = [adv1x, adv1y, adv2x, adv2y, agentx, agenty]

        last_comm = []
        if self.config.edi_mode=='test':
            for a in self.cooperating_agents_mask:
                last_comm.append(obs[a][self.pos_mask])

        score = [0,0] # Score adversaries, score good agents
        communications = 0
        done = [False]*self.n_agents
        episode_step = 0

        episode_sequence = []
        episode_sequence.append(obs)

        self.episode_pars = Episode_parameters(lexi_mode_active, score, communications, done, episode_step, episode_sequence)
        return obs, last_comm

    
    def episode_step_pt1(self, obs, last_comm, i, N_games):
        if self.scenario != 'simple_tag_webots':
            if self.config.is_testing and self.config.render:
                self.env.render()
                time.sleep(0.1)

        if self.config.is_testing and self.config.edi_mode=='test':
            obs, last_comm, self.episode_pars.communications = self.communication_protocol(obs, last_comm, self.episode_pars.communications, self.config.zeta)
        else:
            self.episode_pars.communications += len(self.cooperating_agents_mask)#*(len(self.cooperating_agents_mask)-1)

        if self.config.is_testing:
            if not self.config.noisy:
                actions = self.maddpg_agents.eval_choose_action(obs)
            else:
                actions = self.maddpg_agents.eval_choose_action_noisy(obs, self.config.noise_mode)
        else:
            actions = self.maddpg_agents.choose_action(obs, self.config.greedy, self.par.eps, i/N_games, self.config.decreasing_eps) 
        # print(actions)
        obs_, reward, done, info, n_tags = self.env.step(actions, obs=obs)
        return obs_, last_comm, actions, reward, done

    def episode_step_pt2(self, obs, obs_, actions, reward, done, i):
        state = obs_list_to_state_vector(obs)
        state_ = obs_list_to_state_vector(obs_)

        if self.episode_pars.episode_step >= self.par.max_steps:
            done = [True]*self.n_agents
            self.episode_pars.done = done
            self.session_pars.next_episode = True

        if not self.config.is_testing:
            self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            if self.session_pars.total_steps % 100 == 0:
                if self.config.log:
                    self.maddpg_agents.learn(self.memory, self.episode_pars.lexi_mode_active, self.config.robust_actor_loss, self.writer, i, noise_mode=self.config.noise_mode)
                else:
                    self.maddpg_agents.learn(self.memory, self.episode_pars.lexi_mode_active, self.config.robust_actor_loss, noise_mode=self.config.noise_mode)
                self.maddpg_agents.clear_cache()

        obs = obs_
        self.episode_pars.episode_sequence.append(obs)

        self.episode_pars.score[0] = self.par.gamma*self.episode_pars.score[0] + np.sum(reward[0:self.n_adversaries])
        self.episode_pars.score[1] = self.par.gamma*self.episode_pars.score[1] + np.sum(reward[self.n_adversaries:])
        self.session_pars.total_steps += 1
        self.episode_pars.episode_step += 1

        return obs


    def episode_wrapup(self, i):
        if self.config.edi_mode=='train':
            self.gammanet.learn(self.episode_pars.episode_sequence, self.cooperating_agents_mask)
        
        if self.config.log:
            self.writer.add_scalar("score adversaries", self.episode_pars.score[0], i)
            self.writer.add_scalar("score_agents", self.episode_pars.score[1], i)
            self.writer.add_scalar("communications", self.episode_pars.communications, i)
        self.session_pars.history.append(self.episode_pars.score + [self.episode_pars.communications])
        avg = np.mean(self.session_pars.history[-300:], axis=0)
        self.session_pars.best = [max(els) for els in zip(self.session_pars.best, avg[0:2])]


        if (i % self.par.print_interval == 0 and i > 0):
            if self.config.edi_mode=='test':
                print('episode: ', i, ', average score adversaries:  {:.1f}'.format(avg[0]), 
                        ', best score adversaries: {:.1f}'.format(self.session_pars.best[0]), ', average score good agents: {:.1f}'.format(avg[1]), 
                        ', best score good agents: {:.1f}'.format(self.session_pars.best[1]), ', average communications: {:.1f}'.format(avg[2]))
            else:
                print('episode: ', i, ', average score adversaries:  {:.1f}'.format(avg[0]), 
                        ', best score adversaries: {:.1f}'.format(self.session_pars.best[0]), ', average score good agents: {:.1f}'.format(avg[1]), 
                        ', best score good agents: {:.1f}'.format(self.session_pars.best[1]))
        elif (self.config.is_testing and self.config.render):
            print('episode: ', i, ', score adversaries:  {:.1f}'.format(self.episode_pars.score[0]), 
                        ', best score adversaries: {:.1f}'.format(self.session_pars.best[0]), ', score good agents: {:.1f}'.format(self.episode_pars.score[1]), 
                        ', best score good agents: {:.1f}'.format(self.session_pars.best[1]), ', communications: {:.1f}'.format(self.episode_pars.communications))

        if (i % self.par.autosave_interval == 0 and i > 0):
            if not self.config.is_testing:
                self.maddpg_agents.save_checkpoint()
            if self.config.edi_mode=='train':
                self.gammanet.save()

    def session_wrapup(self):
        if not self.config.is_testing:
            self.maddpg_agents.save_checkpoint()

        if self.config.edi_mode=='train':
            self.gammanet.save()

        if self.config.log:
            self.writer.flush()
            self.writer.close()
        return self.session_pars.history, self.episode_pars.episode_sequence

    def training(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=False, zeta=0.0, greedy=False, decreasing_eps=True, N_games=None, lexi_mode=False, robust_actor_loss=True, log=False, noisy=False, load_alt_location=None, noise_mode=None, run=True):
        if edi_mode=='disabled':
            edi_load = False

        is_testing = False
        if edi_mode!='disabled' and edi_mode!='test' and edi_mode!='train':
            raise Exception('Invalid mode for edi_mode selected')        

        self.config = Config()
        self.config.set(is_testing, edi_mode, load, load_adversaries, edi_load, render, zeta, greedy, decreasing_eps, N_games, lexi_mode, robust_actor_loss, log, noisy, load_alt_location, noise_mode)
        if run:
            return self.run_episodes()

    def testing(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=True, zeta=0.0, greedy=False, decreasing_eps=False, N_games=None, lexi_mode=False, robust_actor_loss=True, log=False, noisy=False, load_alt_location=None, noise_mode=None, run=True):
        if edi_mode=='disabled':
            edi_load = False

        is_testing = True
        if edi_mode!='disabled' and edi_mode!='test' and edi_mode!='train':
            raise Exception('Invalid mode for edi_mode selected')        

        self.config = Config()
        self.config.set(is_testing, edi_mode, load, load_adversaries, edi_load, render, zeta, greedy, decreasing_eps, N_games, lexi_mode, robust_actor_loss, log, noisy, load_alt_location, noise_mode)
        if run:
            return self.run_episodes()

    def ask_save(self):
        answer = False
        while not answer:
            user_input = input("Would you like to save the agent models? (y/n) ")
            if user_input.lower() == 'y':
                self.maddpg_agents.save_checkpoint()
                answer = True
            elif user_input.lower() == 'n':
                answer = True
                pass
            else:
                print("Invalid reply, please respond y or n")


        # Maybe add if statement if even applicable??

        answer = False
        while not answer:
            user_input = input("Would you like to save gamma? (y/n) ")
            if user_input.lower() == 'y':
                self.gammanet.save()
                answer = True
            elif user_input.lower() == 'n':
                answer = True
                pass
            else:
                print("Invalid reply, please respond y or n")


    def load_nets(self, load, edi_load, load_adversaries, load_alt_location):
        load_mask = []
        for i in range(self.n_agents):
            if not self.env.world.agents[i].adversary:
                load_mask.append(i)
            elif load_adversaries:
                load_mask.append(i)

        if load:
            self.maddpg_agents.load_checkpoint(load_mask, load_alt_location)

        if edi_load:
            self.gammanet.load(load_alt_location)


    def communication_protocol(self, obs, last_comm, communications, zeta): # ADAPT WHEN MOVING TO WEBOTS!!        
        agent_knowledge = []
        agent_knowledge_ = []

        for j,n in enumerate(self.cooperating_agents_mask):
            temp = obs[n].copy()
            team_members = list(range(len(self.cooperating_agents_mask)))
            team_members.pop(j)

            for i in range(len(team_members)):
                temp[self.pos_others_mask[i*2]:self.pos_others_mask[i*2]+2] = last_comm[team_members[i]] - temp[self.pos_mask]
            agent_knowledge.append(temp)

            temp2 = temp.copy()
            for i,p in enumerate(self.pos_mask):
                temp2[p] = last_comm[j][i]
            agent_knowledge_.append(temp2)

        
        for i in range(len(self.cooperating_agents_mask)):
            if self.gammanet.communication(agent_knowledge[i], agent_knowledge_[i], zeta):
               # Agent 1 sends update
                last_comm[i] = agent_knowledge[i][self.pos_mask]
                communications += 1


        for j,n in enumerate(self.cooperating_agents_mask):
            temp = obs[n].copy()
            team_members = list(range(len(self.cooperating_agents_mask)))
            team_members.pop(j)

            for i in range(len(team_members)):
                temp[self.pos_others_mask[i*2]:self.pos_others_mask[i*2]+2] = last_comm[team_members[i]] - temp[self.pos_mask]

            obs[n] = temp

        return obs, last_comm, communications
    

    def clear_buffer(self):
        self.memory.init_actor_memory()

    def print_start_message(self):
        print("\n","===============================================================================\n", datetime.datetime.now())
        if not self.config.is_testing:
            msg1 = "Training "
        else:
            msg1 = "Testing "

        if not self.config.lexi_mode:
            msg2 = "regular policy"
        else:
            msg2 = "lexicographic policy"

        if self.config.noisy:
            msg3 = " with noise (mode = "+str(self.config.noise_mode)+")"
        else:
            msg3 = " without noise"

        if self.config.edi_mode == 'train':
            msg4 = ", training gammanet"
        elif self.config.edi_mode == 'test':
            msg4 = ", testing gammanet with zeta = "+str(self.config.zeta)
        else:            msg4 = ""
        
        msg5 = ", for scenario: "+self.scenario

        msg = msg1+msg2+msg3+msg4+msg5
        print(msg)

    def replace_obs(self, obs, obs_webots, vel=None): # obs_webots = [adv1x, adv1y, adv2x, adv2y, agentx, agenty]
        for i, obs_i in enumerate(obs):        
            obs[i][self.pos_mask[0]:self.pos_mask[-1]+1] = [obs_webots[i*2], obs_webots[i*2+1]]
            if vel != None:
                obs[i][0:2] = [vel[i*2], vel[i*2+1]]
            j = list(range(self.n_agents))
            j.pop(i)
            for jid, j in enumerate(j):
                obs[i][self.pos_others_and_agent_mask[jid*2]:self.pos_others_and_agent_mask[jid*2+1]+1] = [obs_webots[j*2]-obs_webots[i*2], obs_webots[j*2+1]-obs_webots[i*2+1]]                 
            if i in self.cooperating_agents_mask and vel != None:
                obs[i][-2:] = vel[-2:]        
        return obs
