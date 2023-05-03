import numpy as np
import time
from MADDPG.maddpg import MADDPG
from MADDPG.buffer import MultiAgentReplayBuffer
from EDI.netutilities import NetUtilities
from MPE.make_env import make_env
from utils import obs_list_to_state_vector, HyperParameters



class Train:
    def __init__(self, scenario):
        self.par = HyperParameters()

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

        if scenario=='simple_tag':
            self.pos_mask = [2,3]
            self.pos_others_mask = list(range(8, 8+(self.n_adversaries-1)*2))
        elif scenario=='simple_adversary':
            self.pos_mask = [0,1]
            self.pos_others_mask = list(range(8, 8+(self.n_good_agents-1)*2))
        else:
            print("WARNING: You picked a scenario for which EDI is not implemented.")

        actor_dims = []
        for i in range(self.n_agents):
            actor_dims.append(self.env.observation_space[i].shape[0])
        critic_dims = sum(actor_dims)

        self.n_actions = self.env.action_space[0].n
        self.maddpg_agents = MADDPG(actor_dims, critic_dims, self.n_agents, self.n_actions, 
                            fc1=64, fc2=64,  
                            lr_actor=0.01, lr_critic=0.01, scenario=scenario, gamma=self.par.gamma,
                            tau=self.par.tau, chkpt_dir='MADDPG/tmp/')

        self.memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                            self.n_actions, self.n_agents, batch_size=1024)
        
        self.gamma_input_dims = self.env.observation_space[1].shape[0]
        


    def run_episodes(self, is_testing, edi_mode, load, edi_load, render, alpha, decreasing_eps):
        self.gammanet = NetUtilities(self.maddpg_agents, self.gamma_input_dims, alpha=alpha, batch_size = self.par.gamma_batch_size)

        total_steps = 0
        history = [] # Score adversaries, score good agents, communications of cooperating agents when applicable
        best = [-1000,-1000] # Best score adversaries, best score good agents

        if is_testing:
            load = True
        self.load_nets(load, edi_load)            

        if is_testing:
            N_games = self.par.N_games_test
        else:
            N_games = self.par.N_games

        for i in range(N_games):
            obs = self.env.reset()
            if edi_mode=='test':
                last_comm = []
                for a in self.cooperating_agents_mask:
                    last_comm.append(obs[a][self.pos_mask])

            score = [0,0] # Score adversaries, score good agents
            communications = 0
            done = [False]*self.n_agents
            episode_step = 0

            episode_sequence = []
            episode_sequence.append(obs)

            while not any(done):
                if is_testing and render:
                    self.env.render()
                    time.sleep(0.1)

                if is_testing and edi_mode=='test':
                    obs, last_comm, communications = self.communication_protocol(obs, last_comm, communications)
                else:
                    communications += len(self.cooperating_agents_mask)#*(len(self.cooperating_agents_mask)-1)

                actions = self.maddpg_agents.choose_action(obs, self.par.eps, i/N_games, decreasing_eps)
                obs_, reward, done, info = self.env.step(actions)

                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)

                if episode_step >= self.par.max_steps:
                    done = [True]*self.n_agents

                if not is_testing:
                    self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)
                    if total_steps % 100 == 0:
                        self.maddpg_agents.learn(self.memory)

                obs = obs_
                episode_sequence.append(obs)

                score[0] += sum(reward[0:self.n_adversaries])
                score[1] += sum(reward[self.n_adversaries:])
                total_steps += 1
                episode_step += 1

            if edi_mode=='train':
                self.gammanet.learn(episode_sequence, self.cooperating_agents_mask)

            history.append(score + [communications])
            avg = np.mean(history[-300:], axis=0)
            best = [max(els) for els in zip(best, avg[0:2])]

            if i % self.par.print_interval == 0 and i > 0:
                if edi_mode=='test':
                    print('episode: ', i, ', average score adversaries:  {:.1f}'.format(avg[0]), 
                            ', best score adversaries: {:.1f}'.format(best[0]), ', average score good agents: {:.1f}'.format(avg[1]), 
                            ', best score good agents: {:.1f}'.format(best[1]), ', average communications: {:.1f}'.format(avg[2]))
                else:
                    print('episode: ', i, ', average score adversaries:  {:.1f}'.format(avg[0]), 
                            ', best score adversaries: {:.1f}'.format(best[0]), ', average score good agents: {:.1f}'.format(avg[1]), 
                            ', best score good agents: {:.1f}'.format(best[1]))

        if edi_mode=='train':
            self.gammanet.save()

        if is_testing:
            return history
        else: 
            self.maddpg_agents.save_checkpoint()
            self.gammanet.save()
            



    def training(self, edi_mode='disabled', load=True, edi_load=True, render=False, alpha=0.0, decreasing_eps=True):
        is_testing = False
        if edi_mode!='disabled' and edi_mode!='test' and edi_mode!='train':
            raise Exception('Invalid mode for edi_mode selected')
        self.run_episodes(is_testing, edi_mode, load, edi_load, render, alpha, decreasing_eps)


    def testing(self, edi_mode='disabled', load=True, edi_load=True, render=True, alpha=0.0, decreasing_eps=False):
        is_testing = True
        if edi_mode!='disabled' and edi_mode!='test' and edi_mode!='train':
            raise Exception('Invalid mode for edi_mode selected')
        history = self.run_episodes(is_testing, edi_mode, load, edi_load, render, alpha, decreasing_eps)
        return history


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


    def load_nets(self, load, edi_load):
        if load:
            self.maddpg_agents.load_checkpoint()

        if edi_load:
            self.gammanet.load()


    def communication_protocol(self, obs, last_comm, communications): # ADAPT WHEN MOVING TO WEBOTS!!        
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
            if self.gammanet.communication(agent_knowledge[i], agent_knowledge_[i]):
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
    
