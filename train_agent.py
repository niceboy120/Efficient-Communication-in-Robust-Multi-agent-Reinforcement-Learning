import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from MADDPG.maddpg import MADDPG
from MADDPG.buffer import MultiAgentReplayBuffer
from EDI.netutilities import NetUtilities
from MPE.make_env import make_env
from utils import obs_list_to_state_vector, HyperParameters
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
        if scenario=='simple_tag' or scenario=='simple_tag_mpc':
            self.pos_mask = [2,3]
            self.pos_others_mask = list(range(4, 4+(self.n_adversaries-1)*2))
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
                            self.par.noise_mode, scenario=scenario, lr_actor=0.01, lr_critic=0.01,   
                            fc1=64, fc2=64, gamma=self.par.gamma,
                            tau=self.par.tau, chkpt_dir='MADDPG'+chkpt_dir)

        self.memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                            self.n_actions, self.n_agents, batch_size=1024)
        
        self.gamma_input_dims = self.env.observation_space[self.cooperating_agents_mask[0]].shape[0]
        


    def run_episodes(self, is_testing, edi_mode, load, load_adversaries, edi_load, render, zeta, greedy, decreasing_eps, N_games, lexi_mode, robust_actor_loss, log, noisy, load_alt_location):
        self.print_start_message(is_testing, edi_mode, lexi_mode, zeta)

        if edi_mode!='disabled':
            self.gammanet = NetUtilities(self.maddpg_agents, self.gamma_input_dims, self.scenario, batch_size = self.par.gamma_batch_size, chkpt_dir=self.chkpt_dir)
        
        if log:
            self.writer = SummaryWriter()

        total_steps = 0
        history = [] # Score adversaries, score good agents, communications of cooperating agents when applicable
        best = [-1000,-1000] # Best score adversaries, best score good agents

        if is_testing:
            load = True
        self.load_nets(load, edi_load, load_adversaries, load_alt_location)            

        if N_games == None:
            if edi_mode=='train':
                    N_games = self.par.N_games_edi
            else:
                if is_testing:
                    N_games = self.par.N_games_test
                else:
                    N_games = self.par.N_games

        for i in range(N_games):
            if lexi_mode and i>self.par.lexi_activate_episode_threshold:
                lexi_mode_active = True
            elif lexi_mode and edi_mode!='disabled':
                lexi_mode_active = True
            else:
                lexi_mode_active = False

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

            n_tags_ep = [0,0,0]

            while not any(done):
                if is_testing and render:
                    self.env.render()
                    time.sleep(0.1)

                if is_testing and edi_mode=='test':
                    obs, last_comm, communications = self.communication_protocol(obs, last_comm, communications, zeta)
                else:
                    communications += len(self.cooperating_agents_mask)#*(len(self.cooperating_agents_mask)-1)

                if is_testing:
                    if not noisy:
                        actions = self.maddpg_agents.eval_choose_action(obs)
                    else:
                        actions = self.maddpg_agents.eval_choose_action_noisy(obs)
                else:
                    actions = self.maddpg_agents.choose_action(obs, greedy, self.par.eps, i/N_games, decreasing_eps) 
                # print(actions)
                obs_, reward, done, info, n_tags = self.env.step(actions)
                for k in range(len(n_tags)):
                    n_tags_ep[k] += n_tags[k]

                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)

                if episode_step >= self.par.max_steps:
                    done = [True]*self.n_agents

                if not is_testing:
                    self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)
                    if total_steps % 100 == 0:
                        if log:
                            self.maddpg_agents.learn(self.memory, lexi_mode_active, robust_actor_loss, self.writer, i)
                        else:
                            self.maddpg_agents.learn(self.memory, lexi_mode_active, robust_actor_loss)
                        self.maddpg_agents.clear_cache()

                obs = obs_
                episode_sequence.append(obs)

                score[0] += np.sum(reward[0:self.n_adversaries])
                score[1] += np.sum(reward[self.n_adversaries:])
                total_steps += 1
                episode_step += 1

            if edi_mode=='train':
                self.gammanet.learn(episode_sequence, self.cooperating_agents_mask)
            
            if log:
                self.writer.add_scalar("score adversaries", score[0], i)
                self.writer.add_scalar("score_agents", score[1], i)
                self.writer.add_scalar("single tags", n_tags_ep[0], i)
                self.writer.add_scalar("double tags", n_tags_ep[1], i)
                self.writer.add_scalar("triple tags", n_tags_ep[2], i)
                self.writer.add_scalar("communications", communications, i)
            history.append(score + [communications])
            avg = np.mean(history[-300:], axis=0)
            best = [max(els) for els in zip(best, avg[0:2])]


            if (i % self.par.print_interval == 0 and i > 0):
                if edi_mode=='test':
                    print('episode: ', i, ', average score adversaries:  {:.1f}'.format(avg[0]), 
                            ', best score adversaries: {:.1f}'.format(best[0]), ', average score good agents: {:.1f}'.format(avg[1]), 
                            ', best score good agents: {:.1f}'.format(best[1]), ', average communications: {:.1f}'.format(avg[2]))
                else:
                    print('episode: ', i, ', average score adversaries:  {:.1f}'.format(avg[0]), 
                            ', best score adversaries: {:.1f}'.format(best[0]), ', average score good agents: {:.1f}'.format(avg[1]), 
                            ', best score good agents: {:.1f}'.format(best[1]))
            elif (is_testing and render):
                print('episode: ', i, ', score adversaries:  {:.1f}'.format(score[0]), 
                            ', best score adversaries: {:.1f}'.format(best[0]), ', score good agents: {:.1f}'.format(score[1]), 
                            ', best score good agents: {:.1f}'.format(best[1]), ', communications: {:.1f}'.format(communications))

            if (i % self.par.autosave_interval == 0 and i > 0):
                if not is_testing:
                    self.maddpg_agents.save_checkpoint()
                if edi_mode=='train':
                    self.gammanet.save()

        if not is_testing:
            self.maddpg_agents.save_checkpoint()

        if edi_mode=='train':
            self.gammanet.save()


        if log:
            self.writer.flush()
            self.writer.close()
        return history
            



    def training(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=False, zeta=0.0, greedy=False, decreasing_eps=True, N_games=None, lexi_mode=False, robust_actor_loss=True, log=False, noisy=False, load_alt_location=None):
        if edi_mode=='disabled':
            edi_load = False

        is_testing = False
        if edi_mode!='disabled' and edi_mode!='test' and edi_mode!='train':
            raise Exception('Invalid mode for edi_mode selected')
        history = self.run_episodes(is_testing, edi_mode, load, load_adversaries, edi_load, render, zeta, greedy, decreasing_eps, N_games, lexi_mode, robust_actor_loss, log, noisy, load_alt_location)
        return history
    

    def testing(self, edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=True, zeta=0.0, greedy=False, decreasing_eps=False, N_games=None, lexi_mode=False, robust_actor_loss=True, log=False, noisy=False, load_alt_location=None):
        if edi_mode=='disabled':
            edi_load = False

        is_testing = True
        if edi_mode!='disabled' and edi_mode!='test' and edi_mode!='train':
            raise Exception('Invalid mode for edi_mode selected')
        history = self.run_episodes(is_testing, edi_mode, load, load_adversaries, edi_load, render, zeta, greedy, decreasing_eps, N_games, lexi_mode, robust_actor_loss, log, noisy, load_alt_location)
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
            self.gammanet.load()


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

    def print_start_message(self, is_testing, edi_mode, lexi_mode, zeta):
        print("\n","===============================================================================\n", datetime.datetime.now())
        if not is_testing:
            msg1 = "Training "
        else:
            msg1 = "Testing "

        if not lexi_mode:
            msg2 = "regular policy"
        else:
            msg2 = "lexicographic policy"

        if edi_mode == 'train':
            msg3 = ", training gammanet"
        elif edi_mode == 'test':
            msg3 = ", testing gammanet with zeta = "+str(zeta)
        else:
            msg3 = ""
        
        msg4 = ", for scenario: "+self.scenario

        msg = msg1+msg2+msg3+msg4
        print(msg)

