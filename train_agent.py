import numpy as np
from MADDPG.maddpg import MADDPG
from MADDPG.buffer import MultiAgentReplayBuffer
from EDI.netutilities import NetUtilities
from make_env import make_env
import time
from utils import obs_list_to_state_vector

## IMORT EDI THINGS

class Train:
    def __init__(self, mode='train', edi_mode='disabled', load=False, save=True, edi_load=False, edi_save=True, print_interval=500, N_games=50000, max_steps=50, alpha=0.0, gamma_batch_size=32, render=True):
        self.mode = mode
        if self.mode=='test':
            self.load = True
            self.save = False
        elif self.mode=='train':
            self.load = load
            self.save = save
        else:
            raise Exception("Wrong mode selected, please choose 'test' or 'train'")
        
        self.edi_mode = edi_mode
        if self.edi_mode=='disabled':
            self.edi_load = False
            self.edi_save = False
        elif self.edi_mode=='test':
            self.edi_load = True
            self.edi_save = False
        elif self.edi_mode=='parallel':
            self.edi_load = edi_load
            self.edi_save = edi_save
        elif self.edi_mode=='sequential': # NOT YET IMPLEMENTED
            self.edi_load = edi_load
            self.edi_save = edi_save
        else:
            raise Exception("Wrong EDI mode selected, please choose 'parallel', 'sequential' 'test' or 'disabled' (sequential training not yet implemented!!!!)")
    
        self.print_interval = print_interval
        self.N_games = N_games
        self.max_steps = max_steps

        scenario = 'simple_adversary'
        self.env = make_env(scenario)
        self.n_agents = self.env.n

        self.alpha = alpha
        self.gamma_batch_size = gamma_batch_size

        self.render = render

        actor_dims = []
        for i in range(self.n_agents):
            actor_dims.append(self.env.observation_space[i].shape[0])
        critic_dims = sum(actor_dims)

        self.n_actions = self.env.action_space[0].n
        self.maddpg_agents = MADDPG(actor_dims, critic_dims, self.n_agents, self.n_actions, 
                            fc1=64, fc2=64,  
                            alpha=0.01, beta=0.01, scenario=scenario,
                            chkpt_dir='MADDPG/tmp/maddpg/')

        self.memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                            self.n_actions, self.n_agents, batch_size=1024)
        

        self.gamma_input_dims = self.env.observation_space[1].shape[0]

        self.gammanet = NetUtilities(self.maddpg_agents, self.gamma_input_dims, alpha = self.alpha, batch_size = self.gamma_batch_size)



        
    def loop(self):
        total_steps = 0
        score_history = []
        communications_history = []
        best_score = 0

        self.load_nets()

        for i in range(self.N_games):
            obs = self.env.reset()
            last_comm1 = obs[1][0:2]
            last_comm2 = obs[2][0:2] 

            score = 0
            communications = 0
            done = [False]*self.n_agents
            episode_step = 0

            episode_sequence = []
            episode_sequence.append(obs)

            while not any(done):
                if self.mode=='test':
                    if self.render:
                        self.env.render()
                        time.sleep(0.1)

                    if self.edi_mode=='test':
                        obs, last_comm1, last_comm2, communications = self.communication_protocol(obs, last_comm1, last_comm2, communications)
                    else:
                        communications += 2
                    actions = self.maddpg_agents.eval_choose_action(obs)
                    
                else:
                    actions = self.maddpg_agents.choose_action(obs)
                obs_, reward, done, info = self.env.step(actions)

                state = obs_list_to_state_vector(obs)
                state_ = obs_list_to_state_vector(obs_)

                if episode_step >= self.max_steps:
                    done = [True]*self.n_agents

                self.memory.store_transition(obs, state, actions, reward, obs_, state_, done)

                if total_steps % 100 == 0 and self.mode=='train':
                    self.maddpg_agents.learn(self.memory)

                obs = obs_
                episode_sequence.append(obs)

                score += sum(reward)
                total_steps += 1
                episode_step += 1

            if self.edi_mode=='parallel':
                self.gammanet.learn(episode_sequence)

            score_history.append(score)
            communications_history.append(communications)
            avg_score = np.mean(score_history[-300:])

            if self.mode=='train':
                if avg_score > best_score:
                    if self.save and i>300:
                        self.maddpg_agents.save_checkpoint()
                        self.gammanet.save()
                        if np.std(score_history[-2000:]) <= 0.3:
                            print("Models trained, switching mode to testing")
                            self.mode = 'test'
                    best_score = avg_score                    
            # else:
            #     print("Score: ", score)
            #     print("Communications: ", communications)
            if i % self.print_interval == 0 and i > 0:
                print('episode', i, 'average score {:.1f}'.format(avg_score))

        
        if self.mode=='test':
            return score_history, communications_history
        
        # print("Final best score: ", best_score)
        # self.ask_save()
        if self.save:
            self.maddpg_agents.save_checkpoint()
        if self.edi_save:
            self.gammanet.save()


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



    def load_nets(self):
        if self.load:
            self.maddpg_agents.load_checkpoint()

        if self.edi_load:
            self.gammanet.load()


    def communication_protocol(self, obs, last_comm1, last_comm2, communications): # ADAPT WHEN MOVING TO WEBOTS!!
        # MAKE THIS NICE FOR A VARIABLE AMOUNT OF AGENTS AND OTHER OBSERVATION SPACES
        
        # for i in range(1, self.maddpg_agents.n_agents):
        #     agent_knowlegde = obs[i]
        #     agent_knowledge[8:] = agent_knowlegde[0:2]-last_comm[]
        
        agent1_knowledge = obs[1].copy()
        agent1_knowledge[8:] = agent1_knowledge[0:2]-last_comm2
        agent1_knowledge_ = agent1_knowledge.copy()
        agent1_knowledge_[0:2] = last_comm1

        agent2_knowledge = obs[2].copy()
        agent2_knowledge[8:] = agent2_knowledge[0:2]-last_comm1
        agent2_knowledge_ = agent2_knowledge.copy()
        agent2_knowledge_[0:2] = last_comm2


        if self.gammanet.communication(agent1_knowledge, agent1_knowledge_):
            # Agent 1 sends update
            last_comm1 = agent1_knowledge[0:2]
            communications += 1

        if self.gammanet.communication(agent2_knowledge, agent2_knowledge_):
            # Agent 2 sends update
            last_comm2 = agent2_knowledge[0:2]
            communications += 1

        obs1 = obs[1].copy()
        obs1[8:] = obs1[0:2]-last_comm2
        obs2 = obs[2].copy()
        obs2[8:] = obs2[0:2]-last_comm1
        obs = [obs[0], obs1, obs2]

        return obs, last_comm1, last_comm2, communications


    # def force_save(self):
    #     self.maddpg_agents.save_checkpoint()


    # @staticmethod
    # def obs_list_to_state_vector(observation):
    #     state = np.array([])
    #     for obs in observation:
    #         state = np.concatenate([state, obs])
    #     return state