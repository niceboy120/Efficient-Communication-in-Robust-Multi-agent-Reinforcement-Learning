from controller import Supervisor, Emitter, Receiver

import sys
sys.path.insert(0, '../../..')

from train_agent import Train



TIME_STEP = 32
HORIZON = 4
# Construct instances
robot = Supervisor()
emitter = Emitter('emitter')
emitter_reset = Emitter('emitter_reset')
receiver = Receiver('receiver')
receiver.enable(TIME_STEP)
receiver_vel_adv1 = Receiver('receiver_vel_adv1')
receiver_vel_adv1.enable(TIME_STEP)
receiver_vel_adv2 = Receiver('receiver_vel_adv2')
receiver_vel_adv2.enable(TIME_STEP)
receiver_vel_agent = Receiver('receiver_vel_agent')
receiver_vel_agent.enable(TIME_STEP)


ENV = 'simple_tag_webots' # 1: simple_tag, 2: simple_tag_elisa, 3: simple_tag_mpc, 3: simple_tag_webots
train = Train(ENV, chkpt_dir='/trained_nets/regular/')
# training(edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=False, zeta=0.0, greedy=False, decreasing_eps=True, N_games=None, lexi_mode=False, robust_actor_loss=True, log=False, noisy=False, load_alt_location=None, noise_mode=None)
# testing(edi_mode='disabled', load=True, load_adversaries=True, edi_load=True, render=True, zeta=0.0, greedy=False, decreasing_eps=False, N_games=None, lexi_mode=False, robust_actor_loss=True, log=False, noisy=False, load_alt_location=None, noise_mode=None):
train.testing(load=True, N_games = 100, greedy=True, decreasing_eps=True, lexi_mode=False, log=False, load_alt_location='simple_tag', run=False)

train.session_setup()
n_episode = 0

adv1_vel = [0,0]
adv2_vel = [0,0] 
agent_vel = [0,0]



i=0
while robot.step(TIME_STEP) != -1:     
    if train.session_pars.next_episode:
        if n_episode != 0:
            train.episode_wrapup(n_episode)        
            emitter_reset.send([1])
        
        if receiver.getQueueLength()==0:
            continue
        obs_webots = receiver.getFloats()
        receiver.nextPacket()
        
        n_episode += 1
        obs, last_comm = train.episode_setup(n_episode, obs_webots) # obs_webots = [adv1x, adv1y, adv2x, adv2y, agentx, agenty]
        train.session_pars.next_episode = False       

   
    if n_episode>train.session_pars.N_games:
        print('================================================================================================\nSession over')
        history, _ = train.session_wrapup()
        with open('results/results_train_regular.pickle', 'wb') as f:
            pickle.dump(history, f)

    if i%HORIZON==0:
        obs_, last_comm, actions, reward, done = train.episode_step_pt1(obs, last_comm, n_episode, train.session_pars.N_games)
        
        #SEND OBS_, IS GOAL
        emitter.send([obs_[0][2], obs_[0][3], obs_[1][2], obs_[1][3], obs_[2][2], obs_[2][3]])
        
        if receiver.getQueueLength()>0:
            obs_webots = receiver.getFloats()
            receiver.nextPacket()
            
        if receiver_vel_adv1.getQueueLength()>0:
            adv1_vel = receiver_vel_adv1.getFloats()
            receiver_vel_adv1.nextPacket()
            
        if receiver_vel_adv2.getQueueLength()>0:
            adv2_vel = receiver_vel_adv2.getFloats()
            receiver_vel_adv2.nextPacket()
            
        if receiver_vel_agent.getQueueLength()>0:
            agent_vel = receiver_vel_agent.getFloats()
            receiver_vel_agent.nextPacket()
        
        vel = adv1_vel + adv2_vel + agent_vel
            
        
        # GET OBS WEBOTS FROM RECEIVER
        obs_ = train.replace_obs(obs_, obs_webots, vel) # obs_webots = [adv1x, adv1y, adv2x, adv2y, agentx, agenty]
        # reward = recalc rewards
        # FIX REWARDS 
        
        obs = train.episode_step_pt2(obs, obs_, actions, reward, done, n_episode)
    
    i += 1
