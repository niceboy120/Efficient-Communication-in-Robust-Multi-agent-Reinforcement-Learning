from controller import Supervisor, Emitter
import random
import math
import pickle
import numpy as np
import time

import sys
sys.path.insert(0, '../../..')

from train_agent import Train
from MPC.controller import SIMPLE_CONTROLLER
from MPC.car import Car
from MPE.multiagent.scenarios.simple_tag import Scenario


TIME_STEP = 64
HORIZON = 4

# Construct instances
robot = Supervisor()
scenario = Scenario()
world = scenario.make_world('simple_tag_webots')
controller = SIMPLE_CONTROLLER(HORIZON)  
emitter1 = Emitter('emitter1')
emitter2 = Emitter('emitter2')
emitter3 = Emitter('emitter3')
emitters = [emitter1, emitter2, emitter3]

# Get robot ids
root_node = robot.getRoot()
children_field = root_node.getField('children')
adv1_node = robot.getFromDef('ADV1')
adv2_node = robot.getFromDef('ADV2')
agent_node = robot.getFromDef('AGENT')
nodes = [adv1_node, adv2_node, agent_node]


def reset_nodes(nodes):
    coords = []
    rotations = []
    for n in range(len(nodes)):
        node = nodes[n]
        translation_field = node.getField('translation')
        rotation_field = node.getField('rotation')
        translation = [random.uniform(-0.9,0.9), random.uniform(-0.9,0.9), 0.0001]
        rotation = random.uniform(0,6.28)
        rotation_field.setSFRotation([0,0,1,rotation])
        translation_field.setSFVec3f(translation)        
        
        coords.append(translation[0])
        coords.append(translation[1])
        rotations.append(rotation)
    return coords, rotations
    
def get_coords(nodes):
    coords = [] # [adv1_x, adv1_y, adv2_x, adv2_y, agent_x, agent_y]
    rotations = [] # [adv1_phi, adv2_phi, agent_phi]
    for n in range(len(nodes)):
        node = nodes[n]
    
        translation = node.getPosition()
        RM = node.getOrientation()
        rotation = math.atan2(RM[3], RM[0])
    
        coords.append(translation[0])
        coords.append(translation[1])
        rotations.append(rotation)
    return coords, rotations


def while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP):
    if train.session_pars.next_episode: # Episode over
        if n_episode != 0:
            train.episode_wrapup(n_episode)  
            for n in range(len(nodes)):
                emitters[n].send([0.0,0.0])
            robot.simulationResetPhysics()
            robot.step(TIME_STEP)
            robot.simulationResetPhysics()
            coords, rotation = reset_nodes(nodes)
            robot.step(TIME_STEP)
            robot.simulationResetPhysics()

        if n_episode != train.session_pars.N_games:
            n_episode += 1
            obs, last_comm = train.episode_setup(n_episode, coords) # obs_webots = [adv1x, adv1y, adv2x, adv2y, agentx, agenty]
            train.session_pars.next_episode = False        
        else:
            n_episode += 1
        
        i=0       

    coords, rotation = get_coords(nodes)# [adv1_x, adv1_y, adv2_x, adv2_y, agent_x, agent_y]
    if i%HORIZON==0:
        if i!=0:
            obs_ = train.replace_obs(obs_, coords, vel)     
            adv_rew, _ = scenario.adversary_reward(world=world, coords=coords)
            agent_rew, _ = scenario.agent_reward(world=world, coords=coords)
            reward = [adv_rew, adv_rew, agent_rew]
            
            obs = train.episode_step_pt2(obs, obs_, actions, reward, done, n_episode) 
            # Output obs is input obs_, is webots valid
        
        obs_, last_comm, actions, reward, done = train.episode_step_pt1(obs, last_comm, n_episode, train.session_pars.N_games)
        # Output obs_ contains the waypoints, is MPE valid
        goal = [[obs_[0][2], obs_[0][3]], [obs_[1][2], obs_[1][3]], [obs_[2][2], obs_[2][3]]]
        
    
    vel = [] # [adv1_vx, adv1_vy, adv2_vx, adv2_vy, agent_vx, agent_vy]
    for n in range(len(nodes)):
            linear_vel, angular_vel = controller.get_control_inputs(np.array([[coords[n*2]],[coords[n*2+1]],[rotation[n]]]), goal[n])
            cars[n].set_robot_velocity(linear_vel, angular_vel)
            vel += [linear_vel*np.cos(rotation[n]), linear_vel*np.sin(rotation[n])]
            emitters[n].send([cars[n].wheel_speed[0][0], cars[n].wheel_speed[1][0]])   
    i+=1            
    return i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel

coords, rotation = reset_nodes(nodes)
vel = [0,0,0,0,0,0]

cars = []
for n in range(len(nodes)):
    cars.append(Car(coords[n*2], coords[n*2+1], rotation[n]))


ENV = 'simple_tag_webots' # 1: simple_tag, 2: simple_tag_elisa, 3: simple_tag_mpc, 3: simple_tag_webots
train = Train(ENV, chkpt_dir='/trained_nets/regular/')
train.testing(load=True, N_games = 1000, greedy=True, decreasing_eps=True, lexi_mode=False, log=False, load_alt_location='simple_tag', run=False)

train.session_setup()
n_episode = 0
i=0
obs = None
obs_ = None
actions = None
reward = None
done = None
last_comm = None
goal = None


# NOISE TEST ===============================================================================================

while robot.step(TIME_STEP) != -1:
    if n_episode>train.session_pars.N_games: # Session over
        if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history_reg_sa, _ = train.session_wrapup()
            break
        
    else:
        i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)


train = Train(ENV, chkpt_dir='/trained_nets/LRRL4/')
train.testing(load=True, N_games = 1000, greedy=True, decreasing_eps=True, lexi_mode=True, log=False, load_alt_location='simple_tag', run=False)
train.session_setup()
n_episode = 0
i=0      
    
while robot.step(TIME_STEP) != -1:
    if n_episode>train.session_pars.N_games: # Session over
        if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history_LRRL_sa, _ = train.session_wrapup()
            break
        
    else:
        i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)


train = Train(ENV, chkpt_dir='/trained_nets/regular/')
train.testing(load=True, N_games = 1000, greedy=True, decreasing_eps=True, lexi_mode=False, log=False, load_alt_location='simple_tag_mpc', run=False)
train.session_setup()
n_episode = 0
i=0      
    
while robot.step(TIME_STEP) != -1:
    if n_episode>train.session_pars.N_games: # Session over
        if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history_reg_mpc, _ = train.session_wrapup()
            break
        
    else:
        i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)


train = Train(ENV, chkpt_dir='/trained_nets/LRRL4/')
train.testing(load=True, N_games = 1000, greedy=True, decreasing_eps=True, lexi_mode=True, log=False, load_alt_location='simple_tag_mpc', run=False)
train.session_setup()
n_episode = 0
i=0      
    
while robot.step(TIME_STEP) != -1:
    if n_episode>train.session_pars.N_games: # Session over
        if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history_LRRL_mpc, _ = train.session_wrapup()
            break
        
    else:
        i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)

    
with open('../../../results/'+ENV+'/results_policy_previous_env.pickle', 'wb') as f:
                pickle.dump([history_reg_sa, history_LRRL_sa, history_reg_mpc, history_LRRL_mpc], f)



# TRAIN MADDPG NOT WORKING ============================================================================================================= 




train = Train(ENV, chkpt_dir='/trained_nets/regular/')
train.training(load=True, N_games = 250, greedy=True, decreasing_eps=True, lexi_mode=False, log=False, load_alt_location='simple_tag_mpc', run=False)
train.session_setup()
n_episode = 0
i=0      
    
while robot.step(TIME_STEP) != -1:
    if n_episode>train.session_pars.N_games: # Session over
        if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history_reg_mpc, _ = train.session_wrapup()
            break
        
    else:
        i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)


train = Train(ENV, chkpt_dir='/trained_nets/LRRL4/')
train.training(load=True, N_games = 250, greedy=True, decreasing_eps=True, lexi_mode=True, log=False, load_alt_location='simple_tag_mpc', run=False)
train.session_setup()
n_episode = 0
i=0      
    
while robot.step(TIME_STEP) != -1:
    if n_episode>train.session_pars.N_games: # Session over
        if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history_LRRL_mpc, _ = train.session_wrapup()
            break
        
    else:
        i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)



# TEST ZETA DIFF =================================================================================

zeta_diff = [[]]
zeta_diff_LRRL = [[]]
for i in range(1000):
    train = Train(ENV, chkpt_dir='/trained_nets/regular/')
    train.testing(load=True, N_games = 1, lexi_mode=False, log=False, load_alt_location='simple_tag', run=False)
    train.session_setup()
    n_episode = 0
    i=0      
        
    while robot.step(TIME_STEP) != -1:
        
        if n_episode>train.session_pars.N_games: # Session over
            # if n_episode == train.session_pars.N_games: 
            print('================================================================================================\n Session over')
            history_LRRL_mpc, sequence = train.session_wrapup()
            break
            
        else:
            i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)
    
    
    start_state = random.randint(0, 20)
    agent_idx = random.randint(0,1)
    for j in range(start_state+1, len(sequence)):
        comm, zeta = train.gammanet.communication(sequence[start_state][agent_idx], sequence[j][agent_idx], 0, return_gamma=True, load_net=True)
        diff = j-start_state
        if diff >= len(zeta_diff):
            zeta_diff.append([])
        zeta_diff[diff].append(zeta)

for i in range(1000):
    train = Train(ENV, chkpt_dir='/trained_nets/LRRL4/')
    train.testing(load=True, N_games = 1, lexi_mode=True, log=False, load_alt_location='simple_tag', run=False)
    train.session_setup()
    n_episode = 0
    i=0      
        
    while robot.step(TIME_STEP) != -1:
        if n_episode>train.session_pars.N_games: # Session over
            # if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history_LRRL_mpc, sequence = train.session_wrapup()
            break
            
        else:
            i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)

    start_state = random.randint(0, 20)
    agent_idx = random.randint(0,1)
    for j in range(start_state+1, len(sequence)):
        comm, zeta = train.gammanet.communication(sequence[start_state][agent_idx], sequence[j][agent_idx], 0, return_gamma=True, load_net=True)
        diff = j-start_state
        if diff >= len(zeta_diff_LRRL):
            zeta_diff_LRRL.append([])
        zeta_diff_LRRL[diff].append(zeta)

with open('../../../results/'+ENV+'/results_zeta_diff.pickle', 'wb+') as f:
    pickle.dump([zeta_diff, zeta_diff_LRRL], f)      



# TESTING COMM RETURN ========================================================================================================================



train = Train(ENV, chkpt_dir='/trained_nets/regular/')
train.testing(load=True, N_games = 250, lexi_mode=False, log=False, load_alt_location='simple_tag', run=False)
train.session_setup()
n_episode = 0
i=0      
    
while robot.step(TIME_STEP) != -1:
    if n_episode>train.session_pars.N_games: # Session over
        if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history, _ = train.session_wrapup()
            break
        
    else:
        i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)

mean_regular = np.mean(history, axis=0)
std_regular = np.std(history, axis=0)
worst_regular = np.min(history, axis=0)


# Testing EDI for different zetas
zeta = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 4]
for z in zeta:
    train.testing(edi_mode='test', zeta=z, load=True, N_games = 250, lexi_mode=False, log=False, load_alt_location='simple_tag', run=False)
    train.session_setup()
    n_episode = 0
    i=0      
        
    while robot.step(TIME_STEP) != -1:
        if n_episode>train.session_pars.N_games: # Session over
            if n_episode == train.session_pars.N_games+1: 
                print('================================================================================================\n Session over')
                history, _ = train.session_wrapup()
                break
            
        else:
            i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)

    mean_regular = np.vstack((mean_regular, np.mean(history, axis=0)))
    std_regular = np.vstack((std_regular, np.std(history, axis=0)))
    worst_regular = np.vstack((worst_regular, np.min(history, axis=0)))



train = Train(ENV, chkpt_dir='/trained_nets/LRRL4/')
train.testing(load=True, N_games = 250, lexi_mode=True, log=False, load_alt_location='simple_tag', run=False)
train.session_setup()
n_episode = 0
i=0      
    
while robot.step(TIME_STEP) != -1:
    if n_episode>train.session_pars.N_games: # Session over
        if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history, _ = train.session_wrapup()
            break
        
    else:
        i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)

mean_LRRL = np.mean(history, axis=0)
std_LRRL = np.std(history, axis=0)
worst_LRRL = np.min(history, axis=0)


# Testing EDI for different zetas
zeta = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 4]
for z in zeta:
    train.testing(edi_mode='test', zeta=z, load=True, N_games = 250, lexi_mode=True, log=False, load_alt_location='simple_tag', run=False)
    train.session_setup()
    n_episode = 0
    i=0      
        
    while robot.step(TIME_STEP) != -1:
        if n_episode>train.session_pars.N_games: # Session over
            if n_episode == train.session_pars.N_games+1: 
                print('================================================================================================\n Session over')
                history, _ = train.session_wrapup()
                break
            
        else:
            i, n_episode, obs, obs_, coords, rotation, last_comm, goal, reward, actions, done, vel = while_loop(train, robot, nodes, n_episode, coords, i, HORIZON, obs_, vel, world, obs, actions, reward, done, last_comm, rotation, goal, cars, emitters, TIME_STEP)

    mean_LRRL = np.vstack((mean_LRRL, np.mean(history, axis=0)))
    std_LRRL = np.vstack((std_LRRL, np.std(history, axis=0)))
    worst_LRRL = np.vstack((worst_LRRL, np.min(history, axis=0)))



# Dumping output
with open('../../../results/'+ENV+'/results_edi.pickle', 'wb+') as f:
    pickle.dump([zeta, mean_regular, std_regular, worst_regular, mean_LRRL, std_LRRL, worst_LRRL],f)      