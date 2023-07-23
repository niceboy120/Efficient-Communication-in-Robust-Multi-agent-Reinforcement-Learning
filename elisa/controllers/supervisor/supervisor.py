from controller import Supervisor, Emitter
import random
import math
import pickle
import numpy as np

import sys
sys.path.insert(0, '../../..')

from train_agent import Train
from MPC.controller import SIMPLE_CONTROLLER
from MPC.car import Car
from MPE.multiagent.scenarios.simple_tag import Scenario


TIME_STEP = 32
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
        translation = [random.uniform(-0.9,0.9), random.uniform(-0.9,0.9), 0]
        rotation = random.uniform(0,6.28)
        translation_field.setSFVec3f(translation)
        rotation_field.setSFRotation([0,0,1,rotation])
        
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
    
coords, rotation = reset_nodes(nodes)
vel = [0,0,0,0,0,0]

cars = []
for n in range(len(nodes)):
    cars.append(Car(coords[n*2], coords[n*2+1], rotation[n]))


ENV = 'simple_tag_webots' # 1: simple_tag, 2: simple_tag_elisa, 3: simple_tag_mpc, 3: simple_tag_webots
train = Train(ENV, chkpt_dir='/trained_nets/regular/')
train.testing(load=True, N_games = 5, greedy=True, decreasing_eps=True, lexi_mode=False, log=False, load_alt_location='simple_tag', run=False)

train.session_setup()
n_episode = 0

while robot.step(TIME_STEP) != -1:
    if n_episode>train.session_pars.N_games: # Session over
        if n_episode == train.session_pars.N_games+1: 
            print('================================================================================================\n Session over')
            history, _ = train.session_wrapup()
            with open('../../../results/'+ENV+'/results_train_regular.pickle', 'wb') as f:
                pickle.dump(history, f)
            n_episode += 1
        
    else:
        if train.session_pars.next_episode: # Episode over
            if n_episode != 0:
                train.episode_wrapup(n_episode)  
                robot.simulationResetPhysics()
                coords, rotation = reset_nodes(nodes)
                robot.simulationResetPhysics()
            
            n_episode += 1
            obs, last_comm = train.episode_setup(n_episode, coords) # obs_webots = [adv1x, adv1y, adv2x, adv2y, agentx, agenty]
            train.session_pars.next_episode = False
            
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
    