from controller import Supervisor, Emitter
import random
import math
import numpy as np

import sys
sys.path.insert(0, '..')
# from MPC.waypoints import WAYPOINT_CONTROLLER
from MPC.test import test



# Construct instances
robot = Supervisor()
emitter1 = Emitter('emitter1')
emitter2 = Emitter('emitter2')
emitter3 = Emitter('emitter3')
emitters = [emitter1, emitter2, emitter3]
# controller = WAYPOINT_CONTROLLER()
test = test()
root_node = robot.getRoot()



# Parameters
TIME_STEP = 32



# Get robot ids
children_field = root_node.getField('children')
adv1_node = robot.getFromDef('ADV1')
adv2_node = robot.getFromDef('ADV2')
agent_node = robot.getFromDef('AGENT')
nodes = [adv1_node, adv2_node, agent_node]


def reset_nodes(nodes, emitters):
    for i in range(len(nodes)):
        node = nodes[i]
        emitter = emitters[i]
        translation_field = node.getField('translation')
        rotation_field = node.getField('rotation')
        translation = [random.uniform(-0.9,0.9), random.uniform(-0.9,0.9), 0]
        rotation = random.uniform(0,6.28)
        translation_field.setSFVec3f(translation)
        rotation_field.setSFRotation([0,0,1,rotation])
        emitter.send([translation[1], translation[0], rotation, translation[1], translation[0]])

reset_nodes(nodes, emitters)

i=0
while robot.step(TIME_STEP) != -1:
    if i%2==0:
        goal1, goal2, goal3 = test.increase_waypoint()
        goal = [goal1, goal2, goal3]
        
    for j in range(len(nodes)):
        node = nodes[j]
        emitter = emitters[j]    
    
        translation = node.getPosition()
        RM = node.getOrientation()
        rotation = math.atan2(RM[3], RM[0])

        emitter.send([translation[1], translation[0], rotation, goal[j][0], goal[j][1]])

    
    i+=1
    