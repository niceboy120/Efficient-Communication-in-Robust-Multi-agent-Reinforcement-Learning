from controller import Supervisor, Emitter, Receiver
import random
import math
import numpy as np

# import sys
# sys.path.insert(0, '..')
# from main.main import Test

TIME_STEP = 32
HORIZON = 4

# Construct instances
robot = Supervisor()
emitter1 = Emitter('emitter1')
emitter2 = Emitter('emitter2')
emitter3 = Emitter('emitter3')
emitters = [emitter1, emitter2, emitter3]

emitter_RL = Emitter('emitter_RL')
receiver_RL = Receiver('receiver_RL')
receiver_RL.enable(TIME_STEP)
receiver_reset = Receiver('receiver_reset')
receiver_reset.enable(TIME_STEP)



root_node = robot.getRoot()





# Get robot ids
children_field = root_node.getField('children')
adv1_node = robot.getFromDef('ADV1')
adv2_node = robot.getFromDef('ADV2')
agent_node = robot.getFromDef('AGENT')
nodes = [adv1_node, adv2_node, agent_node]


def reset_nodes(nodes, emitters):
    coords = []
    for i in range(len(nodes)):
        node = nodes[i]
        emitter = emitters[i]
        translation_field = node.getField('translation')
        rotation_field = node.getField('rotation')
        translation = [random.uniform(-0.9,0.9), random.uniform(-0.9,0.9), 0]
        rotation = random.uniform(0,6.28)
        translation_field.setSFVec3f(translation)
        rotation_field.setSFRotation([0,0,1,rotation])
        emitter.send([translation[0], translation[1], rotation, translation[0], translation[1]])
        coords.append(translation[0])
        coords.append(translation[1])
    return coords

coords = reset_nodes(nodes, emitters)
emitter_RL.send(coords)


i=0
while robot.step(TIME_STEP) != -1:
    if receiver_reset.getQueueLength()>0 and i != 0:
        robot.simulationResetPhysics()
        coords = reset_nodes(nodes, emitters)
        robot.simulationResetPhysics()
        emitter_RL.send(coords)
        receiver_reset.nextPacket()
        i=0

    if i%HORIZON==0: #HARDCODE HORIZON SOMEWHERE
        emitter_RL.send(coords)
        if receiver_RL.getQueueLength()>0:
            goal = receiver_RL.getFloats()
        # goal = [0.9, 0.9, 0.65, 0.32, 0.12, 0.87]
            receiver_RL.nextPacket()
        else:
            if i==0:
                continue
        
    
    coords = []
    for j in range(len(nodes)):
        node = nodes[j]
        emitter = emitters[j]    
    
        translation = node.getPosition()
        RM = node.getOrientation()
        rotation = math.atan2(RM[3], RM[0])
        
        emitter.send([translation[0], translation[1], rotation, goal[j*2], goal[j*2+1]])
        coords.append(translation[0])
        coords.append(translation[1])

    
    i+=1
    