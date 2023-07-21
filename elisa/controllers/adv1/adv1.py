from controller import Robot, DistanceSensor, Motor, Receiver, Emitter
import numpy as np

import sys
sys.path.insert(0, '../../..')
# from MPC.waypoints import WAYPOINT_CONTROLLER
from MPC.controller import SIMPLE_CONTROLLER
from MPC.car import Car


# time in [ms] of a simulation step
TIME_STEP = 32
HORIZON = 4
MAX_SPEED = 130

# create the Robot instance.
robot = Robot()
emitter = Emitter('emitter')
receiver = Receiver('receiver')
receiver.enable(TIME_STEP)

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

light = robot.getDevice('ledrgb')
light.set(0xff0000)



i=0
# feedback loop: step simulation until receiving an exit event
while robot.step(TIME_STEP) != -1:
    if receiver.getQueueLength() > 0:
        message = receiver.getFloats()
        x = message[0:3]
        goal = message[3:]
        receiver.nextPacket()
    
    if i==0:
        car = Car(x[0], x[1], x[2])
        controller = SIMPLE_CONTROLLER(HORIZON)    
    
    linear_vel, angular_vel = controller.get_control_inputs(np.array([[x[0]],[x[1]],[x[2]]]), goal)
    car.set_robot_velocity(linear_vel, angular_vel)
    vel = [linear_vel*np.cos(x[2]), linear_vel*np.sin(x[2])]
    emitter.send(vel)
    
    
    rightMotor.setVelocity(car.wheel_speed[0])
    leftMotor.setVelocity(car.wheel_speed[1])
    
    i += 1
    