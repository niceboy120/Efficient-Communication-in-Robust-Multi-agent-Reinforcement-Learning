from controller import Robot, DistanceSensor, Motor, GPS, Compass

import sys
sys.path.insert(0, '..')
from MPC.waypoints import WAYPOINT_CONTROLLER


# time in [ms] of a simulation step
TIME_STEP = 32
MAX_SPEED = 130

# create the Robot instance.
robot = Robot()

leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)






# feedback loop: step simulation until receiving an exit event
while robot.step(TIME_STEP) != -1:
    # initialize motor speeds at 50% of MAX_SPEED.

    # print(gps.getValues()) #xyz
    # print(get_bearing_in_degrees()) # clockwise from y/north
    
    
    
    controller = WAYPOINT_CONTROLLER()
    controller.initialize(2, 0, robot, TIME_STEP)
    
    controller.set_waypoint(waypoint_adv1 = [0.5, 0.5])
    
    wheel_speed = controller.get_wheel_speed()
    # write actuators inputs
    rightMotor.setVelocity(wheel_speed[0])
    leftMotor.setVelocity(wheel_speed[1])
    