from controller import Robot, DistanceSensor, Motor

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
       
    leftSpeed = 1
    rightSpeed = 1
    # write actuators inputs
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)
    