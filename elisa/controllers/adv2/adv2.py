from controller import Robot, DistanceSensor, Motor, Receiver

# time in [ms] of a simulation step
TIME_STEP = 32

# create the Robot instance.
robot = Robot()
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

wheel_speed = [0,0]

# feedback loop: step simulation until receiving an exit event
while robot.step(TIME_STEP) != -1:
    if receiver.getQueueLength() > 0:
        wheel_speed = receiver.getFloats()
        receiver.nextPacket()
    
    rightMotor.setVelocity(wheel_speed[0])
    leftMotor.setVelocity(wheel_speed[1])