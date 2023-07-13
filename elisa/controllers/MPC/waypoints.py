import math
import numpy as np
from MPC.controller import BEUN
from MPC.car import Car

class WAYPOINT_CONTROLLER:
    waypoint_adv1 = []
    waypoint_adv2 = []
    waypoint_agent = []   
    
    def initialize(self, horizon, car_id, robot, timestep):
        self.controller = BEUN(horizon)               
        
        if car_id == 0:            
            self.gps = robot.getDevice('gps_adv1')
            self.compass = robot.getDevice('compass_adv1')
        elif car_id == 1:            
            self.gps = robot.getDevice('gps_adv2')
            self.compass = robot.getDevice('compass_adv2')
        elif car_id == 2:
            self.gps = robot.getDevice('gps_agent')
            self.compass = robot.getDevice('compass_agent')  
            
        self.gps.enable(timestep)    
        self.compass.enable(timestep)
        self.get_coords()

        self.car = Car(self.y, self.x, self.phi)   
        
        if car_id == 0:            
            self.set_waypoint(waypoint_adv1 = [self.y, self.x])
        elif car_id == 1:            
            self.set_waypoint(waypoint_adv2 = [self.y, self.x])
        elif car_id == 2:
            self.set_waypoint(waypoint_agent = [self.y, self.x])  
                  
           
    def get_wheel_speed(self):
        self.get_coords()
        linear_vel, angular_vel = self.controller.get_control_inputs(np.array([[self.y],[self.x],[self.phi]]), WAYPOINT_CONTROLLER.waypoint_adv1)
        self.car.set_robot_velocity(linear_vel, angular_vel)
        return self.car.wheel_speed
        
    def set_waypoint(self, waypoint_adv1=None, waypoint_adv2=None, waypoint_agent=None):
        if waypoint_adv1 != None:
            WAYPOINT_CONTROLLER.waypoint_adv1 = waypoint_adv1
        
        if waypoint_adv2 != None:
            WAYPOINT_CONTROLLER.waypoint_adv2 = waypoint_adv2
        
        if waypoint_agent != None:
            WAYPOINT_CONTROLLER.waypoint_agent = waypoint_agent
        
    def get_coords(self):
        gps_values = self.gps.getValues()
        self.x = gps_values[1]
        self.y = gps_values[0]
        self.phi = self.get_bearing_in_radians()
                  
    def get_bearing_in_degrees(self):
        north = self.compass.getValues()
        radians = math.atan2(north[1], north[0])
        bearing = (radians) / math.pi * 180
        if bearing<0:
            bearing = bearing +360
        return bearing
        
    def get_bearing_in_radians(self):
        north = self.compass.getValues()
        bearing = math.atan2(north[1], north[0])
        if bearing<0:
            bearing = bearing + 2*math.pi
        return bearing