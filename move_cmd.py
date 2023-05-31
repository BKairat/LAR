from robolab_turtlebot import Turtlebot, Rate, get_time  
import math
import time 

turtle = Turtlebot(rgb=True)

linear_vel = 0.2
angular_vel = 0.314 
 
rate = Rate(10) 


def angleApprox(angle: float) -> float:
    return (angle+61.7)/1.68 


def distApprox(distance: float) -> float:
    return (distance-0.017)/0.86


def rotTime(angle: float) -> float:
    return (angleApprox(angle) * math.pi / (180 * 0.314))*2.2


def rotation(angle: float, start_time: float, turtle):
    if angle == 0:
        # stay
        return

    if angle > 0: 
        # left rotation    
        while get_time()-start_time < rotTime(angle):  #180/pi * 0.314 * 5(time) = 89.9
            turtle.cmd_velocity(angular = angular_vel) 
            rate.sleep() 
             
    if angle < 0:
        # right rotation 
        while get_time()-start_time < rotTime(-angle):  #180/pi * 0.314 * 5(time) = 89.9
            turtle.cmd_velocity(angular = -angular_vel) 
            rate.sleep()
    time.sleep(1)  


def move(distance: float, start_time: float, turtle):                    #1m walk (distance in meters)
    
    if distance < 0.03 and distance > -0.03:
        distance = 0.025 
    
    if distance == 0:
        # stay
        return 

    if distance > 0:
        # forward  
        while get_time() - start_time < distApprox(distance) *5:
            turtle.cmd_velocity(linear=linear_vel)  
            rate.sleep() 
        time.sleep(1) 
        
    if distance < 0: 
        # backward 
        while get_time() - start_time < distApprox(-distance) *5:
            turtle.cmd_velocity(linear=-linear_vel)  
            rate.sleep() 
        time.sleep(1)  
 
 