import cv2
import time

from robolab_turtlebot import Turtlebot, Rate, get_time, detector
from image_analysis import getAngleToColor

from move_cmd import rotation

angular_vel = 0.1
linear_vel = 0.1

WINDOW = 'obstacles'

turtle = Turtlebot(pc=True, depth=True, rgb=True)
rate = Rate(10)


def areas(turtle) -> list:
    depth_image = turtle.get_depth_image()
    total_area_values = []
    area_value1 =0
    area_value2 =0 
    area_value3 =0 
    for row in range(250,260):
        for col in range(20):
            area_value1 += depth_image[row][319+col]   
            area_value2 += depth_image[row][489+col]
            area_value3 += depth_image[row][419+col]
    total_area_values.append(area_value1/200)
    total_area_values.append(area_value2/200)
    total_area_values.append(area_value3/200)

    return total_area_values


def centering(turtle)  -> int:
    cnt = 0
    while cnt < 2 and not turtle.is_shutting_down():
        turtle.wait_for_rgb_image()
        img = turtle.get_rgb_image()
        ang = getAngleToColor(img, 'VIOLET')
        print('ang:',ang)
        if ang[0] < 3 and ang[0] > -3:
            time.sleep(0.5)
            total_area_values = areas(turtle)
            while min(total_area_values) > 400 and not turtle.is_shutting_down():
                total_area_values = areas(turtle)
                print(total_area_values)
                turtle.cmd_velocity(linear=linear_vel)
            time.sleep(0.5)
            rotation(-30, get_time(), turtle)
            cnt += 1
            print('cnt:', cnt)
        # detect markers in the image
        markers = detector.detect_markers(img)
        # draw markers in the image
        detector.draw_markers(img, markers)
        if ang[0] == 45:
            turtle.cmd_velocity(angular = ang[1])    
        else:
            turtle.cmd_velocity(angular = ang[1]*angular_vel*2)
        # show image
        cv2.imshow(WINDOW, img)
        cv2.waitKey(1)
    return 0


def align(turtle) -> int:
    total_area_values = areas(turtle)
    # print("TAV:", total_area_values)

    if abs(total_area_values[0] - total_area_values[1]) < 10 and max(total_area_values) < 1000:
        turtle.wait_for_rgb_image()
        img = turtle.get_rgb_image()
        if getAngleToColor(img, 'YELLOW')[0] != 45:
            time.sleep(1)
            return 1
    
    turtle.cmd_velocity(angular = angular_vel + max(total_area_values)/2000) 
    return 0


def park(turtle) -> int:
    total_area_values = areas(turtle)
    print("TAV:",total_area_values) 
    if min(total_area_values) < 320: #and check_park(turtle,depth_image): 
        time.sleep(0.2)
        return 2 
    turtle.cmd_velocity(linear=linear_vel) 
    return 1


def parking(turtle):  
    state = 0
    while not turtle.is_shutting_down() and state != 2:  
        #Image dimension: 848W 480H 
        if state == 0:
            state = align(turtle)
            print('align ok')
        if state == 1: 
            state = park(turtle)
            print('park ok')
    if check_park(turtle):
        return state
    else:
        print('not right')
        rotation(10, get_time(), turtle)
        parking(turtle)


def check_park(turtle) -> bool:
    rotation(90, get_time(), turtle)

    turtle.wait_for_rgb_image()
    img = turtle.get_rgb_image()
    if getAngleToColor(img, 'YELLOW')[0] == 45: # not find yellow
        rotation(-90, get_time(), turtle)
        print('false')
        return False

    rotation(-180, get_time(), turtle)

    turtle.wait_for_rgb_image()
    img = turtle.get_rgb_image() 
    if getAngleToColor(img, 'YELLOW')[0] == 45: # not find yellow
        rotation(90, get_time(), turtle)
        print('false')
        return False
        
    rotation(90, get_time(), turtle)
    return True