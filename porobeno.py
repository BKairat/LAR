import cv2
import ctypes
# import math
# import time
#
# import numpy as np

from robolab_turtlebot import Turtlebot, Rate, get_time, detector
from image_analysis import analysisRGB, getAngleToColor, isEntry
from map import Map

from move_cmd import rotation, move
import time
from parking_cmd import centering, parking

turtle = Turtlebot(pc=True, depth=True,rgb=True)
rate = Rate(10)

WINDOW = 'obstacles'
bumper_names = ['LEFT', 'CENTER', 'RIGHT']
state_names = ['RELEASED', 'PRESSED']
BUMPER = False


def bumper_cb(msg):
    """Bumber callback."""
    global BUMPER
    # msg.bumper stores the id of bumper 0:LEFT, 1:CENTER, 2:RIGHT
    bumper = bumper_names[msg.bumper]

    # msg.state stores the event 0:RELEASED, 1:PRESSED
    state = state_names[msg.state]
    if state == "PRESSED":
        BUMPER = True

    # Print the event
    print('{} bumper {}'.format(bumper, state))
    if BUMPER:
        print("YAAAAAAAAAAAAAAAAAAAAY!!!!!!")
        quit()


def main():
    # turtle = Turtlebot()
    global BUMPER
    # cv2.namedWindow(WINDOW)
    # # cv2.setMouseCallback(WINDOW, click)

    

    while not turtle.is_shutting_down():
        # get point cloud
        image = turtle.get_rgb_image()

        # wait for image to be ready
        if image is None:
            continue
        ang = getAngleToColor(image)
        print(ang)
        if ang[0] < 3 and ang[0] > -3:
            break
        # detect markers in the image
        markers = detector.detect_markers(image)

        # draw markers in the image
        detector.draw_markers(image, markers)
        if ang[0] == 45:
                turtle.cmd_velocity(angular = ang[1])    
        else:
            turtle.cmd_velocity(angular = ang[1]*0.3)
        # show image
        cv2.imshow(WINDOW, image)
        cv2.waitKey(1)

    time.sleep(1)    

    turtle.wait_for_rgb_image()
    img = turtle.get_rgb_image()



    while True:
        mapa = Map()
        mapa.mapClear()
        garage, obstacles = analysisRGB(img)
        for i in obstacles:
            for j in i:
                mapa.drawObstacle(j.dist, j.angleMap, j.color)
                # cv2.drawContours(img, [j.contour], 0, (0, 255, 0), 1)
        if garage.angleP:
            mapa.drawGarage(garage.dist[0], garage.angleMap[0], garage.angleP, garage.entry)
            # for i in garage.contour:
            #     cv2.drawContours(img, [i], -1, (0, 255, 0), 1)
        elif garage.entry:
            mapa.drawGarage(garage.dist, garage.angleMap, garage.angleP, garage.entry)
        else:
            mapa.drawGarage(garage.dist, garage.angleMap, garage.angleP, garage.entry)

        dist, rot = mapa.bfs()
        mapa.drawRobot()

        print('distance array:', dist, 'rotation array:', rot)

        # for i in garage.contour:
        #     cv2.drawContours(img, [i], -1, (0, 255, 0), 3)
        mapa.showMap()

        for i in range(len(dist)):
            rotation(rot[i], get_time(), turtle)
            move(dist[i] / 100, get_time(), turtle)
            if BUMPER:
                print('bumper')
                quit()
        
        image = 0
        while not turtle.is_shutting_down():
            # get point cloud
            image = turtle.get_rgb_image()

            # wait for image to be ready
            if image is None:
                continue
            ang = getAngleToColor(image)
            print(ang)
            if ang[0] < 3 and ang[0] > -3:
                break
            # detect markers in the image
            markers = detector.detect_markers(image)

            # draw markers in the image
            detector.draw_markers(image, markers)
            if ang[0] == 45:
                turtle.cmd_velocity(angular = ang[1])    
            else:
                turtle.cmd_velocity(angular = ang[1]*0.3)
            # show image
            cv2.imshow(WINDOW, image)
            cv2.waitKey(1)
        if isEntry(image):
            break

    centering(turtle)
    parking(turtle)

if __name__ == '__main__':
    main()