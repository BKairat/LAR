"""
This code used for photo analysis, this file will be updated.
"""

import cv2
import numpy as np
from geometry import square, boxApproximation, lineFrom2Dots, getCenter2Dots
from angles import predictAngle, angle4Map
from distance import dist


colors = {"VIOLET":     [np.array([116, 43, 48]),    np.array([165, 134, 138])],  # garage
          "RED":        [np.array([0, 157, 131]),   np.array([12, 255, 255])],
          "GREEN":      [np.array([40, 61, 92]),    np.array([72, 178, 202])],
          "BLUE":       [np.array([90, 90, 40]),    np.array([116, 255, 255])],
          "YELLOW":     [np.array([20, 129, 116]),   np.array([34, 255, 244])]}

MINAREA = 3000

class Garage:
    def __init__(self, dist: list, angleMap: list, angleP: float, contour: tuple, entry: str):
        self.dist = dist
        self.angleMap = angleMap
        self.angleP = angleP
        self.contour = contour
        self.entry = entry


class Obstacle:
    def __init__(self, color: str, dist: float, angleMap: float, contour: tuple):
        self.color = color
        self.dist = dist
        self.angleMap = angleMap
        self.contour = contour


def compareSquares(box: list, MIN_square: int = 500) -> bool:
    """
    :param box: list with 4 points of square
    :param MIN_square: parameter with witch we will compare square
    :return: True if square of box > MIN_square else False
    """
    return square(box) > MIN_square


def getCenters4Quadrls(quadrilaterals: list) -> list:
    """
    :param quadrilaterals: list of quadrilaterals
    :return: list of quadrilaterals centers [(x1, y1), ...]
    """
    centers = []
    for coords in quadrilaterals:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = coords
        C = (int((x1+x2+x3+x4)/4), int((y1+y2+y3+y4)/4))
        centers.append(C)
    return centers


def getAngleToColor(img: 'numpy.ndarray', color: str = "YELLOW") -> tuple:
    """
    Check is color on image and calculate the angle by which the robot needs to turn.
    :param img: image which we interested in.
    :param color: color which we want to find.
    :return: angle (float), 1 if we have to do left turn -1 if right turn.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # we convert img into hsv format

    THRESH = cv2.inRange(hsv, colors[color][0], colors[color][1])

    CONTOURS = cv2.findContours(THRESH.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    imgCenter = np.shape(img)[1]/2

    objects = []
    for cont in CONTOURS[0]:
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if compareSquares(box):
            objects.append(box)
    approx = []
    for box in objects:
        approx.append(boxApproximation(box))
    if len(getCenters4Quadrls(objects)) == 0:
        return (45, -1)
    center = sum([i[0] for i in getCenters4Quadrls(objects)]) / len(getCenters4Quadrls(objects))
    return (-angle4Map(center), 1 if (center - imgCenter) <= 0 else -1)


def fromPolygonToSqare(figure: 'numpy.ndarray') -> list:
    """
    Divides the contour into quadrilateral contours.
    :param figure: contour [[[x1, y1]], ...]
    :return: list of quadrilateral contours.
    """
    areas = [[], []]
    if len(figure) == 5:
        central_point = max(figure, key=lambda x: x[0][1])
        for i in figure:
            if i[0][0] <= central_point[0][0]:
                areas[0].append(i)
            if i[0][0] >= central_point[0][0]:
                areas[1].append(i)

        k, b = lineFrom2Dots(min(areas[0], key=lambda x: x[0][1])[0], min(areas[1], key=lambda x: x[0][1])[0])
        ToP = np.array([central_point[0][0], round(k * central_point[0][0] + b)])
        for i in range(len(areas)):
            areas[i].append([ToP])
        contour1 = np.array([i for i in areas[0]])
        contour2 = np.array([i for i in areas[1]])

        return[contour1 if cv2.contourArea(contour1) > MINAREA else 0,
               contour2 if cv2.contourArea(contour2) > MINAREA else 0]
    elif len(figure) == 7:
        fig_list = figure.tolist()
        fig_list.sort(key=lambda x: x[0][0])
        areas[0] = fig_list[:2]
        areas[1] = fig_list[-2:]
        fig_list.sort(key = lambda x: x[0][1])

        k, b = lineFrom2Dots(min(fig_list[:3], key=lambda x: x[0][0])[0], max(fig_list[:3], key=lambda x: x[0][0])[0])
        down_dots = fig_list[3:]
        down_dots.sort(key = lambda x: x[0][0])
        down_dots = down_dots[1:3]
        areas[0].append([down_dots[0][0]])
        areas[1].append([down_dots[1][0]])
        new_top_dots = [[[down_dots[0][0][0], round(k*down_dots[0][0][0] + b)]],
                        [[down_dots[1][0][0], round(k*down_dots[1][0][0] + b)]]]
        areas[0].append(new_top_dots[0])
        areas[1].insert(2, new_top_dots[1])
        areas[1].sort(key = lambda x: x[0][0])
        areas[1].reverse()
        contour1 = np.array([np.array(i) for i in areas[0]])
        contour2 = np.array([np.array(i) for i in areas[1]])

        return [contour1 if cv2.contourArea(contour1) > MINAREA else 0,
               contour2 if cv2.contourArea(contour2) > MINAREA else 0]


def defineObstacles(color: str, contours: tuple) -> list:
    """
    Receive color of obstacle and their contours and
    returns list of obstacles (class Obstacle).
    :param color: color of obstacles
    :param contours: list of contours of obstacles
    :return: list of obstacles (class Obstacle)
    """
    objects = []
    for cont in contours:
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if compareSquares(box):
            objects.append(box)
    approx = []
    for box in objects:
        approx.append(boxApproximation(box))

    centerObsts = getCenters4Quadrls(objects)
    ret = []
    for i in range(len(centerObsts)):
        ret.append(Obstacle(color, dist(approx[i], "o"), angle4Map(centerObsts[i][0]), objects[i]))
    return ret


def side(pole: list, objectsY: list) -> str:
    """
    Used in case when we cen see only one violet column of garage.
    :param pole: list of vertices of the quadrilateral.
    :param objectsY: list of 1 and 0.
    :return: LEFT, RIGHT, or TOP depending on where the entrance to the
             garage is relative to the column.
    """
    right = False
    left = False
    poleC = pole.copy()
    poleC.sort(key = lambda x: x[0])
    centr1 = getCenter2Dots(poleC[:2])
    centr2 = getCenter2Dots(poleC[2:])

    for i in objectsY[0]:
        if i[0][0] < centr1[0]:
            left = True
        elif i[0][0] > centr2[0]:
            right = True

    if right and left:
        return "TOP"
    elif right:
        return "RIGHT"
    elif left:
        return "LEFT"


def areaApprox(contoursV: tuple, contoursY: tuple) -> list:
    """
    Used for approximating garage contours.
    :param contoursV: VIOLET contours.
    :param contoursY: YELLOW contours.
    :return: list of quadrilateral.
    """
    objectsV = []
    objectsY = []
    for cont in contoursV:
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if compareSquares(box):
            objectsV.append(box)

    for cont in contoursY:
        if cv2.contourArea(cont) > MINAREA:
            # calc arclentgh
            arclen = cv2.arcLength(cont, True)

            # do approx
            eps = 0.01
            epsilon = arclen * eps
            area = np.array(cv2.approxPolyDP(cont, epsilon, True))
            polygon_vertexies = len(area)

            while polygon_vertexies > 7:
                epsilon = arclen * eps
                area = np.array(cv2.approxPolyDP(cont, epsilon, True))
                polygon_vertexies = len(area)
                eps += 0.001
            try:
                if (len(area)) > 4:
                    for ar in fromPolygonToSqare(area):
                        if type(ar) != int:
                            objectsY.append(ar)
                elif len(area) == 4:
                    objectsY.append(area)
            except:
                continue
    return [objectsV, objectsY]


def defineGarage(contoursV: tuple, contoursY: tuple, img) -> Garage:
    """
    Define garage (class GARAGE) by its contours
    :param contoursV: VIOLET contours.
    :param contoursY: YELLOW contours.
    :param img:
    :return: object of class Garage.
    """
    objectsV, objectsY = areaApprox(contoursV, contoursY)

    if len(objectsV) == 2:
        approx = []
        for box in objectsV:
            approx.append(boxApproximation(box))

        centerCols = getCenters4Quadrls(objectsV)

        center = [(int((centerCols[0][0] + centerCols[0][1]) / 2), int((centerCols[1][0] + centerCols[1][1]) / 2))]

        return Garage([dist(approx, "gv")], [angle4Map(center[0][0])], predictAngle(approx), objectsV, "DOWN")
    elif len(objectsV) == 0:
        obj_coords = [[] for i in range(len(objectsY))]
        for i in range(len(objectsY)):
            for j in range(len(objectsY[i])):
                obj_coords[i].append(objectsY[i][j][0].tolist())

        for i in objectsY:
            cv2.drawContours(img, [i], -1, (0, 0, 255), 2, cv2.LINE_AA)

        dists = []
        angles = []
        for sqr in obj_coords:
            sqr.sort(key = lambda x: x[0])
            for dot in range(0, len(sqr), 2):
                dists.append(dist(sqr[dot:dot+2], "gy"))
                angles.append(angle4Map(getCenter2Dots(sqr[dot:dot + 2])[0]))

        return Garage(dists, angles, None, objectsY, None)
    elif len(objectsV) == 1:
        cv2.drawContours(img, [objectsV[0]], -1, (0, 0, 255), 2, cv2.LINE_AA)
        cur_pole = objectsV[0].tolist()

        cur_pole.sort(key = lambda x: x[0] ** 2 + x[1] ** 2)

        dists = []
        angles = []

        cur_pole.sort(key=lambda x: x[0])
        for dot in range(0, len(cur_pole), 2):
            dists.append(dist(cur_pole[dot:dot + 2], "gy"))
            angles.append(angle4Map(getCenter2Dots(cur_pole[dot:dot + 2])[0]))
        print('hehehehe')
        print(dists, angles)
        return Garage(dists, angles, None, objectsV, side(cur_pole, objectsY))


def isEntry(img: 'numpy.ndarray') -> bool:
    """
    Decides if robot see the entry
    :param img: image that robot have received
    :return: True if there are violet columns
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # we convert img into hsv format

    thresh = cv2.inRange(hsv, colors["VIOLET"][0], colors["VIOLET"][1])

    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = 0
    for cont in contours[0]:
        if cv2.contourArea(cont) > MINAREA/3:
            cnt += 1
    return cnt >= 2


def analysisRGB(img: 'numpy.ndarray') -> tuple:
    """
    Find contours of all colours, and return garage and list of obstacles.
    :param img: image that robot have received.
    :return: garage and list of obstacles.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # we convert img into hsv format

    THRESH = [cv2.inRange(hsv, colors[color][0], colors[color][1]) for color in colors]

    CONTOURS = [cv2.findContours(THRESH[i].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) for i in range(len(colors))]

    obstacles = [defineObstacles("RED", CONTOURS[1][0]),
                 defineObstacles("GREEN", CONTOURS[2][0]),
                 defineObstacles("BLUE", CONTOURS[3][0])]

    garage = defineGarage(CONTOURS[0][0], CONTOURS[4][0], img)

    return (garage, obstacles)
