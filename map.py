import matplotlib.pyplot as plt
import numpy as np
import cv2

from math import pi, tan, sqrt, acos, fabs, asin
from geometry import dist2Points, lineFrom2Dots4Map,\
    getCenter2Dots, getUnitVecrot, getAngeleBetweenArrays
from collections import deque


MAPWIDTH = 400
MAPLENGHT = 400

GDIST = 45

GARAGEW = 58
GARAGEH = 48

DELTANGLE = 30

ROBOT_X = 200
ROBOT_Y = 375
ROBOT_DIAGONAL = 34
OFFSET = 20
OBSTACLE_DIAGONAL = 4

#colors in BGR format
colors = {"WHITE":      (0,   [255, 255, 255]),
          "VIOLET":     (1,   [255, 0, 127]),
          "RED":        (2,   [0, 0, 255]),
          "GREEN":      (3,   [0, 255, 0]),
          "BLUE":       (4,   [255, 0, 0]),
          "YELLOW":     (5,   [0, 255, 255]),
          "ROBOT":      (6,   [0, 0, 0]),
          "WAY":        (7,   [225, 105, 65]),
          "ZONE":       (8,   [136, 189, 255])
          }

MAP = [[0 for i in range(MAPWIDTH)] for j in range(MAPLENGHT)]


class Map(object):
    mapr = [[0 for i in range(MAPWIDTH)] for j in range(MAPLENGHT)]

    def __init__(self):
        # super().__init__()
        self.goal = (0, 0)
        self.position = "RIGHT"
        self.mapInt = [[0 for i in range(MAPWIDTH)] for j in range(MAPLENGHT)]
        self.MAPgr = []

    def getApprPath(self) -> tuple:
        """
        Do approximating the path that was found by bfs algorithm.
        :return: distances and angles, acording to them robot will move.
        """
        self.drawRobot()
        self.MAPgr = []
        for i in range(MAPLENGHT):
            self.MAPgr.append([])
            for j in range(MAPWIDTH):
                self.MAPgr[i].append(list(colors.values())[self.mapr[i][j]][1])

        self.MAPgr = np.array(self.MAPgr).astype(np.uint8)
        hsv = cv2.cvtColor(self.MAPgr, cv2.COLOR_BGR2HSV)
        THRESH = cv2.inRange(hsv, np.array([110, 175, 220]), np.array([120, 185, 230]))

        CONTOURS = cv2.findContours(THRESH.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ar = []
        for cont in CONTOURS[0]:
            arclen = cv2.arcLength(cont, True)

            # do approx
            eps = 0.01
            epsilon = arclen * eps
            area = np.array(cv2.approxPolyDP(cont, epsilon, True))
            polygon_vertexies = len(area)
            ar.append(area)

        a = ar[0].tolist()[:int(len(ar[0]) / 2 + 1)]
        a.reverse()
        dist = []
        vec = [np.array([0, -1])]
        rot = []
        for i in range(0, len(a) - 1):
            dist.append(dist2Points(a[i][0][0], a[i][0][1], a[i + 1][0][0], a[i + 1][0][1]))
            vec.append(getUnitVecrot(a[i][0], a[i + 1][0]))

        for i in range(len(vec) - 1):
            rot.append(getAngeleBetweenArrays(vec[i], vec[i + 1]))

        return dist, rot


    def drawCircle(self, x: int, y: int, r: int, c: str):
        """
        Insert int from 0 to 8 (depends on circle color) to mapInt
        :param x: x coordinate of circle center.
        :param y: y coordinate of circle center.
        :param r: circle radius.
        :param c: circle color.
        :return: mapInt with inserted circle.
        """
        for yi in range(y - r, y + r):
            for xi in range(x - r, x + r):
                # if ((x - xi) ** 2 + (y - yi) ** 2) <= r ** 2:
                    try:
                        if (xi < 400 and xi > 0) and (yi < 400 and yi > 0) and (self.mapr[yi][xi] == 0 or self.mapr[yi][xi] == colors["ZONE"][0]):
                            self.mapr[yi][xi] = colors[c][0]
                    except IndexError:
                        continue
        return self.mapInt

    def drawDotWithZone(self, x: float, y: float, color: str):
        self.drawCircle(round(x), round(y), round(OBSTACLE_DIAGONAL / 2), color)
        self.drawCircle(round(x), round(y), round((ROBOT_DIAGONAL + OFFSET) / 2), "ZONE")

    def defineGoal(self, point1: tuple, point2: tuple, side: str) -> tuple:
        k, b = lineFrom2Dots4Map(point1, point2)
        ko = -1/k if k != 0 else -99999999
        x, y = getCenter2Dots([point1, point2])
        b = y - ko * x
        xg, yg = x, y
        while dist2Points(x, y, xg, yg) < GDIST:
            if side == "TOP" or side == "DOWN":
                yg -= 1 if side == "TOP" else -1
                xg = (yg - b) / ko
            else:
                xg -= 1 if side == "LEFT" else -1
                yg = ko * xg + b
        return (round(xg), round(yg))

    def drawSquareBy2Dots(self, point1: tuple, point2: tuple, h: int, side: str = "LEFT", entry: str = "TOP"):
        k, b = lineFrom2Dots4Map(point1, point2)
        if entry != "DOWN":
            for x in range(point1[0], point2[0]):
                y = round(k * x + b)
                self.drawDotWithZone(x, y, "YELLOW")

        # k, b = lineFrom2Dots4Map(point1, point2)
        ko = -1/k if k != 0 else -99999999
        b1 = point1[1] - ko * point1[0]
        b2 = point2[1] - ko * point2[0]
        x1, y1 = point1
        x2, y2 = point2

        while dist2Points(x1, y1, point1[0], point1[1]) < h and dist2Points(x2, y2, point2[0], point2[1]) < h:
            y1 -= 1
            y2 -= 1
            x1 = (y1 - b1) / ko
            x2 = (y2 - b2) / ko
            if entry == "TOP" or entry == "DOWN" or entry == "RIGHT":
                self.drawDotWithZone(x1, y1, "YELLOW")
            if entry == "TOP" or entry == "DOWN" or entry == "LEFT":
                self.drawDotWithZone(x2, y2, "YELLOW")
        if entry == "RIGHT" or entry == "LEFT" or entry == "DOWN":
            x, y = x1, y1
            b = y - k*x
            while dist2Points(x1, y1, x2, y2) > dist2Points(x, y, x1, y1):
                x += 1 if x1 < x2 else -1
                y = k*x + b
                self.drawDotWithZone(x, y, "YELLOW")
        if entry == "RIGHT":
            self.goal = self.defineGoal((point2[0], point2[1]), (x2, y2), entry)
        elif entry == "LEFT":
            self.goal = self.defineGoal((x1, y1), (point1[0], point1[1]), entry)
        elif entry == "TOP":
            self.goal = self.defineGoal((x1, y1), (x2, y2), entry)
        elif entry == "DOWN":
            self.goal = self.defineGoal(point1, point2, entry)


    def drawRobot(self):
        radius = int(ROBOT_DIAGONAL / 2)
        self.drawCircle(ROBOT_X, ROBOT_Y, radius, "ROBOT")

    def drawObstacle(self, dist: float, angle: float, color: str):
        omega = 90 + angle

        b = ROBOT_Y - tan(pi * (omega / 180)) * ROBOT_X

        x, y = ROBOT_X, ROBOT_Y
        while dist2Points(ROBOT_X, ROBOT_Y, x, y) < dist:
            y -= 1
            x = (y - b) / tan(pi * (omega / 180))
        self.drawDotWithZone(x, y, color)
        return (x, y)

    def drawGarage(self, dist: float, angleMap: float, angleP: float, entry: str):
        if entry == "DOWN":
            omega = 90 + angleMap

            k0 = tan(pi * (omega / 180))
            b = ROBOT_Y - k0 * ROBOT_X

            x, y = ROBOT_X, ROBOT_Y
            while dist2Points(ROBOT_X, ROBOT_Y, x, y) < dist:
                y -= 1
                x = (y - b) / k0

            zeta = -270 + angleP
            k1 = tan(pi * (zeta / 180))
            b = y - k1 * x
            k10 = -1 / k1 if k1 != 0 else -99999999
            xg, yg = x, y
            b0 = yg - k10 * xg
            while dist2Points(xg, yg, x, y) < 30:
                yg += 1
                xg = (yg - b0) / k10

            x1viol, y1viol = x2viol, y2viol = x, y
            while dist2Points(x, y, x1viol, y1viol) < GARAGEW / 2:
                x1viol -= 1
                y1viol = k1 * x1viol + b
                x2viol += 1
                y2viol = k1 * x2viol + b

            self.drawDotWithZone(x1viol, y1viol, "VIOLET")
            self.drawDotWithZone(x2viol, y2viol, "VIOLET")

            self.drawSquareBy2Dots((x1viol, y1viol), (x2viol, y2viol), GARAGEH, entry,"DOWN")
        elif entry:
            point1 = self.drawObstacle(dist[0], angleMap[0], "WHITE")
            point2 = self.drawObstacle(dist[1], angleMap[1], "WHITE")
            dot1 = getCenter2Dots([point1, point2])
            k, b = lineFrom2Dots4Map(point1, point2)
            self.drawDotWithZone(dot1[0], dot1[1], "VIOLET")
            if entry == "LEFT":
                x, y = dot1
                while dist2Points(dot1[0], dot1[1], x, y) < GARAGEW:
                    x -= 1
                    y = k*x + b
                dot2 = (x, y)
            elif entry == "RIGHT":
                x, y = dot1
                while dist2Points(dot1[0], dot1[1], x, y) < GARAGEW:
                    x += 1
                    y = k * x + b
                dot2 = (x, y)
            else:
                x, y = dot1
                while dist2Points(dot1[0], dot1[1], x, y) < GARAGEW:
                    y -= 1
                    x = (y-b) / k
                dot2 = (x, y)

            self.drawDotWithZone(dot2[0], dot2[1], "VIOLET")
            self.drawSquareBy2Dots(dot1, dot2, GARAGEH, entry,"DOWN")
        else:
            LeftPoints = []
            RightPoints =[]
            k0 = 0
            for i in range(0, len(dist), 2):
                omega1 = 90 + angleMap[i]

                b = ROBOT_Y - tan(pi * (omega1 / 180)) * ROBOT_X

                x, y = ROBOT_X, ROBOT_Y
                while dist2Points(ROBOT_X, ROBOT_Y, x, y) < dist[i]:
                    y -= 1
                    x = (y - b) / tan(pi * (omega1 / 180))
                point1 = (round(x), round(y))

                omega = 90 + angleMap[i+1]

                b = ROBOT_Y - tan(pi * (omega / 180)) * ROBOT_X

                x, y = ROBOT_X, ROBOT_Y
                while dist2Points(ROBOT_X, ROBOT_Y, x, y) < dist[i + 1]:
                    y -= 1
                    x = (y - b) / tan(pi * (omega / 180))
                point2 = (round(x), round(y))
                k, b = (lineFrom2Dots4Map(point1, point2))
                if fabs(k) > 1:
                    if len(LeftPoints) != 2:
                        LeftPoints.append(point1)
                        LeftPoints.append(point2)

                    else:
                        if max(LeftPoints, key = lambda x: x[0]) < max([point1, point2], key = lambda x: x[0]):
                            LeftPoints.remove(max(LeftPoints, key = lambda x: x[0]))
                            LeftPoints.insert(0, max([point1, point2], key = lambda x: x[0]))
                            LeftPoints.sort()
                        if min(LeftPoints, key = lambda x: x[0]) > min([point1, point2], key = lambda x: x[0]):
                            LeftPoints.remove(min(LeftPoints, key = lambda x: x[0]))
                            LeftPoints.insert(0, min([point1, point2], key = lambda x: x[0]))
                            LeftPoints.sort()

                else:
                    if len(RightPoints) != 2:
                        RightPoints.append(point1)
                        RightPoints.append(point2)
                    else:
                        if max(RightPoints, key = lambda x: x[0]) < max([point1, point2], key = lambda x: x[0]):
                            RightPoints.remove(max(RightPoints, key = lambda x: x[0]))
                            RightPoints.insert(0, max([point1, point2], key = lambda x: x[0]))
                            RightPoints.sort()
                        if min(RightPoints, key = lambda x: x[0]) > min([point1, point2], key = lambda x: x[0]):
                            RightPoints.remove(min(RightPoints, key = lambda x: x[0]))
                            RightPoints.insert(0, min([point1, point2], key = lambda x: x[0]))
                            RightPoints.sort()

            if len(RightPoints) == 2 and len(LeftPoints) == 2:

                distRight = dist2Points(RightPoints[0][0], RightPoints[0][1], RightPoints[1][0], RightPoints[1][1])
                distLeft = dist2Points(LeftPoints[0][0], LeftPoints[0][1], LeftPoints[1][0], LeftPoints[1][1])
                H = GARAGEH
                if distRight > distLeft:
                    self.drawSquareBy2Dots(RightPoints[0], RightPoints[1], H, entry = "TOP")
                else:
                    self.drawSquareBy2Dots(LeftPoints[0], LeftPoints[1], H, entry = "TOP")
            elif len(LeftPoints) != 2:
                distRight = dist2Points(RightPoints[0][0], RightPoints[0][1], RightPoints[1][0], RightPoints[1][1])
                H = GARAGEH if (fabs(GARAGEW - distRight) < fabs(GARAGEH - distRight)) else GARAGEW
                self.drawSquareBy2Dots(RightPoints[0], RightPoints[1], H, entry = "TOP" if H == GARAGEH else self.position)

            elif len(RightPoints) != 2:
                distLeft = dist2Points(LeftPoints[0][0], LeftPoints[0][1], LeftPoints[1][0], LeftPoints[1][1])
                H = GARAGEH if (fabs(GARAGEW - distLeft) < fabs(GARAGEH - distLeft)) else GARAGEW
                self.drawSquareBy2Dots(LeftPoints[0], LeftPoints[1], H, entry = "TOP" if H == GARAGEH else self.position)

    def bfs(self) -> list:
        start = (ROBOT_X, ROBOT_Y)
        goal = self.goal
        graph = {}

        def getPossibleNodes(x: int, y: int) -> list:
            check_node = lambda x, y: True if 0 <= x < MAPWIDTH and 0 <= y < MAPLENGHT and not self.mapr[y][x] else False
            ways = [-1, 0], [0, -1], [1, 0], [0, 1], [-1, -1], [1, -1], [1, 1], [-1, 1]
            return [(x+ways[i][0], y+ways[i][1]) for i in range(len(ways)) if check_node(x+ways[i][0], y+ways[i][1])]

        for y, row in enumerate(self.mapInt):
            for x, col in enumerate(row):
                if not col:
                    graph[(x, y)] = graph.get((x, y), []) + getPossibleNodes(x, y)

        queue = deque([start])
        visited = {start: None}


        while queue:
            cur_node = queue.popleft()
            if cur_node == goal:
                break
            next_nodes = graph[cur_node]
            for next_node in next_nodes:
                if next_node not in visited:
                    queue.append(next_node)
                    visited[next_node] = cur_node

        path = []
        cell = goal
        while cell and cell in visited:
            self.drawCircle(cell[0], cell[1], int(1), "WAY")
            path.append(cell)
            cell = visited[cell]
        return self.getApprPath() if path else ([], [])

    def showMap(self):

        self.MAPgr = []
        for i in range(MAPLENGHT):
            self.MAPgr.append([])
            for j in range(MAPWIDTH):
                self.MAPgr[i].append(list(colors.values())[self.mapr[i][j]][1])

        self.MAPgr = np.array(self.MAPgr).astype(np.uint8)

        while True:
            cv2.imshow("map",self.MAPgr)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                # self.drawCircle(200, 200, 210, "WHITE")
                # self.clearMAPBGR()
                # self.mapClear()
                break

    def mapClear(self):
        # self.mapInt = None
        # del self.mapInt
        # self.__init__()
        self.mapInt.clear()
        self.mapInt  = [[0 for i in range(MAPWIDTH)] for j in range(MAPLENGHT)]
    # def clearMAPBGR(self):
    #     self.MAPgr = None
    #     # del self.MAPgr
    #     self.MAPgr = []
    #     # self.__init__()
    #     # self.MAPgr.tolist()
    #     # self.MAPgr = []


        # for i in range(len(self.MApr)):
        #     for j in range(len(self.MAPgr[0])):
        #         self.MAPgr[i][j] = colors["WHITE"][1]
