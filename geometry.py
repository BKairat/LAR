"""
Simple geometric calculations, this file will be updated
"""
import numpy as np
from math import sqrt, fabs, pi, asin


def getUnitVecrot(tail: tuple, head: tuple) -> tuple:
    vector = (head[0]-tail[0], head[1]-tail[1])
    return np.array(vector)/np.linalg.norm(vector)


def dist2Points(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    Calculate distance between two points.
    :param x1: x coordinate of first point.
    :param y1: y coordinate of first point.
    :param x2: x coordinate of second point.
    :param y2: y coordinate of second point.
    :return: distance between two points.
    """
    return sqrt((x1-x2)**2+(y1-y2)**2)


def getAngeleBetweenArrays(vecU1: 'numpy.ndarray', vecU2: 'numpy.ndarray') -> float:
    M = np.array([[vecU1[0], -vecU1[1]], [vecU1[1], vecU1[0]]])
    cos, sin = np.linalg.solve(M, vecU2)
    return -asin(sin)*180/pi


def lineFrom2Dots(a: tuple, b: tuple) -> tuple:
    return ((-a[1]+b[1])/(a[0]+b[0]), a[1]-(a[0])/(b[0]-a[0])) if a[0] != b[0] else (a[0])


def square(points: list) -> int:
    A, B, C, D = points
    return (fabs((A[0]-B[0])*(A[1]+B[1]) + (B[0]-C[0])*(B[1]+C[1]) + (C[0]-D[0])*(C[1]+D[1]) + (D[0]-A[0])*(D[1]+A[1])))/2


def getCenter2Dots(dots: list) -> tuple:
    return (round((dots[0][0] + dots[1][0])/2), round((dots[0][1] + dots[1][1])/2))


def height(aprox: list) -> float:
    return sqrt((aprox[0][0]-aprox[1][0])**2 + (aprox[0][1]-aprox[1][1])**2)


def lineFrom2Dots4Map(a: tuple, b:tuple) -> tuple:
    return ((b[1]-a[1])/(b[0]-a[0]), a[1] - (a[0])*(b[1]-a[1])/(b[0]-a[0]))


def boxApproximation(box: 'numpy.ndarray') -> list:
    ret = []
    if type(box) != list:
        box = box.tolist()
    box.sort(key = lambda x: x[0]**2 + x[1]**2)

    for i in range(0, 3, 2):
        ret.append((int((box[i][0]+box[i+1][0])/2), int((box[i][1]+box[i+1][1])/2)))
    return ret


def segmentLengthRatio(dots: list) -> float:
    x1, y1, x2, y2, x3, y3, x4, y4 = dots
    return sqrt((x1-x2)**2+(y1-y2)**2) / sqrt((x3-x4)**2+(y3-y4)**2)