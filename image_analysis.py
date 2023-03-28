"""
This code used for photo analysis, this file will be updated.
"""

import cv2
import numpy as np
from math import fabs



rgb_ref = [68, 59, 102]
# ref_color = [61, 89, 158] #green color
ref_color = [171, 109, 103]
black = [0, 0, 0]
white = [255,255,255]


def square(points: list) -> int:
    A, B, C, D = points
    return (fabs((A[0]-B[0])*(A[1]+B[1]) + (B[0]-C[0])*(B[1]+C[1]) + (C[0]-D[0])*(C[1]+D[1]) + (D[0]-A[0])*(D[1]+A[1])))/2


def comparator(icolor: list) -> tuple:
    # ---green---
    # delta_H = 20
    # delta_S = 55
    # delta_V = 75
    # ---purple---
    delta_H = 55
    delta_S = 50
    delta_V = 70
    if fabs(icolor[0] - ref_color[0]) > delta_H: return (black, 0)
    elif fabs(icolor[1] - ref_color[1]) > delta_S: return (black, 0)
    elif fabs(icolor[2] - ref_color[2]) > delta_V: return (black, 0)
    else: return (white, 1)


def clear(box: 'numpy.ndarray', MIN_square = 500) -> bool:
    return square(box) > MIN_square


def centers(quadrilaterals: list) -> list:
    centers = []
    for coords in quadrilaterals:
        [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] = coords
        C = (int((x1+x2+x3+x4)/4) , int((y1+y2+y3+y4)/4))
        centers.append(C)
    return centers


def analysisRGB(img: 'numpy.ndarray', all = True) -> tuple:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_list = hsv.tolist()
    thresh = []
    tmp = []
    for i in range(len(hsv_list)):
        for j in range(len(hsv_list[0])):
            hsv_list[i][j], pixel = comparator(hsv_list[i][j])
            tmp.append(pixel)
        thresh.append(tmp)
        tmp = []

    thresh = np.array(thresh, np.uint8)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cont in contours:
        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if clear(box):
            objects.append(box)
            cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

    if not all:
        return objects

    return objects, centers(objects), thresh, hsv[0][0]




