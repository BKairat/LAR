"""
Simple geometric calculations, this file will be updated
"""
from math import sqrt
ROT = 0.1


def rotation(size: tuple, centers: list) -> float:
    first, second = centers
    mid_pointX = (first[0]+second[0])/2
    img_midX = size[1]/2
    if mid_pointX < img_midX: return -ROT
    else: return ROT


def center(dots: list) -> tuple:
    return ((dots[0][0] + dots[1][0])/2, (dots[0][1] + dots[1][1])/2)


def box_aploximation(box: 'numpy.ndarray') -> list:
    ret = []
    box = box.tolist()
    box.sort(key = lambda x: x[0]**2 + x[1]**2)
    for i in range(0, 3, 2):
        ret.append((int((box[i][0]+box[i+1][0])/2), int((box[i][1]+box[i+1][1])/2)))
    return ret


def segment_length_ratio(dots: list) -> float:
    x1, y1, x2, y2, x3, y3, x4, y4 = dots
    return sqrt((x1-x2)**2+(y1-y2)**2) / sqrt((x3-x4)**2+(y3-y4)**2)