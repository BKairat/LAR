"""
This Python code is used to create the database from the available images.
Due to the small number of photos, there is also a function for crating "fake" dataset,
based on real data. This mad eit possible to achieve an accuracy of determining the angle of 85%.
"""

from image_analysis import analysisRGB
from geometry import box_aploximation, segment_length_ratio
import numpy as np
import cv2
import os
import random


def create_real_dataset(path_file: str, path_folder: str):
    """
    :param path_file: path to file to which the data will be written.
    :param path_folder: path to folder that contains the images.
    """
    data = open(path_file, 'w')
    for filename in os.listdir(path_folder+"/"):
        if filename.endswith(".jpg"):
            path = os.path.join(path_folder+"/", filename)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            odjects, centers, hsv, color = analysisRGB(img)
            apr = []
            for box in odjects:
                apr.append(box_aploximation(box))
            f = ""
            for i in filename:
                if i == '_': break
                f += i

            a = ""
            for i in str(apr):
                if i not in "[](),": a += i

            data.write(f+" "+a+"\n")
    data.close()


def angle_into_int(angle: int) -> int:
    angles = [30, 45, 60, 90, 120, 135, 150]
    return angles.index(angle)


def get_dataset(path: str) -> list:
    """
    Read data from file and write them in correct form for using them.
    :param path: path to dataset.
    :return: list of data from dataset.
    """
    dataset = open(path, 'r')
    ret = []

    for line in dataset:
        line = [float(x) for x in line[:-1].split(" ")]
        angle = angle_into_int(line[0])
        positions = line[1:]
        arr = np.array([positions], dtype=np.float128)
        ret.append((arr, angle))
    dataset.close()
    return ret


def dataset_generation(path_true: str, path_gen: str, rng: float = 0.05, add: int = 0, ):
    """
    This function used for generating
    :param path_true: path to "true" data file.
    :param path_gen: path to file to which the generated data will be written.
    :param rng: range of differences in generated data.
    :param add: how many data will be generated from one string of true dataset.
    """
    data = open(path_true, 'r')
    dataset = open(path_gen, 'w')
    for line in data:
        line = [int(x) for x in line[:-1].split(" ")]
        if add == 0:
            angle = line[0]
            ratio = segment_length_ratio(line[1:])
            dataset.write(str(angle)+" "+str(ratio)+"\n")
        for i in range(add):
            alpha = random.uniform(-rng, rng)
            angle = line[0]
            ratio = segment_length_ratio(line[1:])+alpha
            dataset.write(str(angle)+" "+str(ratio)+"\n")
    dataset.close()