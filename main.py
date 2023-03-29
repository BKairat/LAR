"""
The main Python file which used for detecting the objects of certain colors ([68, 59, 102] RBG format),
and asle capable to predict at what angle the robot took the picture.
"""

from image_analysis import analysisRGB
import cv2
from geometry import box_aploximation, segment_length_ratio
from neuron_web import predict_angle
import numpy as np


def tmp(a):
    s_ret = ''
    for i in str(a):
        if i not in "()[],": s_ret += i
    ret = [int(x) for x in s_ret.split(' ')]
    return ret


if __name__ == "__main__":
    img_path = "30_1.jpg"  # path to image which we interested in.
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    text_color = (0, 0, 255)
    objects, centers, hsv, color = analysisRGB(img)

    apr =[]

    for box in objects:
        cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
        apr.append(box_aploximation(box))
    apr = (segment_length_ratio(tmp(apr)))

    cv2.putText(img, str(predict_angle(np.array([apr])))+" degrees", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    while True:
        cv2.imshow("other_images", img)  # shows image

        # break if you press esc
        if cv2.waitKey(0) & 0xFF == 27:
            break



