from dataset import create_dist_dataset, get_dataset_for_dist
from geometry import height, dist2Points
import numpy as np


def dist(box: list, mode: str) -> float:
    """
    modes:
    gv - garage entry
    gy - garage wall
    o - obstacle
    """
    DOP = 6
    if mode == "gv":
        hght = (height(box[0])+height(box[1]))/2
    elif mode == "gy":
        hght = dist2Points(box[0][0], box[0][1], box[1][0], box[1][1])
    else:
        hght = height(box)

    polynomial = np.array([1.95319933e-12, -3.13381427e-09,  2.06132788e-06, -7.18746527e-04,
                          1.43043267e-01, -1.61867151e+01,  9.54867505e+02])

    return round(sum(polynomial[k]*(hght**(DOP - k)) for k in range(len(polynomial))), 2)


if __name__ == "__main__":
    create_dist_dataset("datasets/dist.txt", "dist_images", ".png")