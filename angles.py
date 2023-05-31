"""
This Python code used for training the NW and it also contain trained NW which can be used for predictions.
"""

import random
import numpy as np
from dataset import get_dataset
from geometry import segmentLengthRatio

# angles = [30, 45, 60, 75, 90, 105, 120, 135, 150]
angles = [30, 45, 60, 90, 120, 135, 150]

ALPHA = 0.0001  # learning rate
EPOCHS = 1

INPUT_DIM = 1
OUT_DIM = len(angles)
H_DIM = len(angles) + 3


def getOneDimList(a: list) -> list:
    """
    :param a:
    :return:
    """
    s_ret = ''
    for i in str(a):
        if i not in "()[],":
            s_ret += i
    ret = [int(x) for x in s_ret.split(' ')]
    return ret


def relu(t: 'numpy.ndarray') -> 'numpy.ndarray':
    """
    Activation function.
    :param t: array
    :return: return array with [[max(t1, 0) , max(t2,0) ...]]
    """
    return np.maximum(t, 0)


def softmax(t: 'numpy.ndarray') -> 'numpy.ndarray':
    """
    Standard function Softmax
    link: https://en.wikipedia.org/wiki/Softmax_function
    """
    ret = np.exp(t)
    return ret / np.sum(ret)


def sparseCrossEntropy(z: 'numpy.ndarray', y: int) -> float:
    """y is not an array, so we just have to count -log(z[0,y]),
     instead of -sum(yi*log(z[0,yi])), because y contain only zeroes
     and only one 1 we don't have to use sum function."""
    return -np.log(z[0, y])


def toFull(y: int, num_classes: int):
    """
    :param y: index of 1
    :param num_classes: amount of classes
    :return: array of zeroes and one 1 by index of y
    """
    ret = np.zeros((1, num_classes))
    ret[0, y] = 1
    return ret


def reluDeriv(t) -> float:
    """
    Derivation of activation function.
    Return 1 if t >= 0 and 0 if t < 0
    """
    return (t >= 0).astype(float)


def predict(x: 'numpy.ndarray') -> int:
    """Make a prediction """
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


def calcAccuracy():
    """Calculate an accuracy"""
    correct = 0
    for (x, y) in datasetchekc:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(datasetchekc)
    return acc


def getMostProbableAngle(z: list) -> float:
    """
    This function based on equation of center of gravity.
    In our case we can write it in following form:
    most_prob_angle = sum(prob_i * angle_i) / sum(prob_i)
    ( sum(prob_i) allways will be 1 )
    :param z: list of probabilities of angle
    :return: most probable angle
    """
    return sum([z[i] * angles[i] for i in range(len(angles))])


def predictAngle(approx: list) -> float:
    """
    Its trained NW which capable to predict an angel from image with 85% accuracy.
    Prediction based on the ratio of the heights of the found objects.
    :param approx:
    :return: angle [degr]
    """
    slr = (segmentLengthRatio(getOneDimList(approx)))
    x = np.array([slr])

    W1 = np.array([[-2.00018969, 0.2306922, -5.11953054, 1.82200341, 7.43949556]])
    b1 = np.array([[0.90759633, -0.8835846, 7.58510564, -2.12552064, -4.38471096]])
    W2 = np.array([[0.27662751, -1.91819948, 0.99152632, 0.49105279, -0.25077443, 0.67297711, -0.0441832],
                   [0.1310934, -1.95369299, -0.12151098, 1.00547347, 0.94917766, -1.08287209, -0.42368037],
                   [5.53665756, 3.78603098, 3.73590823, 1.80312131, -1.24392236, -2.16185967, -5.65673811],
                   [-0.10198268, -0.99110304, 0.15703252, -0.49595104, -0.15006271, -1.46264658, 2.92438492],
                   [-5.52165422, -2.94493322, -1.86952688, 1.05246194, 2.78808148, 3.63926668, 3.78307236]])
    b2 = np.array([[-0.95076086, 2.58120124, 0.97962599, 0.3145152, 0.47471217, -1.3762197, -1.76939585]])

    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    z = z.tolist()[0]
    z = np.array(z)

    # angle = lSquares4Aglens(z)
    angle = getMostProbableAngle(z)
    return round(angle, 2)


def angle4Map(x: int) -> float:
    """
    This function based on simple linear regression
    :param x: x coordinate of central point
    :return: angle [degr]
    """
    # return (27*x)/308 - 8667/308
    polynomial = [0.08803158, -29.88043477]
    # polynomial = [-0.08094275, 28.10332266]
    return x*polynomial[0] + polynomial[1]


if __name__ == "__main__":
    datsetnw = get_dataset("datasets/datasetlearn.txt")
    datasetchekc = get_dataset("datasets/datasettest.txt")

    W1 = np.random.randn(INPUT_DIM, H_DIM)
    b1 = np.random.randn(1, H_DIM)
    W2 = np.random.randn(H_DIM, OUT_DIM)
    b2 = np.random.randn(1, OUT_DIM)


    loss_arr = []

    for ep in range(EPOCHS):
        random.shuffle(datsetnw)
        for i in range(len(datsetnw)):

                x, y = datsetnw[i]

                # Forward
                t1 = x @ W1 + b1
                h1 = relu(t1)
                t2 = h1 @ W2 + b2
                z = softmax(t2)
                E = sparseCrossEntropy(z, y)

                # Backward
                y_full = toFull(y, OUT_DIM)
                dE_dt2 = z - y_full
                dE_dW2 = h1.T @ dE_dt2
                dE_db2 = dE_dt2
                dE_dh1 = dE_dt2 @ W2.T
                dE_dt1 = dE_dh1 * reluDeriv(t1)
                dE_dW1 = x.T @ dE_dt1
                dE_db1 = dE_dt1

                # Update
                W1 = W1 - ALPHA * dE_dW1
                W2 = W2 - ALPHA * dE_dW2
                b1 = b1 - ALPHA * dE_db1
                b2 = b2 - ALPHA * dE_db2

                loss_arr.append(E)

    print(W1)
    print(b1)
    print(W2)
    print(b2)

    accuracy = calcAccuracy()
    print("Accuracy: ", accuracy)

    import matplotlib.pyplot as plt
    plt.plot(loss_arr)
    plt.title("NW learning process")
    plt.ylabel("inaccuracy")
    plt.xlabel("epochs")
    plt.show()
