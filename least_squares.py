import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataset_for_dist
import cv2
DOP = 1
PRT = 0.01


def lSquares4Aglens(y: list) -> tuple:
    text_color = (255, 10, 10)
    DOP = 7
    x = np.array([30, 45, 60, 90, 120, 135, 150])
    x = x.astype('float64')
    y = y.astype('float64')
    pow_x = []
    for s in range(DOP*2 + 1):
        pow_x.append(sum([xi**s for xi in x]))
    pow_x.reverse()
    matrix_pow = []
    for pm in range(DOP+1):
        matrix_pow.append(pow_x[pm:pm+DOP+1])


    calc = []
    for s in range(DOP+1):
        calc.append(sum([y[i]*(x[i]**s) for i in range(len(x))]))
    calc.reverse()

    M = np.array(matrix_pow)
    V = np.array(calc)

    polynomial = np.linalg.solve(M, V)
    # print(polynomial)
    y_g = []
    x_g = []
    xi = min(x)
    while (xi <= max(x)):
        x_g.append(xi)
        y_g.append(sum(polynomial[k]*(xi**(DOP - k)) for k in range(len(polynomial))))
        xi += PRT
    # return (x_g, y_g)
    # index = 0
    # for i in range(len(y)):
    #     if i == 0:
    #         continue
    #     elif y[i] > y[i-1]:
    #         index = i
    #     else: continue


    # fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    # fig.tight_layout(h_pad=4)
    #
    #
    #
    # ax[0, 0].set_title("angle prediction")
    # ax[0, 0].set_xlabel("angle [deg]")
    # ax[0, 0].set_ylabel("\nprobability [-]")
    # ax[0, 0].bar(x, y, width=10, color='darkorange', label='angle probabilities by NW')
    # ax[0, 0].legend()
    # ax[0, 0].grid()
    #
    #
    # ax[0, 1].set_title("approximation")
    # ax[0, 1].set_xlabel("angle [deg]")
    # ax[0, 1].set_ylabel("probability [-]")
    # ax[0, 1].bar(x, y, width=10, color='darkorange', label='angle probabilities by NW')
    # ax[0, 1].plot(x_g, y_g, label='approximation by a polynomial\n of the 7th degree')
    # ax[0, 1].legend()
    # ax[0, 1].grid()
    #
    #
    #
    # ax[1, 0].set_title("angle prediction with/without approximation")
    # ax[1, 0].set_xlabel("angle [deg]")
    # ax[1, 0].set_ylabel(".\nprobability [-]")
    # ax[1, 0].plot(x_g, y_g, label='approximation by a polynomial\n of the 7th degree')
    # ax[1, 0].bar(x_g[y_g.index([max(y_g)])], max(y_g),width=2, color='darkred', alpha=1, label='angle after approximation')
    # ax[1, 0].bar(x[index], max(y),width=2, color= 'lawngreen', alpha=1, label='angle before approximation')
    # ax[1, 0].legend()
    # ax[1, 0].grid()
    #
    #
    # angle = x_g[y_g.index([max(y_g)])]
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.putText(img, str(round(angle,2)) + " degrees", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    # ax[1, 1].imshow(img)
    # # plt.bar(x, y, width=10, color='r')
    #
    # # plt.plot(x_g, y_g)
    # # plt.show()
    # plt.savefig('grafs/'+name, dpi = 1000)
    return x_g[y_g.index([max(y_g)])]


def lSquares4Dist(x: list, y: list) -> tuple:
    x = x.astype('float64')
    y = y.astype('float64')
    # while(DOP < 61):
    pow_x = []
    for s in range(DOP*2 + 1):
        pow_x.append(sum([xi**s for xi in x]))
    pow_x.reverse()
    matrix_pow = []
    for pm in range(DOP+1):
        matrix_pow.append(pow_x[pm:pm+DOP+1])


    calc = []
    for s in range(DOP+1):
        calc.append(sum([y[i]*(x[i]**s) for i in range(len(x))]))
    calc.reverse()

    M = np.array(matrix_pow)
    V = np.array(calc)

    polynomial = np.linalg.solve(M, V)
    # if DOP == 6:
    print(polynomial)

    y_g = []
    x_g = []
    xi = min(x) - 10
    while (xi <= max(x) + 10):
        x_g.append(xi)
        y_g.append(sum(polynomial[k]*(xi**(DOP - k)) for k in range(len(polynomial))))
        xi += PRT


    return (x_g, y_g)
    # return x_g[y_g.index([max(y_g)])]

if __name__ == "__main__":
    DOP = 1
    # y, x = get_dataset_for_dist('datasets/dist.txt')
    # x = np.array([302.417025617125, 298.2, 293.7, 284.9, 274.5, 264.7, 246.7, 231, 215.9, 200.5, 189,
    #               175.46536520532234, 156.8075810935277, 143.6, 134.19083709739692, 120, 110.9,
    #               103,95, 88.63742824133129])
    # y = np.array([50, 51, 52.0, 54.1, 56.5, 57.8, 63.1, 70, 76.6, 84.9, 95, 106, 120, 159.6, 200, 235, 243, 247, 248.9, 250])
    # x = x.astype('float64')
    # y = y.astype('float64')
    x = np.array([640, 585, 486, 336, 233, 96, 0]).astype('float64')
    y = np.array([30, 20, 10, 0, -10, -20, -30]).astype('float64')

    while DOP < 2:
        xg, yg = lSquares4Dist(x,y)
        plt.plot(xg, yg,)
        plt.plot(x, y, ls = ' ', marker = '+', color = 'r')
        plt.title("Polynomial "+str(DOP))
        plt.xlabel("avg height")
        plt.ylabel("distance")
        # plt.xlabel("angles")
        # plt.ylabel("probability")

    # print(xg[yg.index([max(yg)])])
        print(" ", DOP)
        DOP += 1
        plt.show()
