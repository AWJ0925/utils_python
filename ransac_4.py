import math
import random
import numpy as np
import matplotlib.pyplot as plt


def line_fitting_RANSAC(line_points, iters, sigma, P=0.99):
    best_a, best_b = 0, 0
    pretotal, i = 0, 0

    size = len(line_points)
    while (i < iters):

        sample = random.sample(line_points, 2)
        x1 = sample[0][0]
        y1 = sample[0][1]
        x2 = sample[1][0]
        y2 = sample[1][1]
        a = (x2 - x1) / (y2 - y1)
        b = x1 - a * y1

        # 算出内点数目
        total_inlier = 0
        for index in range(size):
            x_estimate = a * (line_points[index][1]) + b
            if abs(x_estimate - line_points[index][0]) < sigma:
                total_inlier += 1
        i += 1

        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > pretotal:
            if total_inlier < size:
                iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (size), 2))
            pretotal = total_inlier
            best_a = a
            best_b = b

        # 判断是否当前模型已经符合超过一半的点
        if total_inlier > size / 2:
            break

    return [best_b, best_a]


if __name__ == '__main__':
    lane = [(438.00, 344.00), (470.00, 355.00), (503.00, 366.00), (538.00, 377.00), (573.00, 388.00),
            (619.00, 407.00), (655.00, 414.00), (703.00, 431.00), (736.00, 441.00)]
    theta = line_fitting_RANSAC(lane, 1000, 3)

    X = list()
    Y = list()
    for i in range(len(lane)):
            X.append(lane[i][0])
            Y.append(lane[i][1])
    X = np.array(X)
    Y = np.array(Y)
    new_Y = np.linspace(Y[0], Y[-1], 30)
    new_X = theta[1] * new_Y + theta[0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title("RANSAC")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    # 数据显示
    ax1.plot(X, Y, c='r', marker='v')
    ax1.scatter(new_X, new_Y, c='b')
    plt.show()
