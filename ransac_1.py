import math
import random
import numpy as np
import matplotlib.pyplot as plt


def leastsq_mutifunc(x, y, m):
    """
    多项式最小二乘法实现, 矩阵形式
    :param x:输入
    :param y:目标输出
    :param m:多项式阶数
    :return:多项式系数
    """
    x = np.array(x)
    y = np.array(y)

    assert m <= x.shape[0], f"the number of m({m}) need less than x's size({x.shape[0]})"
    assert x.shape[0] == y.shape[0], f"the size of x({x.shape[0]}) must equal to y's size({y.shape[0]}"
    x_mat = np.zeros((x.shape[0], m + 1))
    for i in range(x.shape[0]):
        x_mat_h = np.zeros((1, m + 1))
        for j in range(m + 1):
            x_mat_h[0][j] = x[i] ** (m - j)
        x_mat[i] = x_mat_h
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_mat.T, x_mat)), x_mat.T), y.T)
    return theta


SIZE = 50
X = np.linspace(-20, 20, SIZE)
Y = 4 * (X ** 3) + 3 * (X ** 2) + 4 * X + 10

random_x = []
random_y = []
for i in range(SIZE):
    random_x.append(X[i] + random.uniform(-0.5, 0.5))  # 加噪音
    random_y.append(Y[i] + random.uniform(-0.5, 0.5))
# 添加随机噪声
for i in range(SIZE):
    random_x.append(random.uniform(10, 20))
    random_y.append(random.uniform(10, 20))
RANDOM_X = np.array(random_x)
RANDOM_Y = np.array(random_y)



iters = 10000
sigma = 0.1  # check
best_a0, best_a1, best_a2, best_a3 = 0.001, 0.001, 0.001, 0.001
pretotal = 0
P = 0.99  # 希望的得到正确模型的概率

for iter in range(iters):
    # 采样4个点
    num_sample = 5
    sample_index = random.sample(range(SIZE), num_sample)
    X = [RANDOM_X[sample_index[i]] for i in range(num_sample)]
    Y = [RANDOM_Y[sample_index[i]] for i in range(num_sample)]

    theta = leastsq_mutifunc(X, Y, 3)
    # 获取theta里面是拟合系数
    a0 = theta[0]
    a1 = theta[1]
    a2 = theta[2]
    a3 = theta[3]
    # 算出内点数目
    total_inlier = 0
    for index in range(SIZE):
        y_estimate = a0 * (RANDOM_X[index] ** 3) + a1 * (RANDOM_X[index] ** 2) + a2 * RANDOM_X[index] + a3
        if abs(y_estimate - RANDOM_Y[index]) < sigma:
            total_inlier = total_inlier + 1

    # 判断当前的模型是否比之前估算的模型好
    if total_inlier > pretotal:
        iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE), 2))
        pretotal = total_inlier
        # 保存最好的参数
        best_a0 = a0
        best_a1 = a1
        best_a2 = a2
        best_a3 = a3

    # 判断是否当前模型已经符合超过一半的点
    if total_inlier > (SIZE // 10 * 9):
        break

# 用我们得到的最佳估计画图
Y_fitting = best_a0 * (RANDOM_X ** 3) + best_a1 * (RANDOM_X ** 2) + best_a2 * RANDOM_X + best_a3

# 数据显示
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title("RANSAC")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
# ax1.scatter(X, Y, c='r', marker='v')  # 原始数据
ax1.scatter(RANDOM_X, RANDOM_Y, c='b')  # 加噪数据
ax1.plot(RANDOM_X, Y_fitting, c='y')

text = "best_a0 = " + str(best_a0) + "\nbest_a1 = " + str(best_a1) + \
       "\nbest_a2 = " + str(best_a2) + "\nbest_a3 = " + str(best_a3)
plt.text(5, 10, text, fontdict={'size': 8, 'color': 'r'})
plt.show()
