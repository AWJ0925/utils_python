import math
import random
import numpy as np
import matplotlib.pyplot as plt


# yuanShi = [438.00, 344.00, 470.00, 355.00, 503.00, 366.00, 538.00, 377.00, 573.00, 388.00,
#            619.00, 407.00, 655.00, 414.00, 703.00, 431.00, 736.00, 441.00]
# yuanShi = [543.00, 343.00, 537.00, 352.00, 527.00, 364.00, 535.00, 378.00, 552.00, 394.00, 569.00,
#            405.00, 586.00, 414.00, 620.00, 431.00, 637.00, 439.00, 670.00, 457.00, 687.00, 466.00,
#            708.00, 477.00, 737.00, 491.00, 757.00, 502.00]
# yuanShi =[420.00, 334.00, 438.00, 345.00, 454.00, 355.00, 471.00, 366.00, 487.00, 378.00, 504.00, 392.00,
#           520.00, 406.00, 537.00, 420.00, 554.00, 433.00, 563.00, 440.00, 577.00, 453.00, 591.00, 465.00,
#           605.00, 478.00, 620.00, 491.00, 637.00, 505.00, 653.00, 520.00, 662.00, 527.00, 676.00, 540.00,
#           689.00, 553.00, 705.00, 566.00, 720.00, 579.00, 728.00, 587.00,]
yuanShi = [460.00, 354.00, 471.00, 365.00, 485.00, 382.00, 492.00, 390.00, 502.00, 402.00, 514.00, 414.00,
           523.00, 428.00, 536.00, 442.00, 545.00, 453.00, 556.00, 465.00, 569.00, 482.00, 575.00, 490.00,
           586.00, 504.00, 596.00, 515.00, 606.00, 528.00, 619.00, 543.00, 626.00, 552.00, 638.00, 567.00,
           644.00, 576.00, 656.00, 587.00,
]

X = list()
Y = list()
for i in range(len(yuanShi)):
    if i%2 == 0:
        X.append(yuanShi[i])
    else:
        Y.append(yuanShi[i])
X = np.array(X)
Y = np.array(Y)
print(X)
print(Y)

# 扩展点
X_extend = list()
Y_extend = list()

for i in range(len(X)-1):
    X_extend.append(X[i])
    mid_x = (X[i] + X[i+1])/2
    X_extend.append(mid_x)
    Y_extend.append(Y[i])
    mid_y = (Y[i] + Y[i+1]) / 2
    Y_extend.append(mid_y)
X_extend.append(X[-1])
Y_extend.append(Y[-1])

X = np.array(X_extend)
Y = np.array(Y_extend)
Y = -Y


iters = 10000
sigma = 0.08  # check
best_theta = 0.001
pretotal = 0
P = 0.99  # 希望的得到正确模型的概率
iter = 0

while(iter<iters):
    print('i={}!'.format(iter))
    sample_nums = 5
    sample_index = random.sample(range(len(X)), sample_nums)
    X_sample = [X[sample_index[i]] for i in range(sample_nums)]
    Y_sample = [Y[sample_index[i]] for i in range(sample_nums)]
    theta = np.polyfit(Y_sample, X_sample, sample_nums-1)  # check

    # 算出内点数目
    total_inlier = 0
    p1 = np.poly1d(theta)  # 得到多项式
    for index in range(len(X)):
        x_estimate = p1(Y[index])
        if abs(X[index] - x_estimate) < sigma:
            total_inlier = total_inlier + 1

    iter += 1

    # 判断当前的模型是否比之前估算的模型好
    if total_inlier > pretotal:
        iters = math.log(1 - P) / math.log(1 - pow(total_inlier/len(X), 2))
        pretotal = total_inlier
        best_theta = theta

    # # # 判断是否当前模型已经符合超过一半的点
    # if total_inlier > (len(X)//100*99):
    #     break

p1 = np.poly1d(best_theta)
Y_new = np.linspace(Y[0], Y[-1], 50)
X_new = p1(Y_new)


# ------------多项式拟合------------
theta_poly = np.polyfit(Y, X, 4)  # check
p1 = np.poly1d(theta_poly)
X_new_poly = p1(Y_new)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title("RANSAC")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
# 数据显示
ax1.plot(X, Y, c='r', marker='v', label='原始数据')
ax1.scatter(X_new, Y_new, c='b', label='RANSAC')
ax1.scatter(X_new_poly, Y_new, c='y', label='poly_3')
plt.show()

