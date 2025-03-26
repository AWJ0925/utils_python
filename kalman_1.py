# !/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


# from filterpy.kalman import KalmanFilter
# https://blog.csdn.net/lishan132/article/details/124576990

font = {'family': 'SimSun',  # 宋体
        'weight': 'bold',  # 加粗
        'size': '10.5'  # 五号
        }
plt.rc('font', **font)
plt.rc('axes', unicode_minus=False)

plt.rcParams['figure.facecolor'] = "#FFFFF0"  # 设置窗体颜色
plt.rcParams['axes.facecolor'] = "#FFFFF0"  # 设置绘图区颜色


class KalmanFilter:
    B = 0  # 控制变量矩阵，初始化为0
    u = 0  # 状态控制向量，初始化为0
    K = float('nan')  # 卡尔曼增益无需初始化
    z = float('nan')  # 观测值无需初始化，由外界输入
    P = np.diag(np.ones(4))  # 先验估计协方差

    x = []  # 滤波器输出状态
    G = []  # 滤波器预测状态

    # 状态转移矩阵A，和线性系统的预测机制有关
    A = np.eye(4) + np.diag(np.ones((1, 2))[0, :], 2)

    # 噪声协方差矩阵Q，代表对控制系统的信任程度，预测过程上叠加一个高斯噪声，若希望跟踪的轨迹更平滑，可以调小
    Q = np.diag(np.ones(4)) * 0.1

    # 观测矩阵H：z = H * x，这里的状态是（坐标x， 坐标y， 速度x， 速度y），观察值是（坐标x， 坐标y）
    H = np.eye(2, 4)

    # 观测噪声协方差矩阵R，代表对观测数据的信任程度，观测过程上存在一个高斯噪声，若观测结果中的值很准确，可以调小
    R = np.diag(np.ones(2)) * 0.1

    def init(self, px, py, vx, vy):
        # 本例中，状态x为（坐标x， 坐标y， 速度x， 速度y），观测值z为（坐标x， 坐标y）
        self.B = 0
        self.u = 0
        self.K = float('nan')
        self.z = float('nan')
        self.P = np.diag(np.ones(4))
        self.x = [px, py, vx, vy]
        self.G = [px, py, vx, vy]
        self.A = np.eye(4) + np.diag(np.ones((1, 2))[0, :], 2)
        self.Q = np.diag(np.ones(4)) * 0.1
        self.H = np.eye(2, 4)
        self.R = np.diag(np.ones(2)) * 0.1

    def update(self):
        # Xk_ = Ak*Xk-1+Bk*Uk
        a1 = np.dot(self.A, self.x)
        a2 = self.B * self.u
        x_ = np.array(a1) + np.array(a2)
        self.G = x_

        # Pk_ = Ak*Pk-1*Ak'+Q
        b1 = np.dot(self.A, self.P)
        b2 = np.dot(b1, np.transpose(self.A))
        p_ = np.array(b2) + np.array(self.Q)

        # Kk = Pk_*Hk'/(Hk*Pk_*Hk'+R)
        c1 = np.dot(p_, np.transpose(self.H))
        c2 = np.dot(self.H, p_)
        c3 = np.dot(c2, np.transpose(self.H))
        c4 = np.array(c3) + np.array(self.R)
        c5 = np.linalg.matrix_power(c4, -1)
        self.K = np.dot(c1, c5)

        # Xk = Xk_+Kk(Zk-Hk*Xk_)
        d1 = np.dot(self.H, x_)
        d2 = np.array(self.z) - np.array(d1)
        d3 = np.dot(self.K, d2)
        self.x = np.array(x_) + np.array(d3)

        # Pk = Pk_-Kk*Hk*Pk_
        e1 = np.dot(self.K, self.H)
        e2 = np.dot(e1, p_)
        self.P = np.array(p_) - np.array(e2)

    def accuracy(self, predictions, labels):
        return np.array(predictions) / np.array(labels)


def main():
    # 读取真实路径数据（客观真实的数据，作为滤波器预测结果的对比标签）
    # 比如敌机的真实飞行轨迹
    label_x = [i for i in range(1, 10, 1)]
    label_y = [i + 1 for i in range(1, 10, 1)]
    label_data = np.array(list(zip(label_x, label_y)))

    # 读取检测路径数据（传感器检测到的原始数据，与真实值之间会存在误差，作为滤波器的输入）
    # 比如我方导弹获取的敌机飞行轨迹，只能获取到当前时刻之前的轨迹信息，而不能直接获取未来的轨迹
    detect_x = [i - 1 for i in range(1, 10, 1)]
    detect_y = [i + 3 for i in range(1, 10, 1)]
    detect_data = np.array(list(zip(detect_x, detect_y)))

    # 卡尔曼滤波（根据卡尔曼对当前时刻的预测数据和当前时刻的观测数据，尽可能地输出下一时刻接近真实数据的数据）
    # 实现对敌机未来飞行轨迹的估计，达到跟踪目标的效果
    t = len(detect_data)  # 处理时刻
    kf_data_filter = np.zeros((t, 4))  # 滤波数据
    kf_data_predict = np.zeros((t, 4))  # 预测数据

    # 初始化（创建滤波器，并初始化滤波器状态）
    kf = KalmanFilter()
    kf.init(detect_x[0], detect_y[0], 0, 0)

    # 滤波处理（依次读取每一时刻的数据，输入到卡尔曼滤波器，输出预测结果）
    for i in range(t):
        if i == 0:
            kf.init(detect_x[0], detect_y[i], 0, 0)  # 初始化
        else:
            kf.z = np.transpose([detect_x[i], detect_y[i]])  # 获取当前时刻的观测数据
            kf.update()  # 更新卡尔曼滤波器参数
        kf_data_filter[i, ::] = np.transpose(kf.x)
        kf_data_predict[i, ::] = np.transpose(kf.G)

    # 某段时间内的数据
    kf_filter = kf_data_filter[::, :2]
    kf_predict = kf_data_predict[::, :2]

    # 评价（计算卡尔曼滤波器的预测精度）
    precision_detect = kf.accuracy(detect_data, label_data)
    precision_filter = kf.accuracy(kf_filter, label_data)
    print("-" * 100)
    print("%-4s \t %-20s \t %-20s \t %-20s \t %-20s " % (
        "time", "detect gap x", "filter gap x", "detect gap y", "filter gap y"))
    print("-" * 100)
    for i in range(len(precision_filter)):
        print("%-4s \t %-20s \t %-20s \t %-20s \t %-20s " % (i,
                                                             precision_detect[i][0], precision_filter[i][0],
                                                             precision_detect[i][1], precision_filter[i][1]))
    print("-" * 100)

    # 可视化（对原始数据进行可视化）
    plt.figure()
    # plt.plot(label_x, label_y, 'b-+')
    # plt.plot(detect_x, detect_y, 'r-+')
    # # 可视化（对滤波结果进行可视化）
    plt.plot(kf_filter[::, 0], kf_filter[::, 1], 'g-+')
    plt.plot(kf_predict[::, 1], kf_predict[::, 1], 'm-+')
    # legend = ['reality data', 'detect data', 'filter data', 'predict data']
    # plt.legend(legend, loc="best", frameon=False)
    # plt.title('kalman filter')
    # plt.savefig('result.svg', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
