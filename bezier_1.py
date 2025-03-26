import math
import numpy as np
import matplotlib.pyplot as plt


class Bezier:
    def __init__(self, points, baseRatio=0):
        self.curve = []  # 曲线点
        self.ratio = baseRatio  # 初始比例0   取值(0,1)
        self.Points = points  # 给定拟合的点  6个点即5阶贝塞尔曲线

    def calculation_p(self):
        n = len(self.Points) - 1
        p = len(self.Points)
        L = np.arange(self.ratio, 1, 0.001)  # 计算不比例下的点
        n_fac = math.factorial(n) * np.ones(p)
        k_fac, n_k_fac, coe_t, coe_1_t = [], [], [], []
        for t in L:
            # 方案一
            k_fac.clear()
            n_k_fac.clear()
            coe_t.clear()
            coe_1_t.clear()
            for i in range(p):  # 依据公式计算每个比例下的点
                k_fac.append(math.factorial(i))
                n_k_fac.append(math.factorial(n - i))
                coe_t.append(t ** i)
                coe_1_t.append((1 - t) ** (n - i))
            b = sum((n_fac / (np.array(k_fac) * np.array(n_k_fac)) * np.array(coe_t) * np.array(coe_1_t)).reshape(-1,
                1) * np.array(self.Points))

            '''
            # 方案二
            b = np.zeros(2)
            for i in range(p):  # 依据公式计算每个比例下的点
                temp = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
                b += temp * (1 - t) ** (n - i) * t ** i * np.array(self.Points[i])
            '''

            self.curve.append(b)  # 添加到曲线列表中
        plt.scatter(np.array(self.Points)[:, 0], np.array(self.Points)[:, 1], c='black')  # 画原始点
        plt.plot(np.array(self.Points)[:, 0], np.array(self.Points)[:, 1], c='black')  # 画原始图

        plt.plot(np.array(self.curve)[:-2, 0], np.array(self.curve)[:-2, 1], c='red')  # 画拟合曲线
        plt.show()


if __name__ == '__main__':
    Points = [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125), (6, 196)]
    cur_1 = Bezier(Points)
    cur_1.calculation_p()