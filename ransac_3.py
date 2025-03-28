"""_summary_
RANSAC直线拟合
"""
from copy import copy
import numpy as np
from numpy.random import default_rng

rng = default_rng()


class RANSAC:
    def __init__(self, n=10, k=100, t=0.05, d=10, model=None, loss=None, metric=None):
        self.n = n  # 估计模型所需的最小样本数
        self.k = k  # 最大迭代次数
        self.t = t  # 阈值
        self.d = d
        self.model = model  # 数学模型
        self.loss = loss
        self.metric = metric
        self.best_fit = None
        self.best_error = np.inf

    def fit(self, X, y):
        for _ in range(self.k):
            ids = rng.permutation(X.shape[0])
            maybe_inliers = ids[: self.n]
            maybe_model = copy(self.model).fit(X[maybe_inliers], y[maybe_inliers])

            thresholded = (
                    self.loss(y[ids][self.n:], maybe_model.predict(X[ids][self.n:]))
                    < self.t
            )

            inlier_ids = ids[self.n:][np.flatnonzero(thresholded).flatten()]

            if inlier_ids.size > self.d:
                inlier_points = np.hstack([maybe_inliers, inlier_ids])
                better_model = copy(self.model).fit(X[inlier_points], y[inlier_points])

                this_error = self.metric(
                    y[inlier_points], better_model.predict(X[inlier_points])
                )

                if this_error < self.best_error:
                    self.best_error = this_error
                    self.best_fit = maybe_model

        return self

    def predict(self, X):
        return self.best_fit.predict(X)


def square_error_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2


def mean_square_error(y_true, y_pred):
    return np.sum(square_error_loss(y_true, y_pred)) / y_true.shape[0]


class LinearRegressor:
    def __init__(self):
        self.params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        self.params = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X: np.ndarray):
        r, _ = X.shape
        X = np.hstack([np.ones((r, 1)), X])
        return X @ self.params


if __name__ == "__main__":
    regressor = RANSAC(model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)
    yuanShi = [438.00, 344.00, 470.00, 355.00, 503.00, 366.00, 538.00, 377.00, 573.00, 388.00,
               619.00, 407.00, 655.00, 414.00, 703.00, 431.00, 736.00, 441.00]

    X = list()
    Y = list()
    for i in range(len(yuanShi)):
        if i % 2 == 0:
            X.append(yuanShi[i])
        else:
            Y.append(yuanShi[i])
    X = np.array(X)
    Y = np.array(Y)
    X = np.array(
        [-0.848, -0.800, -0.704, -0.632, -0.488, -0.472, -0.368, -0.336, -0.280, -0.200,
         -0.00800, -0.0840, 0.0240, 0.100, 0.124, 0.148, 0.232, 0.236, 0.324, 0.356, 0.368,
         0.440, 0.512, 0.548, 0.660, 0.640, 0.712, 0.752, 0.776, 0.880, 0.920, 0.944, -0.108,
         -0.168, -0.720, -0.784, -0.224, -0.604, -0.740, -0.0440, 0.388, -0.0200, 0.752, 0.416,
         -0.0800, -0.348, 0.988, 0.776, 0.680, 0.880, -0.816, -0.424, -0.932, 0.272, -0.556, -0.568,
         -0.600, -0.716, -0.796, -0.880, -0.972, -0.916, 0.816, 0.892, 0.956, 0.980, 0.988, 0.992,
         0.00400]).reshape(-1, 1)
    y = np.array(
        [-0.917, -0.833, -0.801, -0.665, -0.605, -0.545, -0.509, -0.433, -0.397, -0.281, -0.205,
         -0.169, -0.0531, -0.0651, 0.0349, 0.0829, 0.0589, 0.175, 0.179, 0.191, 0.259, 0.287, 0.359,
         0.395, 0.483, 0.539, 0.543, 0.603, 0.667, 0.679, 0.751, 0.803, -0.265, -0.341, 0.111, -0.113,
         0.547, 0.791, 0.551, 0.347, 0.975, 0.943, -0.249, -0.769, -0.625, -0.861, -0.749, -0.945,
         -0.493, 0.163, -0.469, 0.0669, 0.891, 0.623, -0.609, -0.677, -0.721, -0.745, -0.885, -0.897,
         -0.969, -0.949, 0.707, 0.783, 0.859, 0.979, 0.811, 0.891, -0.137]).reshape(-1, 1)

    regressor.fit(X, y)

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1)
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title("RANSAC拟合直线")

    plt.scatter(X, y)  # 原始数据
    # 拟合后的数据
    line = np.linspace(-1, 1, num=100).reshape(-1, 1)
    plt.plot(line, regressor.predict(line), c="peru")
    plt.show()
