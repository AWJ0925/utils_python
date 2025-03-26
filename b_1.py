import numpy as np
import matplotlib.pyplot as plt


def de_boor_cox(x, k, i, t):
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    else:
        num1 = (x - t[i]) * de_boor_cox(x, k-1, i, t)
        num2 = (t[i+k+1] - x) * de_boor_cox(x, k-1, i+1, t)
        den1 = t[i+k] - t[i]
        den2 = t[i+k+1] - t[i+1]
        term1 = num1 / den1 if den1 != 0 else 0
        term2 = num2 / den2 if den2 != 0 else 0
        return term1 + term2


"""quasi-uniform b-spline curve"""
def cubic_bspline(t, control_points):
    n = len(control_points)
    k = 3
    x_points = []
    y_points = []

    for x in np.arange(t[k], t[-k-1], 0.01):
        x_point = 0
        y_point = 0
        for i in range(n):
            b = de_boor_cox(x, k, i, t)
            x_point += control_points[i][0] * b
            y_point += control_points[i][1] * b
        x_points.append(x_point)
        y_points.append(y_point)

    return x_points, y_points


def plot_bspline(control_points, x_points, y_points):
    control_points = np.array(control_points)

    plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
    plt.plot(x_points, y_points, 'b-', label='B-spline Curve')
    plt.scatter(x_points, y_points,  label='anwj')

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic B-spline Curve')
    plt.grid()
    plt.show()

"""plot t - f(t), x(t),y(t) """
def plot_points(t_v1,points,title = ""):
    plt.figure()
    plt.plot(t_v1, points, 'b-')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title(title)
    plt.grid()


if __name__ == "__main__":
    control_points = [(0, 0), (2, 3), (4, 3), (6, 0), (8, -3), (10, -1)]
    n = len(control_points)
    k = 3

    # 生成均匀节点向量  len(t) ==  k + n - k + 1 + k  == k + n + 1
    t = np.concatenate((np.zeros(k), np.arange(n - k + 1), np.ones(k) * (n - k)), axis=0)
    print("t = ", t)  # t =  [0. 0. 0. 0. 1. 2. 3. 3. 3. 3.]

    # 计算B样条曲线上的点
    x_points, y_points = cubic_bspline(t, control_points)

    # 绘制曲线
    plot_bspline(control_points, x_points, y_points)

    t_v1 = np.arange(t[k], t[-k - 1], 0.01)
    plot_points(t_v1, x_points, "t - x(t)")
    plot_points(t_v1, y_points, "t - y(t)")
    plt.show()

