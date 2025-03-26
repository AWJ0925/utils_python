# 计算两条直线的交点
import numpy as np


def cross_point(line1, line2):  # 计算交点函数
    x1, y1 = line1[0], line1[1]
    x2, y2 = line1[2], line1[3]

    x3, y3 = line2[0], line2[1]
    x4, y4 = line2[2], line2[3]

    if x2 - x1 == 0:  # L1 直线斜率不存在
        k1 = None
        b1 = 0
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k1 is None and k2 is None:  # L1与L2直线斜率都不存在，两条直线均与y轴平行
        if x1 == x3:  # 两条直线实际为同一直线
            return [x1, y1]  # 均为交点，返回任意一个点
        else:
            return None  # 平行线无交点
    elif k1 is not None and k2 is None:  # 若L2与y轴平行，L1为一般直线，交点横坐标为L2的x坐标
        x = x3
        y = k1 * x * 1.0 + b1 * 1.0
    elif k1 is None and k2 is not None:  # 若L1与y轴平行，L2为一般直线，交点横坐标为L1的x坐标
        x = x1
        y = k2 * x * 1.0 + b2 * 1.0
    else:  # 两条一般直线
        if k1 == k2:  # 两直线斜率相同
            if b1 == b2:  # 截距相同，说明两直线为同一直线，返回任一点
                return [x1, y1]
            else:  # 截距不同，两直线平行，无交点
                return None
        else:  # 两直线不平行，必然存在交点
            x = (b2 - b1) * 1.0 / (k1 - k2)
            y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


if __name__ == '__main__':
    line1 = [1, 1, 2, 2]  # y = x
    line2 = [0, 3, 3, 0]  # y = -x +3
    print(cross_point(line1, line2))