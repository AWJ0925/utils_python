import scipy.interpolate as spi

def cubic_one(lane):
    lane.sort(key=lambda x: x[1])  # 排序

    X = []
    Y = []
    for i in range(len(lane)):
        X.append(lane[i][0])
        Y.append(lane[i][1])

    ipo1 = spi.splrep(np.array(Y), np.array(X), k=1)
    new_Y = np.arange(Y[0], Y[-1], 5)
    new_X = spi.splev(new_Y, ipo1)

    if len(new_Y) > 4:
        cubic_y_min = max((Y[0]//10+1)*10, 5)
        cubic_y_max = ((Y[-1]+3)//10)*10

        ipo3 = spi.splrep(new_Y, new_X, k=3)
        txt_cubic_Y = np.arange(cubic_y_min, cubic_y_max, 10)
        txt_cubic_X = spi.splev(txt_cubic_Y, ipo3)
        txt_cubic = list(zip(txt_cubic_X[::-1], txt_cubic_Y[::-1]))
    else:
        txt_cubic = []

    return txt_cubic

def spline_inter(lanes):
    '''样条拟合'''
    new_lanes = []
    for i in range(len(lanes)):
        new_lanes.append(cubic_one(lanes[i]))
    return new_lanes


