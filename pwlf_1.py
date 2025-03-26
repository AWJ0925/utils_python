import numpy as np
import matplotlib.pyplot as plt


X = np.linspace(-10, 10, 30)
Y1 = 2*(X[:10]**2) + 3*X[:10] + 5
Y2 = X[10:20] + 0
Y3 = -(X[20:]**3)
Y = list()
for i in range(10):
    Y.append(Y1[i])

for i in range(10):
    Y.append(Y2[i])

for i in range(10):
    Y.append(Y3[i])


# initialize piecewise linear fit with your x and y data
import pwlf
my_pwlf = pwlf.PiecewiseLinFit(X, Y)
# fit the data for four line segments
# this performs 3 multi-start optimizations
res = my_pwlf.fitfast(4, pop=3)
# predict for the determined points
xHat = np.linspace(-10, 10, num=20)
yHat = my_pwlf.predict(xHat)

# plot the results
plt.figure()
plt.scatter(X, Y)
plt.plot(xHat, yHat, '-')
plt.show()