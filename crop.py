import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
Y = np.arange(0, 1080, 100)
X = -2480 - 3.09*Y
plt.scatter(X, Y)
plt.show()