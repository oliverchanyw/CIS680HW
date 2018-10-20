import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

'''
Define the sigmoid function given some bias and weight
'''
def sig(bias, weight):
    input = 1  # For some fixed input, we will plot the surface
    return (1 + math.exp(-(weight * input + bias))) ** -1


x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
ACTIVATION = np.vectorize(sig)(X, Y)

def l2loss(bias, weight):
    return (sig(bias, weight) - 0.5) ** 2

LOSS = np.vectorize(l2loss)(X, Y)

def lossgradient(bias, weight):
    input = 1  # For some fixed input, we will plot the surface
    eterm = math.exp(-(weight * input + bias))
    return 2 * (sig(bias, weight) - 0.5) * sig(bias, weight) ** 2 * eterm

GRADIENT = np.vectorize(lossgradient)(X, Y)

def crossentropyloss(bias, weight):
    input = 1  # For some fixed input, we will plot the surface
    yhat = sig(bias, weight)
    y = 0.5
    return -(y * math.log(yhat, 2) + (1 - y) * math.log(1 - yhat, 2))

CROSSENTROPYLOSS = np.vectorize(crossentropyloss)(X, Y)

def crossentropygradient(bias, weight):
    input = 1  # For some fixed input, we will plot the surface
    eterm = math.exp(-(weight * input + bias))
    yhat = sig(bias, weight)
    y = 0.5
    return -0.5 * (yhat ** -1 - (1 - yhat) ** - 1) * yhat ** 2 * eterm

CROSSENTROPYGRADIENT = np.vectorize(crossentropygradient)(X, Y)

# Sub in ACTIVATION or LOSS or GRADIENT to plot the right one
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, CROSSENTROPYGRADIENT, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.5, 0.5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# Add title
fig.suptitle('Gradient of Cross-entropy loss with y = 0.5', fontsize=14)

# Add axis labels
ax.set_xlabel('Bias')
ax.set_ylabel('Weight')
ax.set_zlabel('Gradient of cross-entropy loss')

# Display
plt.show()
