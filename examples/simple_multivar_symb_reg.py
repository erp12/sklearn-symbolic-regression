import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from ..base import SymbolicRegressor

X_train = np.arange(-1.5, 1.5, 0.05)
np.random.shuffle(X_train)
X_train = np.reshape(X_train, (-1, 2))


def target_func(a, b): return (a ** 6) + (2 * b**4) + (a**2)


Y_train = []
for r in X_train:
    Y_train.append(target_func(*r))

print(X_train)
print(Y_train)
print

symbReg = SymbolicRegressor(verbose=1)
symbReg.fit(X_train, Y_train)

print(symbReg.best_error_)
print(symbReg.best_)


X_test = np.mgrid[-1:1.1:0.33, -1:1.1:0.33]
preds = symbReg.predict(X_train)

#
# Plotting
#

x_min, x_max = X_train.min() - .1, X_train.max() + .1
y_min, y_max = min(Y_train) - .1, max(Y_train) + .1

fig = plt.figure(1)

plt.subplot(2, 1, 1)
plt.scatter(X_train[:, 0], Y_train,  color='black')
plt.scatter(X_train[:, 0], preds, color='blue')

plt.xlabel('X1')
plt.ylabel('Y')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.subplot(2, 1, 2)
plt.scatter(X_train[:, 1], Y_train,  color='black')
plt.scatter(X_train[:, 1], preds, color='blue')

plt.xlabel('X2')
plt.ylabel('Y')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()
