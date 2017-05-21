import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from ..base import SymbolicRegressor

X_train = np.arange(-1.5, 1.5, 0.05)
np.random.shuffle(X_train)
X_train = np.reshape(X_train, (-1, 2))


def target_func(row): return (row[0] ** 6) + (2 * row[1]**4) + (row[0]**2)

Y_train = []
for r in X_train:
    Y_train.append(target_func(r))

print(X_train)
print(Y_train)
print

symbReg = SymbolicRegressor(verbose=1)
symbReg.fit(X_train, Y_train)

print(symbReg.best_error_)
print(symbReg.best_)

X1_test = np.arange(-1.5, 1.5, 0.1)
X2_test = np.arange(-1.5, 1.5, 0.1)
X1_test, X2_test = np.meshgrid(X1_test, X2_test)
X_test = np.hstack((X1_test.reshape(-1, 1), X2_test.reshape(-1, 1)))

print(X_test.shape)

Y_hat = symbReg.predict(X_test)
Y_true = np.apply_along_axis(target_func, 1, X_test)

print(Y_hat.shape)
print(Y_true.shape)

#
# Plotting
#

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
ax.scatter(X_test[:,0], X_test[:,1], Y_hat, linewidth=0, c='red',
                antialiased=False, alpha=0.5)
ax.scatter(X_test[:,0], X_test[:,1], Y_true, linewidth=0, c='blue',
                antialiased=False, alpha=0.5)

plt.show()

# x_min, x_max = X_train.min() - .1, X_train.max() + .1
# y_min, y_max = min(Y_train) - .1, max(Y_train) + .1
#
# fig = plt.figure(1)
#
# plt.subplot(2, 1, 1)
# plt.scatter(X_train[:, 0], Y_train,  color='black')
# plt.scatter(X_train[:, 0], preds, color='blue')
#
# plt.xlabel('X1')
# plt.ylabel('Y')
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
#
# plt.subplot(2, 1, 2)
# plt.scatter(X_train[:, 1], Y_train,  color='black')
# plt.scatter(X_train[:, 1], preds, color='blue')
#
# plt.xlabel('X2')
# plt.ylabel('Y')
#
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
#
# plt.show()
