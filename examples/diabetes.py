import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from ..base import SymbolicRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data,
                                                    diabetes.target,
                                                    test_size=0.5)

symbReg = SymbolicRegressor(verbose=1, generations=75, population_size=100)
symbReg.fit(X_train, y_train)
y_pred = symbReg.predict(X_test)
print('SymbReg MSE', mean_squared_error(y_test, y_pred))


parameters = {'C':[0.01, 0.1, 1.0, 10, 100]}
svr = SVR()
clf = GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('SVR MSEs', mean_squared_error(y_test, y_pred))
