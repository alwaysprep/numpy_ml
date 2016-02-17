import matplotlib.pyplot as plt
from sklearn import datasets

from npml.LinearRegression import LinearRegressionModel

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data[:, 2:3]

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

regr = LinearRegressionModel()

regr.fit(diabetes_X_train, diabetes_y_train)

plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')

plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
