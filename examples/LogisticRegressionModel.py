import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split


from npml.LogisticRegression import LogisticRegressionModel

iris = datasets.load_iris()

X, x, Y, y = train_test_split(iris.data[0:100, :2], iris["target"][0:100], test_size=0.20)


logreg = LogisticRegressionModel()

logreg.fit(X, Y)

h = .02 
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)


# Plot also the training points
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
