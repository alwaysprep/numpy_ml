import numpy as np

class LinearRegressionModel(object):

	def __init__(self, alpha=0.001):
		self.alpha = alpha

	def initialiseQ(self, X):
		self.Q = np.zeros(X.shape[1] + 1)
		self.prevQ = float("inf")

	def initialiseX(self, X):
		self.X = np.insert(X, 0, 1, axis=1)

	def fit(self, X, Y):
		self.initialiseX(X)
		self.initialiseQ(X)

		while (sum(abs(self.Q - self.prevQ)) > 10**-3):
			self.updateQ(X, Y)

	def updateQ(self, X,  Y):
		self.prevQ = self.Q
		temp = np.zeros(len(self.Q))
		for i in range(len(self.Q)):
			h_x = np.dot(self.X, self.Q)
			temp[i] = self.Q[i] - np.sum( (h_x - Y) * self.X[:,i]) * self.alpha
		self.Q = temp

	def predict(self, test):		
		test = np.insert(test, 0, 1, axis=1)
		return np.dot(test, self.Q)

