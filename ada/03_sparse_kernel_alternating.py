# coding: utf-8

__author__ = "Hiromi Nakagawa"

import numpy as np
import matplotlib.pyplot as plt


class GaussKernelModel:

	def __init__(self, h=0.3, _lambda=0.1):

		self.h = h
		self._lambda = _lambda
		self.target_sigma = 0.1

	def fit(self, train_x, train_y):

		self.train_x = train_x
		K = self.kernel_matrix(train_x, train_x)
		self.theta = np.random.rand(K.shape[0])
		z = np.random.rand(K.shape[0])
		u = np.random.rand(K.shape[0])
		sigma = float("inf")

		#count = 0
		while sigma > self.target_sigma:
			#count += 1
			#print("step%03d/  sigma: %.3f" % (count, sigma))

			# theta
			theta_hat = np.linalg.inv(K.T.dot(K) + np.identity(K.shape[0])).dot((K.T).dot(train_y) + z - u)
			sigma = np.linalg.norm(self.theta - theta_hat)
			self.theta = theta_hat

			# z

			if np.linalg.norm(self.theta + u - self._lambda) > 0:
				z = self.theta + u - self._lambda
			elif np.linalg.norm(self.theta + u + self._lambda) < 0:
				z = self.theta + u + self._lambda
			else:
				z = np.zeros(K.shape)

			# u 
			u += self.theta - z


	def predict(self, test_x):

		y_hat = np.dot(self.kernel_matrix(test_x, self.train_x), self.theta)
		return y_hat


	def gauss_kernel(self, x, c):

		return np.exp(- (x - c)**2 / (2*self.h**2))
		#return np.exp(- np.power(x-c, 2).sum() / (2*self.h**2))


	def kernel_matrix(self, X, X_j):

		matrix = [[self.gauss_kernel(x, x_j) for x_j in X_j] for x in X]
		matrix = np.array(matrix).reshape((len(X), len(X_j)))

		return matrix


def sin_func(x):

	pix = np.pi*x
	y = np.sin(pix)/pix + 0.1*x + 0.05*np.random.randn(len(x))	
	return y


np.random.seed(1234)
train_x, test_x = np.linspace(-1, 1, 500), np.linspace(-1, 1, 50)
train_y, test_y = sin_func(train_x), sin_func(test_x)
model = GaussKernelModel()
model.fit(train_x, train_y)

pred_y = model.predict(test_x)

plt.plot(test_x, test_y, label="test_y")
plt.plot(test_x, pred_y, label="pred_y")
plt.legend()
plt.show()
