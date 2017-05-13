# coding: utf-8

__author__ = "Hiromi Nakagawa"

import numpy as np
import pandas as pd 
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

class GaussKernelModel:

	def __init__(self, h=0.3, _lambda=0.1):

		self.h = h
		self._lambda = _lambda


	def fit(self, train_X, train_y):

		K = self.kernel_matrix(train_X, train_X)
		signed_y = np.array([self.sign(train_y, _class) for _class in range(10)]).T
		W = np.linalg.inv(K.T.dot(K) + self._lambda*np.identity(K.shape[0])).dot(K.T).dot(signed_y)
		self.W = W.T


	def predict(self, test_X):
		"""
		W: (10, n_train)
		kernel_matrix(train_X, test_X): (n_train, n_test)
		"""
		predict_matrix = np.dot(self.W, self.kernel_matrix(train_X, test_X))	# (10, n_test)
		prediction = np.argmax(predict_matrix, axis=0)

		return prediction


	def gauss_kernel(self, x, c):

		return np.exp(-np.power(x - c, 2).sum() / (2*self.h**2))


	def kernel_matrix(self, X, X_j):

		matrix = [[self.gauss_kernel(x, x_j) for x_j in X_j] for x in X]
		matrix = np.array(matrix).reshape((len(X), len(X_j)))

		return matrix


	def sign(self, y, _class):

		res = -np.ones(y.shape)
		res[np.where(y == _class)] = 1

		return res



def load_data():

	n_train_per_class = 500	# max: 500
	n_test_per_class = 200	# max: 200


	DIR = "./digit/"
	train_Xs = []
	for i in range(10):
		train_Xs.append(pd.read_csv(DIR + "digit_train%i.csv" % i, header=None)[:n_train_per_class])
	train_X = pd.concat(train_Xs)
	train_X = np.array(train_X)
	train_y = np.array([np.tile(i, n_train_per_class) for i in range(10)]).ravel()

	DIR = "./digit/"
	test_Xs = []
	for i in range(10):
		test_Xs.append(pd.read_csv(DIR + "digit_test%i.csv" % i, header=None)[:n_test_per_class])
	test_X = pd.concat(test_Xs)
	test_X = np.array(test_X)
	test_y = np.array([np.tile(i, n_test_per_class) for i in range(10)]).ravel()

	return train_X, train_y, test_X, test_y


if __name__ == "__main__":

	train_X, train_y, test_X, test_y = load_data()
	model = GaussKernelModel()
	model.fit(train_X, train_y)
	pred_y = model.predict(test_X)

	print(classification_report(test_y, pred_y))
	print(confusion_matrix(test_y, pred_y))

