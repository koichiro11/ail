# coding: utf-8

__author__ = "Hiromi Nakagawa"

import numpy as np
import sys

def gauss_kernel(x, c, h):

	return np.exp(- (x - c)**2 / (2*h**2))


def kernel_matrix(X, X_j, h):

	matrix = [[gauss_kernel(x, x_j, h) for x_j in X_j] for x in X]
	#matrix = np.array(matrix)
	matrix = np.array(matrix).reshape((len(X), len(X_j)))

	return matrix


def train_test_split(x, y, start, valid_size):

	valid_start = start * valid_size
	train_start = valid_start + valid_size
	train_x = np.concatenate([x[train_start:], x[:valid_start]])
	train_y = np.concatenate([y[train_start:], y[:valid_start]])
	valid_x = x[valid_start:train_start]
	valid_y = y[valid_start:train_start]

	return train_x, train_y, valid_x, valid_y


def cross_validation(N, CV, l, s):

	costs = []
	valid_size = N // CV
	train_size = N - valid_size
	for i in range(CV):
		train_x, train_y, valid_x, valid_y = train_test_split(x, y, i, valid_size)
		K = kernel_matrix(train_x, train_x, s)
		theta_hat = np.linalg.inv(K**2 + np.identity(train_size)*l).dot(K.T).dot(train_y)
		y_hat = np.dot(kernel_matrix(valid_x, train_x, s), theta_hat)
		cost = np.sum((y_hat - valid_y)**2) / len(valid_y)
		costs.append(cost)

	return np.mean(costs)


# パラメータの一覧
params = {
	"lambda": [0.01, 0.1, 1],	# 正則化パラメータ
	"sigma": [0.01, 0.1, 1]		# ガウス分布のバンド幅
}

np.random.seed(1234)
N = 100
CV = 5
x = np.random.randn(N, 1)
pix = np.pi*x
y = np.sin(pix)/pix + 0.1*x + 0.05*np.random.randn(N, 1)


results = {}
for lamb in params["lambda"]:
	for sig in params["sigma"]:
		cost = cross_validation(N, CV, lamb, sig)
		results[(lamb, sig)] = cost
		print("lambda: %.2f, sigma: %.2f, cost: %.5f" % (lamb, sig, cost))

best = sorted(results.items(), key=lambda n: n[1])[0]
print("BEST PARAMS:  lambda: %.2f, sigma: %.2f" % (best[0][0], best[0][1]))
print("MINIMAM COST:  %.5f" % (best[1]))
