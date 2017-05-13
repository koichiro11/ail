# coding: utf-8

__author__ = "Hiromi Nakagawa"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calc_theta(Phi, W, train_y):
	a = np.linalg.inv(Phi.T.dot(W).dot(Phi))
	b = Phi.T.dot(W).dot(train_y)
	return np.dot(a, b)


def func(train_x, theta):
	return 1*theta[0] + train_x*theta[1]

def update_W(W, theta, train_x, train_y, eta):
	r = func(train_x, theta) - train_y
	for i in range(n):
		if np.abs(r[i]) > eta:
			W[i,i] = 0
		else:
			W[i,i] = (1-r[i]**2/eta**2)**3
	return W


# データ生成
np.random.seed(1234)
n = 10
N = 1000
train_x = np.linspace(-3, 3, n).reshape(n,1)
train_y = train_x + 0.2*np.random.randn(n,1)
train_y[n-1] = -4
train_y[n-2] = -4
train_y[1] = -4


# Phi, W, theta, etaの設定
Phi = np.ones((n, 2))
for i in range(n):
	Phi[i,1] = train_x[i]

W = np.zeros((n,n))
W[np.arange(n), np.arange(n)] = 1/6

theta = calc_theta(Phi, W, train_y)

eta = 1

# 繰り返し最小二乗
for step in range(N):
	W = update_W(W, theta, train_x, train_y, eta)
	theta = calc_theta(Phi, W, train_y)

pred_y = func(train_x, theta)
plt.scatter(np.arange(n), train_y, label="train_y")
plt.plot(pred_y, label="pred_y")
plt.legend()
plt.show()


