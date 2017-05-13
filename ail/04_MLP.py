
def homework(train_X, train_y, test_X):

	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.utils import shuffle

	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	def deriv_sigmoid(x):
		return sigmoid(x)*(1-sigmoid(x))

	def tanh(x):
		return np.tanh(x)

	def deriv_tanh(x):
		return 1-tanh(x)**2

	def softmax(x):
		exp_x = np.exp(x)
		return exp_x / np.sum(exp_x, axis=1, keepdims=True)

	def deriv_softmax(x):
		return softmax(x)*(1-softmax(x))

	def multiclass_entropy(y, t):
		return -np.sum(t*np.log(y)) / len(y)

	def fprop(x, W1, b1, W2, b2, W3, b3, act_func):
		u1 = np.matmul(x, W1) + b1
		z1 = act_func(u1)
		u2 = np.matmul(z1, W2) + b2
		z2 = act_func(u2)
		u3 = np.matmul(z2, W3) + b3
		z3 = softmax(u3)
		y = z3
		return u1, z1, u2, z2, u3, z3, y


	def train(x, t, W1, b1, W2, b2, W3, b3, activation="tanh", eps=1.0):

		act_func, deriv_act_func = act_dict[activation]

		u1, z1, u2, z2, u3, z3, y = fprop(x, W1, b1, W2, b2, W3, b3, act_func)

		cost = multiclass_entropy(y, t)

		delta_3 = y-t
		delta_2 = deriv_act_func(u2)*np.matmul(delta_3, W3.T)
		delta_1 = deriv_act_func(u1)*np.matmul(delta_2, W2.T)

		dW1 = np.matmul(x.T, delta_1)
		db1 = np.matmul(np.ones(len(x)), delta_1)
		W1 = W1 - eps*dW1
		b1 = b1 - eps*db1

		dW2 = np.matmul(z1.T, delta_2)
		db2 = np.matmul(np.ones(len(z1)), delta_2)
		W2 = W2 - eps*dW2
		b2 = b2 - eps*db2

		dW3 = np.matmul(z2.T, delta_3)
		db3 = np.matmul(np.ones(len(z2)), delta_3)
		W3 = W3 - eps*dW3
		b3 = b3 - eps*db3


		return cost, W1, b1, W2, b2, W3, b3

	def validate(x, t, W1, b1, W2, b2, W3, b3, activation="tanh"):
		act_func, deriv_act_func = act_dict[activation]
		u1, z1, u2, z2, u3, z3, y = fprop(x, W1, b1, W2, b2, W3, b3, act_func)
		cost = multiclass_entropy(y, t)

		return cost

	def predict(x, W1, b1, W2, b2, W3, b3, activation="tanh"):
		act_func, deriv_act_func = act_dict[activation]
		u1, z1, u2, z2, u3, z3, y = fprop(x, W1, b1, W2, b2, W3, b3, act_func)

		return y

	def one_hot_encode(x, n_label):
		one_hot = np.zeros((x.shape[0], n_label))
		one_hot[np.arange(len(x)), x] = 1
		return one_hot

	act_dict = {
		"sigmoid": (sigmoid, deriv_sigmoid),
		"tanh": (tanh, deriv_tanh),
		}

	def get_init_weight(n_in, n_out, default_weight=True):
		if default_weight:
			w = 0.08
		else:
			w = np.sqrt(6. / (n_in + n_out))
		
		return np.random.uniform(
			low=-w, 
			high=w, 
			size=(n_in, n_out)).astype('float32')

	np.random.seed(1234)

	# params
	train_size = 0.7
	eps = 1e-2
	max_epoch = 100
	n_hidden1 = 150
	n_hidden2 = 50
	batch_size = 20

	# preprocess
	n_feature = train_X.shape[1]
	n_label = len(np.unique(train_y))
	train_y = one_hot_encode(train_y, n_label)


	# Layer1 weights		
	W1 = get_init_weight(n_feature, n_hidden1)
	b1 = np.zeros(n_hidden1).astype('float32')

	# Layer2 weights
	W2 = get_init_weight(n_hidden1, n_hidden2)
	b2 = np.zeros(n_hidden2).astype('float32')

	# Layer3 weights
	W3 = get_init_weight(n_hidden2, n_label)
	b3 = np.zeros(n_label).astype('float32')

	stop_count = 0
	min_valid_cost = float("inf")
	for epoch in range(max_epoch):
		train_costs, valid_costs = [], []
		train_X, train_y = shuffle(train_X, train_y)
		for start in range(0, train_X.shape[0], batch_size):
			batch_X, batch_y = train_X[start:start+batch_size], train_y[start:start+batch_size]
			batch_train_X, batch_valid_X, batch_train_y, batch_valid_y = train_test_split(batch_X, batch_y, train_size=train_size, random_state=1234)
			train_cost, W1, b1, W2, b2, W3, b3 = train(batch_train_X, batch_train_y, W1, b1, W2, b2, W3, b3, eps=eps)
			train_costs.append(train_cost)
			valid_cost = validate(batch_valid_X, batch_valid_y, W1, b1, W2, b2, W3, b3)			
			valid_costs.append(valid_cost)

		avg_train_cost, avg_valid_cost = np.mean(train_costs), np.mean(valid_costs)
		#if epoch % 5 == 0:
		#	print("E %03d, train:%.5f, valid: %.5f" % (epoch, avg_train_cost, avg_valid_cost))
		if avg_valid_cost < min_valid_cost:
			min_valid_cost = avg_valid_cost
			stop_count = 0
		else:
			stop_count += 1

		if stop_count >= 5:
			#print("Early Stopping")
			break

	pred_y = predict(test_X, W1, b1, W2, b2, W3, b3)
	pred_y = np.argmax(pred_y, axis=1)
	return pred_y
