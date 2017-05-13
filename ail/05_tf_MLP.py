	# no decay
	# no dropout
	# no custom optimizer


	def homework(train_X, train_y, test_X):
	    
		import numpy as np
		import tensorflow as tf 
		from sklearn.utils import shuffle
		from sklearn.metrics import f1_score
		from sklearn.datasets import fetch_mldata
		from sklearn.model_selection import train_test_split

		def one_hot_encode(x, n_label):
			one_hot = np.zeros((x.shape[0], n_label))
			one_hot[np.arange(len(x)), x] = 1
			return one_hot

		def get_init_weight(rng, n_in, n_out, default_weight=None):

			if default_weight:
				w = default_weight
			else:
				w = np.sqrt(6. / (n_in + n_out))
			
			return rng.uniform(
				low=-w, 
				high=w, 
				size=(n_in, n_out)).astype('float32')


		class EarlyStopping:

			def __init__(self):

				self.best_loss = float('inf')
				self.stop_count = 0

			def check(self, loss):

				if loss < self.best_loss:
					self.best_loss = loss
					self.stop_count = 0
				else:
					self.stop_count += 1

				return self.stop_count


		class Model:

			def __init__(self, n_hiddens=[400, 200], act_funcs=["tanh", "tanh"], lr=0.01):
			

				tf.reset_default_graph()
				self.n_hiddens = n_hiddens
				activations = {
					"sigmoid": tf.nn.sigmoid,
					"tanh": tf.tanh,
					"relu": tf.nn.relu,
					"elu": tf.nn.elu,
				}
				self.act_funcs = [activations[act_func] for act_func in act_funcs]
				self.act_funcs.append(tf.nn.softmax)
				self.lr = lr
				self.train_size = 0.7
				self.rng = np.random.RandomState(1234)

			def make_graph(self,):

				# placeholders
				self.x = tf.placeholder(tf.float32, [None, 784], name="x")
				self.t = tf.placeholder(tf.float32, [None, 10], name="t")

				# variables
				ins = np.concatenate([[784], self.n_hiddens])
				outs = np.concatenate([self.n_hiddens, [10]])

				# fprop
				u = self.x
				params = []
				for idx, (n_in, n_out) in enumerate(zip(ins, outs)):
					W = tf.Variable(get_init_weight(rng=self.rng, n_in=n_in, n_out=n_out, default_weight=0.08), name="W%d" % (idx+1,))
					b = tf.Variable(np.zeros(n_out).astype('float32'), name="b%d" % (idx+1,))
					params += [W, b]
					u = tf.matmul(u, W) + b
					u = self.act_funcs[idx](u)
				y = u
				clipped_y = tf.clip_by_value(y, 1e-10, 1.0)
				self.cost = -tf.reduce_mean(tf.reduce_sum(self.t*tf.log(clipped_y), axis=1))

				# updates
				gparams = tf.gradients(self.cost, params)
				updates = [param.assign_add(-self.lr*gparam) for param, gparam in zip(params, gparams)]

				# functions
				self.train = tf.group(*updates)
				self.valid = tf.argmax(y, axis=1)

			def batch_training(self, train_X, train_y, batch_size=100):

				train_costs, valid_costs = [], []
				for start in range(0, train_X.shape[0], batch_size):
					batch_X, batch_y = train_X[start:start+batch_size], train_y[start:start+batch_size]
					batch_train_X, batch_valid_X, batch_train_y, batch_valid_y = train_test_split(batch_X, batch_y, train_size=self.train_size, random_state=1234)
					sess.run(self.train, feed_dict={self.x: batch_train_X, self.t: batch_train_y})
					pred_y, valid_cost = sess.run([self.valid, self.cost], feed_dict={self.x: batch_valid_X, self.t: batch_valid_y})
					valid_costs.append(valid_cost)
				return np.mean(valid_costs)


		n_hiddens = [400, 200]
		act_funcs = ["tanh", "tanh"]
		lr = 0.1
		model = Model(n_hiddens=n_hiddens, act_funcs=act_funcs, lr=lr)
		model.make_graph()

		train_y = one_hot_encode(train_y, 10)
		max_epoch = 1000
		early_stopping = EarlyStopping()

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(max_epoch):
				train_costs, valid_costs = [], []
				train_X, train_y = shuffle(train_X, train_y)
				loss = model.batch_training(train_X, train_y)
				if epoch % 5 == 0:
					print("E %03d, valid_loss: %.5f" % (epoch, loss))
				stop_count = early_stopping.check(loss)
				if stop_count >= 5:
					break

			pred_y = sess.run(model.valid, feed_dict={model.x: test_X})

	    return pred_y

