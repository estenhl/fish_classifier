import random
import numpy as np
import tensorflow as tf

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128

class CNN:
	def __init__(self, id, input_shape, classes, class_weights=None):
		self.id = id
		self.input_shape = input_shape
		self.classes = classes
		self.variables = {}

		height, width, channels = input_shape
		input_size = height * width * channels

		self.graph = tf.Graph()
		with tf.Session(graph=self.graph) as sess:
			self.x = tf.placeholder(tf.float32, [None, input_size], name='x_placeholder')
			self.y = tf.placeholder(tf.float32, [None, classes], name='y_placeholder')

			weights = self.weights(input_shape)
			biases = self.biases()

			if class_weights is None:
				class_weights = np.ones(classes) / np.sum(np.ones(classes))

			self.pred, self.layers = self.conv_net(self.x, input_shape, weights, biases)
			self.weighted_pred = tf.mul(self.pred, class_weights, name='weighted_pred')

			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.weighted_pred, self.y, name='softmax'), name='reduce_mean')
			self.optimizer = tf.train.AdamOptimizer(learning_rate=DEFAULT_LEARNING_RATE, name='adam').minimize(self.cost)

			correct_pred = tf.equal(tf.argmax(self.weighted_pred, 1), tf.argmax(self.y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	def weights(self, input_shape):
		height, width, channels = input_shape
		return {
			'wc1': tf.Variable(tf.random_normal([5, 5, channels, 32]), name='wc1'),
			'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='wc2'),
			'wc3': tf.Variable(tf.random_normal([5, 5, 64, 64]), name='wc3'),
			'wc4': tf.Variable(tf.random_normal([5, 5, 64, 128]), name='wc4'),
			'wc5': tf.Variable(tf.random_normal([5, 5, 128, 128]), name='wc5'),
			'wc6': tf.Variable(tf.random_normal([5, 5, 128, 256]), name='wc6'),
			'wd1': tf.Variable(tf.random_normal([256, 1024]), name='wd1'),
			'wd2': tf.Variable(tf.random_normal([1024, 512]), name='wd2'),
			'out': tf.Variable(tf.random_normal([512, self.classes]), name='out_weight')
		}

	def biases(self):
		return {
			'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
			'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
			'bc3': tf.Variable(tf.random_normal([64]), name='bc3'),
			'bc4': tf.Variable(tf.random_normal([128]), name='bc4'),
			'bc5': tf.Variable(tf.random_normal([128]), name='bc5'),
			'bc6': tf.Variable(tf.random_normal([256]), name='bc6'),
			'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
			'bd2': tf.Variable(tf.random_normal([512]), name='bd2'),
			'out': tf.Variable(tf.random_normal([self.classes]), name='out_bias')
		}

	def conv2d(self, x, W, b, strides=1, name=None):
		conv = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', name=name)
		conv = tf.nn.bias_add(conv, b)

		return tf.nn.relu(conv)

	def maxpool2d(self, x, k=2, name=None):
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

	def conv_net(self, x, input_shape, weights, biases):
		height, width, channels = input_shape
		x = tf.reshape(x, shape=[-1, height, width, channels])
		layers = []

		# Conv1
		conv1 = self.conv2d(x, weights['wc1'], biases['bc1'], name='conv1')
		depth = weights['wc1'].get_shape().as_list()[3]
		size = str(input_shape[0]) + 'x' + str(input_shape[1]) + 'x' + str(depth)
		layers.append({'layer': conv1, 'name': 'conv1', 'size': size})

		# Pool1
		k1 = 2
		pool1 = self.maxpool2d(conv1, k=k1, name='pool1')
		size = str(int(input_shape[0]/k1)) + 'x' +  str(int(input_shape[1]/k1)) + 'x' + str(depth)
		layers.append({'layer': pool1, 'name': 'pool1', 'size': size})

		# Conv2
		conv2 = self.conv2d(pool1, weights['wc2'], biases['bc2'], name='conv2')
		depth = weights['wc2'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/k1)) + 'x' +  str(int(input_shape[1]/k1)) + 'x' + str(depth)
		layers.append({'layer': conv2, 'name': 'conv2', 'size': size})

		# Pool2
		k2 = 2
		pool2 = self.maxpool2d(conv2, k=k2, name='pool2')
		size = str(int(input_shape[0]/(k1*k2))) + 'x' +  str(int(input_shape[1]/(k1*k2))) + 'x' + str(depth)
		layers.append({'layer': pool2, 'name': 'pool2', 'size': size})

		# Conv3
		conv3 = self.conv2d(pool2, weights['wc3'], biases['bc3'], name='conv3')
		depth = weights['wc3'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/k1*k2)) + 'x' +  str(int(input_shape[1]/k1*k2)) + 'x' + str(depth)
		layers.append({'layer': conv3, 'name': 'conv3', 'size': size})

		# Conv4
		conv4 = self.conv2d(pool2, weights['wc4'], biases['bc4'], name='conv4')
		depth = weights['wc4'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/k1*k2)) + 'x' +  str(int(input_shape[1]/k1*k2)) + 'x' + str(depth)
		layers.append({'layer': conv3, 'name': 'conv4', 'size': size})

		# Pool3
		k3 = 2
		pool3 = self.maxpool2d(conv4, k=k3, name='pool3')
		size = str(int(input_shape[0]/(k1*k2*k3))) + 'x' +  str(int(input_shape[1]/(k1*k2*k3))) + 'x' + str(depth)
		layers.append({'layer': pool3, 'name': 'pool3', 'size': size})

		# Conv5
		conv5 = self.conv2d(pool3, weights['wc5'], biases['bc4'], name='conv5')
		depth = weights['wc5'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/(k1*k2*k3))) + 'x' +  str(int(input_shape[1]/(k1*k2*k3))) + 'x' + str(depth)
		layers.append({'layer': conv4, 'name': 'conv5', 'size': size})

		# Conv6
		conv4 = self.conv2d(conv5, weights['wc6'], biases['bc6'], name='conv6')
		depth = weights['wc6'].get_shape().as_list()[3]
		size = str(int(input_shape[0]/(k1*k2*k3))) + 'x' +  str(int(input_shape[1]/(k1*k2*k3))) + 'x' + str(depth)
		layers.append({'layer': conv4, 'name': 'conv4', 'size': size})

		# Flatten
		k4 = input_shape[0]/(k1*k2*k3)
		flatten = self.maxpool2d(conv4, k=k4, name='pool4')
		size = '1x1x' + str(depth)
		layers.append({'layer': flatten, 'name': 'flatten', 'size': size})

		# Fully connected 1
		fc1 = tf.reshape(flatten, [-1, weights['wd1'].get_shape().as_list()[0]])
		fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
		fc1 = tf.nn.relu(fc1)
		size = str(weights['wd1'].get_shape().as_list()[1])
		layers.append({'layer': fc1, 'name': 'fc1', 'size': size})

		# Fully connected 2
		fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
		fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
		fc2 = tf.nn.relu(fc2)
		fc2 = tf.nn.dropout(fc2, 1, name='dropout')
		size = str(weights['wd2'].get_shape().as_list()[1])
		layers.append({'layer': fc1, 'name': 'fc2', 'size': size})

		# Output
		out = tf.add(tf.matmul(fc2, weights['out']), biases['out'], name='out')
		size = str(weights['out'].get_shape().as_list()[1])
		layers.append({'layer': out, 'name': 'out', 'size': size})

		return out, layers

	def split_data(self, X, y=None):
		batches = []

		for i in range(0, int(len(X) / DEFAULT_BATCH_SIZE) + 1):
			start = (i * DEFAULT_BATCH_SIZE)
			end = min((i + 1) * DEFAULT_BATCH_SIZE, len(X))
			batch_X = X[start:end]
			if y is None:
				batches.append({'x': batch_X})
			else:
				batch_y = y[start:end]
				batches.append({'x': batch_X, 'y': batch_y})

		return batches

	def checkpoint_variables(self, sess):
		print(str([n.name for n in tf.global_variables()]))
		for var in tf.global_variables():
			print('Initializing ' + var.name)
			self.variables[var] = sess.run(var)

	def initialize_session(self):
		sess = tf.Session(graph=self.graph)

		if len(self.variables) > 0:
			for var in self.variables:
				sess.run(var.assign(self.variables[var]))

		return sess

	def train_epoch(self, sess, batches, steps=1):
		for i, batch in enumerate(batches):
			sess.run(self.optimizer, feed_dict={self.x: batch['x'], self.y: batch['y']})
					
			if i == len(batches) - 1:
				loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: batch['x'], self.y: batch['y']})
					
				print("Training step " + str(steps * DEFAULT_BATCH_SIZE) + ", training loss: " + \
					"{:.2f}".format(loss) + ", training acc.: {:.4f}".format(acc))
			steps += 1

		return steps

	def validate_epoch(self, sess, val_batches, total_len):
		loss = 0.0
		acc = 0.0

		for val_batch in val_batches:
			batch_loss, batch_acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: val_batch['x'], self.y: val_batch['y']})
			loss += batch_loss * len(val_batch['x'])
			acc += batch_acc * len(val_batch['x'])

			v = val_batch

		loss /= total_len
		acc /= total_len
		
		return loss, acc

	def fit(self, train_X, train_y, val_X, val_y, epochs=DEFAULT_EPOCHS):
		height, width, channels = self.input_shape
		train_X = np.reshape(train_X, [-1, height * width * channels])
		val_X = np.reshape(val_X, [-1, height * width * channels])

		batches = self.split_data(train_X, train_y)
		val_batches = self.split_data(val_X, val_y)

		print('Started training with ' + str(len(train_X)) + ' images')
		with self.initialize_session() as sess:
			sess.run(tf.global_variables_initializer())
			steps = 1
			for epoch in range(0, epochs):
				random.shuffle(batches)

				steps = self.train_epoch(sess, batches, steps=steps)
				loss, acc = self.validate_epoch(sess, val_batches, len(val_X))
				print("Epoch " + str(epoch + 1) + ", val loss: " + \
					"{:.2f}".format(loss) + ", val acc.: " + \
					"{:.4f}".format(acc))

			self.checkpoint_variables(sess)
			

	def save(self, path):
		with self.initialize_session() as sess:
			saver = tf.train.Saver(tf.global_variables())
			saver.save(sess, path)

			return True

		return False

	def load(self, path):
		graph_path = path + '.meta'
		with tf.Session(graph=self.graph) as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.import_meta_graph(graph_path)
			saver.restore(sess, path)

			self.checkpoint_variables(sess)

	def predict(self, X):
		height, width, channels = self.input_shape
		input_size = height * width * channels
		X = np.reshape(X, (-1, input_size))
		batches = self.split_data(X)
		predictions = []

		with self.initialize_session() as sess:
			for batch in batches:
				batch_preds = sess.run(self.pred, feed_dict={self.x: batch['x']})
				if len(predictions) == 0:
					predictions = batch_preds
				else:
					predictions = np.concatenate((predictions, batch_preds))
		
		return predictions



