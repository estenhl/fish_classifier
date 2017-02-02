import tensorflow as tf
from .nn import NN

class SingleLayerNN(NN):
	def __init__(self, id, input_shape, classes, class_weights=None):
		super().__init__(self, id, input_shape, classes)

	def weights(self):
		return {
			'hidden': tf.Variable(tf.random_normal([self.input_shape, int(self.input_shape * (2/3)) + self.classes]), name='hidden_weight'),
			'out': tf.Variable(tf.random_normal([int(self.input_shape * (2/3)) + self.classes, self.classes]), name='out_weight')
		}

	def biases(self):
		return {
			'hidden': tf.Variable(tf.random_normal([self.input_shape]), name='hidden_bias'),
			'out': tf.Variable(tf.random_normal([self.classes]), name='out_bias')
		}

	def net(self, x, input_shape, weights, biases):
		self.x = tf.placeholder(tf.float32, [None, input_shape], name='x_placeholder')
		self.y = tf.placeholder(tf.float32, [None, classes], name='y_placeholder')

		hidden = tf.reshape(fc, [-1, weights['hidden']].get_shape().as_list()[0])
		hidden = tf.add(tf.matmul(fc, weights['hidden']), biases['hidden'])
		hidden = tf.nn.relu(fc)
		size = str(weights['hidden'].get_shape().as_list()[1])
		layers.append({'name': 'hidden', 'size': size})

		out = tf.add(tf.matmul(fc2, weights['fc' + len(layers) - 1]), biases['fc' + len(layers) - 1], name='out')

		return out, layers