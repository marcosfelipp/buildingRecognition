import tensorflow as tf
import numpy as np 
import time
from log import Log

class NeuralNetwork():
	def __init__(self):
		# Parameters
		self.learning_rate = 0.005
		self.pixels_x = 100
		self.pixels_y = 100
		self.num_classes = 2
		self.max_steeps = 1000
		# Network Parameters
		self.n_hidden_1 = 100
		self.n_hidden_2 = 2000
		self.n_hidden_3 = 1000
		self.n_hidden_4 = 100

		# tf Graph input
		self.input_matrix = tf.placeholder(dtype=tf.float32, shape=[self.pixels_y, self.pixels_y])
		self.output_expected = tf.placeholder(dtype=tf.float32, shape=[1, self.num_classes])


	def create_model_multilayer_perceptron(self):
		'''
        :param input_matrix:
        :return:
        '''
		weights = {
			'h1': tf.Variable(tf.random_normal([self.pixels_x, self.n_hidden_1])),
			'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
			'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3])),
			'h4': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_hidden_4])),
			'out': tf.Variable(tf.random_normal([self.n_hidden_4, self.num_classes]))
		}
		biases = {
			'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
			'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
			'b3': tf.Variable(tf.random_normal([self.n_hidden_3])),
			'b4': tf.Variable(tf.random_normal([self.n_hidden_4])),
			'out': tf.Variable(tf.random_normal([self.num_classes]))
		}

		layer_1_multiplication = tf.matmul(self.input_matrix, weights['h1'])
		layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
		layer_1_activation = tf.nn.relu(layer_1_addition)

		layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
		layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
		layer_2_activation = tf.nn.relu(layer_2_addition)

		layer_3_multiplication = tf.matmul(layer_2_activation, weights['h3'])
		layer_3_addition = tf.add(layer_3_multiplication, biases['b3'])
		layer_3_activation = tf.nn.relu(layer_3_addition)

		layer_4_multiplication = tf.matmul(layer_3_activation, weights['h4'])
		layer_4_addition = tf.add(layer_4_multiplication, biases['b4'])
		layer_4_activation = tf.nn.relu(layer_4_addition)

		out_layer_multiplication = tf.matmul(layer_4_activation, weights['out'])
		out_layer_addition = out_layer_multiplication + biases['out']

		return out_layer_addition

	# Reads an image from a file, decodes it into a dense tensor, and resizes it
	# to a fixed shape.
	def _parse_function(filename, label):
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_image(image_string)
		image_resized = tf.image.resize_images(image_decoded, [28, 28])
		return image_resized, label


	def neural_network_run(self, label_number, num_epochs):
		'''
        Method that build a neural network
        :param label_number : number of labels
        :param num_epochs
        :return: None
        '''

		# Construct model
		model = self.create_model_multilayer_perceptron()

		# Compare out value with output expected:
		loss_op = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=self.output_expected))

		# Computing gradients and apply gradients automatic:
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss_op)

		init = tf.global_variables_initializer()

		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)

			# A vector of filenames.
			filenames = tf.constant(["./img" + str(x) + ".png" for x in range(100)])

			# `labels[i]` is the label for the image in `filenames[i].
			labels = tf.constant([1 for x in range(100)])

			dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
			dataset = dataset.map(self._parse_function)

			# Training cycle
			for epoch in range(num_epochs):
				avg_cost = 0.
				print('ok')
		# 		# Loop over all tuples:
		# 		for tuple_position in range(self.input_train_len):
		# 			# Correct shape's problem -- convert vector in matrix
		# 			input_m = []
		# 			input_m.append(self.input_matrix[tuple_position])
        #
		# 			c, _ = sess.run([loss_op, optimizer],
		# 							feed_dict={input_matrix: input_m,
		# 									   output_expected: self.output_matrix[tuple_position]})
        #
		# 			avg_cost += c / self.input_train_len
		# 		Log.info("Accurace in epoch %d" % epoch + " : %f" % avg_cost)
        #
		# 	Log.info("Optimization finished")
        #
		# 	# Test model
		# 	correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(output_expected, 1))
        #
		# 	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #
		# 	result = 0
		# 	for tuple in range(self.input_test_len):
		# 		input_m = []
		# 		input_m.append(self.input_test[tuple])
		# 		result += accuracy.eval({input_matrix: input_m, output_expected: self.output_test[tuple]})
        #
		# 	result = result / self.input_test_len
        #
		# 	Log.info("RESULT: %f" % result)
        #
		# return result

if __name__ == "__main__":

	rede = NeuralNetwork()
	model = rede.neural_network_run(10,10)
