from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import tensorflow as tf 
import numpy as np 
#import pandas as pd 
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main()__":
	tf.app.run()


def input_data_retrieval(input_data):
	fea_hog_train = input_data['fea_hog_train']
	fea_scatter_train = input_data['fea_scat_train']
	X = input_data['images_train']
	y= input_data['labels_train']
	return(fea_hog_train, fea_scatter_train, X, y)

def cnn_model_1(features, labels, mode):
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

	conv_1 = tf.layers.conv2d(inputs = input_layer, filters = 32, 
		kernel_size = [5,5],padding = "same", activation= tf.nn.relu)
	print(conv_1)
	pool_1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides =2)
	print(pool_1)
	conv_2 = tf.layers.conv2d(inputs = pool_1, filters= 64, 
		kernel_size = [5,5], padding="same", activation= tf.nn.relu)
	print(conv_2)
	pool_2 = tf.layers.max_pooling2d(inputs = conv_2, pool_size=[2,2], strides =2)
	print(pool_2)
	pool2_flat = tf.reshape(pool2, [-1,7*7*64])
	
	#Dense
	dense = tf.layers.dense(inputs = pool2_flat, units= 1024, activation= tf.nn.relu)
	#Dropout
	dropout = tf.layers.dropout(inputs = dense, rate = 0.01, training_mode = tf.estimator.ModeKeys.TRAIN)
	
	#logit layer
	logits = tf.layers.dense(inputs= dropout, units = 6)

	predictions={
		#Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		#Add softmax_tensor to the graph. It is used for predict and by the logging_hook
		"probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
		}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
	return(tf.estimator.EstimatorSpec(mode= mode, loss= loss, train_op = train_op))


	eval_metric_ops = {
		"accuracy":tf.metric.accuracy(labels = labels, predictions= predictions["classes"])}

	return(tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops))



def main():
	input_data = loadmat("object_train.mat")
	fea_hog_train, fea_scatter_train, X, y = input_data_retrieval(input_data)
	X = X.reshape(5000, 784)
	print(X.shape)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state= 42)
	mnist_classifier = tf.estimator.Estimator(model_fn = cnn_model_1, model_dir="myModel")
	tensor_to_log = {"probabilities":" softmax_tensor"}
	logging_hook = tf.train.loggingTensorHook(tensors = tensor_to_log, every_n_iter=50)
	train_input_fn = tf.estimato.inputs.numpy_input_fn(x= {"x": X_train}, y = y_train, batch_size = 100, num_epochs = 1, shuffle = True)
	mnist_classifier.train(input_fn = train_input_fn, steps = 200000, hooks = [logging_hook])
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":X_test}, y = y_test, num_epochs = 1, shuffle = False)
	eval_results = 	mnist_classifier.evaluate(input_fn= eval_input_fn)
	print(eval_results)

if __name__ == '__main__':
	main()