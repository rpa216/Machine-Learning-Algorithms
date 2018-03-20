#Multilayer multiclass perceptron
import numpy as np 
from scipy.io import loadmat

#function divides the data into development, train and test set for both images and labesl for the mnist data file.
def data_division(input_data, ratio_test_set):
	#definng the ratio of the training set from the dev set
	train_set_ratio = (1- ratio_test_set)
	#reading the features and labels from the mat file
	train_data = (input_data['images_train'])
	test_data  = (input_data['images_test'])
	labels_train = (input_data['labels_train'])
	labels_test = (input_data['labels_test'])

	# Dividing the data in int -train and int-dev
	int_train = (train_data[:int(len(train_data)*train_set_ratio)])
	int_train_label = labels_train[:int(len(labels_train)*train_set_ratio)]

	int_dev = train_data[int(len(train_data)*train_set_ratio):]
	int_dev_label = labels_train[int(len(labels_train)*train_set_ratio):]

	#calculating the vector for label class
	output_y = np.unique(int_train_label)
	return(int_train, int_dev, int_dev_label, int_train_label, test_data, labels_test,output_y, train_data, labels_train)


def sigmoid_activation_function(activation):
	hidden_layer = (1/(1+np.exp((-activation))))
	return(hidden_layer)

def signed_zero_one(int_train):
	for i in range(len(int_train)):
		for j in range(len(int_train[i])):
			if int_train[i][j] > 0:
				int_train[i][j] = 1
	return(int_train)


def test_perceptron_average(int_dev, weight, v, int_dev_label):
	#initializing prediction list
	prediction = []
	# going over each test data image
	for X in range(len(int_dev)):
		w_X = np.dot(w_H1, int_dev[X])
		a = np.zeros((30,1))
		for i in range(len(a)):
			a[i] = w_X[i]
		# for Y in range(len(w_H1)):
		# 	w_X = (np.dot(w_H1[Y], int_train[X]))
		h1_a = np.tanh(a)
		# print(np.array(h1_a))
		w_X_O = np.tanh(np.dot(w_V,h1_a))
		activation_o= np.argmax(w_X_O)
		prediction.append(activation_o)
	print(prediction)
	print(len(prediction))
	#returning the prediction of each test image
	return(np.array(prediction))

def accuracy_perceptron(int_dev_label, prediction):
		#Initializing final output array list
		final_output = []
		# going over each predicted value and comparing it with the original value
		for i in range(len(int_dev_label)):
			#finding the error difference
			error= int_dev_label[i]- prediction[i]
			final_output.append(error)
		#converting the list to array for further calculation
		final_output = np.array(final_output) 
		#calculating the final accuracy using the final output and total label in dev set
		accuracy = ((final_output == 0).sum()/(len(int_dev_label))*100)
		return(accuracy)


def one_hot_encoder(y):
	for i in range(len(y)):
		for j in range(len(y)):
			if i == j:
				y[i][j] = 1
	return(y)

input_data = loadmat('mnist_colmajor.mat')
int_train, int_dev, int_dev_label, int_train_label, test_data, labels_test,y, train_data, labels_train= data_division(input_data, 0.5)
int_train = signed_zero_one(int_train)
int_dev = signed_zero_one(int_dev)
y_label = np.zeros((y.shape[0], y.shape[0]))
y_label = one_hot_encoder(y_label)
eta1 = 0.00001
eta2 = 0.00001
max_iteration = 40
w_H1 = np.random.uniform(-0.1, 0.1,(30, int_train.shape[1]))
w_V = np.random.uniform(-0.1,0.1,(y.shape[0], len(w_H1)))
G_input = np.zeros((30, int_train.shape[1]))
G_output = np.zeros((y.shape[0], len(w_H1)))
int_train = np.array(int_train)
h1_a  = np.zeros((30,1))
for _ in range(max_iteration):
	for X in range(len(int_train[:10000])):
		w_X = np.dot(w_H1, int_train[X])
		a = np.zeros((30,1))
		for i in range(len(a)):
			a[i] = w_X[i]
		h1_a = np.tanh(a)
		w_X_O = (np.dot(w_V, h1_a))
		activation_o = w_X_O
		error = np.zeros((len(y_label),1))
		y = y_label[int_train_label[X]]
		y = np.transpose(y)
		# print(y.shape)
		for j in range(len(y)):
			error[j] = y[j] - activation_o[j]
		error_hidden_layer_output = np.dot(h1_a, np.transpose(error))
		G_output = G_output - np.transpose(error_hidden_layer_output)

		for i in range(len(G_input)):
			tan_function = np.array(h1_a[i]**2)
			tan_function = 1 - tan_function
			outer_weight = np.array(w_V[:,i])
			outer_weight = np.reshape(outer_weight, (10, 1))
			error_outer_layer_product = np.dot(np.transpose(outer_weight), (error))
			function_calculator = np.dot(error_outer_layer_product, tan_function)
			R = np.array(int_train[X])
			R = R.reshape((1,-1))
			delta = np.dot(function_calculator,R)
			G_input[i] = G_input[i] - delta
	w_H1 -=eta1*G_input
	w_V-=eta2*G_output



prediction = test_perceptron_average(int_dev[12000], w_H1, w_V, int_dev_label[:12000])
accuracy = accuracy_perceptron(prediction, int_dev_label[:12000])
print(accuracy)

