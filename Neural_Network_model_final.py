#Multilayer multiclass perceptron
import numpy as np 
from scipy.io import loadmat
import math

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
	return(int_train, int_dev, int_dev_label, int_train_label, test_data, labels_train,output_y)


def sigmoid(activation):
	a = np.zeros((len(activation),1))
	for i in range(len(activation)):
		a[i] = (1/(1+np.exp((-activation[i]))))
	return(a)

def sigmoid_prime(activation):
	# a = np.zeros((len(activation),1))
	# for i in range(len(a)):
	# 	a[i] = activation
	return((activation)*(1- (activation)))

def signed_zero_one(int_train):
	for i in range(len(int_train)):
		for j in range(len(int_train[i])):
			if int_train[i][j] > 0:
				int_train[i][j] = 1
	return(int_train)


def test_perceptron_average(int_dev, weight_hidden, weight_output, int_dev_label):
	#initializing prediction list
	prediction = []
	# going over each test data image
	for X in range(len(int_dev)):
		w_X = np.dot(weight_hidden, int_dev[X])
		a = np.zeros((len(w_X),1))
		for i in range(len(a)):
			a[i] = w_X[i]
		# for Y in range(len(w_H1)):
		# 	w_X = (np.dot(w_H1[Y], int_train[X]))
		h1_a = tanh(a)
		# print(np.array(h1_a))
		w_X_O = (np.dot(weight_output,h1_a))
		activation_o= np.argmax(sigmoid(w_X_O))
		prediction.append(activation_o)
	#print(prediction)
	print(len(prediction))
	#returning the prediction of each test image
	return(np.array(prediction))

def accuracy_perceptron(int_dev_label, prediction):
		#Initializing final output array list
		count_1 = 0
		final_output = []
		print(prediction.shape)
		print(int_dev_label.shape)
		# going over each predicted value and comparing it with the original value
		for i in range(len(int_dev_label)):
			#finding the error difference
			error= np.subtract(prediction[i],int_dev_label[i])
			if error == 0:
				count_1+=1
			else:
				pass
		
		#converting the list to array for further calculation
		#calculating the final accuracy using the final output and total label in dev set
		print(count_1)
		accuracy = ((count_1)/(len(int_dev_label))*100)
		return(accuracy)

def tanh(X):
	return(np.tanh(X))

def tanh_prime(X):
	return((1-(X)**2))

def one_hot_encoder(y):
	for i in range(len(y)):
		for j in range(len(y)):
			if i == j:
				y[i][j] = 1
	return(y)

def mean_square_loss(activation, y_label):
	return((np.sum(activation - y_label)**2)/y_label.size)

def derivative_loss(activation, y_label):
	a = np.zeros((len(y_label),1))
	for i in range(len(a)):
		a[i] = y_label[i] - activation[i]
	return(a)

def output_backpropogated(X):
	a = np.zeros((len(X),1))
	b = np.argmax(X)
	for i in range(len(a)):
		if i == b:
			a[i] = 1
		else:
			a[i] = 0
	return(a)



def feed_forward_function(int_train, weight_hidden, weight_output):
	activation_hidden = tanh(np.dot(weight_hidden, int_train))
	activation_output = sigmoid(np.dot(weight_output, activation_hidden))
	return(activation_output, activation_hidden)

def backpropogation_function(int_train,activation_output, activation_hidden, weight_hidden, weight_output,gradient_hidden, gradient_output, y_label, int_train_label):
	y = y_label[int_train_label]
	y = y.reshape(-1,1)
	#print(y.shape)
	#print(loss)
	activation_encoded = output_backpropogated(activation_output)
	#print(activation_encoded)
	loss_prime = derivative_loss(activation_encoded, y)
	#print(loss_prime.shape)
	error_output = (loss_prime*(sigmoid_prime(activation_output)))
	#error_output = loss_prime
	#print(error_output.shape)
	activation_hidden = np.array(activation_hidden)
	activation_hidden = activation_hidden.reshape(1,-1)
	error_output_1 = np.dot(error_output, activation_hidden)
	#print(error_output.shape)
	gradient_output = gradient_output - error_output_1 
	for i in range(len(activation_hidden)):
		activation_hidden_prime = tanh_prime(activation_hidden[i])
		outer_weight = np.array(weight_output[:,i])
		outer_weight = np.reshape(outer_weight, (10, 1))
		error_1 = np.dot(np.transpose(outer_weight),(error_output))
		error_2 = error_1*activation_hidden_prime[i]
		R = np.array(int_train)
		R = R.reshape(1,-1)
		gradient_hidden[i] = gradient_hidden[i] - np.dot(error_2, R)
	return(gradient_hidden, gradient_output, y)


def main():
	input_data = loadmat('mnist_colmajor.mat')
	int_train, int_dev, int_dev_label, int_train_label, test_data, labels_test,y= data_division(input_data, 0.5)
	y_label = np.zeros((y.shape[0], y.shape[0]))
	y_label = one_hot_encoder(y_label)
	eta1 = 0.000001
	eta2 = 0.000001
	eta = np.zeros((2,1))
	NT = 0.00001
	E = (1*(10**-15))
	max_iteration = 0
	weight_hidden = np.random.uniform(-0.05, 0.05,(90, int_train.shape[1]))
	weight_output = np.random.uniform(-0.05,0.05,(y.shape[0], len(weight_hidden)))
	gradient_list_1 = []
	gradient_list_2 = []
	max_iteration = 80
	for _ in range(max_iteration):
		gradient_hidden = np.zeros((90, int_train.shape[1]))
		gradient_output = np.zeros((y.shape[0], len(weight_hidden)))
		for S in range(0,len(int_train[:20000]),50):
			for X in range(S, S+50):
				activation_output, activation_hidden = feed_forward_function(int_train[X], weight_hidden, weight_output)
				gradient_hidden, gradient_output,y = backpropogation_function(int_train[X],activation_output, activation_hidden, 
					weight_hidden, weight_output, gradient_hidden, gradient_output, y_label, int_train_label[X])
				#print(gradient_hidden.shape, gradient_output.shape)
			
			loss = mean_square_loss(activation_output, y)
			print(loss)
			# gradient_list_1.append(gradient_output**2)
			# gradient_list_2.append(gradient_hidden**2)
			# G_learn_1 = np.sum(gradient_list_1)
			# G_learn_1 = np.sum(gradient_list_2)
			# eta[0] = NT/(np.sqrt(E+G_learn_1))
			# eta[1] = NT/(np.sqrt(E+G_learn_1))
			weight_output-= eta1*gradient_output
			weight_hidden-= eta2*gradient_hidden

	prediction = test_perceptron_average(int_dev[:10000], weight_hidden, weight_output, int_dev_label[:10000])
	accuracy = accuracy_perceptron(int_dev_label[:10000], prediction)
	print(accuracy)

if __name__ == '__main__':
	main()