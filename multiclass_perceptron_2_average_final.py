#Multiclass perceptron
import numpy as np 
from scipy.io import loadmat

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

def one_hot_encoder(y):
	for i in range(len(y)):
		for j in range(len(y)):
			if i == j:
				y[i][j] = 1
	return(y)

def average_perceptron_train(max_iteration, int_train, int_train_label, w, b, u, beta, y_label):
	c = 1
	for _ in range(max_iteration):
		for X in range(len(int_train)):
			w_x = (np.dot(w,int_train[X]))
			addition  = np.zeros((10,1))
			for j in range(len(addition)):
				addition[j] = w_x[j] + b[j]
			activation = np.argmax(addition)
			# print(activation)
			# print(int_train_label[X])
			if int_train_label[X]- activation != 0:
				w[int_train_label[X]] = w[int_train_label[X]] + int_train[X]
				w[activation] = w[activation] -int_train[X]
				b[int_train_label[X]] = b[int_train_label[X]] + 1
				b[activation] = b[activation] - 1
				u[int_train_label[X]] = u[int_train_label[X]] + int_train[X]*c
				u[activation] = u[activation]-int_train[X]*c
				beta[int_train_label[X]]= beta[int_train_label[X]] + c
				beta[activation] = beta[activation] - c
			c = c+1
	weights = []
	for i in range(len(w)):
		o1 = w[i] -(u[i]/c)
		weights.append(o1)
	weights = np.array(weights)

	bias = []
	for i in range(len(b)):
		o2 = b[i] - (beta[i]/c)
		bias.append(o2)
	bias = np.array(bias)
	print(bias)
	return(weights, bias)


def test_perceptron_average(int_dev, weight, bias, int_dev_label):
	#initializing prediction list
	prediction = []
	# going over each test data image
	for X in range(len(int_dev)):
		w_x = (np.dot(weight, int_dev[X]))
		activation = np.zeros((10,1))
		for i in range(len(activation)):
			activation[i] = w_x[i] + bias[i]
		prediction.append(np.argmax(activation))
	print(prediction)
	print(len(prediction))
	#returning the prediction of each test image
	return(np.array(prediction))

def accuracy_perceptron(int_dev_label, prediction):
    count_1 = 0
    for i in range(len(int_dev_label)):
        error = np.subtract(int_dev_label[i], prediction[i])
        if error == 0:
            count_1+=1
    print(count_1)
		#converting the list to array for further calculation 
		#calculating the final accuracy using the final output and total label in dev set
    accuracy = (((count_1)/len(int_dev_label))*100)
    return(accuracy)


input_data = loadmat('mnist_colmajor.mat')
int_train, int_dev, int_dev_label, int_train_label, test_data, labels_test,y, train_data, labels_train= data_division(input_data, 0.2)
y_label = np.zeros((len(y), len(y)))
y_label = one_hot_encoder(y_label)
print(y_label)
max_iteration =10
w = np.zeros((y.shape[0], int_train.shape[1]))
b = np.zeros((y.shape[0], 1))
print(b)
u = np.zeros((y.shape[0], int_train.shape[1]))
beta =np.zeros((y.shape[0],1))
N  =0.0001
weight, bias = average_perceptron_train(max_iteration, train_data, labels_train, w, b,u,beta, y_label)
print(weight.shape)
print(bias.shape)
prediction = test_perceptron_average(test_data, weight, bias, labels_test)
print(prediction.shape)
accuracy = accuracy_perceptron(labels_test, prediction)
print(accuracy)


