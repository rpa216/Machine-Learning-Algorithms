#Multiclass_perceptron
# Name: Rajat Patel
# Student ID: NC69336
# email ID: rpatel12@umbc.edu
# Data Source : MNIST number dataset with colmajor file

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
	return(int_train, int_dev, int_dev_label, int_train_label, test_data, labels_train,output_y)


def perceptron_train(max_iteration, int_train, int_train_label, weight):
	# Initially taking the max number of iteration as you want
	for i in range(max_iteration):
		#going over each and every image in the list or dataset of training image set
		for X in range(len(int_train)):
			# performing the weight and image vector dot product and adding the bias term
			activation = np.argmax(np.dot(weight, int_train[X]))
			#checking the error by taking the difference between the original and the predicted output
			if int_train_label[X] - activation != 0:							
			#if the error is not zero we will update the correct weight and lower the incorrect
				weight[int_train_label[X]] = weight[int_train_label[X]] + int_train[X]
				weight[activation] = weight[activation]-int_train[X]
				#bias[int_train_label[X]] = int_train_label[X] + bias[int_train_label[X]] # also update the bias 
		#return the weight and the bias vectors for final prediction on test sets
		return(weight)

# test prediction function for standard multiclass perceptron
def test_perceptron(int_dev, weight):
	#initializing prediction list
	prediction = []
	# going over each test data image
	for i in range(len(int_dev)):
		activation = np.argmax(np.dot(weight, int_dev[i]))
		prediction.append(activation)
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


# The main function predicting the output
def main():
	input_data = loadmat('mnist_colmajor.mat')
	int_train, int_dev, int_dev_label, int_train_label, test_data, labels_train,y= data_division(input_data, 0.5)
	print(y)
	max_iteration = 10
	w= np.zeros((y.shape[0], int_train.shape[1]))
	w1= perceptron_train(max_iteration, int_train, int_train_label,w)
	prediction = test_perceptron(int_dev, w1)
	accuracy = accuracy_perceptron(int_dev_label, prediction)
	print(accuracy)


if __name__ == '__main__':
	main()
	print("Thank you for running this perceptron..!!!")


