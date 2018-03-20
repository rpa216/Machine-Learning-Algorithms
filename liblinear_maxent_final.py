# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from liblinear import *
#import liblinear
import scipy
import numpy as np
from scipy.io import loadmat
from liblinearutil import *
from liblinear import *


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


def one_hot_encoder(y):
	for i in range(len(y)):
		for j in range(len(y)):
			if i == j:
				y[i][j] = 1
	return(y)



def cross_validation_accuracy(int_train, int_train_label):
    CV_ACC = train(int_train_label, int_train, '-v 3')
    return(CV_ACC)

def best_c_best_R(int_train, int_train_label):
    return(train(int_train_label, int_train, '-c -s 0'))


def train_liblinear(a, int_train, int_train_label,prob,param):
    if a == 1:
        model_train = train(int_train_label, int_train, '-c 5')
        return(model_train)
    if a == 2:
        model_train = train(prob, param)
        #model_train = train(int_train_label, int_train, param )
        return(model_train)
    if a == 3:
        model_train = train(int_train_label, int_train, '-s 3 -c 5 -q')
        return(model_train)
    
    
def predict_value(int_dev, int_dev_label, model):
    p_labels, p_acc, p_vals = predict(int_dev_label, int_dev, model)
    return(p_labels, p_acc, p_vals)

def evaluation_data(p_labels,int_dev_label):
    int_dev_label = np.array(int_dev_label)
    p_labels = np.array(p_labels)
    ACC, MSE, SCC = evaluations(p_labels, int_dev_label, useScipy = True)
    return(ACC, MSE, SCC)

input_data = loadmat('mnist_colmajor.mat')
int_train, int_dev, int_dev_label, int_train_label, test_data, labels_test,y, train_data, labels_train= data_division(input_data, 0.2)
int_train_label = int_train_label.reshape(len(int_train_label),)
int_dev_label = int_dev_label.reshape(len(int_dev_label),)
labels_train = labels_train.reshape(len(labels_train),)
labels_test = labels_test.reshape(len(labels_test),)
#prob = problem(int_train_label,int_train)
prob = problem(labels_train, train_data)
#prob.set_bias(1)
param = parameter()
param.set_to_default_values()
model = train_liblinear(1, train_data, labels_train, prob, param)
#model = train_liblinear(int_train, int_train_label,prob,param)
CV_ACC = cross_validation_accuracy(train_data,labels_train)
print("This is the cross validation accuracy: ", CV_ACC)
p_labels, p_acc, p_vals = predict_value(test_data, labels_test, model)
#print(p_labels)
print("This is the accuracy:", p_acc)
#print(p_vals)
ACC, MSE, SCC = evaluation_data(p_labels, np.array(labels_test))
print("This is the accuracy:", ACC)
print("This is the mean_square error: ", MSE)
print("This is correlation score: ", SCC)