import numpy as np 
from scipy.io import loadmat
import math



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


def Relu(Z):
    a = np.zeros((len(Z),1))
    for i in range(len(Z)):
        a[i] = max(0, Z[i])
    return(a)
        
def Relu_Prime(Z):
  a = np.zeros((len(Z),1))
  for i in range(len(Z)):
    if Z[i] > 0:
        a[i] = 1
    else:
        a[i] = 0
  return(a)

def softmax(X):
  a = np.zeros((len(X),1))
  #print("This is softmax_output")
  for i in range(len(X)):
    shiftx = X[i] - max(X)
    a[i] = (np.exp(shiftx))/(np.sum(np.exp(X)))
  return(a)

def softmax_prime(X):
  a = np.zeros((len(X),1))
  #print("This is softmax_prime_output")
  for i in range(len(X)):
    a[i] = (X[i])*(1-(X[i]))
  return(a)
  
def cross_entroy_loss(y_label, activation):
  indices = np.argmax(y_label, axis = 1).astype(int)
  probability = activation[np.arange(len(activation)), indices]
  # print(probability)
  log_value = np.log(probability)
  #print(log_value)
  loss = -1.0 * np.sum(log_value) / len(log_value)
  #print(loss)
  return(loss)


def cross_entropy_delta(activation, y_label):
  a = np.zeros((len(activation),1))
  #print("This is cross_entropy_loss_derivative output")
  for i in range(len(activation)):
    a[i] = activation[i] - y_label[i]
  return(a)
  
def cost_function(y_label, activation):
    cost_function = (np.sum((activation - y_label)**2)/activation.size)
    return(cost_function)

def cost_function_prime(y_label, activation):
    prime_out = (np.sum(activation - y_label))
    return(prime_out)


def feedfoward(int_train,weight_hidden,weight_output):
    Z_H = np.dot(np.transpose(weight_hidden),int_train)
    activation_inner = Relu(Z_H)
    Z_O = np.dot(np.transpose(weight_output), activation_inner)
    activation_output = softmax(Z_O)
    return(activation_output, activation_inner)
def output_backpropogated(X):
  a = np.zeros((len(X),1))
  b = np.argmax(X)
  for i in range(len(a)):
    if i == b:
      a[i] = 1
    else:
      a[i] = 0
  return(a)

def backpropagation(activation_output,activation_inner,y_label,weight_output, weight_hidden, image,output_gradient,inner_gradient):
    #print(loss_function)
    activation_encoded = output_backpropogated(activation_output)
    prime_out = cross_entropy_delta(activation_encoded, y_label)
    #print(prime_out.shape)
    #print(prime_out)
    activation_output_prime = softmax_prime(activation_output)
    #print("This is the derivative of the output activation")
    #print(activation_output_prime.shape)
    #print(activation_output_prime)
    Error_output = ((prime_out)*activation_output_prime)
    #print("This is the error value of the output layer to be back propogated")
    #print(Error_output.shape)
    inner_gradient = np.dot(Error_output, np.transpose(activation_inner))
    #print(inner_gradient)
    #print(inner_gradient.shape)
    #print("This is the derivative value of the error backpropagated from the hidden layer to input weights")
    Error_input = np.multiply(np.dot(weight_output,Error_output), Relu_Prime(activation_inner))
    #print(Error_input)
    #print(Error_input.shape)
    X_image = np.array(image)
    X_image = X_image.reshape(784,1)
    output_gradient = np.dot((Error_input),np.transpose(X_image))
    #print(output_gradient.shape)
    return(output_gradient, inner_gradient, y_label)
    
def one_hot_encoder(y):
    for i in range(len(y)):
        for j in range(len(y)):
            if i == j :
                y[i][j] =1
    return(y)

def signed_zero_one(int_train):
    for i in range(len(int_train)):
        for j in range(len(int_train[i])):
            if int_train[i][j] > 0:
                int_train[i][j] = 1
    return(int_train)

# def sigmoid(X):
#     return(1/(1+np.exp(X))

# def sigmoid_prime(X):
#   a = np.zeros((10,1))
#   for i in range(len(X)):
#       a[i] = sigmoid(X[i])(1-sigmoid(X[i]))
#   return(a)

# test prediction function for standard multiclass perceptron
def test_neural_network(int_dev, weight_hidden, weight_output, int_dev_label):
  prediction = []
  for X in range(len(int_dev)):
    input_activation = np.dot(np.transpose(weight_hidden),int_dev[X])
    activation_inner_test = Relu(input_activation)
    output_activation = np.dot(np.transpose(weight_output), activation_inner_test)
    activation_outer_test = np.argmax(softmax(output_activation))
    prediction.append(activation_outer_test)
  return(np.array(prediction))


def accuracy_neural_network(int_dev_label, prediction):
	#Initializing final output array list
	final_output = []
  # going over each predicted value and comparing it with the original value
	for i in range(len(int_dev_label)):
		#finding the error difference
		error= int_dev_label[i]- prediction[i]
		final_output.append(error)
	#converting the list to array for further calculation
	#final_output = np.array(final_output) 
	print(final_output.count(0))
  #calculating the final accuracy using the final output and total label in dev set
	accuracy = ((final_output.count(0))/(len(int_dev_label)))*100
	return(accuracy)
      
def Robin_monro(X):
  alpha = ((X*10 + 100000)**0.9)
  return(alpha)

 
  
def main():
    input_data = loadmat('mnist_colmajor.mat')
    int_train, int_dev, int_dev_label, int_train_label, test_data, labels_test,y, train_data, test_data = data_division(input_data, 0.5)
    #print(int_dev_label[0])
    y_encoded = np.zeros((y.shape[0], y.shape[0]))
    y_encoded = one_hot_encoder(y_encoded)
    #print(y_encoded)
    max_iter = 40
    # alpha1 = 0.00001
    alpha = 0.0009
    #alpha = np.zeros((2,1))
    print(alpha)
    NT = 0.0001
    E = (1*(10**-8))
    gradient_list_1 = []
    gradient_list_2 = []
    weight_hidden = np.random.uniform(-0.1,0.1, (int_train.shape[1],30))
    weight_output = np.random.uniform(-0.1,0.1,(weight_hidden.shape[1], 10))
    #int_train = signed_zero_one(int_train)
    #int_dev = signed_zero_one(int_dev)
    for _ in range(max_iter):
      inner_gradient = np.zeros((weight_hidden.shape[1], y.shape[0]))
      output_gradient = np.zeros((int_train.shape[1], 10))
      for z in range(0,len(int_train[:1000]), 50):
          for X in range(z+50):
              activation_output, activation_inner = feedfoward(int_train[X],weight_hidden, weight_output)
              output_gradient, inner_gradient,y_label = backpropagation(activation_output, activation_inner, 
                                                                  (y_encoded[:,int_train_label[X]]), 
                                                                  weight_output, weight_hidden, int_train[X], 
                                                                  output_gradient, inner_gradient)
            
          loss_function = cross_entroy_loss(y_label, activation_output)
          print(loss_function)
        # gradient_list_1.append(output_gradient**2)
        # gradient_list_2.append(inner_gradient**2)
        # G_learn_1 = np.sum(gradient_list_1)
        # G_learn_2 = np.sum(gradient_list_2)
        # alpha[0] = NT/np.sqrt(E+G_learn_1)
        # alpha[1] = NT/np.sqrt(E+G_learn_2)
          weight_hidden-= alpha*np.transpose(output_gradient)
          weight_output-= alpha*np.transpose(inner_gradient)


    prediction = test_neural_network(int_train[:100], weight_hidden, weight_output, int_train_label[:100])
    accuracy = accuracy_neural_network(prediction, int_train_label[:100])
    print(accuracy)
    
main()
