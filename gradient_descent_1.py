import numpy as np
import matplotlib.pyplot as plt 
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import math


def compute_cost(y,W, N):
	a = sum(y)
	F = (-a + N*(math.log(np.exp(W[0])+np.exp(W[1]))))
	return(F)

def gradient_1(W, N, num_zeros):
	G_F = (-(num_zeros) + N*((np.exp(W[0])/(np.exp(W[1])+ np.exp(W[0])))))
	return(G_F)

def gradient_2(W,N, num_ones):
	G_F = (-(num_ones) + N*((np.exp(W[1])/(np.exp(W[1])+np.exp(W[0])))))
	return(G_F)

def descent(G_F, W, alpha):
	W = W - alpha*G_F
	return(W)

def main():
	Num_iteration  = 1000
	N = 50
	y = np.random.randint(2, size=N)
	print(y)
	W = np.zeros((2,1))
	value_1 = []
	value_2  = []
	precision = 0.00000000
	W = np.random.normal(0.5,1,(2,1))
	print(W[0])
	print(W[1])
	print(W.shape)
	L = np.zeros((Num_iteration,1))
	for i in range(Num_iteration):
		value_1.append(W.item(0))
		value_2.append(W.item(1))
		G_F = np.zeros((2,1))
		L[i] = compute_cost(y,W,N)
		print(L[i])
		if L[i] - L[i-1] == float(precision):
			print(L[:i])
			plt.plot(L[:i])
			plt.title('Working of gradient descent')
			plt.ylabel('Function value of F')
			plt.xlabel('Number of Iteration')
			print("The convergence condition at"+ str(L[i]) + "and" + str(W[0])+','+ str(W[1])+ "at Iteration "+ str(i))
			#print("The convergence condition at"+ str(L[i]) + "and" + str(W[0])+','+ str(W[1]))
			plt.show()
			exit()
		alpha = (((i+1)+15560)**-0.50)
		num_zeros = (y == 0).sum()
		num_ones = (y ==1).sum()
		G_F[0] = gradient_1(W,N,num_zeros)
		G_F[1] = gradient_2(W,N,num_ones)
		W = descent(G_F,W, alpha)
	print(L)
	plt.plot(L)
	plt.title('Working of gradient descent')
	plt.ylabel('Function value of F')
	plt.xlabel('Number of Iteration')
	plt.show()


if __name__ == '__main__':
	main()