#gradient_descent_algorithm
import numpy as np
import matplotlib.pyplot as plt 
import sys
import math

def cost_function(x,c,d):
	return(sum(((c-x[:-1])**2)+(d*((x[1:]-(x[:-1])**2)**2))))

def gradient(x,c,d,k):
	D_H = np.zeros((k,1))
	X_a = x[1:-1]
	X_b = x[:-2]
	X_c = x[2:]
	D_H[1:-1] = 2*(c-X_a) -4*d*(X_c -((X_a)**2)) + 2*d*(X_a - ((X_b)**2))
	D_H[0] = 2*(c-x[0]) - 4*d*x[0]*(x[1]-((x[0])**2))
	D_H[-1] = 2*d*(x[-1]-((x[-2])**2))
	return(D_H)

def descent(D_H, alpha, X):
	X = X - alpha*D_H
	return(X)

def main():
	Num_iteration = 10
	c = 1
	d = 100
	k = 10
	x = np.zeros((k,1))
	E = ((1*(10**-8)))
	NT = 0.8
	G = np.zeros((k,1))
	x = np.random.normal(500,1,(k,1))
	print(x)
	D = np.zeros((Num_iteration,1))
	precision = 0.001
	for i in range(Num_iteration):
		alpha = np.zeros((k,1))
		D[i] = cost_function(x,c,d)
		if abs(float(D[i] - D[i-1])) <= float(precision):
			print(D[:i])
			plt.plot(D[:i])
			plt.ylabel('Cost_function')
			plt.xlabel('Number of Iteration')
			plt.show()
			print("The convergence condition at cost function value"+ str(D[i]) + "at iteration" + str(i))
			exit()
		A = gradient(x,c,d,k)
		print(A)
		A = np.square(A)
		G+=A
		for j in range(len(G)):
			alpha[j] = (NT/(math.sqrt(E+G[j])))
		x = descent(A, alpha, x)
	print(D)
	#print(x)
	#print(len(alpha))
	plt.plot(D)
	plt.ylabel('Cost_function')
	plt.xlabel('Number of iteration')
	plt.show()

if __name__ == '__main__':
	main()














