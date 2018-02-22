import numpy as np
import matplotlib.pyplot as plt 
import sys
import math

def compute_cost(y,W,N):
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
	Num_iteration  = 100
	N = 50
	y = np.random.randint(2, size=N)
	print(y)
	W = np.zeros((2,1))
	W = np.random.normal(0.9,1,(2,1))
	P_1 = []
	P_2  = []
	precision = 0.000000
	print(W)
	L = np.zeros((Num_iteration,1))
	alpha  = np.zeros((2,1))
	E = ((3*(10**-8)))
	NT = 0.5
	gradient_list_1 = []
	gradient_list_2 = []
	for i in range(Num_iteration):
		P_1.append(W.item(0))
		P_2.append(W.item(1))
		G_F = np.zeros((2,1))
		L[i] = compute_cost(y,W,N)
		if abs(float(L[i] - L[i-1])) == float(precision):
			print(L[:i])
			plt.plot(L[:i])
			plt.ylabel('Parameter W_0')
			plt.xlabel('Cost_function')
			print("The convergence condition at"+ str(L[i]) + "and" + str(W[0])+','+ str(W[1])+ "at iteration"+ str(i))
			plt.show()
			exit()
		num_zeros = (y == 0).sum()
		num_ones = (y ==1).sum()
		G_F[0] = gradient_1(W, N, num_zeros)
		G_F[1] = gradient_2(W, N, num_ones)
		gradient_list_1.append((G_F.item(0)**2))
		gradient_list_2.append((G_F.item(1)**2))
		G1 = sum(gradient_list_1)
		G2 = sum(gradient_list_2)
		alpha[0] = NT/(math.sqrt(E+G1))
		alpha[1] = NT/(math.sqrt(E+G1))
		W = descent(G_F, W, alpha)
	print(L)
	plt.plot(L)
	plt.xlabel("Number of Iteration")
	plt.ylabel("Cost function")
	plt.show()

if __name__ == '__main__':
	main()