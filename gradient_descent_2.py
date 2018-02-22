#gradient descent algorithm
import numpy as np
import sys
import matplotlib.pyplot as plt
import math


def cost_function(z,a,b):
	A = (((a-z[0])**2) + (b*(z[1]-(z[0]**2)**2)))
	return(A)

def gradient_1(z,a,b):
	d_f_1 = (2*z[0] - 2*a + (4*b*((z[0]**3) - z[1]*z[0])))
	return(d_f_1)

def gradient_2(z,a,b):
	d_f_2 = (2*b*(z[1]-(z[0]**2)))
	return(d_f_2)

def descent(d_F,z,alpha):
	z = z - alpha*d_F
	return(z)

def main():
	Num_iteration = 1000
	a = 1
	b = 100
	z = np.zeros((2,1))
	P_1 = []
	P_2 = []
	z = np.random.normal(0.5,0.1,(2,1))
	print(z)
	print(z.shape)
	precision = float(0.00005)
	F = np.zeros((Num_iteration, 1))
	alpha = np.zeros((2,1))
	for i in range(Num_iteration):
		P_1.append(z.item(0))
		P_2.append(z.item(1))
		F[i] = cost_function(z,a,b)
		#print(F[i])
		if abs(float(F[i] - F[i-1])) <= float(0.0005):
			print(F[:i])
			plt.plot(F[:i])
			plt.ylabel('Cost_function')
			plt.xlabel('Number of Iteration')
			plt.title("Convergence and Cost function variation graph")
			print("The convergence condition at cost function value"+ str(F[i]) + "and parameter values" + str(z[0])+','+ str(z[1])+ "at iteration "+ str(i))
			plt.show()
			exit()
		d_f = np.zeros((2,1))
		alpha = (((i+1)+10000)**-0.95)
		d_f[0] = gradient_1(z, a, b)
		d_f[1]= gradient_2(z, a, b)
		z = descent(d_f,z,alpha)
	print(F)
	plt.plot(F)
	plt.ylabel('Cost Function')
	plt.xlabel('Number of iteration')
	plt.show()

if __name__ == '__main__':
	main()


