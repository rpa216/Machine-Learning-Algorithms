#finding alpha
import numpy as np
import math
alpha = 0
for i in range(10):
	alpha = (((i+1)+1000)**-1)
	print(alpha)

def rosenbrock_general_form(x):
	return(np.sum())

y = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1])
num_zeros = (y == 0).sum()
num_ones = (y ==1).sum()
print(num_zeros)
print(num_ones)

# A = np.random.randint(10, size=5)
# A= np.zeros((3,1))
# A[0] =2 
# A[1] = 3
# A[2]= 4
# N = 0.1
# E = (1*(10**-8))
# alpha = np.zeros((3,1))
# print(np.square(A))
# G = np.zeros((3,1))
# for i in range(2):
# 	G+=(np.square(A))
# 	print(G)
# for i in range(len(G)):
# 	alpha[i] = (N/(math.sqrt(E+G[i])))
# print(alpha)
# print("This is the value of the product",alpha*G)
W = np.zeros((2,1))
W = np.random.normal(0.5,0.4,2)
print(W.transpose())
print(W.shape)

		if i >= 30:
			if F[-1] - F[-2] == 0 or F[-1] ==0:
				if F[-2] - F[-3] == 0:
					if F[-3] - F[-4] ==0:
						print("The convergence condition at"+ str(F[i]) + "and" + str(z[0])+','+ str(z[1]))
						print(F)
						print(P_1)
						print(P_2)
						P_1 = np.array((P_1))
						P_2 = np.array((P_2))
						v1 = P_1.transpose()
						v2 = P_2.transpose()
						fig = plt.figure()
						ax = fig.add_subplot(111, projection='3d')
						ax.plot_surface(F, v1, v2)
						#ax.plot_wireframe(F, v1, v2)
						#ax.plot(F, v1, v2)
						#plt.xlabel('Cost Function F(z)')
						plt.ylabel('Parameter z2')
						plt.xlabel('Parameter z1')
						plt.show()
						exit()