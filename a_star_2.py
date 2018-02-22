
#I have created the function Solve, which uses a*algorithm for solving the puzzle and finding the optimum path. 
# the current selection of node in function is taken the bases of two parameter indexe value and the heuristic distance
#Thus, the f(n) = g(n) + h(n) , where g(n)= index of the value of the list and heuristic = distance from the goal state
#Reference pseudo code this particular algorithm was taken from following website:
#http://web.mit.edu/eranki/www/tutorials/search/
#syntax for the python are referred from official python documentation site: https://docs.python.org/
#Initially I have created a vertex function which return value, index, heuristic of particular nodes of the tree.
class vertex():
	def __init__(self, value,index, hueristic): #initialization
		self.name = value
		self.index = index
		self.hueristic = hueristic
		self.leftchild = None
		self.rightchild = None
#Insert operation for left and Right nodes.
	def insert_left(self,value):
		self.leftchild = value
		#print("new leftchild")

	def insert_right(self, value):
		self.rightchild = value
		#print("new rightchild")

#Solve function with take list as input
def solve(A):
	closed_list = []
	closed_list_traverse_index = []			#initialised my closed list
	shortest_path = [] 			#intialised my shortest path to record the path with node changes
	open_list = [] 				#initializing open list
	index = 0 					#intializing index value to zero
	hueristic = len(A)-index
	root_node = vertex(A[0],0,hueristic) #initializing the root node
	open_list.append(root_node)			 #appending root node to open list
	#b = None
	#a = None
	for i in A:
		while open_list !=[]:
			b = None					#initial value for left and right child nodes
			a = None
			min_v = len(A)				#taken the length of the list
			for v in open_list:
				if v.hueristic <= (min_v): #comparing the intial hueristic value with len of the list to not overestimate it
					min_v = v.hueristic
					pop_n = v
			#print("current_open_list: ",open_list)
			open_list.remove(pop_n)			#removing the initial node from the list
			root_node = pop_n
			if root_node.name == 0 or root_node.name >= len(A): #checking for value zero with node value and the element value greater than lenght of list to exit the code
				#return(True)
				if root_node.index == len(A)-1:
					x = ""
					x = x.join(shortest_path)
					return(x)
				else:
					return("No solution found")
			q = pop_n.index 				#taking only the index value of the node as the cost function 
			closed_list.append(pop_n)		#appending the value of node closed list to make sure we do not visit again
			closed_list_traverse_index.append(q) #closed list for traverse indexes
			if q >= len(A) :				#base case
				#print("We have reached goal state")
				break
			if q-A[q] > 0:									#checking the condition to insert left node
				b = vertex(A[q-A[q]],q-A[q],len(A)-(q-A[q]))
				if b not in open_list and b not in closed_list:
					open_list.append(b)
					root_node.insert_left(b)
			else:
				#print("left node does not exist")
				#print("Go Right")
				shortest_path.append("R")					#path creation
			if q+A[q]<len(A):								#checking the condition to insert right node
				a = vertex(A[q+A[q]], q+A[q],len(A)-(q+A[q]))
				if a not in open_list and a not in closed_list:
					open_list.append(a)
					root_node.insert_right(a)
			else:
				#print("right node does not exist")
				#print("Go Left")
				shortest_path.append("L")					
			#print("Hear Hear",open_list)
			if a!= None and b!=None:					#intial condition to check which is the optimal node to visit next
				#print(a.hueristic)
				#print(b.hueristic)
				if a.hueristic < b.hueristic:			#comparing the value of the heuristic function to take a optimal path
					root_node = root_node.rightchild	#conditon is which ever node is closer to the goal state the that the optimal path
					shortest_path.append("R")
				if b.hueristic < a.hueristic:
					root_node = root_node.leftchild
					shortest_path.append("L")
			elif a == None and b!= None:				#if the value of node greater than the value of the current location then we use these conditions
				root_node = root_node.leftchild				
			elif b == None and a!=None:					#checking if leftnode is absent to insert right node		
				root_node = root_node.rightchild
			else:
				root_node = root_node

#Main function created
def main():

	#A = [3,6,4,1,3,45,87,0]
	#A = [3,6,4,1,99,99,5,0,]
	#A=[3,6,4,1,3,4,2,5,3,0]
	A = [3,6,4,1,3,4,2,5,3,0]
	#A = [3,4,5,4,2,8,9,0]
	print(solve(A))
if __name__ == '__main__':
	main()



