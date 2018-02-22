import sys

class Node():
	def __init__(self, value):
		self.name = value
		self.neighbour = list()
		self.distance = sys.maxsize
		self.color = 'Black'

	def add_neighbour(self, value):
		if value not in self.neighbour:
			self.neighbour.append(value)
			self.neighbour.sort()

class graph():
	vertices = {}
	def add_vertex(self, vertex):
		if isinstance(vertex, Node) and vertex.name not in self.vertices:
			self.vertices[vertex.name] = vertex
			return(True)
		else:
			return False

	def add_edge(self, u,v):
		if u in self.vertices and v in self.vertices:
			for keys, value  in self.vertices.items():
				if keys == u:
					value.add_neighbour(v)
				if keys == v:
					value.add_neighbour(u)
			return(True)

	def bfs(self,vert):
		q = list() 				#closed list
		vert.distance = 0
		vert.color = 'Red'

		for v in vert.neighbour:
			self.vertices[v].distance = vert.distance +1
			q.append(v)
			print(q)

		while len(q)>0:
			u = q.pop(0)
			node_u = self.vertices[u]
			node_u.color = 'Red'


			for v in node_u.neighbour:
				node_v = self.vertices[v]
				if node_u.color == 'Black':
					q.append(v)
					print(list(v))
					if node_v.distance >= node_u.distance +1:
						node_v.distance = node_u.distance +1


	def print_graph(self):
		for key in sorted(list(self.vertices.keys())):
			print(key + str(self.vertices[key].neighbour)+ " " +str(self.vertices[key].distance))
g = graph()
a = Node('A')
g.add_vertex(a)
g.add_vertex('B')

for i in range(ord('A'), ord('K')):
	g.add_vertex(Node(chr(i)))

edges = ['AB', 'AE', 'BF', 'CG', 'DE', 'DH', 'EH', 'FG', 'FI', 'FJ', 'GJ', 'HI']
for edge in edges:
	g.add_edge(edge[:1], edge[1:])


g.bfs(Node('A'))
g.print_graph()
print(list(graph.vertices))



