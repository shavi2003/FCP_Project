import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import network as nx 
import argparse

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

	def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes 

	def get_mean_degree(self):
		total_degree = sum(len(node.connections) for node in self.nodes)
        	get_mean_degree = total_degree / len(self.nodes)
        	return get_mean_degree

	def get_mean_clustering(self):
		total_clustering_coefficient = 0
        	for node in self.nodes:
            		neighbors = [self.nodes[idx] for idx, connection in enumerate(node.connections) if connection == 1]
            		if len(neighbors) < 2:
                		continue
            		possible_connections = (len(neighbors) * (len(neighbors) - 1)) / 2
            		actual_connections = sum(sum(
                		self.nodes[nidx].connections[nnidx] for nnidx in range(nidx + 1, len(neighbors)) if
                		self.nodes[nidx].connections[nnidx] == 1) for nidx in range(len(neighbors)))
            		clustering_coefficient = actual_connections / possible_connections
            		total_clustering_coefficient += clustering_coefficient
        	get_mean_clustering = total_clustering_coefficient / len(self.nodes)
        	return get_mean_clustering
	
	def get_mean_path_length(self):
		total_path_length = 0
        	num_paths = 0
        	for source_node in self.nodes.values():
            		if sum(source_node.connections.values()) == 0:
                		continue
            		path_lengths = []
            		for target_index, connection in source_node.connections.items():
                		if connection == 1:
                    			path_lengths.append(nx.shortest_path_length(self.to_networkx_graph(), source=source_node.number, target=target_index))
            			total_path_length += sum(path_lengths)
            			num_paths += len(path_lengths)
        		if num_paths == 0:
            			return float('inf')
        		get_mean_path_length = total_path_length / num_paths
        	return get_mean_path_length

	def make_random_network(self, N, connection_probability):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1


	def make_ring_network(self, N, neighbour_range=1):
		#Your code  for task 4 goes here
        ring_network = nx.Graph()
        for i in range(N):
            for j in range(1, neighbour_range + 1):
                ring_network.add_edge(i, (i + j) % N)
                ring_network.add_edge(i, (i - j) % N)
        return ring_network

	def make_small_world_network(self, N, re_wire_prob=0.2):
		#Your code for task 4 goes here
		ring_network = self.make_ring_network(N)
                for edge in ring_network.edges():
                     if random.random() < re_wire_prob:
                        ring_network.remove_edge(*edge)
                        new_destination = random.choice(list(ring_network.nodes()))
                        while new_destination == edge[0] or ring_network.has_edge(edge[0], new_destination):
                             new_destination = random.choice(list(ring_network.nodes()))
                        ring_network.add_edge(edge[0], new_destination)
                     	return ring_network
	def main():
	    parser = argparse.ArgumentParser(description='Generate different types of networks.')
            parser.add_argument('-ring_network', type=int, help='Create a ring network with specified size')
            parser.add_argument('-small_world', type=int, help='Create a small-worlds network with specified size')
            parser.add_argument('-re_wire', type=float, default=0.2, help='Rewiring probability for small-worlds network')
            parser.add_argument('-show_plot', action='store_true', help='Display the generated network plot')
            args = parser.parse_args()
            if args.ring_network:
               ring_network = create_ring_network(args.ring_network, range=2)  # Default range of 2 for ring network
               if args.show_plot:
                  nx.draw(ring_network, with_labels=True)
                  plt.title('Ring Network')
                  plt.show()

            if args.small_world:
               small_world_network = create_small_world_network(args.small_world, range=2, rewire_prob=args.re_wire)
               if args.show_plot:
                  nx.draw(small_world_network, with_labels=True)
                  plt.title('Small World Network')
                  plt.show()

	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()



		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''

	n_rows, n_cols = population.shape
    agreement = 0
    current_value = population[row, col]
    neighbors = [
        (row - 1) % n_rows, col,  # North
        (row + 1) % n_rows, col,  # South
        row, (col - 1) % n_cols,  # West
        row, (col + 1) % n_cols   # East
    ]

    for i in range(0, len(neighbors), 2):
        neighbor_value = population[neighbors[i], neighbors[i+1]]
        agreement += current_value * neighbor_value

    # Adding external influence
    agreement += external * current_value

    return agreement

def ising_step(population, external=0.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	
	n_rows, n_cols = population.shape
    	for _ in range(n_rows * n_cols):  # Iterate through each cell once per step
        	row = np.random.randint(n_rows)
        	col = np.random.randint(n_cols)
        	agreement = calculate_agreement(population, row, col, external)
        
        	# Flip logic: if agreement is negative, flip the cell
        	if agreement < 0:
            		population[row, col] *= -1
        	elif agreement == 0:  # If agreement is zero, flip based on temperature
           		if np.random.random() < np.exp(-alpha):
                 		population[row, col] *= -1

    	return population

def plot_ising(im, population):
	'''
	This function will display a plot of the Ising model
	'''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)

def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==14), "Test 9"
    assert(calculate_agreement(population,1,1,-10)==-6), "Test 10"

    print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

class Node:
    def __init__(self, value, number):
        self.index = number
        self.value = value

class Network:
    def __init__(self, nodes=None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes

    def update_opinions(self, T, beta):
        i = np.random.randint(0, len(self.nodes))
        # Neighbors in a 1D grid (periodic boundary conditions)
        j = (i + 1) % len(self.nodes) if np.random.rand() > 0.5 else (i - 1) % len(self.nodes)

        if abs(self.nodes[i].value - self.nodes[j].value) < T:
            mean_value = (self.nodes[i].value + self.nodes[j].value) / 2
            self.nodes[i].value += beta * (mean_value - self.nodes[i].value)
            self.nodes[j].value += beta * (mean_value - self.nodes[j].value)

    def plot(self, timesteps):
        # Plot evolution of opinions over time
        plt.figure(figsize=(10, 5))
        for t, timestep in enumerate(timesteps):
            plt.scatter([t] * len(timestep), timestep, c='blue', s=1)
        plt.xlabel("Time")
        plt.ylabel("Opinion")
        plt.title("Evolution of Opinions Over Time")
        plt.show()

        # Plot final distribution
        plt.figure()
        plt.hist(timesteps[-1], bins=30, color='blue', alpha=0.7)
        plt.title("Final Distribution of Opinions")
        plt.xlabel("Opinion")
        plt.ylabel("Frequency")
        plt.show()

def simulate_deffuant(N, num_iterations, beta, T):
    nodes = [Node(np.random.rand(), i) for i in range(N)]
    network = Network(nodes)
    timesteps = []

    for _ in range(num_iterations):
        network.update_opinions(T, beta)
        timesteps.append([node.value for node in network.nodes])

    network.plot(timesteps)

def deffuant_main():
    parser = argparse.ArgumentParser(description="Simulate Opinion Dynamics")
    parser.add_argument("-deffuant", action="store_true", help="Run Deffuant model")
    parser.add_argument("-beta", type=float, default=0.2, help="Set beta value for Deffuant model")
    parser.add_argument("-threshold", type=float, default=0.2, help="Set threshold value for Deffuant model")
    parser.add_argument("-test_deffuant", action="store_true", help="Run tests for Deffuant model")
    args = parser.parse_args()

    if args.deffuant:
        simulate_deffuant(100, 1000, args.beta, args.threshold)
    if args.test_deffuant:
        test_deffuant()

def test_deffuant():
    # Placeholder for actual test implementation
    print("Testing Deffuant model...")
    # Implement tests based on the expectations described

if __name__ == "__deffuant_main__":
    deffuant_main()
'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''
def main():
    parser = argparse.ArgumentParser(description="Network Simulation")
    parser.add_argument("-network", type=int, help="Create and plot a network of size N")
    parser.add_argument("-test_networks", action="store_true", help="Run test functions")

    args = parser.parse_args()

    if args.test_networks:
        test_networks()

    if args.network:
        N = args.network
        network = Network()
        network.make_random_network(N, connection_probability=0.5)  # You can adjust the connection probability as needed
        network.plot()
        mean_degree = network.get_mean_degree()
        mean_path_length = network.get_mean_path_length()
        mean_clustering = network.get_mean_clustering()
        print(f"Mean_degree: {mean_degree}")
        print(f"Mean_path_length: {mean_path_length}")
        print(f"Clustering coefficient: {mean_clustering}")

if __name__ == "__main__":
    main()
