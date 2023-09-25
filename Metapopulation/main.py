"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 23 September 2023

--------------------------------------------------------------------

Metapopulation approach to SIR model of epidemic spreading

"""
from functions_SIR_metapop import *
from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Number of rows and columns in the lattice
N_row = 4
N_col = 4

# Define node position in the lattice with a square topology
G, dict_nodes = initialize_lattice(N_row, N_col)
lab_nodes = list(dict_nodes.keys())
pos_nodes = list(dict_nodes.values())

N = len(G.nodes)
# Compute distance matrix of every node with all the others
DistanceMatrix = distance_matrix(G, pos_nodes)

# Create node list with each node randomly populated
populationTot = 1e4
Nfix = 3
percentage_FixNodes = 80
# choice_bool = 0 : uniform distribution
# choice_bool = 1 : Nfix nodes have percentage of population equal to percentage_FixNodes %
choice_bool = 0
node_list = create_node_list(G, populationTot, Nfix, percentage_FixNodes, choice_bool)
node_population = np.array([node_list[i].Npart for i in G.nodes])
print('node population: ', node_population)
node_density = node_population / populationTot  # population density vector

strongConnection = False
contFalse = 0
while strongConnection == False and contFalse < 1000:
    contFalse = contFalse + 1
    # Transition matrix
    TransitionMatrix = transition_matrix(G, DistanceMatrix, node_density)
    weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
    weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0 ]
    # Add weighted edges to networks : only edges with weight != 0 are added
    for i in range(N):
        for j in range(N):
            if TransitionMatrix[i,j] != 0:
                G.add_edge(i, j, weight=TransitionMatrix[i, j])
    # Edge dictionary
    dict_edges = nx.get_edge_attributes(G, name = 'weight')
    # Control strongly connected graph
    strongConnection = nx.is_strongly_connected(G)
    if strongConnection == False:
        for i in range(N):
            for j in range(N):
                if TransitionMatrix[i, j] != 0:
                    G.remove_edge(i, j)
print(strongConnection)
print(contFalse)
print('weightNonZero: ',weightNonZero)
print('dictEdges: ',dict_edges)
# ------------ From now one I work with a strongly connected graph ------------
# Plot network
plot_network(G, node_population, dict_nodes, dict_edges, weightNonZero)


rho0, rho0check = perron_frobenius_theorem(TransitionMatrix)
