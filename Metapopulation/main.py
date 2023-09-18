"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 07 September 2023

--------------------------------------------------------------------

Metapopulation approach to SIR model of epidemic spreading

"""
from functions_SIR_metapop import *
from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Number of rows and columns in the lattice
N_row = 3
N_col = 3

# Define node position in the lattice with a square topology
G, pos, labels = initialize_lattice(N_row, N_col)
N = len(G.nodes)
# Compute distance matrix of every node with all the others
DistanceMatrix = distance_matrix(G, pos)

# Create node list with each node randomly populated
populationTot = 1e3
Nfix = 3
percentage_FixNodes = 80
# choice_bool = 0 : uniform distribution
# choice_bool = 1 : Nfix nodes have percentage of population equal to percentage_FixNodes %
choice_bool = 0
node_list = create_node_list(G, populationTot, Nfix, percentage_FixNodes, choice_bool)
node_population = np.array([node_list[i].Npart for i in G.nodes])
print(node_population)
node_density = node_population / populationTot  # population density vector

# Transition matrix
TransitionMatrix = transition_matrix(G, DistanceMatrix, node_density)
weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]

# Add weights to the network
for i in range(N):
    for j in range(N):
        G.add_edge(i, j, weight=TransitionMatrix[i, j])

# Plot network
plot_network(G, pos, node_population, labels, weight)

rho0, rho0check = perron_frobenius_theorem(TransitionMatrix)
