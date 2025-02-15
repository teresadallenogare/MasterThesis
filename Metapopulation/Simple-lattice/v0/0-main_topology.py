"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 15 October 2023

--------------------------------------------------------------------

Generate a strongly connected network with populations in nodes that are
randomly selected from a multinomial distribution.
Generate the transition (stochastic matrix) imposing row normalization to 1 by fixing the self loop value.
This transition matrix is fixed.

"""

from functions_SIR_metapop import *
from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
import seaborn as sns

datadir = os.getcwd()
start = time.time()
plt.figure(figsize=(8, 8))

seed = None
np.random.seed(seed)
# ------------------------------------------------ Parameters  -------------------------------------------------
# Number of rows and columns in the lattice
N_row = 3
N_col = 3

# Average population per node (fixed)
avg_popPerNode = 1e4

# Number of fixed nodes containing the percentage percentage_FixNodes of population
Nfix = 3
percentage_FixNodes = 60

# choice_bool = 0 : uniform distribution
# choice_bool = 1 : Nfix nodes have percentage of population equal to percentage_FixNodes %
choice_bool = 0

# Parameters to establish the connectivity and the self loops
a = 0.2  # establish connectivity
b = 0.5  # establish self loop (low b means very high self loops)

# ------------------------------------------------ Lattice initialization  -------------------------------------------------

# Define node position in the lattice with a square topology
G, dict_nodes = initialize_lattice(N_row, N_col)
lab_nodes = list(dict_nodes.keys())
pos_nodes = list(dict_nodes.values())
# Number of nodes
N = len(G.nodes)
# Total population
populationTot = N * avg_popPerNode
# Compute distance matrix of every node with all the others
DistanceMatrix = distance_matrix(G, pos_nodes)
# Populate nodes and set initial conditions for infection
initialize_node_population(G, populationTot, Nfix, percentage_FixNodes, choice_bool, seed)

node_population0 = nx.get_node_attributes(G, name='Npop')
node_population0 = np.array(list(node_population0.values()))
# / np.mean(node_population) : I divide by a constant number so I keep fluctuations
node_density0 = node_population0 / np.mean(node_population0)
# Cycle until I have a strongly connected graph but with population initialized at the beginning
# Calculate transition matrix
strongConnection = False
contFalse = 0
while strongConnection == False and contFalse < 1000:
    contFalse = contFalse + 1
    TransitionMatrix, c1 = transition_matrix(G, DistanceMatrix, node_density0, a, b)
    print('c1: ', c1)
    weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
    weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0]
    # Add weighted edges to networks : only edges with weight != 0 are added
    for i in range(N):
        for j in range(N):
            if TransitionMatrix[i, j] != 0:
                G.add_edge(i, j, weight=TransitionMatrix[i, j])
    # Edge dictionary
    dict_edges = nx.get_edge_attributes(G, name='weight')
    # Control strongly connected graph
    strongConnection = nx.is_strongly_connected(G)
    print('Strong connection : ', strongConnection)
    if strongConnection == False:
        for i in range(N):
            for j in range(N):
                if TransitionMatrix[i, j] != 0:
                    G.remove_edge(i, j)

#Check PF convergence
#check_convergence(TransitionMatrix)
# rho0, rho0check = perron_frobenius_theorem(TransitionMatrix)

# Check network
#plot_network(G, node_population0, dict_nodes, weightNonZero, node_state0)

# Look at transition matrix
sns.heatmap(TransitionMatrix)

# Look at properties of the graph
#plot_centralities(G)

in_degrees = [G.in_degree(n) for n in G.nodes()]
plt.figure()
plt.bar(*np.unique(in_degrees, return_counts=True))
plt.title("Degree centrality input edges")  # out degrees = in_degrees by construction
plt.xlabel(" Input degree")
plt.ylabel("Frequency")
c1 = 1
# -------------------------------------------- Write topology file  ---------------------------------------------
write_topology_file(N_row, N_col, N, avg_popPerNode, choice_bool, Nfix, percentage_FixNodes, c1, node_population0)

folder_topology = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Topology'
np.save(folder_topology + '/pos_nodes', pos_nodes)
# save graph object to file
pickle.dump(G, open(folder_topology + '/G.pickle', 'wb'))
pickle.dump(dict_nodes, open(folder_topology + '/dict_nodes.pickle', 'wb'))
np.save(folder_topology + '/DistanceMatrix', DistanceMatrix)
np.save(folder_topology + '/TransitionMatrix', TransitionMatrix)
plt.figure()
plot_static_network(G, node_population0, dict_nodes, weightNonZero)
