"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 27 November 2023

--------------------------------------------------------------------

Generate a square lattice network with populations in nodes that are
randomly selected from a multinomial distribution.
Generate the transition (stochastic matrix) imposing row normalization to 1 by fixing the self loop value.
This transition matrix is fixed.

"""

from functions_square_network_v1 import *
from functions_square_output_v1 import write_topology_file
from functions_square_visualization_v1 import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

datadir = os.getcwd()

plt.figure(figsize=(8, 8))

seed = None
np.random.seed(seed)

# ------------------------------------------------ Parameters  -------------------------------------------------
# Number of rows and columns in the lattice
N_row = 50
N_col = 50

# Average population per node (fixed)
avg_popPerNode = 1e4

# Number of fixed nodes containing the percentage percentage_FixNodes of population
Nfix = math.ceil(10 / 100 * N_col * N_row)
percentage_FixNodes = 20

# choice_bool = 0 : uniform distribution
# choice_bool = 1 : Nfix nodes have percentage of population equal to percentage_FixNodes %
choice_bool = 0

# Parameters to establish the connectivity and the self loops
alpha_vals = [0.2, 1]  # Strength of the loop (alpha = 1 means low self loops, alpha = 1/2  means high self-loops)

# ------------------------------------------ Lattice initialization  -----------------------------------------

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
idxNfix = initialize_node_population(G, populationTot, Nfix, percentage_FixNodes, choice_bool, seed)

node_population0 = nx.get_node_attributes(G, name='Npop')
node_population0 = np.array(list(node_population0.values()))
# / np.mean(node_population) : I divide by a constant number so I keep fluctuations
node_density0 = node_population0 / avg_popPerNode

for alpha in alpha_vals:
    b = alpha * np.exp(- 1/N_row)
    c1 = 0 if alpha == 1 else 1
    print('c1: ', c1)
    folder_topology = datadir + f'/Data_squareLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    TransitionMatrix, c1_real = transition_matrix_square(G, DistanceMatrix, node_density0, b)
    print('c1_real :', c1_real)
    weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
    weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0]
    # Add weighted edges to networks : only edges with weight != 0 are added
    for i in range(N):
        for j in range(N):
            if TransitionMatrix[i, j] != 0:
                G.add_edge(i, j, weight=TransitionMatrix[i, j])
    # Edge dictionary
    dict_edges = nx.get_edge_attributes(G, name='weight')

    # Input degree
    in_degrees = [G.in_degree(n) for n in G.nodes()]

    # Check PF convergence
    rho0, k_list, diff_list = PF_convergence(TransitionMatrix, N)
    # Stationary density vector of people per node
    print('rho0: ', rho0)

    # Control if the topology file exists.
    # If it exists, then ask if I want to overwrite data
    # If it does not exist, save data
    check_file = os.path.isfile(folder_topology + 'topologyFile.txt')
    if check_file:
        print('Files already saved ')
        overwrite = int(input('Do you want to overwrite files? 0 for no, 1 for yes : '))
        if overwrite == 0:
            sys.exit('Files no overwritten')
        elif overwrite == 1:
            print('Overwrite files')

# Write topology file
    write_topology_file(N_row, N_col, N, pos_nodes, avg_popPerNode, populationTot, choice_bool, Nfix, idxNfix,
                        percentage_FixNodes, c1,
                        node_population0, b, rho0, k_list, diff_list, in_degrees)

    # Plot network
    if N_row == 3 or N_row == 5 or N_row == 10:
        plot_static_network(G, node_population0, dict_nodes, weightNonZero, N_row, N_col, choice_bool, c1)
        plot_TransitionMatrix(TransitionMatrix, N_row, N_col, choice_bool, c1)

    # Save parameters
    np.save(folder_topology + 'pos_nodes', pos_nodes)
    np.save(folder_topology + 'avg_popPerNode', avg_popPerNode)
    np.save(folder_topology + 'choice_bool', choice_bool)

    np.save(folder_topology + 'b', b)
    np.save(folder_topology + 'c1_real', c1_real)
    if choice_bool == 1:
        np.save(folder_topology + 'Nfix', Nfix)
        np.save(folder_topology + 'percentage_FixNodes', percentage_FixNodes)
        np.save(folder_topology + 'idxNfix', idxNfix)
    np.save(folder_topology + 'rho0', rho0)
    np.save(folder_topology + 'k_list', k_list)
    np.save(folder_topology + 'diff_list', diff_list)
    # Save graph object
    pickle.dump(G, open(folder_topology + 'G.pickle', 'wb'))
    pickle.dump(dict_nodes, open(folder_topology + 'dict_nodes.pickle', 'wb'))
    np.save(folder_topology + 'DistanceMatrix', DistanceMatrix)
    np.save(folder_topology + 'TransitionMatrix', TransitionMatrix)
    print('Files saved or overwritten')
    # Remove edges of G after I saved all data
    for i in range(N):
        for j in range(N):
            if TransitionMatrix[i, j] != 0:
                G.remove_edge(i, j)

