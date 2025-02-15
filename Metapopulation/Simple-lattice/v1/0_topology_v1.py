"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 20 November 2023

--------------------------------------------------------------------

Generate a strongly connected network with populations in nodes that are
randomly selected from a multinomial distribution.
Generate the transition (stochastic matrix) imposing row normalization to 1 by fixing the self loop value.
This transition matrix is fixed.
Idea is to provide the same topology for the case of large and small self loops (that are controlled by c1)

"""

from functions_network_v1 import *
from functions_output_v1 import write_topology_file
from functions_visualization_v1 import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

datadir = os.getcwd()

seed = None
np.random.seed(seed)

# ------------------------------------------------ Parameters  -------------------------------------------------
# Number of rows and columns in the lattice
N_row = 50
N_col = 50

# Average population per node (fixed)
avg_popPerNode = 1e4

# Number of fixed nodes containing the percentage percentage_FixNodes of population
# consider the 20% of population in the 10% of nodes
Nfix = math.ceil(10 / 100 * N_col * N_row)
print('Nfix: ', Nfix)
percentage_FixNodes = 20

# choice_bool = 0 : uniform distribution
# choice_bool = 1 : Nfix nodes have percentage of population equal to percentage_FixNodes %
choice_bool = 1

# Parameters to establish the connectivity and the self loops
a = 0.2  # Fixed parameter for the connectivity
alpha_vals = [0.2, 1]  # Strength of the loop (alpha = 1 means low self loops, alpha = 0.2 means high self-loops)
nbr_repetitions = 10

for repeat in range(1, nbr_repetitions):
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

    # Density in principle can be density per node or global density in the system, that is I can have density obtained by dividing
    # by the average population per node or the density obtained dividing  by the whole population. What changes is a factor
    # N at the denominator.
    # Density calculated like this for both choice_bool 0 and 1 to keep track of presence of more populated nodes.
    node_population0 = nx.get_node_attributes(G, name='Npop')
    node_population0 = np.array(list(node_population0.values()))
    node_density0 = node_population0 / avg_popPerNode
    # Control on strong connection regards the connectivity not the weights
    strongConnection = False
    contFalse = 0
    while strongConnection == False and contFalse < 1000:
        contFalse = contFalse + 1
        print('contFalse:', contFalse)
        AdjacencyMatrix = adjacency_matrix(G, DistanceMatrix, node_density0, a)
        for i in range(N):
            for j in range(N):
                if AdjacencyMatrix[i, j] != 0:
                    G.add_edge(i, j, weight=AdjacencyMatrix[i, j])
        # Edge dictionary
        dict_edges = nx.get_edge_attributes(G, name='weight')
        # Control strongly connected graph
        strongConnection = nx.is_strongly_connected(G)
        print('Strong connection : ', strongConnection)
        if not strongConnection:
            for i in range(N):
                for j in range(N):
                    if AdjacencyMatrix[i, j] != 0:
                        G.remove_edge(i, j)
    # Remove edges in G after controlling the strong connection
    for i in range(N):
        for j in range(N):
            if AdjacencyMatrix[i, j] != 0:
                G.remove_edge(i, j)

    # Provided the same adjacency matrix I consider two different weights
    for alpha in alpha_vals:
        weight = []
        weightNonZero = []
        in_degrees = []
        b = alpha * np.exp(- 1 / N_row)
        c1 = 0 if alpha == 1 else 1
        print('c1: ', c1)
        folder_topology = datadir + f'/Data_simpleLattice_v1/Repeated_topologies/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
        # Cycle until I have a strongly connected graph but with population initialized at the beginning
        # Calculate transition matrix
        TransitionMatrix, c1_real = transition_matrix(G, AdjacencyMatrix, DistanceMatrix, node_density0, b)
        print('c1: ', c1_real)
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

        print('------------ Transition matrix ------------ ')
        # Check PF convergence
        rho0, k_list, diff_list = PF_convergence(TransitionMatrix, N)
        # Stationary density vector of people per node
        print('rho0: ', rho0)

        # Control if the topology file exists.
        # If it exists, then ask if I want to overwrite data
        # If it does not exist, save data
        check_file = os.path.isfile(folder_topology + f'topologyFile_rep{repeat}.txt')
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
                            node_population0, strongConnection, a, b, rho0, k_list, diff_list, in_degrees, repeat)

        # Plot network
        #if N_row == 3 or N_row == 5 or N_row == 10:
        #    plot_static_network(G, node_population0, dict_nodes, weightNonZero, N_row, N_col, choice_bool, c1)
        #    plot_TransitionMatrix(TransitionMatrix, N_row, N_col, choice_bool, c1)

        # Save parameters
        np.save(folder_topology + f'pos_nodes_rep{repeat}', pos_nodes)
        np.save(folder_topology + f'avg_popPerNode_rep{repeat}', avg_popPerNode)
        np.save(folder_topology + f'choice_bool_rep{repeat}', choice_bool)
        np.save(folder_topology + f'a_rep{repeat}', a)
        np.save(folder_topology + f'b_rep{repeat}', b)
        np.save(folder_topology + f'c1_real_rep{repeat}', c1_real)
        if choice_bool == 1:
            np.save(folder_topology + f'Nfix_rep{repeat}', Nfix)
            np.save(folder_topology + f'percentage_FixNodes_rep{repeat}', percentage_FixNodes)
            np.save(folder_topology + f'idxNfix_rep{repeat}', idxNfix)
        np.save(folder_topology + f'rho0_rep{repeat}', rho0)
        np.save(folder_topology + f'k_list_rep{repeat}', k_list)
        np.save(folder_topology + f'diff_list_rep{repeat}', diff_list)
        # Save graph object
        pickle.dump(G, open(folder_topology + f'G_rep{repeat}.pickle', 'wb'))
        pickle.dump(dict_nodes, open(folder_topology + f'dict_nodes_rep{repeat}.pickle', 'wb'))
        np.save(folder_topology + f'DistanceMatrix_rep{repeat}', DistanceMatrix)
        np.save(folder_topology + f'AdjacencyMatrix_rep{repeat}', AdjacencyMatrix)
        np.save(folder_topology + f'TransitionMatrix_rep{repeat}', TransitionMatrix)
        print('Files saved or overwritten')
        # Remove edges of G after I saved all data
        for i in range(N):
            for j in range(N):
                if TransitionMatrix[i, j] != 0:
                    G.remove_edge(i, j)
