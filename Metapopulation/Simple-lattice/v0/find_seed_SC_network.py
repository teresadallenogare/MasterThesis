"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 07 October 2023

--------------------------------------------------------------------

Find seeds to have a strongly connected network to use then in the main
so that I generate a strongly connected network on which doing simulations

"""
from functions_SIR_metapop import *
from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import gcd
from functools import reduce

# --------------------------------------------- Parameter initialization ----------------------------------------------
tot_seed_strongConn = []
max_seed_idx = 1000
for seed_idx in range(0,max_seed_idx):
    seed = seed_idx
    np.random.seed(seed)


    # Number of rows and columns in the lattice
    N_row = 10
    N_col = 10

    # Number of fixed nodes containing the percentage percentage_FixNodes of population
    Nfix = 3
    percentage_FixNodes = 60
    # choice_bool = 0 : uniform distribution
    # choice_bool = 1 : Nfix nodes have percentage of population equal to percentage_FixNodes %
    choice_bool = 0

    # ------------------------------------------------ Network definition -------------------------------------------------
    # Define node position in the lattice with a square topology
    G, dict_nodes = initialize_lattice(N_row, N_col)
    lab_nodes = list(dict_nodes.keys())
    pos_nodes = list(dict_nodes.values())
    # Number of nodes
    N = len(G.nodes)
    # Average population per node (fixed)
    avg_popPerNode = 1e3
    # Total population
    populationTot = N * avg_popPerNode

    # Compute distance matrix of every node with all the others
    DistanceMatrix = distance_matrix(G, pos_nodes)
    # Populate nodes and set initial conditions for infection
    initialize_nodes(G, populationTot, Nfix, percentage_FixNodes, choice_bool, seed)
    node_population = nx.get_node_attributes(G, name = 'Npop')
    node_population = np.array(list(node_population.values()))
    node_density = node_population / populationTot  # population density vector


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
    #print('Strong connection : ', strongConnection)
    # Control periodicity (the graph should be aperiodic)
    #cycles = list(nx.algorithms.cycles.simple_cycles(G))
    #cycles_sizes = [len(c) for c in cycles]
    #cycles_gcd = reduce(gcd, cycles_sizes)
    #is_periodic = cycles_gcd > 1
    #print("is_periodic: {}".format(is_periodic))

    if strongConnection == True: # and is_periodic == False:
        tot_seed_strongConn.append(seed_idx)

print(tot_seed_strongConn)


