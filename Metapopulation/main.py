"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 27 September 2023

--------------------------------------------------------------------

Metapopulation approach to SIR model of epidemic spreading

"""
from functions_SIR_metapop import *
from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------- Parameter initialization ----------------------------------------------

# Number of rows and columns in the lattice
N_row = 2
N_col = 2
# Total population
populationTot = 1e6
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
# Compute distance matrix of every node with all the others
DistanceMatrix = distance_matrix(G, pos_nodes)
# Populate nodes and set initial conditions for infection
initialize_nodes(G, populationTot, Nfix, percentage_FixNodes, choice_bool)
node_population = nx.get_node_attributes(G, name = 'Npop')
node_population = np.array(list(node_population.values()))
print(type(node_population))
print('node population: ', node_population)
node_density = node_population / populationTot  # population density vector
print('node density: ', node_density)

# Add edges modeling human mobility through a gravity law
# c is the parameter for the gravity law. I stop as soon as I find the first strongly connected graph
c_min = 0
c_max = 200
# Cycle until the network is strongly connected
strongConnection = False
for c in range(c_min, c_max):
    contFalse = 0
    while strongConnection == False and contFalse < 1000:
        contFalse = contFalse + 1
        # Transition matrix
        TransitionMatrix = transition_matrix(G, DistanceMatrix, node_density, c)
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
    if strongConnection == True:
        print(f'Strong connection for c = {c} : ', strongConnection)
        print(f'False iterations for c = {c}:', contFalse)
        #print('weightNonZero: ', weightNonZero)
        #print('dictEdges: ', dict_edges)
        break
    else:
        print(f'No strongly connected graph for c={c}')


# ------------ From now one I work with a strongly connected graph ------------
# Plot network
plot_network(G, node_population, dict_nodes, dict_edges, weightNonZero)


rho0, rho0check = perron_frobenius_theorem(TransitionMatrix)
