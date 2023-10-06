"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 03 October 2023

--------------------------------------------------------------------

Metapopulation approach to SIR model of epidemic spreading

"""
from functions_SIR_metapop import *
from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from math import gcd
from functools import reduce
import pickle
import time

start = time.time()

# --------------------------------------------------- Colors ----------------------------------------------------------

new_cmap = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6',
            '#6A3D9A', '#ECEC28', '#B15928', '#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6',
            '#6A3D9A', '#ECEC28', '#B15928', '#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6',
            '#6A3D9A', '#ECEC28', '#B15928']
#new_cmap = ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de", "#0571b0"]
rtg_r = LinearSegmentedColormap.from_list("rtg", new_cmap)
colors = rtg_r(np.linspace(0, 1, 100))

# --------------------------------------------- Parameter initialization ----------------------------------------------

seed = 66
np.random.seed(seed)

# Number of rows and columns in the lattice
N_row = 3
N_col = 3

# Average population per node (fixed)
avg_popPerNode = 1e3

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
# Total population
populationTot = N * avg_popPerNode

# Compute distance matrix of every node with all the others
DistanceMatrix = distance_matrix(G, pos_nodes)
# Populate nodes and set initial conditions for infection
initialize_nodes(G, populationTot, Nfix, percentage_FixNodes, choice_bool, seed)
node_population = nx.get_node_attributes(G, name = 'Npop')
node_NS = nx.get_node_attributes(G, name = 'N_S')
node_NI = nx.get_node_attributes(G, name = 'N_I')
node_NR = nx.get_node_attributes(G, name = 'N_R')
node_population = np.array(list(node_population.values()))
node_NS = np.array(list(node_NS.values()))
node_NI = np.array(list(node_NI.values()))
node_NR = np.array(list(node_NR.values()))
print('node population: ', node_population)
print('node NS : ', node_NS)
print('node NI : ', node_NI)
print('node NR : ', node_NR)
node_density = node_population / populationTot  # population density vector
print('node density: ', node_density)

# Calculate transition matrix
TransitionMatrix = transition_matrix(G, DistanceMatrix, node_density)
weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0 ]
# Add weighted edges to networks : only edges with weight != 0 are added
for i in range(N):
    for j in range(N):
        if TransitionMatrix[i,j] != 0:
            G.add_edge(i, j, weight=TransitionMatrix[i, j])


# save graph object to file
#pickle.dump(G, open('G55.pickle', 'wb'))
# load graph object from file
#G = pickle.load(open('G55.pickle', 'rb'))

# Edge dictionary
dict_edges = nx.get_edge_attributes(G, name = 'weight')
# Control periodicity (the graph should be aperiodic)
#cycles = list(nx.algorithms.cycles.simple_cycles(G))
stopT1 = time.time()
durationT1 = stopT1 - start
print('duration T1:', durationT1)
#cycles_sizes = [len(c) for c in cycles]
#cycles_gcd = reduce(gcd, cycles_sizes)
#is_periodic = cycles_gcd > 1
#print("is_periodic: {}".format(is_periodic))
# Control strongly connected graph
strongConnection = nx.is_strongly_connected(G)
print('Strong connection : ', strongConnection)
stop1 = time.time()
duration1 = stop1 - start
print('Duration up to computation Transition matrix: ', duration1)
# Plot network
plot_network(G, node_population, dict_nodes, weightNonZero)
stop2 = time.time()
duration2 = stop2 - start
print('Duration up to plot of network: ', duration2)
check_convergence(TransitionMatrix)
stop3 = time.time()
duration3 = stop3 - start
print('Duration up to check convergence: ', duration3)
#rho0, rho0check = perron_frobenius_theorem(TransitionMatrix)

plot_network(G, node_population, dict_nodes, weightNonZero)
# ------------------------------------------------ Dynamics -------------------------------------------------
# total simulation length
T = 10
T_sim = np.linspace(0, T, T)

idx_node = 0

fig2 = plt.figure(figsize=(10,10))
for idx_node in range(1):
    popNode_idx = []
    popDensity_idx = []
    for t in range(T):
        Nij, Nij_S, Nij_I, Nij_R = choice_particle_to_move(G, TransitionMatrix)
        move_particle(G, Nij, Nij_S, Nij_I, Nij_R)
        node_population = nx.get_node_attributes(G, name = 'Npop')
        node_NS = nx.get_node_attributes(G, name='N_S')
        node_NI = nx.get_node_attributes(G, name='N_I')
        node_NR = nx.get_node_attributes(G, name='N_R')
        node_population = np.array(list(node_population.values()))
        node_NS = np.array(list(node_NS.values()))
        node_NI = np.array(list(node_NI.values()))
        node_NR = np.array(list(node_NR.values()))

        node_density = node_population/populationTot
        popNode_idx.append(node_population[idx_node])
        popDensity_idx.append(node_density[idx_node])
        #print('node_pop after:', node_population)
        # Control that the total population is exactly the same as the initial one
        print('total pop: before -> ', populationTot, 'after ->', node_population.sum())
        # Plot temporal evolution of network
        plt.clf()
        plot_network(G, node_NI, dict_nodes, weightNonZero)
        plt.pause(1)  ###(10 figures per second) in second the time a figure lasts
    #plt.close()
    plt.plot(T_sim, popDensity_idx, color = colors[idx_node])
plt.axhline(y = avg_popPerNode/populationTot, color = 'black', linestyle = '--', label = 'Fixed average density per node')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Node density')
plt.show()