"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 07 October 2023

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
N_row = 10
N_col = 10

# Average population per node (fixed)
avg_popPerNode = 1e3

# number of infected individuals in one node
popI_node = 1
# list of index of nodes initially containing popI_node infected individuals
idx_nodes_I_init = [0]

# Number of fixed nodes containing the percentage percentage_FixNodes of population
Nfix = 3
percentage_FixNodes = 60

# choice_bool = 0 : uniform distribution
# choice_bool = 1 : Nfix nodes have percentage of population equal to percentage_FixNodes %
choice_bool = 0

# infection rate and recovery rate
beta = 0.9
mu = 0.2

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
initialize_nodes(G, populationTot, popI_node, idx_nodes_I_init, Nfix, percentage_FixNodes, choice_bool, seed)
node_population = nx.get_node_attributes(G, name = 'Npop')
node_NS = nx.get_node_attributes(G, name = 'N_S')
node_NI = nx.get_node_attributes(G, name = 'N_I')
node_NR = nx.get_node_attributes(G, name = 'N_R')
node_state = nx.get_node_attributes(G, name = 'state')
node_population = np.array(list(node_population.values()))
node_NS = np.array(list(node_NS.values()))
node_NI = np.array(list(node_NI.values()))
node_NR = np.array(list(node_NR.values()))
node_state = np.array(list(node_state.values()))
print('----- Initial setup -----')
t = 0
print('t: ', t, 'Npop:', node_population)
print('t: ', t, 'NS: ', node_NS)
print('t: ', t, 'NI: ', node_NI)
print('t: ', t, 'NR: ', node_NR)
print('t: ', t, 'state: ', node_state)
print('-------------------------')
node_density = node_population / populationTot  # population density vector

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

check_convergence(TransitionMatrix)
#rho0, rho0check = perron_frobenius_theorem(TransitionMatrix)


# ------------------------------------------------ Dynamics -------------------------------------------------
# total simulation length
T = 100
T_sim = np.linspace(1, T, T-1)

idx_node = 0

#fig2 = plt.figure(figsize=(10,10))

# Population inside the node with index idx at each time step
popNode_idx = []
popDensity_idx = []
sDensity_idx = []
iDensity_idx = []
rDensity_idx = []
# t starts from 1 because t = 0 is the initial condition
for t in range(1, T):
    # Motion of particles chosen between nodes
    Nij, Nij_S, Nij_I, Nij_R = choice_particle_to_move(G, TransitionMatrix)
    move_particle(G, Nij, Nij_S, Nij_I, Nij_R)
    print('----- Motion -----')
    node_population = nx.get_node_attributes(G, name='Npop')
    node_NS = nx.get_node_attributes(G, name='N_S')
    node_NI = nx.get_node_attributes(G, name='N_I')
    node_NR = nx.get_node_attributes(G, name='N_R')
    node_state = nx.get_node_attributes(G, name='state')
    node_population = np.array(list(node_population.values()))
    node_NS = np.array(list(node_NS.values()))
    node_NI = np.array(list(node_NI.values()))
    node_NR = np.array(list(node_NR.values()))
    node_state = np.array(list(node_state.values()))
    print('t: ', t, 'Npop:', node_population)
    print('t: ', t, 'NS: ', node_NS)
    print('t: ', t, 'NI: ', node_NI)
    print('t: ', t, 'NR: ', node_NR)
    print('t: ', t, 'state: ', node_state)
    # Control that the total population is exactly the same as the initial one
    print('total pop: before -> ', populationTot, 'after ->', node_population.sum())
    # Infection step
    infection_step_node(G, beta, mu)
    print('----- Infection -----')
    node_population = nx.get_node_attributes(G, name='Npop')
    node_NS = nx.get_node_attributes(G, name='N_S')
    node_NI = nx.get_node_attributes(G, name='N_I')
    node_NR = nx.get_node_attributes(G, name='N_R')
    node_state = nx.get_node_attributes(G, name='state')
    node_population = np.array(list(node_population.values()))
    node_NS = np.array(list(node_NS.values()))
    node_NI = np.array(list(node_NI.values()))
    node_NR = np.array(list(node_NR.values()))
    node_state = np.array(list(node_state.values()))
    print('t: ', t, 'Npop:', node_population)
    print('t: ', t, 'NS: ', node_NS)
    print('t: ', t, 'NI: ', node_NI)
    print('t: ', t, 'NR: ', node_NR)
    print('t: ', t, 'state: ', node_state)
    node_density = node_population / populationTot
    nodeS_density = node_NS / populationTot
    nodeI_density = node_NI / populationTot
    nodeR_density = node_NR / populationTot
    popNode_idx.append(node_population[idx_node])
    popDensity_idx.append(node_density[idx_node])
    sDensity_idx.append(nodeS_density[idx_node])

    iDensity_idx.append(nodeI_density[idx_node])
    rDensity_idx.append(nodeR_density[idx_node])

    # Plot temporal evolution of network after infection step
    # #plt.clf()
    #plot_network(G, node_population, dict_nodes, weightNonZero, node_state)
    #plt.pause(0.4)


#plt.close()
plt.plot(T_sim, popDensity_idx, color = 'grey', label = 'population density')
plt.plot(T_sim, sDensity_idx, color = 'blue', label = 'S density')
plt.plot(T_sim, iDensity_idx, color = 'red', label = 'I density')
plt.plot(T_sim, rDensity_idx, color = 'green', label = 'R density')
plt.axhline(y = avg_popPerNode/populationTot, color = 'black', linestyle = '--', label = 'Fixed average density per node')
#plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Node density')
plt.title(f'SIR density node {idx_node}')
plt.show()

