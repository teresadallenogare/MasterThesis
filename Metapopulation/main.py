"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 10 October 2023

--------------------------------------------------------------------

Metapopulation approach to SIR model of epidemic spreading.
File to generate data.
Still have to implement repetitions

"""
from functions_SIR_metapop import *
from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time

datadir = os.getcwd()
start = time.time()
plt.figure(figsize=(8, 8))

# --------------------------------------------- Parameter initialization ----------------------------------------------

seed = 30
np.random.seed(seed)

# Number of rows and columns in the lattice
N_row = 10
N_col = 10

# Average population per node (fixed)
avg_popPerNode = 1e4
# number of infected individuals in one node
popI_node = 1000
# list of index of nodes initially containing popI_node infected individuals
idx_nodes_I_init = [0]

# Number of fixed nodes containing the percentage percentage_FixNodes of population
Nfix = 3
percentage_FixNodes = 60

# choice_bool = 0 : uniform distribution
# choice_bool = 1 : Nfix nodes have percentage of population equal to percentage_FixNodes %
choice_bool = 0

# infection rate and recovery rate
beta = 0.55
mu = 0.52

# total simulation length
T = 100
T_sim = np.linspace(0, T, T)

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
initialize_nodes(G, populationTot, popI_node, idx_nodes_I_init, Nfix, percentage_FixNodes, choice_bool, seed)

# ------------------------------------------------ Colors  -------------------------------------------------
grad_gray = []
grad_red = []
grad_blue = []
grad_green = []

for x in range(N):
    #                                dark           light
    grad_gray.append(colorFader('#505050', '#EAE9E9', x/N))
    grad_red.append(colorFader('#E51C00', '#FCE0DC', x/N))
    grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x/N))
    grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x/N))

for t in range(T):
    # ------------------------------------------------ Network definition -------------------------------------------------
    if t == 0:
        node_population0 = nx.get_node_attributes(G, name='Npop')
        node_NS0 = nx.get_node_attributes(G, name='N_S')
        node_NI0 = nx.get_node_attributes(G, name='N_I')
        node_NR0 = nx.get_node_attributes(G, name='N_R')
        node_state0 = nx.get_node_attributes(G, name='state')
        node_population0 = np.array(list(node_population0.values()))
        node_NS0 = np.array(list(node_NS0.values()))
        node_NI0 = np.array(list(node_NI0.values()))
        node_NR0 = np.array(list(node_NR0.values()))
        node_state0 = np.array(list(node_state0.values()))
        # / np.mean(node_population) : I divide by a constant number so I keep fluctuations
        node_density0 = node_population0 / np.mean(node_population0)
        nodeS_density0 = node_NS0 / np.mean(node_population0)
        nodeI_density0 = node_NI0 / np.mean(node_population0)
        nodeR_density0 = node_NR0 / np.mean(node_population0)
        # Calculate transition matrix
        TransitionMatrix, c1 = transition_matrix(G, DistanceMatrix, node_density0)
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

        #Check PF convergence
        check_convergence(TransitionMatrix)
        # rho0, rho0check = perron_frobenius_theorem(TransitionMatrix)

        # Check network
        #plot_network(G, node_population0, dict_nodes, weightNonZero, node_state0)
        # -------------------------------------------- Write topology file  ---------------------------------------------
        write_topology_file(N_row, N_col, N, seed, avg_popPerNode, choice_bool, Nfix, percentage_FixNodes, c1, beta, mu, T,
                        node_population0, node_NS0, node_NI0, node_NR0, node_state0)
        folder_data = f'/choice_bool-{choice_bool}/c1-{int(c1)}/beta-{beta}mu-{mu}'
        folder_topology = datadir+f'/Data-simpleLattice/{N_row}x{N_col}'+folder_data+'/Topology'
        np.save(folder_topology + '/DistanceMatrix', DistanceMatrix)
        np.save(folder_topology + '/TransitionMatrix', TransitionMatrix)
    else:
        # ------------------------------------------------ Dynamics -------------------------------------------------

        # 1- choice of particles
        Nij, Nij_S, Nij_I, Nij_R = choice_particle_to_move(G, TransitionMatrix)

        # 2- motion of particles
        move_particle(G, Nij, Nij_S, Nij_I, Nij_R)

        # 3- infection step
        infection_step_node(G, beta, mu)

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

        node_density = node_population / np.mean(node_population)
        nodeS_density = node_NS / np.mean(node_population)
        nodeI_density = node_NI / np.mean(node_population)
        nodeR_density = node_NR / np.mean(node_population)

        if t == 1:
            # populations
            node_population_time = np.vstack((node_population0, node_population))
            node_NS_time = np.vstack((node_NS0, node_NS))
            node_NI_time = np.vstack((node_NI0, node_NI))
            node_NR_time = np.vstack((node_NR0, node_NR))
            # densities
            node_density_time = np.vstack((node_density0, node_density))
            nodeS_density_time = np.vstack((nodeS_density0, nodeS_density))
            nodeI_density_time = np.vstack((nodeI_density0, nodeI_density))
            nodeR_density_time = np.vstack((nodeR_density0, nodeR_density))
        else:
            # populations
            node_population_time = np.vstack((node_population_time, node_population))
            node_NS_time = np.vstack((node_NS_time, node_NS))
            node_NI_time = np.vstack((node_NI_time, node_NI))
            node_NR_time = np.vstack((node_NR_time, node_NR))
            # densities
            node_density_time = np.vstack((node_density_time, node_density))
            nodeS_density_time = np.vstack((nodeS_density_time, nodeS_density))
            nodeI_density_time = np.vstack((nodeI_density_time, nodeI_density))
            nodeR_density_time = np.vstack((nodeR_density_time, nodeR_density))

            # Plot temporal evolution of network after infection step
            #plt.clf()
            #plot_network(G, node_population, dict_nodes, weightNonZero, node_state)
            #plt.pause(1)
# ---------------------------------- Save data after time evolution ----------------------------------------------------
folder_dynamics = datadir+f'/Data-simpleLattice/{N_row}x{N_col}'+folder_data+'/Dynamics'
np.save(folder_dynamics + '/node_population_time', node_population_time)
np.save(folder_dynamics + '/node_NS_time', node_NS_time)
np.save(folder_dynamics + '/node_NI_time',node_NI_time)
np.save(folder_dynamics + '/node_NR_time', node_NR_time)



#plt.close()
for idx_node in range(N):
    if idx_node == 0:
        plt.plot(T_sim, node_density_time[:, idx_node], color = grad_gray[idx_node], label = 'population density')
        plt.plot(T_sim, nodeS_density_time[:, idx_node], color = grad_blue[idx_node], label = 'S density')
        plt.plot(T_sim, nodeI_density_time[:, idx_node], color = grad_red[idx_node], label = 'I density')
        plt.plot(T_sim, nodeR_density_time[:, idx_node], color = grad_green[idx_node], label = 'R density')
    else:
        plt.plot(T_sim, node_density_time[:, idx_node], color=grad_gray[idx_node])
        plt.plot(T_sim, nodeS_density_time[:, idx_node], color=grad_blue[idx_node])
        plt.plot(T_sim, nodeI_density_time[:, idx_node], color=grad_red[idx_node])
        plt.plot(T_sim, nodeR_density_time[:, idx_node], color=grad_green[idx_node])
plt.axhline(y = avg_popPerNode/avg_popPerNode, color = 'black', linestyle = '--', label = 'Fixed average density per node')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Node density')
#plt.title(f'SIR density node {idx_node}')
plt.show()
