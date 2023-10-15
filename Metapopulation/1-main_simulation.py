"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 15 October 2023

--------------------------------------------------------------------

Perform simulations on a network already defined

"""

from functions_SIR_metapop import *
from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle

datadir = os.getcwd()
start = time.time()
plt.figure(figsize=(8, 8))

seed = None
np.random.seed(seed)

# ------------------------------------------------ Parameters  -------------------------------------------------
# Parameters I have control on : give in input to load data of the lattice I want
N_row = 3
N_col = 3
choice_bool = 0
datadir = os.getcwd()

c1 = 0  # for now

beta = 0.4
mu = 0.2

# total simulation length
T = 100
T_sim = np.linspace(0, T, T+1)

# Number of infected individuals in one node
popI_node = 1
# List of index of nodes initially containing popI_node infected individuals
idx_nodes_I_init = [0]

# Number of repetitions of a simulation with fixed parameters
nbr_repetitions = 10

# ------------------------------------------------ Colors  -------------------------------------------------
grad_gray = []
grad_red = []
grad_blue = []
grad_green = []

for x in range(N_row*N_col):
    #                                dark           light
    grad_gray.append(colorFader('#505050', '#EAE9E9', x/(N_row * N_col)))
    grad_red.append(colorFader('#E51C00', '#FCE0DC', x/(N_row * N_col)))
    grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x/(N_row * N_col)))
    grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x/(N_row * N_col)))


# -------------------------------------- Load data --------------------------------------
folder_topology = datadir+f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Topology'

# load graph object from file
G = pickle.load(open(folder_topology + '/G.pickle', 'rb'))
dict_nodes = pickle.load(open(folder_topology + '/dict_nodes.pickle', 'rb'))
N = len(G.nodes)

DistanceMatrix = np.load(folder_topology + '/DistanceMatrix.npy')
TransitionMatrix = np.load(folder_topology + '/TransitionMatrix.npy')
weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0]

avg_popPerNode = np.load(folder_topology + '/avg_popPerNode.npy')
if choice_bool == 1:
    Nfix = np.load(folder_topology + '/Nfix.npy')
    percentage_FixNodes = np.load(folder_topology + '/percentage_FixNodes.npy')
else:
    Nfix = 0
    percentage_FixNodes = 0

# --------------------------------------------- Set initial configuration ------------------------------------------------

node_population0 = nx.get_node_attributes(G, name='Npop')
node_population0 = np.array(list(node_population0.values()))
initial_configuration_SIR(G, node_population0, popI_node, idx_nodes_I_init, Nfix, percentage_FixNodes, choice_bool, seed)
node_NS0 = nx.get_node_attributes(G, name='N_S')
node_NI0 = nx.get_node_attributes(G, name='N_I')
node_NR0 = nx.get_node_attributes(G, name='N_R')
node_state0 = nx.get_node_attributes(G, name='state')
node_NS0 = np.array(list(node_NS0.values()))
node_NI0 = np.array(list(node_NI0.values()))
node_NR0 = np.array(list(node_NR0.values()))
node_state0 = np.array(list(node_state0.values()))
# / np.mean(node_population) : I divide by a constant number so I keep fluctuations
node_density0 = node_population0 / np.mean(node_population0)
nodeS_density0 = node_NS0 / np.mean(node_population0)
nodeI_density0 = node_NI0 / np.mean(node_population0)
nodeR_density0 = node_NR0 / np.mean(node_population0)

plot_static_network(G, node_population0, dict_nodes, weightNonZero)


for t in range(T):
    # 1- choice of particles
    Nij, Nij_S, Nij_I, Nij_R = choice_particle_to_move(G, TransitionMatrix)

    # 2- motion of particles
    move_particle(G, Nij, Nij_S, Nij_I, Nij_R)

    # 3- infection step
    NI_new = infection_step_node(G, beta, mu)

    # Number of individuals
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

    # Density per node
    node_density = node_population / np.mean(node_population)
    nodeS_density = node_NS / np.mean(node_population)
    nodeI_density = node_NI / np.mean(node_population)
    nodeR_density = node_NR / np.mean(node_population)

    if t == 0:
        # Populations (matrix - row : node #, column : time)
        node_population_time = np.vstack((node_population0, node_population))
        node_NS_time = np.vstack((node_NS0, node_NS))
        node_NI_time = np.vstack((node_NI0, node_NI))
        node_NR_time = np.vstack((node_NR0, node_NR))
        # Densities  (matrix - row : node #, column : time)
        node_density_time = np.vstack((node_density0, node_density))
        nodeS_density_time = np.vstack((nodeS_density0, nodeS_density))
        nodeI_density_time = np.vstack((nodeI_density0, nodeI_density))
        nodeR_density_time = np.vstack((nodeR_density0, nodeR_density))

        new_I_time = np.vstack((node_NI0, NI_new))
    else:
        # Populations (matrix - row : node #, column : time)
        node_population_time = np.vstack((node_population_time, node_population))
        node_NS_time = np.vstack((node_NS_time, node_NS))
        node_NI_time = np.vstack((node_NI_time, node_NI))
        node_NR_time = np.vstack((node_NR_time, node_NR))
        # Densities  (matrix - row : node #, column : time)
        node_density_time = np.vstack((node_density_time, node_density))
        nodeS_density_time = np.vstack((nodeS_density_time, nodeS_density))
        nodeI_density_time = np.vstack((nodeI_density_time, nodeI_density))
        nodeR_density_time = np.vstack((nodeR_density_time, nodeR_density))

        new_I_time = np.vstack((new_I_time, NI_new))

    node_NI_prev = node_NI
    # Plot temporal evolution of network after infection step
    # plt.clf()
    # plot_network(G, node_population, dict_nodes, weightNonZero, node_state)
    # plt.pause(1)
    # ---------------------------------- Save data after time evolution ----------------------------------------------------
folder_simulation = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/'

write_simulation_file(N_row, N_col, choice_bool, c1, node_population0, node_NS0, node_NI0, node_NR0, node_state0, T,
                      beta, mu, nbr_repetitions)
np.save(folder_simulation + f'beta-{beta}mu-{mu}/T', T)
np.save(folder_simulation + f'beta-{beta}mu-{mu}/node_population_time', node_population_time)
np.save(folder_simulation + f'beta-{beta}mu-{mu}/node_NS_time', node_NS_time)
np.save(folder_simulation + f'beta-{beta}mu-{mu}/node_NI_time', node_NI_time)
np.save(folder_simulation + f'beta-{beta}mu-{mu}/node_NR_time', node_NR_time)

plt.plot( new_I_time)
plt.show()

plot_in_degree_dist(G)
# plt.close()
for idx_node in range(N):
    if idx_node == 0:
        plt.plot(T_sim, node_density_time[:, idx_node], color=grad_gray[idx_node], label='population density')
        plt.plot(T_sim, nodeS_density_time[:, idx_node], color=grad_blue[idx_node], label='S density')
        plt.plot(T_sim, nodeI_density_time[:, idx_node], color=grad_red[idx_node], label='I density')
        plt.plot(T_sim, nodeR_density_time[:, idx_node], color=grad_green[idx_node], label='R density')
    else:
        plt.plot(T_sim, node_density_time[:, idx_node], color=grad_gray[idx_node])
        plt.plot(T_sim, nodeS_density_time[:, idx_node], color=grad_blue[idx_node])
        plt.plot(T_sim, nodeI_density_time[:, idx_node], color=grad_red[idx_node])
        plt.plot(T_sim, nodeR_density_time[:, idx_node], color=grad_green[idx_node])
plt.axhline(y=avg_popPerNode / avg_popPerNode, color='black', linestyle='--', label='Fixed average density per node')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Node density')
# plt.title(f'SIR density node {idx_node}')
plt.show()



