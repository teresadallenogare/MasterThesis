"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 15 November 2023

--------------------------------------------------------------------

Metapopulation simulation of SIR epidemics on network G

"""
from functions_SIR_metapop_v1 import *
from functions_output_v1 import write_simulation_file
from functions_visualization_v1 import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import math

datadir = os.getcwd()
plt.figure(figsize=(8, 8))

seed = None
np.random.seed(seed)

# ------------------------------------------------ Parameters  -------------------------------------------------
N_row = 10
N_col = 10

choice_bool = 0
c1 = 0

# Infection and recovery rate
beta = 1.2
mu = 0.3

# Total simulation length
T = 120
T_sim = np.linspace(0, T-1, T)

# Number of infected individuals in one node
popI_node = 2

# List of index of nodes initially containing popI_node infected individuals
idx_nodes_I_init = [0]

# Number of repetitions of a simulation with fixed parameters
nbr_repetitions = 10

folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
folder_simulation = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'

np.save(folder_simulation + f'mu-{mu}/beta-{beta}/T', T)
np.save(folder_simulation + f'mu-{mu}/beta-{beta}/nbr_repetitions', nbr_repetitions)

# ------------------------------------------------ Simulations -------------------------------------------------
nbr_sim_not_start = 0
idx_sim_not_start = []

for sim in range(nbr_repetitions):
    # Load graph and initial conditions at every new simulation
    G = pickle.load(open(folder_topology + 'G.pickle', 'rb'))
    dict_nodes = pickle.load(open(folder_topology + 'dict_nodes.pickle', 'rb'))
    N = len(G.nodes)
    TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
    weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
    weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0]
    avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')
    popTot = avg_popPerNode * N
    if choice_bool == 1:
        Nfix = np.load(folder_topology + 'Nfix.npy')
        percentage_FixNodes = np.load(folder_topology + 'percentage_FixNodes.npy')
        idxNfix = np.load(folder_topology + 'idxNfix.npy')
        pop_FixNodes = math.floor(percentage_FixNodes / 100 * popTot)
        pop_others = popTot - pop_FixNodes

        avg_popPerNode_Nfix = pop_FixNodes / Nfix
        avg_popPerNode_Others = pop_others / (N-Nfix)
    else:
        Nfix = 0
        percentage_FixNodes = 0
    # I have population in nodes in both choice_bool = 0,1 cases
    node_population0 = nx.get_node_attributes(G, name='Npop')
    node_population0 = np.array(list(node_population0.values()))

    initial_configuration_SIR(G, node_population0, popI_node, idx_nodes_I_init, Nfix, percentage_FixNodes, choice_bool, seed)

    # Initial populations
    node_NS0 = nx.get_node_attributes(G, name='N_S')
    node_NI0 = nx.get_node_attributes(G, name='N_I')
    node_NR0 = nx.get_node_attributes(G, name='N_R')
    node_state0 = nx.get_node_attributes(G, name='state')
    node_NS0 = np.array(list(node_NS0.values()))
    node_NI0 = np.array(list(node_NI0.values()))
    node_NR0 = np.array(list(node_NR0.values()))
    node_state0 = np.array(list(node_state0.values()))

    # Note : if I maintain the idea of dividing by the temporal average I should only divide by node_population0,
    # not by the mean. Otherwise, I should always divide by both the network average + temporal average
    # In alternative, I cal always divide by the initial population
    #nodeS_density0 = node_NS0 / np.mean(node_population0)
    nodeS_density0 = node_NS0 / node_population0

    idx_node = 0
    delta_NS_time = []
    for t in range(1, T):

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

        nodeS_density = node_NS / node_population0

        if t == 1:
            # Populations (matrix - row : node #, column : time)
            node_population_time = np.vstack((node_population0, node_population))
            node_NS_time = np.vstack((node_NS0, node_NS))
            node_NI_time = np.vstack((node_NI0, node_NI))
            node_NR_time = np.vstack((node_NR0, node_NR))
            node_state_time = np.vstack((node_state0, node_state))

            # Densities  (matrix - row : node #, column : time)
            nodeS_density_time = np.vstack((nodeS_density0, nodeS_density))

            # New infected per node in time
            new_I_time = np.vstack((node_NI0, NI_new))

        else:
            # Populations (matrix - row : node #, column : time)
            node_population_time = np.vstack((node_population_time, node_population))
            node_NS_time = np.vstack((node_NS_time, node_NS))
            node_NI_time = np.vstack((node_NI_time, node_NI))
            node_NR_time = np.vstack((node_NR_time, node_NR))
            node_state_time = np.vstack((node_state_time, node_state))

            # Densities  (matrix - row : node #, column : time)
            nodeS_density_time = np.vstack((nodeS_density_time, nodeS_density))

            # New infected per node in time
            new_I_time = np.vstack((new_I_time, NI_new))

            # Control if simulation started
            delta_NS = np.abs(node_NS_time[t, idx_node] - node_NS_time[t - 1, idx_node])
            delta_NS_time.append(delta_NS)
        node_NI_prev = node_NI
    max_deltaNS = np.max(delta_NS_time)
    max_delta_densityNS = max_deltaNS / np.mean(node_population)

    if nodeS_density_time[int(T - 5), idx_node] < (avg_popPerNode / avg_popPerNode + 2.0 * max_delta_densityNS) and nodeS_density_time[int(T - 5), idx_node] > (avg_popPerNode / avg_popPerNode - 2.0 * max_delta_densityNS):
        print('Simulation did not start')
        nbr_sim_not_start += 1
        idx_sim_not_start.append(sim)

    # Save data
    np.save(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_new_I_time', new_I_time)
    np.save(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_population_time', node_population_time)
    np.save(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NS_time', node_NS_time)
    np.save(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NI_time', node_NI_time)
    np.save(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NR_time', node_NR_time)
    np.save(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_state_time', node_state_time)

percent_sim_not_started = nbr_sim_not_start / nbr_repetitions * 100
print('percent sim not started: ', percent_sim_not_started, ' %')

np.save(folder_simulation + f'mu-{mu}/beta-{beta}/nbr_sim_not_start', nbr_sim_not_start)
np.save(folder_simulation + f'mu-{mu}/beta-{beta}/idx_sim_not_start', idx_sim_not_start)
print('Nbr simulations not start: ', nbr_sim_not_start)
print('Idx simulations not start: ', idx_sim_not_start)

# Write simulation file
write_simulation_file(N_row, N_col, choice_bool, c1, node_population0, node_NS0, node_NI0, node_NR0, node_state0, T,
                      beta, mu, nbr_repetitions, nbr_sim_not_start, idx_sim_not_start)


bool_density = 0
idx_sims = [0,1,2,3,4,5,6,7,8,9]



idx_nodes = list(idxNfix)
idx_nodes.append(6)
print(idxNfix)
plot_SIR_timeseries(N_row, N_col, choice_bool, c1, beta, mu, idx_sims, idx_nodes, T_sim,
                    avg_popPerNode, avg_popPerNode_Nfix, avg_popPerNode_Others)



