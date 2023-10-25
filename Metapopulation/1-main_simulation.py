"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 25 October 2023

--------------------------------------------------------------------

Perform simulations on a network already defined. Do repeated simulations.
Added node state time

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

c1 = 1  # for now

beta = 0.9
mu = 0.1

# Total simulation length
T = 150
T_sim = np.linspace(0, T, T+1)

# Number of infected individuals in one node
popI_node = 8
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


folder_topology = datadir+f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Topology/'
folder_simulation = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/'

nbr_sim_not_start = 0
idx_sim_not_start = []
# ------------------------------------------------ Simulations -------------------------------------------------
# Need to reload the initial condition at every repetition!!
for sim in range(nbr_repetitions):
    # load graph object from file
    # -------------------------------------- Load topology data --------------------------------------
    G = pickle.load(open(folder_topology + 'G.pickle', 'rb'))
    dict_nodes = pickle.load(open(folder_topology + 'dict_nodes.pickle', 'rb'))
    N = len(G.nodes)

    TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
    weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
    weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0]

    avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')
    if choice_bool == 1:
        Nfix = np.load(folder_topology + 'Nfix.npy')
        percentage_FixNodes = np.load(folder_topology + 'percentage_FixNodes.npy')
    else:
        Nfix = 0
        percentage_FixNodes = 0

    # --------------------------------------------- Set initial configuration ------------------------------------------------

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

    # Initial densities
    # / np.mean(node_population) : I divide by a constant number so I keep fluctuations
    nodeS_density0 = node_NS0 / np.mean(node_population0)


    #plot_static_network(G, node_population0, dict_nodes, weightNonZero)

    # Temporal evolution and control if simulation started or not
    idx_node = 0
    delta_NS_time = []
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

        # Density per node : ok np.mean because it is for fixed time and averages over the population in all nodes.
        # This is just used as a control to ensure whether the simulation started or not.
        nodeS_density = node_NS / np.mean(node_population)

        if t == 0:
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
            delta_NS = np.abs(node_NS_time[t, idx_node] - node_NS_time[t-1, idx_node])
            delta_NS_time.append(delta_NS)

        node_NI_prev = node_NI
    max_deltaNS = np.max(delta_NS_time)
    max_delta_densityNS = max_deltaNS / np.mean(node_population)


    if nodeS_density_time[int(T - 5), idx_node] < (avg_popPerNode / avg_popPerNode + 1.5 * max_delta_densityNS) and nodeS_density_time[int(T - 5), idx_node] > (avg_popPerNode / avg_popPerNode - 1.5 * max_delta_densityNS):
        print('Simulation did not start')
        nbr_sim_not_start += 1
        idx_sim_not_start.append(sim)

        # When end the simulation, control if it started or not
        # Find the maximum fluctuation of S for each of the nodes in the simulation. If, at the end of the simulation,
        # the number of S is inside 2 times the maximum fluctuation, then the simulation did not start
        # For now I only control for node 0 because I see that nodes have all the same trend.



    # ---------------------------------- Save data after time evolution ----------------------------------------------------
    if sim == 0:
        write_simulation_file(N_row, N_col, choice_bool, c1, node_population0, node_NS0, node_NI0, node_NR0, node_state0, T,
                              beta, mu, nbr_repetitions, nbr_sim_not_start)
        np.save(folder_simulation + f'beta-{beta}mu-{mu}/T', T)
        np.save(folder_simulation + f'beta-{beta}mu-{mu}/nbr_repetitions', nbr_repetitions)
    np.save(folder_simulation + f'beta-{beta}mu-{mu}/sim_{sim}_new_I_time', new_I_time)
    np.save(folder_simulation + f'beta-{beta}mu-{mu}/sim_{sim}_node_population_time', node_population_time)
    np.save(folder_simulation + f'beta-{beta}mu-{mu}/sim_{sim}_node_NS_time', node_NS_time)
    np.save(folder_simulation + f'beta-{beta}mu-{mu}/sim_{sim}_node_NI_time', node_NI_time)
    np.save(folder_simulation + f'beta-{beta}mu-{mu}/sim_{sim}_node_NR_time', node_NR_time)
    np.save(folder_simulation + f'beta-{beta}mu-{mu}/sim_{sim}_node_state_time', node_state_time)


np.save(folder_simulation + f'beta-{beta}mu-{mu}/nbr_sim_not_start', nbr_sim_not_start)
np.save(folder_simulation + f'beta-{beta}mu-{mu}/idx_sim_not_start', idx_sim_not_start)
print('Nbr simulations not start: ', nbr_sim_not_start)
print('Idx simulations not start: ', idx_sim_not_start)