"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 22 October 2023

--------------------------------------------------------------------
File to create dictionaries and data in the correct format to then do analysis

"""

import numpy as np
import os
import pickle


# ------------------------------------------------ Parameters  -------------------------------------------------

N_row = [10]#, 30]
N_col = [10]#, 30]

choice_bool = 0
datadir = os.getcwd()
c1 = 0  # for now
beta_vals  = [0.4, 0.3, 0.9, 0.35, 0.75]
mu_vals = [0.2, 0.1, 0.1, 0.3, 0.6]


normalization = 1

for row, col in zip(N_row, N_col):
    N = row * col
    for beta, mu in zip(beta_vals, mu_vals):
        folder_topology = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Topology/'
        folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
        folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'

        T = np.load(folder_simulation + 'T.npy')
        T_sim = np.linspace(0, T, T+1)
        nbr_repetitions = np.load(folder_simulation + 'nbr_repetitions.npy')

        pos_nodes = np.load(folder_topology + 'pos_nodes.npy')

        # Extract coordinates of nodes
        x_nodes = [pos_nodes[i][0] for i in range(N)]
        y_nodes = [pos_nodes[i][1] for i in range(N)]

        for sim in range(nbr_repetitions):
            node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
            node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
            node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
            node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

            if normalization == 0:
                # -------------------------------------- CREATE DICTIONARY WITH TIME IN KEY --------------------------------------------
                # dict :
                # - <key> = timestep
                # - <value> = matrix with a node for row. Each row has (x_node, y_node, #S, #I, #R)

                dict_5d_nodes = {}
                for t in range(T):
                    for i in range(N):
                        if i == 0:
                            array_0 = np.array([x_nodes[i], y_nodes[i], node_NS_time[t, i], node_NI_time[t, i], node_NR_time[t, i]])
                        elif i == 1:
                            array_1 = np.array([x_nodes[i], y_nodes[i], node_NS_time[t, i], node_NI_time[t, i], node_NR_time[t, i]])
                            mtrx_node_t = np.vstack((array_0, array_1))
                        else:
                            array_t = np.array([x_nodes[i], y_nodes[i], node_NS_time[t, i], node_NI_time[t, i], node_NR_time[t, i]])
                            mtrx_node_t = np.vstack((mtrx_node_t, array_t))
                    dict_5d_nodes[t] = mtrx_node_t

                # Save dictionary. It goes in the folder of the corresponding beta and mu
                pickle.dump(dict_5d_nodes, open(folder_dict + f'/dict_data-{row}x{col}-sim{sim}.pickle', 'wb'))


            elif normalization == 1:

                # 1. Normalization position nodes : divide by the number of rows
                x_nodes_normalized = [pos_nodes[i][0] / row for i in range(N)]
                y_nodes_normalized = [pos_nodes[i][1] / col for i in range(N)]

                # 2. Normalization density
                # density calculated dividing by the average over time of the population in each fixed node
                density_population_time = node_population_time / np.mean(node_population_time, axis = 0)
                density_NS_time = node_NS_time / np.mean(node_population_time, axis = 0)
                density_NI_time = node_NI_time / np.mean(node_population_time, axis=0)
                density_NR_time = node_NR_time / np.mean(node_population_time, axis=0)

                # -------------------------------------- CREATE DICTIONARY WITH TIME IN KEY --------------------------------------------
                # dict :
                # - <key> = timestep
                # - <value> = matrix with a node for row. Each row has (x_node, y_node, #S, #I, #R)

                dict_5d_densities = {}
                for t in range(T):
                    for i in range(N):
                        if i == 0:
                            array_0 = np.array(
                                [x_nodes_normalized[i], y_nodes_normalized[i], density_NS_time[t, i], density_NI_time[t, i], density_NR_time[t, i]])
                        elif i == 1:
                            array_1 = np.array(
                                [x_nodes_normalized[i], y_nodes_normalized[i], density_NS_time[t, i], density_NI_time[t, i], density_NR_time[t, i]])
                            mtrx_node_t = np.vstack((array_0, array_1))
                        else:
                            array_t = np.array(
                                [x_nodes_normalized[i], y_nodes_normalized[i], density_NS_time[t, i], density_NI_time[t, i], density_NR_time[t, i]])
                            mtrx_node_t = np.vstack((mtrx_node_t, array_t))
                    dict_5d_densities[t] = mtrx_node_t

                # Save dictionary. It goes in the folder of the corresponding beta and mu
                pickle.dump(dict_5d_densities, open(folder_dict + f'/Normalized/dict_data_normalized-{row}x{col}-sim{sim}.pickle', 'wb'))

            else:
                print('wrong value of normalization inserted')
