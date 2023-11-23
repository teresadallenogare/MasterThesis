"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 22 November 2023

--------------------------------------------------------------------
File to create dictionaries and data in the correct format to then do analysis
normalization = 0 -> (X, Y, #S, #I, #R)
normalization = 1 -> (X/Nrow, Y/Ncol, #S/nodePop0, #I/nodePop0, #R/nodePop0). Normalization such that the sum of
densities of the network gives me the total number of nodes.

"""
import numpy as np
import os
import pickle

datadir = os.getcwd()
# ------------------------------------------------ Parameters  -------------------------------------------------

N_row = [50]
N_col = [50]

choice_bool_lst = [0, 1]
c1_lst = [0, 1]

# Normalization by hand
normalization = 0

# Infection and recovery rate
beta_vals_3_5_10 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2, 0.23, 0.24, 0.3, 0.4, 0.6, 0.8, 0.345, 0.36, 0.45, 0.6, 0.9, 1.2]
mu_vals_3_5_10 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

beta_vals_30_50 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for row, col in zip(N_row, N_col):
    N = row * col
    if row == 3 or row == 5 or row == 10:
        beta_vals = beta_vals_3_5_10
        mu_vals = mu_vals_3_5_10
    else:
        beta_vals = beta_vals_30_50
        mu_vals = mu_vals_30_50

    for choice_bool in choice_bool_lst:
        for c1 in c1_lst:
            folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
            folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'
            folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'

            avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')
            pos_nodes = np.load(folder_topology + 'pos_nodes.npy')
            # Extract coordinates of nodes
            x_nodes = [pos_nodes[i][0] for i in range(N)]
            y_nodes = [pos_nodes[i][1] for i in range(N)]

            sim = 0
            for beta, mu in zip(beta_vals, mu_vals):
                folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
                T = np.load(folder_simulation + 'T.npy')
                T_sim = np.linspace(0, T - 1, T)
                node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
                node_population_time0 = node_population_time[0,:]
                node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
                node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
                node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

                # No normalized dictionary
                if normalization == 0:
                    # -------------------------------------- CREATE DICTIONARY WITH TIME IN KEY --------------------------------------------
                    # dict :
                    # - <key> = timestep
                    # - <value> = matrix with a node for row. Each row has (x_node, y_node, #S, #I, #R)
                    dict_5d_nodes = {}
                    for t in range(T):
                        for i in range(N):
                            if i == 0:
                                array_0 = np.array([x_nodes[i], y_nodes[i], node_NS_time[t, i], node_NI_time[t, i],
                                                    node_NR_time[t, i]])
                            elif i == 1:
                                array_1 = np.array([x_nodes[i], y_nodes[i], node_NS_time[t, i], node_NI_time[t, i],
                                                    node_NR_time[t, i]])
                                mtrx_node_t = np.vstack((array_0, array_1))
                            else:
                                array_t = np.array([x_nodes[i], y_nodes[i], node_NS_time[t, i], node_NI_time[t, i],
                                                    node_NR_time[t, i]])
                                mtrx_node_t = np.vstack((mtrx_node_t, array_t))
                        dict_5d_nodes[t] = mtrx_node_t

                        # Save dictionary. It goes in the folder of the corresponding beta and mu
                        pickle.dump(dict_5d_nodes,
                                    open(folder_dict_noNorm + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'wb'))

                # Normalized by hand
                elif normalization == 1:
                    # 1. Normalization position nodes : divide by the number of rows
                    x_nodes_normalized = [pos_nodes[i][0] / row for i in range(N)]
                    y_nodes_normalized = [pos_nodes[i][1] / col for i in range(N)]

                    # 2. Normalization density
                    # density calculated dividing by the average over time of the population in each fixed node
                    density_population_time = node_population_time / avg_popPerNode
                    density_NS_time = node_NS_time / avg_popPerNode
                    density_NI_time = node_NI_time / avg_popPerNode
                    density_NR_time = node_NR_time / avg_popPerNode

                    dict_5d_densities = {}
                    for t in range(T):
                        for i in range(N):
                            if i == 0:
                                array_0 = np.array(
                                    [x_nodes_normalized[i], y_nodes_normalized[i], density_NS_time[t, i],
                                     density_NI_time[t, i], density_NR_time[t, i]])
                            elif i == 1:
                                array_1 = np.array(
                                    [x_nodes_normalized[i], y_nodes_normalized[i], density_NS_time[t, i],
                                     density_NI_time[t, i], density_NR_time[t, i]])
                                mtrx_node_t = np.vstack((array_0, array_1))
                            else:
                                array_t = np.array(
                                    [x_nodes_normalized[i], y_nodes_normalized[i], density_NS_time[t, i],
                                     density_NI_time[t, i], density_NR_time[t, i]])
                                mtrx_node_t = np.vstack((mtrx_node_t, array_t))
                        dict_5d_densities[t] = mtrx_node_t

                    # Save dictionary. It goes in the folder of the corresponding beta and mu
                    pickle.dump(dict_5d_densities,
                                open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'wb'))
