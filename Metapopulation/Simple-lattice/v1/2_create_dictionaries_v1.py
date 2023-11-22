"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 22 November 2023

--------------------------------------------------------------------
File to create dictionaries and data in the correct format to then do analysis

"""
import numpy as np
import os
import pickle

datadir = os.getcwd()
# ------------------------------------------------ Parameters  -------------------------------------------------

N_row = [3, 5, 10]
N_col = [3, 5, 10]

choice_bool_lst = [0, 1]
c1_lst = [0, 1]

# Infection and recovery rate
beta_vals_3_5_10 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2, 0.23, 0.24, 0.3, 0.4, 0.6, 0.8, 0.345, 0.36, 0.45, 0.6, 0.9, 1.2]
mu_vals_3_5_10 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

beta_vals_30_50 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Total simulation length
T_3_5_10 = [1000, 800, 600, 300, 250, 200, 120, 120, 800, 600, 250, 200, 150, 120, 500, 500, 250, 150, 120, 120]
T_30_50 = [1000, 800, 600, 300, 250, 200, 120, 120]

normalization = 0

for row, col in zip(N_row, N_col):
    N = row * col
    if row == 3 or row == 5 or row == 10:
        beta_vals = beta_vals_3_5_10
        mu_vals = mu_vals_3_5_10
        T = T_3_5_10
    else:
        beta_vals = beta_vals_30_50
        mu_vals = mu_vals_30_50
        T = T_30_50

    for choice_bool in choice_bool_lst:

        for c1 in c1_lst:
            folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
            folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'

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
                                    open(folder_dict_noNorm + f'/dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'wb'))
