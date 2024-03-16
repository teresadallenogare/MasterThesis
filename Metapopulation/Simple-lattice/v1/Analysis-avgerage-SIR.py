"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 18 February 2024

--------------------------------------------------------------------

Compute the average SIR
"""

from functions_TDA_v1 import *
from functions_heatmap_v1 import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

row = 30
col = 30
N = row * col

choice_bool = 0
c1 = 0

beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

nbr_repetitions = 10

datadir = os.getcwd()

step0Analysis = 1

step1AvgData = 0
step2Dict = 0
step3NPE = 0
step4check = 0
step5heatmapAvg = 0

if step0Analysis == 1:
    idx_sims_started = []
    start = 0
    avg_epi = []
    std_epi = []
    avg_size = []
    std_size = []
    avg_NI_max = []
    std_NI_max = []
    avg_t = []
    std_t = []
    for beta, mu in zip(beta_vals, mu_vals):
        f, ax = plt.subplots(figsize=(14, 7))
        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
        idx_sims_not_start = np.load(folder_simulation + f'idx_sim_not_start.npy')
        epidemic_duration = []
        jj = 0
        final_size_sim = []
        R0_vals = []
        # Number of recovered at long time limit inside every node
        final_size_node = np.zeros(shape=(len(idx_sims_started), N))
        # Population of the node at long time limit
        final_population_node = np.zeros(shape=(len(idx_sims_started), N))
        # is 1 if node was infected, is 0 otherwise
        mtrx_nodes_infected = np.zeros(shape=(len(idx_sims_started), N))

        nbr_nodes_infected = []
        NI_max_sim = []
        t_sim = []
        for sim in range(nbr_repetitions):
            if sim not in idx_sims_not_start:
                idx_sims_started.append(int(sim))
                node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')/ (N*10**4)
                NI_time = node_NI_time.sum(axis=1)
                # Set end after the pick has occurred (at network level)
                t_peak = np.argmax(NI_time)

                # End of the epidemics is set to be 3 timesteps after the peak value. (How to justify? When it starts to decrease)
                end = t_peak + 3
                duration = end - start
                epidemic_duration.append(duration)

                total_population = 10**4 * N
                node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')
                node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
                # Number of recovered per node (one node per column and one R0 per row)

                final_size = node_NR_time[-1].sum()
                final_size_sim.append(final_size/total_population)
                NI_max_sim.append(NI_time[t_peak])
                t_sim.append(t_peak)
                print('size:', final_size)

                jj = jj + 1

        avg_epi.append(np.array(epidemic_duration).mean())
        std_epi.append((np.array(epidemic_duration).std(ddof=1)))

        avg_size.append(np.array(final_size_sim).mean())
        std_size.append(np.array(final_size_sim).std(ddof = 1))

        avg_NI_max.append(np.array(NI_max_sim).mean())
        std_NI_max.append(np.array(NI_max_sim).std(ddof = 1))

        avg_t.append(np.array(t_sim).mean())
        std_t.append(np.array(t_sim).std(ddof = 1))
    #print('avg epi:', avg_epi)
    #print('std dev:', std_epi)
    print('final size:', avg_size)
    print('std dev:', std_size)
    print('NI_max:', avg_NI_max)
    print('std dev:', std_NI_max)

    print('t: ', avg_t)
    print('std dev: ', std_t)

if step1AvgData == 1:
    idx_sims_started = []
    for beta, mu in zip(beta_vals, mu_vals):
        f, ax = plt.subplots(figsize=(14, 7))
        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
        idx_sims_not_start = np.load(folder_simulation + f'idx_sim_not_start.npy')
        node_population_lst = []
        node_NS_lst = []
        node_NI_lst = []
        node_NR_lst = []
        node_NInew_lst = []
        for sim in range(nbr_repetitions):
            if sim not in idx_sims_not_start:
                idx_sims_started.append(int(sim))
                node_population_time_sim = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
                node_NS_time_sim = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
                node_NI_time_sim = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
                node_NR_time_sim = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')
                node_NInew_time_sim = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
                if beta == 0.115 or beta == 0.12:
                    node_population_time_sim = node_population_time_sim[:1000]
                    node_NS_time_sim = node_NS_time_sim[:1000]
                    node_NI_time_sim = node_NI_time_sim[:1000]
                    node_NR_time_sim = node_NR_time_sim[:1000]
                    node_NInew_time_sim = node_NInew_time_sim[:1000]
                node_population_lst.append(node_population_time_sim)
                node_NS_lst.append(node_NS_time_sim)
                node_NI_lst.append(node_NI_time_sim)
                node_NR_lst.append(node_NR_time_sim)
                node_NInew_lst.append(node_NInew_time_sim)
        T = np.load(folder_simulation + f'T.npy')
        if row == 30:
            if beta == 0.115 or beta == 0.12:
                T = 1000
        T_sim = np.linspace(0, T - 1, T)
        # Create 3d matrices (simulations along axis 2)
        node_population = np.stack(node_population_lst, axis = 2)
        node_NS = np.stack(node_NS_lst, axis = 2)
        node_NI = np.stack(node_NI_lst, axis = 2)
        node_NR = np.stack(node_NR_lst, axis = 2)
        node_NInew = np.stack(node_NInew_lst, axis = 2)

        avg_node_population = np.mean(node_population, axis = 2)
        avg_node_NS = np.mean(node_NS, axis = 2)
        avg_node_NI = np.mean(node_NI, axis = 2)
        avg_node_NR = np.mean(node_NR, axis = 2)
        avg_node_NInew = np.mean(node_NInew, axis = 2)

        avg_population = np.sum(avg_node_population, axis = 1)
        avg_NS = np.sum(avg_node_NS, axis = 1)
        avg_NI = np.sum(avg_node_NI, axis = 1)
        avg_NR = np.sum(avg_node_NR, axis = 1)
        avg_NInew = np.sum(avg_node_NInew, axis = 1)

        np.save(folder_simulation + f'avg_population_node-mu{mu}-beta{beta}', avg_node_population)
        np.save(folder_simulation + f'avg_node_NS-mu{mu}-beta{beta}', avg_node_NS)
        np.save(folder_simulation + f'avg_node_NI-mu{mu}-beta{beta}', avg_node_NI)
        np.save(folder_simulation + f'avg_node_NR-mu{mu}-beta{beta}', avg_node_NR)
        np.save(folder_simulation + f'avg_node_NInew-mu{mu}-beta{beta}', avg_node_NInew)
        plt.plot(T_sim, avg_NS, color = 'b')
        plt.plot(T_sim, avg_NI, color = 'r')
        plt.plot(T_sim, avg_NR, color = 'g')

        #plt.show()

if step2Dict == 1:
    folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'

    for beta, mu in zip(beta_vals, mu_vals):
        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
        if beta == 0.115 or beta == 0.12:
            T = 1000
        else:
            T = np.load(folder_simulation + 'T.npy')

        avg_node_population = np.load(folder_simulation + f'avg_population_node-mu{mu}-beta{beta}.npy')
        avg_node_NS = np.load(folder_simulation + f'avg_node_NS-mu{mu}-beta{beta}.npy')
        avg_node_NI = np.load(folder_simulation + f'avg_node_NI-mu{mu}-beta{beta}.npy')
        avg_node_NR = np.load(folder_simulation + f'avg_node_NR-mu{mu}-beta{beta}.npy')
        avg_node_NInew = np.load(folder_simulation + f'avg_node_NInew-mu{mu}-beta{beta}.npy')

        avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')
        pos_nodes = np.load(folder_topology + 'pos_nodes.npy')
        # Extract coordinates of nodes
        x_nodes = [pos_nodes[i][0] for i in range(N)]
        y_nodes = [pos_nodes[i][1] for i in range(N)]

        # 1. Normalization position nodes : divide by the number of rows
        x_nodes_normalized = [pos_nodes[i][0] / row for i in range(N)]
        y_nodes_normalized = [pos_nodes[i][1] / col for i in range(N)]

        # 2. Normalization density
        # density calculated dividing by the average over time of the population in each fixed node
        avg_density_population = avg_node_population / avg_popPerNode
        avg_density_NS = avg_node_NS / avg_popPerNode
        avg_density_NI = avg_node_NI / avg_popPerNode
        avg_density_NR = avg_node_NR / avg_popPerNode
        avg_density_NInew = avg_node_NInew / avg_popPerNode
        # dict :
        # - <key> = timestep
        # - <value> = matrix with a node for row. Each row has (x_node, y_node, #S, #I, #R)
        dict_5d_nodes = {}
        for t in range(T):
            for i in range(N):
                if i == 0:
                    array_0 = np.array([x_nodes_normalized[i], y_nodes_normalized[i], avg_density_NS[t, i], avg_density_NI[t, i],
                                        avg_density_NR[t, i], avg_density_NInew[t, i]])
                elif i == 1:
                    array_1 = np.array([x_nodes_normalized[i], y_nodes_normalized[i], avg_density_NS[t, i], avg_density_NI[t, i],
                                        avg_density_NR[t, i], avg_density_NInew[t, i]])
                    mtrx_node_t = np.vstack((array_0, array_1))
                else:
                    array_t = np.array([x_nodes_normalized[i], y_nodes_normalized[i], avg_density_NS[t, i], avg_density_NI[t, i],
                                        avg_density_NR[t, i], avg_density_NInew[t, i]])
                    mtrx_node_t = np.vstack((mtrx_node_t, array_t))
            dict_5d_nodes[t] = mtrx_node_t

        # Save dictionary. It goes in the folder of the corresponding beta and mu
        pickle.dump(dict_5d_nodes,
                    open(folder_dict_normHand + f'dict_avg_data_beta{beta}-mu{mu}.pickle', 'wb'))
        print('hello')


if step3NPE == 1:
    columns = ['X', 'Y', 'S', 'I', 'R']
    id = 'XYSIR'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'
    for beta, mu in zip(beta_vals, mu_vals):
        dict_data = pickle.load(
            open(folder_dict_normHand + f'dict_avg_data_beta{beta}-mu{mu}.pickle', 'rb'))
        df_dict_data = data_2_pandas(dict_data)
        #### PE calculation
        for normalize_entropy in [True]:
            print('normalize_entropy: ', normalize_entropy)
            ph, dgms, entropy_H0, entropy_H1 = entropy_calculation(df_dict_data, columns, normalize_entropy)
            np.save(folder_entropy + f'entropy_avg-H0-nrm{normalize_entropy}-beta{beta}-mu{mu}', entropy_H0)
            np.save(folder_entropy + f'entropy_avg-H1-nrm{normalize_entropy}-beta{beta}-mu{mu}', entropy_H1)

if step4check == 1:
    columns = ['X', 'Y', 'S', 'I', 'R']
    id = 'XYSIR'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'
    for beta, mu in zip(beta_vals, mu_vals):
        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
        T = np.load(folder_simulation + f'T.npy')
        if row == 30:
            if beta == 0.115 or beta == 0.12:
                T = 1000
        T_sim = np.linspace(0, T - 2, T-1)
        avgH0 = np.load(folder_entropy + f'entropy_avg-H0-nrm{True}-beta{beta}-mu{mu}.npy')
        avgH1 = np.load(folder_entropy + f'entropy_avg-H1-nrm{True}-beta{beta}-mu{mu}.npy')

        plt.plot(T_sim, avgH0, color = 'r')
        plt.plot(T_sim, avgH1, color = 'b')

        plt.show()


if step5heatmapAvg == 1:
    beta = 1.2
    mu = 0.1

    bool_static = 0
    bool_Inew = 0

    time = 21

    folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

    if beta == 0.115 or beta == 0.12:
        T = 1000
    else:
        T = np.load(folder_simulation + 'T.npy')

    heatmap_time_avg_infecteds(row, col, choice_bool, c1, beta, mu, bool_static, bool_Inew, time)

