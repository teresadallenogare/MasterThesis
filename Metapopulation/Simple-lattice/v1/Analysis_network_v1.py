"""

--------------------------------------------------------------------

Author  :   Teresa Dalle Nogare
Version :   22 November 2023

--------------------------------------------------------------------

Analysis of network topology computing the key characteristic properties of graphs.

"""
from functions_network_v1 import path_analysis
from functions_output_v1 import write_network_file
from functions_visualization_v1 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.optimize import curve_fit
from scipy.stats import poisson

datadir = os.getcwd()
plt.figure(figsize=(8, 8))

# ------------------------------------------------ Parameters  -------------------------------------------------
N_row = [3, 5, 10, 30, 50]
N_col = [3, 5, 10, 30, 50]

choice_bool_lst = [0]
c1_lst = [0]

beta_vals_3_5_10 = [0.115]#, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2, 0.23, 0.24, 0.3, 0.4, 0.6, 0.8, 0.345, 0.36, 0.45, 0.6, 0.9, 1.2]
mu_vals_3_5_10 = [0.1]#, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

beta_vals_30_50 = [0.115]#, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1]#, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

population_analysis = 0
degree_analysis = 1
distance_analysis = 0
clustering_analysis = 0
weight_analysis = 0
PF_convergence = 0
show_transition_matrix = 0

write_file = 0


# --------------------------------------------------------------------------------------------------------------

def Poisson_funct(k, lamb):
    # poisson probability mass function
    return poisson.pmf(k, lamb)

def nth_moment_v2(g,n):
    degree_np = np.array(list(dict(g.in_degree).values()))
    return (sum(degree_np**n)/len(g))

for row, col in zip(N_row, N_col):
    N = row * col
    for choice_bool in choice_bool_lst:
        for c1 in c1_lst:
            folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'
            folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
            G = pickle.load(open(folder_topology + 'G.pickle', 'rb'))
            TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
######################################################################################################################

            if population_analysis == 1:
                # Node population
                avg_population = np.load(folder_topology + 'avg_popPerNode.npy')
                total_population = N * avg_population
                node_population_0 = nx.get_node_attributes(G, name='Npop')
                node_population_0 = np.array(list(node_population_0.values()))
                if choice_bool == 0:

                    # Mean and average from the multinomial distribution
                    prob_sample1 = 1/N
                    mean_population_multi1 = total_population * prob_sample1
                    stdDev_population_multi1 = np.sqrt(total_population * prob_sample1 * (1-prob_sample1))

                    N_fix = 0
                    mean_population_multi2 = 0
                    stdDev_population_multi2 = 0
                    idx_Nfix = 0
                elif choice_bool == 1:
                    N_fix = np.load(folder_topology + 'Nfix.npy')
                    idx_Nfix = np.load(folder_topology + 'idxNfix.npy')
                    percentage_Nfix = np.load(folder_topology + 'percentage_FixNodes.npy')

                    N_fix_population = total_population * percentage_Nfix/100
                    N_other_population = total_population - N_fix_population

                    prob_sample1 = 1 / N_fix
                    prob_sample2 = 1 / (N - N_fix)


                    mean_population_multi1 =  N_fix_population * prob_sample1
                    stdDev_population_multi1 = np.sqrt(N_fix_population * prob_sample1 * (1. - prob_sample1))

                    mean_population_multi2 = N_other_population * prob_sample2
                    stdDev_population_multi2 = np.sqrt(N_other_population * prob_sample2 * (1. - prob_sample2))

                plot_node_population_0(N, N_fix, idx_Nfix,node_population_0, mean_population_multi1, stdDev_population_multi1, mean_population_multi2, stdDev_population_multi2, choice_bool)

                print('hello')
            ######################################################################################################################

            if degree_analysis == 1:
                # [Degree properties]
                # Input degrees
                in_degrees = np.array([G.in_degree(n) for n in G.nodes()])
                # Total number of links
                L = in_degrees.sum()
                L_max = N * (N - 1) / 2
                # Average input degree
                avg_in_degree = L / N
                second_moment = nth_moment_v2(G, 2)
                print('avg1:', avg_in_degree)
                print('second moment: ', second_moment)
                # In-degree distribution
                # No normalized
                Pk_noNorm = np.unique(in_degrees, return_counts=True)

                k_vals = Pk_noNorm[0]
                N_k = Pk_noNorm[1]

                print('avg2:', nth_moment_v2(G, 1))
                # Normalization : the Pk divided by the total number of nodes st sum(pk) = 1
                Pk_norm = N_k / N
                # Fit with Poisson distribution
                guess = avg_in_degree
                param, cov_matrix = curve_fit(Poisson_funct, k_vals, Pk_norm, p0 = guess)
                print('param:', param)
                SE = np.sqrt(np.diag(cov_matrix))
                SE_A = SE[0]

                plot_degree_distribution(row, col, choice_bool, c1, k_vals, Pk_norm, avg_in_degree, Poisson_funct,
                                         param)

                # Matrix of connections: total number of edges between vertices of degree k and vertices of degree k′ N_kk'
                print('k_vals:', k_vals)
                ContactMatrix = np.zeros(shape = (max(k_vals) + 1, max(k_vals) + 1))

                for i in range(N):
                    for j in range(N): # self-loops??
                        if TransitionMatrix[i, j] != 0:
                            k = G.in_degree(i)
                            k_prime = G.in_degree(j)
                            # Cont the number of edges from k to k' and viceversa
                            ContactMatrix[k, k_prime] = ContactMatrix[k, k_prime] + 1
                # Delete if both row and col are zeros
                max_dim = max(k_vals) + 1
                idx_delete = []
                for i in range(max_dim):
                    sum_row = np.sum(ContactMatrix[i, :])
                    sum_col = np.sum(ContactMatrix[:, i])
                    if sum_row == 0 and sum_col == 0:
                        idx_delete.append(i)

                ContactMatrix = np.delete(ContactMatrix, idx_delete, 0) # delete i-th row
                ContactMatrix = np.delete(ContactMatrix, idx_delete, 1) # delete i-th col

                # Joint probability P(k, k')
                P_joint = ContactMatrix/(avg_in_degree * N)

                # Conditional probability P(k|k')
                P_cond = np.zeros(shape = (len(k_vals), len(k_vals)))
                for k in range(len(k_vals)):
                        P_cond[k, :] = ContactMatrix[k, :] / (k_vals[k] * N_k[k])

                # ANND k_barn_nn
                k_bar_nn = []
                for k in range(len(k_vals)):
                    sum_k = 0
                    for k_prime in range(len(k_vals)):
                        sum_k = sum_k + k_vals[k_prime] * P_cond[k, k_prime]
                    k_bar_nn.append(sum_k)


                k_bar_nn_non_corr = second_moment / avg_in_degree
                plt.plot(k_vals, k_bar_nn, marker = 'o')
                plt.axhline(y = k_bar_nn_non_corr, linestyle = '--', color = 'k')
                plt.xlabel('k')
                plt.ylabel(r'$\bar{k}_{nn}(k)$')

                plt.show()
                #print(f'ch_bool: {choice_bool}, c1: {c1}, {row}x{col}, avg_k:', avg_in_degree, 'L_in: ', L, 'L_max: ',
                #      L_max, 'Perc. link: ', np.round(L / L_max * 100, 2), '%')
                print('hello')
######################################################################################################################

            if distance_analysis == 1:
                # [Paths and distances] (referred to the number of edges composing a path not to the Euclidan distance)
                max_dist = 10
                d_vals = np.linspace(0, max_dist, max_dist + 1)
                diameter, avg_distance, pd_norm = path_analysis(G, max_dist)
                print('sum pd: ', pd_norm.sum())
                plot_distance_distribution(row, col, choice_bool, c1, d_vals, pd_norm, avg_distance)

                print('diameter:', diameter, 'avg_distance', np.round(avg_distance, 3))

######################################################################################################################

            if clustering_analysis == 1:
                # [Clustering coefficient]
                print('TO DO')

######################################################################################################################

            if PF_convergence == 1:
                idx_node = np.linspace(0, N-1, N)

                folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
                avg_population = np.load(folder_topology + 'avg_popPerNode.npy')
                rho0 = np.load(folder_topology + '/rho0.npy')
                rho0 = rho0 * N
                # [PF convergence]
                # Plot the error as a function of the dimension
                k_list = np.load(folder_topology + 'k_list.npy')
                diff_list = np.load(folder_topology + 'diff_list.npy')

                plt.plot(k_list, diff_list, '-o')
                # add labels and plot multiple dimensions in one to see how the decay of the error to zero changes as
                # a function of the network dimension.
                plt.xlabel('Power law')
                plt.ylabel('Error')
                plt.show()

                # Plot the difference between rho0 and the density of people in the final time
                if row == 3 or row == 5 or row == 10:
                    beta_vals = beta_vals_3_5_10
                    mu_vals = mu_vals_3_5_10
                else:
                    beta_vals = beta_vals_30_50
                    mu_vals = mu_vals_30_50

                for beta, mu in zip(beta_vals, mu_vals):
                    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
                    node_population_time = np.load(folder_simulation + 'sim_0_node_population_time.npy')
                    node_population_final = node_population_time[-1, :]
                    node_density_final = node_population_final / avg_population

                    diff_density = node_density_final - rho0

                    plt.scatter(idx_node, diff_density, color = 'k')
                    plt.axhline(y=0, linestyle='--', color='k')
                    plt.xlabel('Index node')
                    plt.ylabel(r'$\rho_{\infty} - \rho_0$')
                    plt.show()
                    print('hello')
            ######################################################################################################################

            if weight_analysis == 1:
                TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
                TM_ravel = TransitionMatrix.ravel()
                TM_round = np.round(TM_ravel, 2) # keep 2 decimals for plot
                TM_removed = TM_round[TM_round != 0.00]

                plt.hist(TM_removed,  color='#0504aa', alpha=0.7, align = 'mid', bins = 100 )
                plt.yscale('log')
                plt.xlabel('weight')
                plt.ylabel('Frequency')
                plt.title(f'dim = {row}x{col}, choice_bool = {choice_bool}, c1 = {c1}')
                plt.show()
######################################################################################################################
            if show_transition_matrix == 1:
                TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
                print('hello')
######################################################################################################################

            if write_file == 1:
                write_network_file(row, col, choice_bool, c1, in_degrees, avg_in_degree, L, L_max,
                                   np.round(L / L_max * 100, 2),
                                   k_vals, Pk_norm, Pk_noNorm, diameter, avg_distance, param)
