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

degree_analysis = 1
distance_analysis = 0
clustering_analysis = 0
weight_analysis = 0
PF_convergence = 0

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

                # plot_degree_distribution(row, col, choice_bool, c1, k_vals, Pk_norm, avg_in_degree, Poisson_funct,
                #                         param)

                # Matrix of connections: total number of edges between vertices of degree k and vertices of degree k′ N_kk'
                print('k_vals:', k_vals)
                ContactMatrix = np.zeros(shape = (max(k_vals) + 1, max(k_vals) + 1))

                for i in range(N):
                    for j in range(N): # self-loops??
                        if TransitionMatrix[i, j] != 0: # take tridiagonal superior (input) and consider that an edge exists
                            k = G.in_degree(i)
                            k_prime = G.in_degree(j)

                            ContactMatrix[k, k_prime] = ContactMatrix[k, k_prime] + 1
                # Delete if both row and col are zeros
                max_dim = max(k_vals) + 1
                idx_delete = []
                for i in range(max_dim):
                    sum_row = np.sum(ContactMatrix[i, :])
                    print('sum_row :', sum_row)
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
                    P_cond[k, :] = P_joint[k, :] / (k_vals[k] * N_k[k])

                # ANND k_barn_nn
                k_bar_nn = []
                for k in range(len(k_vals)):
                    sum_k = 0
                    for k_prime in range(len(k_vals)):
                        sum_k = sum_k + k_vals[k_prime] * P_cond[k, k_prime]
                    k_bar_nn.append(sum_k)


                k_bar_nn_non_corr = second_moment / avg_in_degree
                plt.plot(k_vals, k_bar_nn, marker = 'o')
                #plt.axhline(y = k_bar_nn_non_corr, linestyle = '--', color = 'k')
                plt.xlabel('k')
                plt.ylabel(r'\bar{k}_{nn}(k)')

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
                # [PF convergence]
                k_list = np.load(folder_topology + 'k_list.npy')
                diff_list = np.load(folder_topology + 'diff_list.npy')

                plt.plot(k_list, diff_list, '-o')
                # add labels and plot multiple dimensions in one to see how the decay of the error to zero changes as
                # a function of the network dimension.
                plt.show()

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

            if write_file == 1:
                write_network_file(row, col, choice_bool, c1, in_degrees, avg_in_degree, L, L_max,
                                   np.round(L / L_max * 100, 2),
                                   k_vals, Pk_norm, Pk_noNorm, diameter, avg_distance, param)
