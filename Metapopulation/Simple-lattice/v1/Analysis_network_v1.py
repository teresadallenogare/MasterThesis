"""

--------------------------------------------------------------------

Author  :   Teresa Dalle Nogare
Version :   20 November 2023

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
N_row = [30, 50]
N_col = [30, 50]

choice_bool_lst = [0, 1]
c1_lst = [0, 1]

degree_analysis = 1
path_analysis = 0
clustering_analysis = 0
PF_convergence = 0

# --------------------------------------------------------------------------------------------------------------

def Poisson_funct(k, lamb):
    # poisson probability mass function
    return poisson.pmf(k, lamb)

for row, col in zip(N_row, N_col):
    N = row * col
    for choice_bool in choice_bool_lst:
        for c1 in c1_lst:
            folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'
            folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
            G = pickle.load(open(folder_topology + 'G.pickle', 'rb'))
            if degree_analysis == 1:
                # [Degree properties]
                # Input degrees
                in_degrees = np.array([G.in_degree(n) for n in G.nodes()])
                # Total number of links
                L = in_degrees.sum()
                L_max = N * (N-1)/2
                # Average input degree
                avg_in_degree = L/N
                # In-degree distribution
                # No normalized
                Pk_noNorm = np.unique(in_degrees, return_counts=True)
                k_vals = Pk_noNorm[0]
                # Normalization : the Pk divided by the total number of nodes st sum(pk) = 1
                Pk_norm = Pk_noNorm[1] / N
                # Fit with Poisson distribution
                guess = (10)
                param, cov_matrix = curve_fit(Poisson_funct, k_vals, Pk_norm, p0 = guess)
                print('param:', param)
                SE = np.sqrt(np.diag(cov_matrix))
                SE_A = SE[0]

                plot_degree_distribution(row, col, choice_bool, c1, k_vals, Pk_norm, avg_in_degree, Poisson_funct, param)
                print(f'ch_bool: {choice_bool}, c1: {c1}, {row}x{col}, avg_k:', avg_in_degree, 'L_in: ', L, 'L_max: ',
                      L_max, 'Perc. link: ', np.round(L / L_max * 100, 2), '%')



            if path_analysis == 1:
                # [Paths and distances] (referred to the number of edges composing a path not to the Euclidan distance)
                max_dist = 10
                d_vals = np.linspace(0, max_dist, max_dist + 1)
                diameter, avg_distance, pd_norm = path_analysis(G, max_dist)
                print('sum pd: ', pd_norm.sum())
                plot_distance_distribution(row, col, choice_bool, c1, d_vals, pd_norm, avg_distance)

                print('diameter:', diameter, 'avg_distance', np.round(avg_distance, 3))

            if clustering_analysis == 1:
                # [Clustering coefficient]
                print('TO DO')

            if PF_convergence == 1:
                # [PF convergence]
                k_list = np.load(folder_topology + 'k_list.npy')
                diff_list = np.load(folder_topology + 'diff_list.npy')

                plt.plot(k_list, diff_list, '-o')
                # add labels and plot multiple dimensions in one to see how the decay of the error to zero changes as
                # a function of the network dimension.
                plt.show()

            if degree_analysis == 1 and path_analysis == 1:
                write_network_file(row, col, choice_bool, c1, in_degrees, avg_in_degree, L, L_max, np.round(L/L_max * 100, 2),
                                    k_vals, Pk_norm, Pk_noNorm, diameter, avg_distance)









