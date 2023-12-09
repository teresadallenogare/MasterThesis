"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 23 November 2023

--------------------------------------------------------------------

Control of persistent entropy results

"""

from functions_TDA_v1 import *
import os
import numpy as np
import matplotlib.pyplot as plt

N_row = [30]
N_col = [30]

choice_bool_lst = [0, 1]
c1_lst = [0, 1]

# Infection and recovery rate
beta_vals_3_5_10 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2, 0.23, 0.24, 0.3, 0.4, 0.6, 0.8, 0.345, 0.36, 0.45, 0.6, 0.9, 1.2]
mu_vals_3_5_10 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

beta_vals_30_50 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


sim = 0

# normalization = 0 -> no normalized data
#               = 1 -> standard scaler normalization
#               = 2 -> normalization by hand
normalization = 2
id = 'XYSIR'

datadir = os.getcwd()


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
            folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'
            if normalization == 0:
                folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/No-normalized/{id}/'
            elif normalization == 1:
                folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'
            elif normalization == 2:
                folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-scaler/{id}/'
            for beta, mu in zip(beta_vals, mu_vals):
                T = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/T.npy')
                print('row: ', row, 'col: ', col, 'choice_bool: ', choice_bool, 'c1: ', c1)
                print('beta: ', beta, 'mu: ', mu)

                for normalize_entropy in [False]:#, True]:
                    plt.figure(figsize=(8, 8))
                    entropy_H0 = np.load(folder_entropy + f'entropy_H0-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}.npy')
                    entropy_H1 = np.load(folder_entropy + f'entropy_H1-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}.npy')

                    x = range(0, len(entropy_H0))

                    plt.plot(x, entropy_H0, color = 'r', label = 'entropy at H0')
                    plt.plot(x, entropy_H1, color = 'b', label = 'entropy at H1')

                    plt.xlabel('Time')
                    plt.ylabel('Persistent Entropy')
                    plt.title(f'PE choice_bool:{choice_bool}, c1:{c1}, beta:{beta}, mu:{mu}, normE:{normalize_entropy}, normDat:{normalization}')
                    plt.legend()
                    plt.show()
