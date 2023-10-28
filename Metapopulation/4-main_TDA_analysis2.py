"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 26 October 2023

--------------------------------------------------------------------
Compute the persistent entropy (H0 and H1) from persistence diagrams obtained using the ripser function.
Each persistence diagram consists of a pair (birth time, death time).

"""

import numpy as np
import os
import pickle
import ripser


from persim.persistent_entropy import *

N_row = [10, 30]
N_col = [10, 30]

choice_bool = 0
datadir = os.getcwd()
c1 = 0  # for now

#beta_vals = [0.3, 0.4, 0.9, 0.35, 0.75]
#mu_vals = [0.1, 0.2, 0.1, 0.3, 0.6]

beta_vals = [0.4, 0.3, 0.9, 0.35, 0.75]
mu_vals = [0.2, 0.1, 0.1, 0.3, 0.6]

nbr_simulations = 10

normalization = 2

for row, col in zip(N_row, N_col):
    for beta, mu in zip(beta_vals, mu_vals):
        # --------------------------------------------- Folders ---------------------------------------------
        folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
        folder_analysis = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis/'
        folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'

        T = np.load(folder_simulation + 'T.npy')
        T_sim = np.linspace(0, T, T + 1)

        for sim in range(nbr_simulations):
            print('row: ', row, 'col: ',  col)
            print('beta: ', beta, 'mu: ', mu)
            print('sim: ', sim)
            # ------------------------------------ Load data ------------------------------------
            if normalization == 0:
                print('No normalization')
                dict_vals = pickle.load(open(folder_dict + f'No-normalized/dict_data-{row}x{col}-sim{sim}.pickle', 'rb'))
            elif normalization == 1:
                print('Normalization')
                dict_vals = pickle.load(open(folder_dict + f'Normalized/dict_data_normalized-{row}x{col}-sim{sim}.pickle', 'rb'))
            elif normalization == 2:
                print('Standard scaler')
                dict_vals = pickle.load(open(folder_dict + f'Normalized-scaler/dict_data_normalized-{row}x{col}-sim{sim}.pickle', 'rb'))

            else:
                print('Wrong normalization value')
            entropy_H0 = []
            entropy_H1 = []
            for t in dict_vals.keys():
                # Compute persistence diagrams
                dgms = ripser.ripser(dict_vals[t], maxdim = 1)['dgms']
                entropy = persistent_entropy(dgms)
                entropy_H0.append(entropy[0])
                entropy_H1.append(entropy[1])
            if normalization == 0:
                np.save(folder_analysis + f'No-normalized/entropy_H0-sim{sim}', entropy_H0)
                np.save(folder_analysis + f'No-normalized/entropy_H1-sim{sim}', entropy_H1)
            elif normalization == 1:
                np.save(folder_analysis + f'Normalized/entropy_H0-sim{sim}', entropy_H0)
                np.save(folder_analysis + f'Normalized/entropy_H1-sim{sim}', entropy_H1)
            elif normalization == 2:
                np.save(folder_analysis + f'Normalized-scaler/entropy_H0-sim{sim}', entropy_H0)
                np.save(folder_analysis + f'Normalized-scaler/entropy_H1-sim{sim}', entropy_H1)
            else:
                print('Wrong normalization value')





