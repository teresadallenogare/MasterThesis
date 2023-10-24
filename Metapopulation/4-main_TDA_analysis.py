"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 22 October 2023

--------------------------------------------------------------------
First analysis concerning the application of TDA to data obtained from the simulations

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import ripser

from persim import plot_diagrams
from persim.persistent_entropy import *


# ------------------------------------------------ Parameters  -------------------------------------------------

N_row = [10, 30]
N_col = [10, 30]

choice_bool = 0
datadir = os.getcwd()
c1 = 0  # for now
beta_outbreak = [0.3, 0.4, 0.9]
beta_no_outbreak = [0.35, 0.75]
mu_outbreak = [0.1, 0.2, 0.1]
mu_no_outbreak = [0.3, 0.6]

nbr_simulations = 10

# --------------------------------------------- Load data ---------------------------------------------------

# ------- OUTBREAK CASE -------
# Fix lattice dimension
for row, col in zip(N_row, N_col):
    # OUTBREAK CASE
    for beta, mu in zip(beta_outbreak, mu_outbreak):
        # Fix simulation
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
            # ------------------------------------ Load data for entropy analysis -----------------------------------------
            # Load dictionary for outbreak case
            dict_outbreak = pickle.load(open(folder_dict + f'dict_data-{row}x{col}-sim{sim}.pickle', 'rb'))

            # ------------------------------------ Persistent entropy algorithm --------------------------------------------

            entropy_H0_outbreak = []
            entropy_H1_outbreak = []
            for k in dict_outbreak.keys():
                entropy_outbreak = persistent_entropy(ripser.ripser(dict_outbreak[k])['dgms'])
                entropy_H0_outbreak.append(entropy_outbreak[0])
                entropy_H1_outbreak.append(entropy_outbreak[1])

            np.save(folder_analysis + f'entropy_H0_outbreak-sim{sim}', entropy_H0_outbreak)
            np.save(folder_analysis + f'entropy_H1_outbreak-sim{sim}', entropy_H1_outbreak)


# ------- NO OUTBREAK CASE -------
# Fix lattice dimension
for row, col in zip(N_row, N_col):
    # NO OUTBREAK CASE
    for beta, mu in zip(beta_no_outbreak, mu_no_outbreak):
        # Fix simulation
        # --------------------------------------------- Folders ---------------------------------------------
        folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
        folder_before_beta_mu = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/'
        folder_analysis = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis/'
        folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'

        T = np.load(folder_simulation + 'T.npy')
        T_sim = np.linspace(0, T, T + 1)

        for sim in range(nbr_simulations):
            print('row: ', row, 'col: ',  col)
            print('beta: ', beta, 'mu: ', mu)
            print('sim: ', sim)
            # ------------------------------------ Load data for entropy analysis -----------------------------------------
            # Load dictionary for no outbreak case
            dict_no_outbreak = pickle.load(open(folder_dict + f'dict_data-{row}x{col}-sim{sim}.pickle', 'rb'))

            # ------------------------------------ Persistent entropy algorithm --------------------------------------------

            entropy_H0_no_outbreak = []
            entropy_H1_no_outbreak = []
            for k in dict_no_outbreak.keys():
                entropy_no_outbreak = persistent_entropy(ripser.ripser(dict_no_outbreak[k])['dgms'])
                entropy_H0_no_outbreak.append(entropy_no_outbreak[0])
                entropy_H1_no_outbreak.append(entropy_no_outbreak[1])

            np.save(folder_analysis + f'entropy_H0_no_outbreak-sim{sim}', entropy_H0_no_outbreak)
            np.save(folder_analysis + f'entropy_H1_no_outbreak-sim{sim}', entropy_H1_no_outbreak)


