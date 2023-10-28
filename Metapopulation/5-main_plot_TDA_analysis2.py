"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 25 October 2023

--------------------------------------------------------------------
First analysis concerning the application of TDA to data obtained from the simulations.

Here make distinction of outbreak and non-outbreak cases
"""

from functions_SIR_metapop import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


# ------------------------------------------------ Parameters  -------------------------------------------------

N_row = [30]
N_col = [30]

choice_bool = 0
datadir = os.getcwd()
c1 = 0  # for now
beta_outbreak = [0.4, 0.3, 0.9]
mu_outbreak = [0.2, 0.1, 0.1]

#beta_outbreak = [0.9]
#mu_outbreak = [0.1]

beta_no_outbreak = [0.35, 0.75]
mu_no_outbreak = [0.3, 0.6]

nbr_simulations = 10

normalization = 2
outbreak = 1


# ------------------------------------------------ Colors  -------------------------------------------------
grad_gray = []
grad_red = []
grad_blue = []
grad_green = []

for x in range(nbr_simulations):
    #                                dark           light
    grad_gray.append(colorFader('#505050', '#EAE9E9', x/nbr_simulations))
    grad_red.append(colorFader('#E51C00', '#FCE0DC', x/nbr_simulations))
    grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x/nbr_simulations))
    grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x/nbr_simulations))

# ------------ Persistent entropy calculation -----------
# Outbreak case
for row, col in zip(N_row, N_col):
    if outbreak == 0:
        beta_vals = beta_no_outbreak
        mu_vals = mu_no_outbreak
    elif outbreak == 1:
        beta_vals = beta_outbreak
        mu_vals = mu_outbreak

    for beta, mu in zip(beta_vals, mu_vals):
        fig, ax = plt.subplots()
        folder_analysis = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis/'
        folder_analysis_normalized = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis/Normalized/'
        folder_analysis_scaler = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis/Normalized-scaler/'

        folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
        folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'
        folder_dict_normalized = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/Normalized/'
        folder_dict_scaler = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/Normalized-scaler/'

        for sim in range(nbr_simulations):
            if normalization == 0:
                dict_vals = pickle.load(open(folder_dict + f'dict_data-{row}x{col}-sim{sim}.pickle', 'rb'))
                entropy_H0 = np.load(folder_analysis + f'entropy_H0_outbreak-sim{sim}.npy')
                entropy_H1 = np.load(folder_analysis + f'entropy_H1_outbreak-sim{sim}.npy')
            elif normalization == 1:
                dict_vals = pickle.load(open(folder_dict_normalized + f'dict_data_normalized-{row}x{col}-sim{sim}.pickle', 'rb'))
                entropy_H0 = np.load(folder_analysis_normalized + f'entropy_H0-sim{sim}.npy')
                entropy_H1 = np.load(folder_analysis_normalized + f'entropy_H1-sim{sim}.npy')
            elif normalization == 2:
                dict_vals = pickle.load(open(folder_dict_scaler + f'dict_data_normalized-{row}x{col}-sim{sim}.pickle', 'rb'))
                entropy_H0 = np.load(folder_analysis_scaler + f'entropy_H0-sim{sim}.npy')
                entropy_H1 = np.load(folder_analysis_scaler + f'entropy_H1-sim{sim}.npy')

            x = range(0, len(dict_vals.keys()))
            y_H0 = entropy_H0
            y_H1 = entropy_H1

            ax.plot(x, y_H0, label=f'PE at H0, sim {sim}', color = grad_green[sim] if outbreak == 0 else grad_blue[sim])
            ax.plot(x, y_H1, label=f'PE at H1, sim {sim}',color = grad_red[sim] if outbreak == 0 else grad_red[sim])
        ax.set_xlabel("Time")
        ax.set_ylabel("Persistent Entropy")
        ax.set_title(f"PE for {row}x{col}, beta = {beta}, mu = {mu}" )
        #ax.legend()
        plt.show()
