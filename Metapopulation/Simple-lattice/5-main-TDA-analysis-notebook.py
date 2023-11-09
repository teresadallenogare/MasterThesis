"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 04 November 2023

--------------------------------------------------------------------
Analysis of persistent entropy data obtained in 4-TDA-note4book.ipynb
All simulations are considered in this analysis file.

In the notebook there are 2 normalizations:
1. NORMALIZATION OF DATA
    - no normalized data : original dictionary
    - normalized data according to scaler

2. NORMALIZATION OF ENTROPY
    - no normalized persistent entropy
    - normalized persistent entropy according to the function
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

beta_no_outbreak = [0.35, 0.75]
mu_no_outbreak = [0.3, 0.6]

nbr_simulations = 10

normalization_by_hand = 2 # normalization of data 'by hand'
scaler = 1
normalization_entropy = 1
outbreak = 1

id = 'SIR'
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


for row, col in zip(N_row, N_col):
    if outbreak == 0:
        beta_vals = beta_no_outbreak
        mu_vals = mu_no_outbreak
    elif outbreak == 1:
        beta_vals = beta_outbreak
        mu_vals = mu_outbreak
    for beta, mu in zip(beta_vals, mu_vals):
        fig, ax = plt.subplots()
        folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'
        # Exclude simulations that did not start
        if beta == 0.3 and mu == 0.1:
            idx_simulations = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif beta == 0.35 and mu == 0.3:
            idx_simulations = [0, 2, 3, 4, 5, 7, 9]
        elif beta == 0.75 and mu == 0.6:
            idx_simulations = [0, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            idx_simulations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Fix folder containing non-normalized (0) or normalized (1) entropy data
        if normalization_entropy == 0:
            folder_entropy = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis-notebook-TDA/No-normalized-entropy/'
        elif normalization_entropy == 1:
            folder_entropy = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis-notebook-TDA/Normalized-entropy/'
        # Run over all simulations
        for sim in idx_simulations:
            print('row: ', row, 'col: ', col)
            print('beta: ', beta, 'mu: ', mu)
            print('sim: ', sim)
            if scaler == 0:
                entropy_H0 = np.load(folder_entropy + f'/Entropy-{id}/entropy_H0-sim{sim}.npy')
                entropy_H1 = np.load(folder_entropy + f'/Entropy-{id}/entropy_H1-sim{sim}.npy')
            elif scaler == 1:
                entropy_H0 = np.load(folder_entropy + f'/Entropy-scaler-{id}/entropy_H0-sim{sim}.npy')
                entropy_H1 = np.load(folder_entropy + f'/Entropy-scaler-{id}/entropy_H1-sim{sim}.npy')

            if normalization_by_hand == 1:
                folder_entropy = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis/Normalized/'
                entropy_H0 = np.load(folder_entropy + f'entropy_H0-sim{sim}.npy')
                entropy_H1 = np.load(folder_entropy + f'entropy_H1-sim{sim}.npy')
            x = range(0, len(entropy_H0))
            y_H0 = entropy_H0
            y_H1 = entropy_H1
            if sim == 2:
                ax.plot(x, y_H0, label=f'PE at H0', color=grad_green[sim] if outbreak == 0 else grad_blue[sim])
                ax.plot(x, y_H1, label=f'PE at H1', color=grad_red[sim] if outbreak == 0 else grad_red[sim])
            else:
                ax.plot(x, y_H0, color=grad_green[sim] if outbreak == 0 else grad_blue[sim])
                ax.plot(x, y_H1, color=grad_red[sim] if outbreak == 0 else grad_red[sim])
        ax.set_xlabel("Time")
        ax.set_ylabel("Persistent Entropy")
        ax.set_title(f"PE for {row}x{col}, beta = {beta}, mu = {mu}, R0 = {round(beta/mu,2)}")
        ax.legend()
        plt.show()