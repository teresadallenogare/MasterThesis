"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 26 October 2023

--------------------------------------------------------------------
Analysis of TDA data normalized with standard scaler as Matteo Rucco did
"""

from functions_SIR_metapop import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

beta_vals = [0.4, 0.3, 0.9, 0.35, 0.75]
mu_vals = [0.2, 0.1, 0.1, 0.3, 0.6]

nbr_simulations = 10


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

def plot_data_phase_space(dict_vals, dict_norm_vals, beta, mu, sim, id, scaler, normalize_entropy):
    fig, ax = plt.subplots()





def plot_entropy_time_series(entropy_h0, entropy_h1, beta, mu, sim, id, scaler, normalize_entropy):
    y_h0 = entropy_h0
    y_h1 = entropy_h1
    x = range(0, len(y_h0))

    fig, ax = plt.subplots()
    ax.plot(x, y_h0, label='Persistent entropy at H0')
    ax.plot(x, y_h1, label='Persistent entropy at H1')
    ax.set_xlabel("Time")
    ax.set_ylabel("Persistent Entropy")
    if normalize_entropy == 0:
        if scaler == 0:
            ax.set_title(f"PE id: {id}, beta: {beta}, mu: {mu} - sim: {sim} ")
        elif scaler == 1:
            ax.set_title(f"PE id: {id}, beta: {beta}, mu: {mu} - sim: {sim} - scaled data")
    elif normalize_entropy == 1:
        if scaler == 0:
            ax.set_title(f"Normalized PE id: {id}, beta: {beta}, mu: {mu} - sim: {sim} ")
        elif scaler == 1:
            ax.set_title(f"Normalized PE id: {id}, beta: {beta}, mu: {mu} - sim: {sim} - scaled data")
    #ax.legend()
    plt.show()

# Select data to load

row = 10
col = 10

beta = 0.35
mu = 0.3

sim = 0

scaler = 0
normalize_entropy = 0

for id in ['S', 'I', 'R']:
    for sim in range(nbr_simulations):
        if normalize_entropy == 0:
            folder = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis-notebook-TDA/No-normalized-entropy/'
            if scaler == 0:
                entropy_h0 = np.load(folder + f'/Entropy-XY{id}/entropy_H0-sim{sim}.npy')
                entropy_h1 = np.load(folder + f'/Entropy-XY{id}/entropy_H1-sim{sim}.npy')
            elif scaler == 1:
                entropy_h0 = np.load(folder + f'/Entropy-scaler-XY{id}/entropy_H0-sim{sim}.npy')
                entropy_h1 = np.load(folder + f'/Entropy-scaler-XY{id}/entropy_H1-sim{sim}.npy')
        elif normalize_entropy == 1:
            folder = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis-notebook-TDA/Normalized-entropy/'
            if scaler == 0:
                entropy_h0 = np.load(folder + f'/Entropy-XY{id}/entropy_H0-sim{sim}.npy')
                entropy_h1 = np.load(folder + f'/Entropy-XY{id}/entropy_H1-sim{sim}.npy')
            elif scaler == 1:
                entropy_h0 = np.load(folder + f'/Entropy-scaler-XY{id}/entropy_H0-sim{sim}.npy')
                entropy_h1 = np.load(folder + f'/Entropy-scaler-XY{id}/entropy_H1-sim{sim}.npy')

        plot_entropy_time_series(entropy_h0, entropy_h1, beta, mu, sim, id, scaler, normalize_entropy)

# Plot phase space data I normalized by hand without scaler
row = 10
col = 10
beta = 0.35
mu = 0.3
# --------------------------------------------- Folders ---------------------------------------------
folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'

T = np.load(folder_simulation + 'T.npy')
T_sim = np.linspace(0, T, T + 1)
nbr_simulations = 1
for sim in range(nbr_simulations):
    print('row: ', row, 'col: ',  col)
    print('beta: ', beta, 'mu: ', mu)
    print('sim: ', sim)
    ax = plt.axes(projection='3d')
    dict_data_no_norm = pickle.load(open(folder_dict + f'No-normalized/dict_data-{row}x{col}-sim{sim}.pickle', 'rb'))
    dict_no_norm_vals = list(dict_data_no_norm.values())
    dict_data = pickle.load(open(folder_dict + f'Normalized/dict_data_normalized-{row}x{col}-sim{sim}.pickle', 'rb'))
    dict_vals = list(dict_data.values())
    for t in range(T):
        mtrx_t = dict_vals[t]
        x = mtrx_t[:, 0]
        y = mtrx_t[:, 1]
        z = mtrx_t[:, 3]
        sc = ax.scatter3D(x, y, z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('I')


    plt.show()

    plt.figure()
    ax = plt.axes(projection='3d')
    for t in range(T):
        mtrx_t = dict_no_norm_vals[t]
        x = mtrx_t[:, 0]
        y = mtrx_t[:, 1]
        z = mtrx_t[:, 3]
        sc = ax.scatter3D(x, y, z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('I')


    plt.show()

