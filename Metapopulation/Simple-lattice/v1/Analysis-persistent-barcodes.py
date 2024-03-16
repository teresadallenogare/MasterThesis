"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 06 February 2024

--------------------------------------------------------------------

Plot persistence barcodes and persistence diagram

10x10 examplw: r0 = 12, ch 0 and c1 0
"""

from functions_TDA_v1 import *
from functions_visualization_v1 import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datadir = os.getcwd()

nbr_repetitions = 1

choice_bool = 0
c1 = 0

row = 30
col = 30

N = row * col

beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

beta_vals = [0.3]
mu_vals = [0.1]

id = 'XYSIR'
columns = ['X', 'Y', 'S', 'I', 'R']

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})
#sns.set(rc={"axes.labelsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16})

normalize_entropy = True

sim = 0 # exclude 4

clr = ['#FF8080', '#FF7F2A', '#D38D5F', '#87DE87', '#5FD3BC', '#80B3FF', '#8787DE', '#AA00D4', '#FF80E5', '#D35F8D']

for beta, mu in zip(beta_vals, mu_vals):
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'

    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'

    min_H0_sims = np.load(folder_entropy + f'min_H0_sims-mu{mu}-beta{beta}.npy')
    min_H1_sims = np.load(folder_entropy + f'min_H1_sims-mu{mu}-beta{beta}.npy')

    t_star = min_H0_sims[1, sim]

    print('t_star:', t_star)
    T = np.load(folder_simulation + 'T.npy')
    if beta == 0.115 or beta == 0.12:
        T = 1000
    T_sim = np.linspace(0, T - 1, T)

    dict_data = pickle.load(
        open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    df_dict_data = data_2_pandas(dict_data)

    # Dataframe at fixed time value
    S_t = df_dict_data.loc[df_dict_data['Time'] == t_star]  ##### ATTENTION HERE : t0 or t1??
    # Select only data indexed by "column"
    S_data_t = S_t[columns]
    # Euclidean distance between pairs of points (Each point corresponds to a row. The dimension of the space is given by the number of columns.)
    pers_homology = ripser.ripser(S_data_t)
    dgms = pers_homology['dgms']

    # Persistence diagrams for H0
    dgms0 = dgms[0]
    print('type dgms', type(dgms0))
    print('len dgms', len(dgms0))
    # Sort features in decreasing order
    brith0_upDown = dgms0[:, 0]
    birth0 = brith0_upDown[::-1]
    end0_upDown = dgms0[:, 1]
    end0 = end0_upDown[::-1]

    # Persistence diagrams for H1 (not sorted)
    dgms1 = dgms[1]
    birth1_upDown = dgms1[:, 0]
    end1_upDown = dgms1[:, 1]
    length_1 = end1_upDown - birth1_upDown
    # Get the indices that would sort the difference vector in decreasing order
    sorted_indices = np.argsort(length_1)[::-1]

    # Sort both vectors based on the sorted indices
    birth1 = birth1_upDown[sorted_indices]
    end1 = end1_upDown[sorted_indices]

    y0 = np.linspace(len(dgms0), 0, len(dgms0) + 1)
    y1 = np.linspace(len(dgms1), 0, len(dgms1) + 1)

    nbr_bars = 100

    plot_barcodes(birth0[:nbr_bars], end0[:nbr_bars], birth1[:nbr_bars], end1[:nbr_bars], y0[:nbr_bars], y1[:nbr_bars], t_star, beta/mu, sim, clr[sim])#[:100]
    #plot_barcodes(birth0[:], end0[:], birth1[:], end1[:], y0[:], y1[:], t_star, beta/mu, sim)#[:100]

    #plot_persistence_diagram(birth0, end0, birth1, end1)


