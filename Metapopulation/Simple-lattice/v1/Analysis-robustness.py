"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 18 February 2024

--------------------------------------------------------------------

Analysis of robustness : compute NPE from the topological features only. 
"""

from functions_TDA_v1 import *
from functions_visualization_v1 import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

acquire_data = 0

datadir = os.getcwd()

nbr_repetitions = 10

choice_bool = 0
c1 = 0

row = 30
col = 30

N = row * col

beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

beta_vals = [1.2]
mu_vals = [0.1]

id = 'XYSIR'
columns = ['X', 'Y', 'S', 'I', 'R']
normalize_entropy = True

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})
#sns.set(rc={"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

## Fix simulation (sim = 5 for choice_bool = 1, c1 = 1)
sim = 0


for beta, mu in zip(beta_vals, mu_vals):
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'

    T = np.load(folder_simulation + 'T.npy')
    if beta == 0.115 or beta == 0.12:
        T = 1000
    T_sim = np.linspace(0, T - 1, T)

    # Load dictionary
    dict_data = pickle.load(
        open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    df_dict_data = data_2_pandas(dict_data)

    ## THESE ARE EXTRACTED AT t0 AND t1 RESPECTIVELY
    ## Load data H0
    nbr_cc0 = np.load(folder_entropy + f'nbr_cc0-mu{mu}-beta{beta}-sim{sim}.npy')
    ## Load data H1
    nbr_cc1 = np.load(folder_entropy + f'nbr_cc1-mu{mu}-beta{beta}-sim{sim}.npy')
    # Select the persistence diagram of the number of most relevant features
    nbr_cc0 = 15
    nbr_cc1 = 15
    nbr_00 = 15
    nbr_11 = 300
    PEH0_3cc_time = []
    PEH1_3cc_time = []
    PEH0_inter_time = []
    PEH1_inter_time = []
    for t in range(T-1):
        # Dataframe at fixed time value
        S_t = df_dict_data.loc[df_dict_data['Time'] == t]
        # Select only data indexed by "column"
        S_data_t = S_t[columns]
        # print('t:', t, 'S_t:', S_data_t)
        # Euclidean distance between pairs of points (Each point corresponds to a row. The dimension of the space is given by the number of columns.)
        pers_homology = ripser.ripser(S_data_t)
        dgms = pers_homology['dgms']

        # Persistence diagrams for H0
        dgms0 = dgms[0]
        # Sort features in decreasing order
        brith0_upDown = dgms0[:, 0]
        birth0 = brith0_upDown[::-1]
        end0_upDown = dgms0[:, 1]
        end0 = end0_upDown[::-1]
        # Select the most relevant features
        birth0_3cc = birth0[1:nbr_cc0 ]
        end0_3cc = end0[1:nbr_cc0 ]
        # Intermediate between all and topological features
        birth0_inter = birth0[1:nbr_00]
        end0_inter = end0[1:nbr_00]
        # Diagram with sorted most relevant features
        dgm0_3cc = np.array([list(pair) for pair in zip(birth0_3cc, end0_3cc)])
        dgm0_inter = np.array([list(pair) for pair in zip(birth0_inter, end0_inter)])
        # Persistence diagrams for H1
        dgms1 = dgms[1]
        birth1_upDown = dgms1[:, 0]
        end1_upDown = dgms1[:, 1]
        # Intermediate between all and topological features
        length_1 = end1_upDown - birth1_upDown

        # Get the indices that would sort the difference vector in decreasing order
        sorted_indices = np.argsort(length_1)[::-1]

        # Sort both vectors based on the sorted indices (by length)
        birth1 = birth1_upDown[sorted_indices]
        end1 = end1_upDown[sorted_indices]

        # Select only births and deaths of longest cc
        birth1_3cc = birth1[1:nbr_cc1 ]
        end1_3cc = end1[1:nbr_cc1 ]
        birth1_inter = birth1[1: nbr_11]
        end1_inter = end1[1:nbr_11]
        dgm1_3cc = np.array([list(pair) for pair in zip(birth1_3cc, end1_3cc)])
        dgm1_inter = np.array([list(pair) for pair in zip(birth1_inter, end1_inter)])

        # Persistence diagram at a certain t from which computing PE
        dgms_3cc = [dgm0_3cc, dgm1_3cc]

        dgms_inter = [dgm0_inter, dgm1_inter]

        PE_3cc = persistent_entropy(dgms_3cc, normalize=True)
        PE_inter = persistent_entropy(dgms_inter, normalize = True)
        PE_H0_3cc = PE_3cc[0]
        PE_H1_3cc = PE_3cc[1]
        PE_H0_inter = PE_inter[0]
        PE_H1_inter = PE_inter[1]
        PEH0_3cc_time.append(PE_H0_3cc)
        PEH1_3cc_time.append(PE_H1_3cc)
        PEH0_inter_time.append(PE_H0_inter)
        PEH1_inter_time.append(PE_H1_inter)



    PEH0 = np.load(folder_entropy + f'entropy_H0-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}.npy')
    PEH1 = np.load(folder_entropy + f'entropy_H1-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}.npy')
    # Exclude last point because it excludes it in the calculation of entropy for how the code is implemented
    diff_H0 = PEH0 - PEH0_3cc_time
    diff_H1 = PEH1 - PEH1_3cc_time
    #plt.plot(T_sim[:T-1], diff_H0, color = 'red', label = 'H0')
    #plt.plot(T_sim[:T - 1], diff_H1, color='blue', label = 'H1')
    #plt.legend()
    #plt.show()
    f, ax = plt.subplots(figsize = (16, 6))
    plt.plot(T_sim[:T-1], PEH0, color='r', linestyle = ':', label = r'$H(\mathcal{H}_0)$', alpha = 0.7)
    plt.plot(T_sim[:T-1], PEH1, color='b', linestyle = ':', label = r'$H(\mathcal{H}_1)$', alpha = 0.7)
    plt.plot(T_sim[:T-1], PEH0_inter_time, color='r', linestyle = '--', label = r'$H(\mathcal{H}_0)$ 600 c.c', alpha = 0.8)
    plt.plot(T_sim[:T-1], PEH1_inter_time, color='b', linestyle = '--', label = r'$H(\mathcal{H}_1)$ 600 holes', alpha = 0.8)
    plt.plot(T_sim[:T-1], PEH0_3cc_time, color = 'r', label = r'$H(\mathcal{H}_0)$ topological features')
    plt.plot(T_sim[:T-1], PEH1_3cc_time, color = 'b', label = r'$H(\mathcal{H}_1)$ topological features')
    plt.xlabel('Time', fontsize=22)
    plt.ylabel('NPE', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=24)  # Adjust the size of major ticks
    ax.patch.set_alpha(0.8)
    plt.legend(fontsize = 18)
    plt.tight_layout()

    plt.show()

    print('hello')
