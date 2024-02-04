"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 04 February 2023

--------------------------------------------------------------------

PE repeated trials :
- Plot NPE for repeated simulations and average trend
- Extract min value of average NPE
- Extract min value of repeated simulations NPE
"""

from functions_TDA_v1 import *
from functions_visualization_v1 import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datadir = os.getcwd()

nbr_repetitions = 10

choice_bool = 0
c1 = 0

row = 30
col = 30

N = row * col

beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

id = 'XYSIR'
normalize_entropy = True

plt.figure(figsize=(10, 8))
# sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})

for beta, mu in zip(beta_vals, mu_vals):
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'

    idx_sims_not_start = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/idx_sim_not_start.npy')

    entropy_H0_repeat = []
    entropy_H1_repeat = []
    i = 0

    t_min_H0 = []
    t_min_H1 = []
    min_H0 = []
    min_H1 = []
    idx_sims_started = []
    for sim in range(nbr_repetitions):
        if sim not in idx_sims_not_start:
            idx_sims_started.append(int(sim))
            T = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/T.npy')
            print('row: ', row, 'col: ', col, 'choice_bool: ', choice_bool, 'c1: ', c1)
            print('beta: ', beta, 'mu: ', mu)

            entropy_H0_sim = np.load(
                folder_entropy + f'entropy_H0-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}.npy')
            entropy_H1_sim = np.load(
                folder_entropy + f'entropy_H1-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}.npy')
            if choice_bool == 0 and c1 == 1 and sim == 0:
                entropy_H0_sim = entropy_H0_sim[:999]
                entropy_H1_sim = entropy_H1_sim[:999]
            len_H0 = len(entropy_H0_sim)
            len_H1 = len(entropy_H1_sim)

            print('len H0: ', len_H0)
            print('len H1: ', len_H1)

            ## Extract minimum of entropy and corresponding time
            idx_min_H0_sims = np.argmin(entropy_H0_sim)
            t_min_H0.append(idx_min_H0_sims)
            min_H0.append(entropy_H0_sim[idx_min_H0_sims])

            idx_min_H1_sims = np.argmin(entropy_H1_sim)
            t_min_H1.append(idx_min_H1_sims)
            min_H1.append(entropy_H0_sim[idx_min_H1_sims])

            entropy_H0_repeat.extend(entropy_H0_sim)
            entropy_H1_repeat.extend(entropy_H1_sim)

            x = range(0, len(entropy_H0_sim))
            plt.plot(x, entropy_H0_sim, color='silver', lw=0.6)
            plt.plot(x, entropy_H1_sim, color='silver', lw=0.6)

            plt.xlabel('Time')
            plt.ylabel('Persistent Entropy')

            i = i + 1

    min_H0_sims = [idx_sims_started, t_min_H0, min_H0]
    min_H1_sims = [idx_sims_started, t_min_H1, min_H1]

    min_H0_sims = np.array(min_H0_sims)
    min_H1_sims = np.array(min_H1_sims)
    entropy_H0 = np.array(entropy_H0_repeat).reshape(i, len(entropy_H0_sim))
    entropy_H1 = np.array(entropy_H1_repeat).reshape(i, len(entropy_H1_sim))
    avg_H0 = np.mean(entropy_H0, axis=0)
    avg_H1 = np.mean(entropy_H1, axis=0)
    t_min_avg_H0 = np.argmin(avg_H0)
    t_min_avg_H1 = np.argmin(avg_H1)
    # Get the corresponding x value
    min_avg_H0 = np.array([t_min_avg_H0, avg_H0[t_min_avg_H0]])
    min_avg_H1 = np.array([t_min_avg_H1, avg_H1[t_min_avg_H1]])


    print('x_min avg H0: ', t_min_avg_H0)
    print('min_avg H0: ', min_avg_H0)
    print('-------------------------------')
    print('x_min avg H1: ', t_min_avg_H1)
    print('min_avg H1: ', min_avg_H1)

    plt.plot(x, avg_H0, color='r', lw=1, label=r'H($\mathcal{H}_0$)')
    plt.plot(x, avg_H1, color='b', lw=1, label=r'H($\mathcal{H}_1$)')
    plt.scatter(min_avg_H0[0], min_avg_H0[1])
    plt.scatter(min_avg_H1[0], min_avg_H1[1])

    plt.legend()
    plt.show()

    # NEED TO SAVE FOR CH1 AND C11

    # Save average behaviour
   # np.save(folder_entropy + f'avg_H0-mu{mu}-beta{beta}', avg_H0)
   # np.save(folder_entropy + f'avg_H1-mu{mu}-beta{beta}', avg_H1)

    # Save min entropy simulations
    np.save(folder_entropy + f'min_H0_sims-mu{mu}-beta{beta}', min_H0_sims)
    np.save(folder_entropy + f'min_H1_sims-mu{mu}-beta{beta}', min_H1_sims)
    np.save(folder_entropy + f'min_avg_H0-mu{mu}-beta{beta}', min_avg_H0)
    np.save(folder_entropy + f'min_avg_H1-mu{mu}-beta{beta}', min_avg_H1)




