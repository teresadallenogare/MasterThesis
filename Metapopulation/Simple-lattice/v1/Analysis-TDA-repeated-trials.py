"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 04 February 2024

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

single_NPE = 1
multiple_NPEs = 0
derivative_entropy = 0

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})

#sns.set(rc={"axes.labelsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16, "axes.titlesize": 16})

if single_NPE == 1:

    nbr_repetitions = 10

    choice_bool = 0
    c1 = 0

    row = 30
    col = 30

    N = row * col

    beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
    mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    beta_vals = [0.4]
    mu_vals = [0.1]
    id = 'XYSIR'
    normalize_entropy = True

    for beta, mu in zip(beta_vals, mu_vals):
        f, ax = plt.subplots(figsize=(14, 7))
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
        t_max_H0 = []
        t_max_H1 = []
        max_H0 = []
        max_H1 = []
        idx_sims_started = []

        for sim in range(4,5):
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

                ## Extract minimum of entropy and corresponding time
                idx_min_H0_sims = np.argmin(entropy_H0_sim)
                t_min_H0.append(idx_min_H0_sims)
                min_H0.append(entropy_H0_sim[idx_min_H0_sims])

                idx_min_H1_sims = np.argmin(entropy_H1_sim)
                t_min_H1.append(idx_min_H1_sims)
                min_H1.append(entropy_H0_sim[idx_min_H1_sims])

                ## Extract maximum of entropy and corresponding time
                idx_max_H0_sims = np.argmax(entropy_H0_sim)
                t_max_H0.append(idx_max_H0_sims)
                max_H0.append(entropy_H0_sim[idx_max_H0_sims])

                idx_max_H1_sims = np.argmax(entropy_H1_sim[1:])
                t_max_H1.append(idx_max_H1_sims)
                max_H1.append(entropy_H1_sim[idx_max_H1_sims])

                entropy_H0_repeat.extend(entropy_H0_sim)
                entropy_H1_repeat.extend(entropy_H1_sim)

                x = range(0, len(entropy_H0_sim))
                if sim == 0:
                    plt.plot(x, entropy_H0_sim, color='silver', lw=0.6)# 0.6
                    plt.plot(x, entropy_H1_sim, color='silver', lw=0.6)
                else:
                    plt.plot(x, entropy_H0_sim, color='silver', lw=0.6)
                    plt.plot(x, entropy_H1_sim, color='silver', lw=0.6)

                plt.xlabel('Time', fontsize = 24)
                plt.ylabel('NPE', fontsize = 24)
                plt.title(fr'NPE for $R_0$ = {np.round(beta/mu, 2)}', fontsize = 26)
                ax.tick_params(axis='both', which='major', labelsize=22)  # Adjust the size of major ticks
                ax.patch.set_alpha(0.8)
                i = i + 1

        min_H0_sims = [idx_sims_started, t_min_H0, min_H0]
        min_H1_sims = [idx_sims_started, t_min_H1, min_H1]

        max_H0_sims = [idx_sims_started, t_max_H0, max_H0]
        max_H1_sims = [idx_sims_started, t_max_H1, max_H1]
        print('t min H0:', t_min_H0)
        print('t min H1: ', t_min_H1)

        print('t max H0:', t_max_H0)
        print('t max H1:', t_max_H1)

        print('avg t max H1:', np.array(t_max_H1).mean())

        min_H0_sims = np.array(min_H0_sims)
        min_H1_sims = np.array(min_H1_sims)
        entropy_H0 = np.array(entropy_H0_repeat).reshape(i, len(entropy_H0_sim))
        entropy_H1 = np.array(entropy_H1_repeat).reshape(i, len(entropy_H1_sim))
        avg_H0 = np.mean(entropy_H0, axis=0)
        avg_H1 = np.mean(entropy_H1, axis=0)
        #print('avgavg H1', avg_H1[750:].mean(axis = 0))

        t_min_avg_H0 = np.argmin(avg_H0)
        t_min_avg_H1 = np.argmin(avg_H1)
        t_max_avg_H1 = np.argmax(avg_H1[1:])
        # Get the corresponding x value
        min_avg_H0 = np.array([t_min_avg_H0, avg_H0[t_min_avg_H0]])
        min_avg_H1 = np.array([t_min_avg_H1, avg_H1[t_min_avg_H1]])

        line1_H0 = t_min_avg_H0 - 4
        line2_H0 = t_min_avg_H0 + 4

        line1_H1 = t_max_avg_H1 - 4
        line2_H1 = t_max_avg_H1 + 4

        print('x_min avg H0: ', t_min_avg_H0)
        print('min_avg H0: ', min_avg_H0)
        print('-------------------------------')
        print('x_min avg H1: ', t_min_avg_H1)
        print('min_avg H1: ', min_avg_H1)

        plt.plot(x, avg_H0, color='r', lw=2, label=r'H($\mathcal{H}_0$)')
        plt.plot(x, avg_H1, color='b', lw=2, label=r'H($\mathcal{H}_1$)')
        #plt.fill_betweenx(plt.ylim(), line1_H0, line2_H0, alpha=0.1, color='gray')
        #plt.fill_betweenx(plt.ylim(), line1_H1, line2_H1, alpha=0.2, color='silver')
       # plt.scatter(min_avg_H0[0], min_avg_H0[1])
       # plt.scatter(min_avg_H1[0], min_avg_H1[1])

        plt.legend(fontsize = 22)
        plt.tight_layout()
        plt.show()

        # NEED TO SAVE FOR CH1 AND C11

        # Save average behaviour
        np.save(folder_entropy + f'avg_H0-mu{mu}-beta{beta}', avg_H0)
        np.save(folder_entropy + f'avg_H1-mu{mu}-beta{beta}', avg_H1)

        # Save min entropy simulations
        np.save(folder_entropy + f'min_H0_sims-mu{mu}-beta{beta}', min_H0_sims)
        np.save(folder_entropy + f'min_H1_sims-mu{mu}-beta{beta}', min_H1_sims)
        np.save(folder_entropy + f'min_avg_H0-mu{mu}-beta{beta}', min_avg_H0)
        np.save(folder_entropy + f'min_avg_H1-mu{mu}-beta{beta}', min_avg_H1)


if multiple_NPEs == 1:
    sns.set(rc={"axes.labelsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16, "axes.titlesize": 16})
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})

    row = 30
    col = 30

    N = row * col

    choice_bool = 0
    c1 = 0

    beta_vals = [0.12, 0.4, 1.2]
    mu = 0.1

    nbr_repetitions = 10

    id = 'XYSIR'
    normalize_entropy = True


    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'

    idx_sims_not_start0 = np.load(folder_simulation + f'mu-{mu}/beta-{beta_vals[0]}/idx_sim_not_start.npy')
    idx_sims_not_start1 = np.load(folder_simulation + f'mu-{mu}/beta-{beta_vals[1]}/idx_sim_not_start.npy')
    idx_sims_not_start2 = np.load(folder_simulation + f'mu-{mu}/beta-{beta_vals[2]}/idx_sim_not_start.npy')

    T0 = np.load(folder_simulation + f'mu-{mu}/beta-{beta_vals[0]}/T.npy')
    T1 = np.load(folder_simulation + f'mu-{mu}/beta-{beta_vals[1]}/T.npy')
    T2 = np.load(folder_simulation + f'mu-{mu}/beta-{beta_vals[2]}/T.npy')

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    entropy_H0_repeat_0 = []
    entropy_H1_repeat_0 = []
    entropy_H0_repeat_1 = []
    entropy_H1_repeat_1 = []
    entropy_H0_repeat_2 = []
    entropy_H1_repeat_2 = []
    i = 0
    ### First plot
    for sim in range(nbr_repetitions):
        if sim not in idx_sims_not_start0:
            entropy_H0_sim_0 = np.load(
                folder_entropy + f'entropy_H0-nrm{normalize_entropy}-beta{beta_vals[0]}-mu{mu}-sim{sim}.npy')
            entropy_H1_sim_0 = np.load(
                folder_entropy + f'entropy_H1-nrm{normalize_entropy}-beta{beta_vals[0]}-mu{mu}-sim{sim}.npy')
            if choice_bool == 0 and c1 == 1 and sim == 0:
                entropy_H0_sim_0 = entropy_H0_sim_0[:999]
                entropy_H1_sim_0 = entropy_H1_sim_0[:999]

            entropy_H0_repeat_0.extend(entropy_H0_sim_0)
            entropy_H1_repeat_0.extend(entropy_H1_sim_0)

            x = range(0, len(entropy_H0_sim_0))
            ax1.plot(x, entropy_H0_sim_0, color='silver', lw=0.6)
            ax1.plot(x, entropy_H1_sim_0, color='silver', lw=0.6)

            ax1.set_xlabel('Time', fontsize = 16)
            ax1.set_ylabel('NPE', fontsize = 16)
            ax1.tick_params(axis='both', which='major', labelsize=14)  # Adjust the size of major ticks
            ax1.patch.set_alpha(0.8)
            i = i + 1
    entropy_H0 = np.array(entropy_H0_repeat_0).reshape(i, len(entropy_H0_sim_0))
    entropy_H1 = np.array(entropy_H1_repeat_0).reshape(i, len(entropy_H1_sim_0))
    avg_H0 = np.mean(entropy_H0, axis=0)
    avg_H1 = np.mean(entropy_H1, axis=0)

    ax1.plot(x, avg_H0, color='r', lw=1, label=r'H($\mathcal{H}_0$)')
    ax1.plot(x, avg_H1, color='b', lw=1, label=r'H($\mathcal{H}_1$)')
    ax1.set_title(rf'NPE for $R_0$ = { np.round(beta_vals[0] / mu, 2)}', fontsize = 16)

    i = 0
    ### Second plot
    for sim in range(nbr_repetitions):
        if sim not in idx_sims_not_start1:
            entropy_H0_sim_1 = np.load(
                folder_entropy + f'entropy_H0-nrm{normalize_entropy}-beta{beta_vals[1]}-mu{mu}-sim{sim}.npy')
            entropy_H1_sim_1 = np.load(
                folder_entropy + f'entropy_H1-nrm{normalize_entropy}-beta{beta_vals[1]}-mu{mu}-sim{sim}.npy')
            if choice_bool == 0 and c1 == 1 and sim == 0:
                entropy_H0_sim_1 = entropy_H0_sim_1[:999]
                entropy_H1_sim_1 = entropy_H1_sim_1[:999]

            entropy_H0_repeat_1.extend(entropy_H0_sim_1)
            entropy_H1_repeat_1.extend(entropy_H1_sim_1)

            x = range(0, len(entropy_H0_sim_1))
            ax2.plot(x, entropy_H0_sim_1, color='silver', lw=0.6)
            ax2.plot(x, entropy_H1_sim_1, color='silver', lw=0.6)

            ax2.set_xlabel('Time', fontsize = 16)
            ax2.set_ylabel('NPE', fontsize = 16)
            ax2.tick_params(axis='both', which='major', labelsize=14)  # Adjust the size of major ticks

            i = i + 1

    entropy_H0 = np.array(entropy_H0_repeat_1).reshape(i, len(entropy_H0_sim_1))
    entropy_H1 = np.array(entropy_H1_repeat_1).reshape(i, len(entropy_H1_sim_1))
    avg_H0 = np.mean(entropy_H0, axis=0)
    avg_H1 = np.mean(entropy_H1, axis=0)

    ax2.plot(x, avg_H0, color='r', lw=1, label=r'H($\mathcal{H}_0$)')
    ax2.plot(x, avg_H1, color='b', lw=1, label=r'H($\mathcal{H}_1$)')
    ax2.set_title(rf'NPE for $R_0$ = {np.round(beta_vals[1] / mu, 2)}', fontsize = 16)
    ax2.patch.set_alpha(0.8)
    i = 0
    ### Third plot
    for sim in range(nbr_repetitions):
        if sim not in idx_sims_not_start2:
            entropy_H0_sim_2 = np.load(
                folder_entropy + f'entropy_H0-nrm{normalize_entropy}-beta{beta_vals[2]}-mu{mu}-sim{sim}.npy')
            entropy_H1_sim_2 = np.load(
                folder_entropy + f'entropy_H1-nrm{normalize_entropy}-beta{beta_vals[2]}-mu{mu}-sim{sim}.npy')
            if choice_bool == 0 and c1 == 1 and sim == 0:
                entropy_H0_sim_2 = entropy_H0_sim_2[:999]
                entropy_H1_sim_2 = entropy_H1_sim_2[:999]

            entropy_H0_repeat_2.extend(entropy_H0_sim_2)
            entropy_H1_repeat_2.extend(entropy_H1_sim_2)

            x = range(0, len(entropy_H0_sim_2))
            ax3.plot(x, entropy_H0_sim_2, color='silver', lw=0.6)
            ax3.plot(x, entropy_H1_sim_2, color='silver', lw=0.6)

            ax3.set_xlabel('Time', fontsize = 16)
            ax3.set_ylabel('NPE', fontsize = 16)
            ax3.patch.set_alpha(0.8)
            i = i + 1
    entropy_H0 = np.array(entropy_H0_repeat_2).reshape(i, len(entropy_H0_sim_2))
    entropy_H1 = np.array(entropy_H1_repeat_2).reshape(i, len(entropy_H1_sim_2))
    avg_H0 = np.mean(entropy_H0, axis=0)
    avg_H1 = np.mean(entropy_H1, axis=0)
    ax3.plot(x, avg_H0, color='r', lw=1, label=r'H($\mathcal{H}_0$)')
    ax3.plot(x, avg_H1, color='b', lw=1, label=r'H($\mathcal{H}_1$)')
    ax3.set_title(rf'NPE for $R_0$ = {np.round(beta_vals[2] / mu, 2)}', fontsize = 16)
    ax3.tick_params(axis='both', which='major', labelsize=14)  # Adjust the size of major ticks

    ax1.legend(fontsize = 14)
    ax2.legend(fontsize = 14)
    ax3.legend(fontsize = 14)
    plt.tight_layout()
    plt.show()


if derivative_entropy == 1:

    nbr_repetitions = 1

    choice_bool = 0
    c1 = 0

    row = 30
    col = 30

    N = row * col

    beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
    mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    id = 'XYSIR'
    normalize_entropy = True

    for beta, mu in zip(beta_vals, mu_vals):
        f, ax = plt.subplots(figsize=(10, 8))
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
        t_max_H0 = []
        t_max_H1 = []
        max_H0 = []
        max_H1 = []
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


                entropy_H0_repeat.extend(entropy_H0_sim)
                entropy_H1_repeat.extend(entropy_H1_sim)

                ## Numerical derivative of entropy H0 and H1
                dt = 1
                dEntropy_H0 = np.gradient(entropy_H0_sim, dt)
                dEntropy_H1 = np.gradient(entropy_H1_sim, dt)

                x = range(0, len(entropy_H0_sim))
                plt.plot(x, dEntropy_H0, color = 'r')
                plt.plot(x, dEntropy_H1, color = 'b')

                plt.xlabel('Time')
                plt.ylabel('dH/dt')

                plt.show()
