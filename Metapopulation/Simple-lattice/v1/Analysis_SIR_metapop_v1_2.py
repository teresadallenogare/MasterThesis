"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 23 January 2024

--------------------------------------------------------------------

Compute observables to characterize the SIR epidemics simulated as a function of time or of R0

"""
from functions_SIR_metapop_v1 import *
from functions_output_v1 import write_simulation_file
from functions_visualization_v1 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.integrate import odeint

datadir = os.getcwd()
plt.figure(figsize=(8, 6))
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})

lineStyle = ['-', '--', ':']

row = 30
col = 30

N = row * col

choice_bool = 0

sim = 0

beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9,1.2]
mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

ana_prevalence = 0
ana_attack_rate = 0
ana_max = 1
ana_max2 = 0
ana_max3 = 0

folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{0}/Topology/'
avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')

# ------------------------------------------------ Colors  -------------------------------------------------
grad_green = []

for x in range(8):
    grad_green.append(colorFader('#9DAC93', '#6600ff', x / 16))


grad_green = ['#374845', '#536c67', '#6f918a', '#93aca7','#b7c8c4', '#d5d5ff', '#aaaaff', '#8787de' ]
###########################################################################
if ana_prevalence == 1:
    f, ax = plt.subplots(figsize=(13, 8))

    i = 0
    for (beta, mu) in zip(beta_vals, mu_vals):
        c1 = 0
        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
        node_new_NI_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

        T = np.load(folder_simulation + f'T.npy')
        T_sim = np.linspace(0, T - 1, T)
        ax2 = plt.subplot(2, 1, 1)
        network_new_NI_time = node_new_NI_time.sum(axis= 1)
        rho_network_new_NI_time = network_new_NI_time / (N* avg_popPerNode)
        plt.plot(T_sim[:400], rho_network_new_NI_time[:400], color = grad_green[i])
        plt.xlabel('Time', fontsize=14)
        plt.ylabel(r'Prevalence', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.text(350, 0.2, 'HOM no confined', fontsize=12)

        c1 = 1
        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
        node_new_NI_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

        T = np.load(folder_simulation + f'T.npy')
        T_sim = np.linspace(0, T - 1, T)
        ax1 = plt.subplot(2, 1, 2)
        network_new_NI_time = node_new_NI_time.sum(axis=1)
        rho_network_new_NI_time = network_new_NI_time / (N * avg_popPerNode)
        plt.plot(T_sim[:400], rho_network_new_NI_time[:400], label=fr'$R_0$= {np.round(beta / mu, 2)}',
                 color=grad_green[i])
        plt.xlabel('Time', fontsize=14)
        plt.ylabel(r'Prevalence', fontsize=14)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.text(350, 0.15, 'HOM confined', fontsize=12)
        # Adjust the position of subplots to the left
        #plt.subplots_adjust(left=0.08, right=0.9, wspace=0.4)
        i = i+1

    # Adjust layout to prevent overlapping titles
    #plt.tight_layout()
    legend = f.legend(loc='center', bbox_to_anchor=(0.95, 0.5), fontsize = 14, frameon=False)
    plt.show()


if ana_attack_rate == 1:
    f, ax = plt.subplots(figsize=(10, 8))

    i = 0
    for (beta, mu) in zip(beta_vals, mu_vals):
        c1 = 0
        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
        node_new_NI_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

        T = np.load(folder_simulation + f'T.npy')
        T_sim = np.linspace(0, T - 1, T)

        network_new_NI_time = node_new_NI_time.sum(axis = 1)
        network_NS_time = node_NS_time.sum(axis = 1)

        attack_rate = network_new_NI_time / network_NS_time

        plt.plot(T_sim[:400], attack_rate[:400], label=fr'$R_0$= {np.round(beta / mu, 2)}',
                 color=grad_green[i])
        plt.xlabel('Time', fontsize=14)
        plt.ylabel(r'Attack rate', fontsize=14)

        i = i + 1

    plt.show()
########################################################################################################################
if ana_max == 1:
    row = 30
    col = 30
    N = row * col
    beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
    mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    choice_bool = 0
    c1_lst = [0]

    nbr_simulations = 10
    # Non confined case

    avg_popPerNode = 10**4
    for c1 in c1_lst:

        for (beta, mu) in zip(beta_vals, mu_vals):
            i = 0
            f, ax = plt.subplots(figsize=(12, 8))
            folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

            idx_max_I_varyR0 = []
            idx_max_Inew_varyR0 = []
            max_I_varyR0 = []
            max_Inew_varyR0 = []
            idx_max_Rnew_varyR0 = []
            NI_new_time_repeat = []
            NR_new_time_repeat = []
            idx_sims_not_start = np.load(folder_simulation + f'idx_sim_not_start.npy')
            for sim in range(nbr_simulations):
                if sim not in idx_sims_not_start:
                    node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
                    node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')/(N*avg_popPerNode)
                    node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')/(N*avg_popPerNode)
                    node_new_NI_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')/(N*avg_popPerNode)
                    node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')/(N*avg_popPerNode)
                    node_NR_new_time = np.load(folder_simulation + f'sim_{sim}_new_R_time.npy')/(N*avg_popPerNode)
                    if beta == 0.115 or beta == 0.12:
                        node_new_NI_time = node_new_NI_time[:1000]
                        node_NR_new_time = node_NR_new_time[:1000]

                    # Sum to work at network level
                    NI_time = node_NI_time.sum(axis=1)
                    NI_new_time = node_new_NI_time.sum(axis=1)
                    NR_new_time = node_NR_new_time.sum(axis=1)

                    # List to work with repeated trials
                    NI_new_time_repeat.extend(NI_new_time)
                    NR_new_time_repeat.extend(NR_new_time)

                    # Extract maximum values
                    idx_max_I = np.argmax(NI_time)
                    idx_max_I_new = np.argmax(NI_new_time)
                    idx_max_I_varyR0.append(idx_max_I)
                    idx_max_Inew_varyR0.append(idx_max_I_new)
                    idx_max_Rnew = np.argmax(NR_new_time)
                    idx_max_Rnew_varyR0.append(idx_max_Rnew)

                    T = np.load(folder_simulation + f'T.npy')
                    if beta == 0.115 or beta == 0.12:
                        T = 1000
                    T_sim = np.linspace(0, T - 1, T)

                    plt.plot(T_sim, NI_new_time, color = '#e9afaf', linewidth = 1)
                    plt.plot(T_sim, NR_new_time, color = '#93ac93', linewidth = 1)

                    plt.xlabel('Time', fontsize = 32)
                    plt.ylabel(r'Variation', fontsize = 32)
                    ax.tick_params(axis='both', which='major', labelsize=32)  # Adjust the size of major ticks
                    if beta == 0.115 or beta == 0.12 or beta == 0.15 or beta == 0.2:
                        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

                    ax.patch.set_alpha(0.8)

                    i = i + 1

            max_I_varyR0.append(NI_time[idx_max_I])
            max_Inew_varyR0.append(NI_new_time[idx_max_I_new])
            print('------------------------------------------')
            print('beta: ', beta)
            #print('idx I:', idx_max_I_varyR0)
            #print('idx I_new:', idx_max_I_new)
            #print('idx R_new:', idx_max_Rnew)
            print('t I max:', np.array(idx_max_I_varyR0).mean())
            print('avg I max:', np.array(max_I_varyR0).mean())
            print('std I max:', np.array(max_I_varyR0).std(ddof = 1))
            #print('avg t I new max:', np.array(idx_max_Inew_varyR0).mean())

            NI_new_time_end = np.array(NI_new_time_repeat).reshape(i, len(NI_new_time))
            NR_new_time_end = np.array(NR_new_time_repeat).reshape(i, len(NR_new_time))
            avg_NI_new = np.mean(NI_new_time_end, axis=0)
            avg_NR_new = np.mean(NR_new_time_end, axis=0)

            plt.plot(T_sim, avg_NI_new, color='#d40000', lw=2, label = r'$\langle \Delta I(t) \rangle$')
            plt.plot(T_sim, avg_NR_new, color='#005522', lw=2, label = r'$\langle\Delta R(t) \rangle$')
            ax.yaxis.get_offset_text().set_fontsize(32)

            plt.legend(fontsize = 32)
            plt.show()

if ana_max2 == 1:

    row = 30
    col = 30
    choice_bool = 0
    c1 = 0

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    ##### First plot

    beta = 0.12
    mu = 0.1
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
    idx_sims_not_start = np.load(folder_simulation + f'idx_sim_not_start.npy')

    sim = 0

    if sim not in idx_sims_not_start:

        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy') / (N * avg_popPerNode)
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy') / (N * avg_popPerNode)
        node_new_NI_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy') / (N * avg_popPerNode)
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy') / (N * avg_popPerNode)
        node_NR_new_time = np.load(folder_simulation + f'sim_{sim}_new_R_time.npy') / (N * avg_popPerNode)
        if beta == 0.115 or beta == 0.12:
            node_new_NI_time = node_new_NI_time[:1000]
            node_NR_new_time = node_NR_new_time[:1000]
        # Sum to work at network level
        NI_time = node_NI_time.sum(axis=1)
        NI_new_time = node_new_NI_time.sum(axis=1)
        NR_new_time = node_NR_new_time.sum(axis=1)
        # Extract maximum values
        idx_max_I = np.argmax(NI_time)
        idx_max_I_new = np.argmax(NI_new_time)
        idx_max_Rnew = np.argmax(NR_new_time)
        T = np.load(folder_simulation + f'T.npy')
        if beta == 0.115 or beta == 0.12:
            T = 1000
        T_sim = np.linspace(0, T - 1, T)

        ax1.plot(T_sim, NI_new_time, color='#d40000', linewidth=1.5, label=r'$\Delta I(t)$')
        ax1.plot(T_sim, NR_new_time, color='#005522', linewidth=1.5, label=r'$\Delta R(t)$')

        ax1.set_xlabel('Time', fontsize=20)
        ax1.set_ylabel(r'Variation', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=18)  # Adjust the size of major ticks
        if beta == 0.115 or beta == 0.12 or beta == 0.15 or beta == 0.2:
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax1.yaxis.get_offset_text().set_fontsize(18)

        ax1.patch.set_alpha(0.8)
    ax1.legend(fontsize=18)
    print('idx I:', idx_max_I)
    print('idx I_new:', idx_max_I_new)

    ##### Second plot

    beta = 0.4
    mu = 0.1
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
    idx_sims_not_start = np.load(folder_simulation + f'idx_sim_not_start.npy')

    sim = 0

    if sim not in idx_sims_not_start:

        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy') / (N * avg_popPerNode)
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy') / (N * avg_popPerNode)
        node_new_NI_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy') / (N * avg_popPerNode)
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy') / (N * avg_popPerNode)
        node_NR_new_time = np.load(folder_simulation + f'sim_{sim}_new_R_time.npy') / (N * avg_popPerNode)
        if beta == 0.115 or beta == 0.12:
            node_new_NI_time = node_new_NI_time[:1000]
            node_NR_new_time = node_NR_new_time[:1000]
        # Sum to work at network level
        NI_time = node_NI_time.sum(axis=1)
        NI_new_time = node_new_NI_time.sum(axis=1)
        NR_new_time = node_NR_new_time.sum(axis=1)
        # Extract maximum values
        idx_max_I = np.argmax(NI_time)
        idx_max_I_new = np.argmax(NI_new_time)
        idx_max_Rnew = np.argmax(NR_new_time)
        T = np.load(folder_simulation + f'T.npy')
        if beta == 0.115 or beta == 0.12:
            T = 1000
        T_sim = np.linspace(0, T - 1, T)

        ax2.plot(T_sim, NI_new_time, color='#d40000', linewidth=1.5, label=r'$\Delta I(t)$')
        ax2.plot(T_sim, NR_new_time, color='#005522', linewidth=1.5, label=r'$\Delta R(t)$')

        ax2.set_xlabel('Time', fontsize=20)
        #ax2.set_ylabel(r'Variation', fontsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=18)  # Adjust the size of major ticks
        if beta == 0.115 or beta == 0.12 or beta == 0.15 or beta == 0.2:
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax2.patch.set_alpha(0.8)
        ax2.yaxis.get_offset_text().set_fontsize(18)

    ax2.legend(fontsize=18)
    print('idx I:', idx_max_I)
    print('idx I_new:', idx_max_I_new)

    ##### Third plot

    beta = 1.2
    mu = 0.1
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
    idx_sims_not_start = np.load(folder_simulation + f'idx_sim_not_start.npy')

    sim = 0

    if sim not in idx_sims_not_start:

        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy') / (N * avg_popPerNode)
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy') / (N * avg_popPerNode)
        node_new_NI_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy') / (N * avg_popPerNode)
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy') / (N * avg_popPerNode)
        node_NR_new_time = np.load(folder_simulation + f'sim_{sim}_new_R_time.npy') / (N * avg_popPerNode)
        if beta == 0.115 or beta == 0.12:
            node_new_NI_time = node_new_NI_time[:1000]
            node_NR_new_time = node_NR_new_time[:1000]
        # Sum to work at network level
        NI_time = node_NI_time.sum(axis=1)
        NI_new_time = node_new_NI_time.sum(axis=1)
        NR_new_time = node_NR_new_time.sum(axis=1)
        # Extract maximum values
        idx_max_I = np.argmax(NI_time)
        idx_max_I_new = np.argmax(NI_new_time)
        idx_max_Rnew = np.argmax(NR_new_time)
        T = np.load(folder_simulation + f'T.npy')
        if beta == 0.115 or beta == 0.12:
            T = 1000
        T_sim = np.linspace(0, T - 1, T)

        ax3.plot(T_sim, NI_new_time, color='#d40000', linewidth=1.5, label=r'$\Delta I(t)$')
        ax3.plot(T_sim, NR_new_time, color='#005522', linewidth=1.5, label=r'$\Delta R(t)$')
        ax3.yaxis.get_offset_text().set_fontsize(24)

        ax3.set_xlabel('Time', fontsize=20)
        #ax3.set_ylabel(r'Variation', fontsize=20)
        ax3.tick_params(axis='both', which='major', labelsize=18)  # Adjust the size of major ticks
        if beta == 0.115 or beta == 0.12 or beta == 0.15 or beta == 0.2:
            ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        ax3.patch.set_alpha(0.8)
    ax3.legend(fontsize=18)
    print('idx I:', idx_max_I)
    print('idx I_new:', idx_max_I_new)

    plt.tight_layout()
    plt.show()

if ana_max3 == 1:
    row = 30
    col = 30
    N = row * col
    beta_vals = [0.12, 0.4, 1.2]
    mu_vals = [0.1, 0.1, 0.1]
    choice_bool = 0
    c1= 0

    nbr_simulations = 1
    # Non confined case

    avg_popPerNode = 10**4
    f, ax = plt.subplots(figsize=(12, 8))
    for (beta, mu) in zip(beta_vals, mu_vals):
        i = 0
        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

        idx_sims_not_start = np.load(folder_simulation + f'idx_sim_not_start.npy')
        for sim in range(nbr_simulations):
            if sim not in idx_sims_not_start:
                node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
                node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')/(N*avg_popPerNode)
                node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')/(N*avg_popPerNode)
                node_new_NI_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')/(N*avg_popPerNode)
                node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')/(N*avg_popPerNode)
                node_NR_new_time = np.load(folder_simulation + f'sim_{sim}_new_R_time.npy')/(N*avg_popPerNode)
                if beta == 0.115 or beta == 0.12:
                    node_new_NI_time = node_new_NI_time[:1000]
                    node_NR_new_time = node_NR_new_time[:1000]

                # Sum to work at network level
                NI_time = node_NI_time.sum(axis=1)
                NI_new_time = node_new_NI_time.sum(axis=1)
                NR_new_time = node_NR_new_time.sum(axis=1)
                # Extract maximum values
                idx_max_I = np.argmax(NI_time)
                idx_max_I_new = np.argmax(NI_new_time)
                idx_max_Rnew = np.argmax(NR_new_time)

                T = np.load(folder_simulation + f'T.npy')
                if beta == 0.115 or beta == 0.12:
                    T = 1000
                T_sim = np.linspace(0, T - 1, T)

                plt.plot(T_sim, NI_new_time, color='#d40000', linewidth=1.5)#, label=r'$\langle\Delta I(t) \rangle$')
                plt.plot(T_sim, NR_new_time, color='#005522', linewidth=1.5)#, label=r'$\langle\Delta R(t) \rangle$')
        #if beta == 0.115 or beta == 0.12 or beta == 0.15 or beta == 0.2:
        #    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.xlabel('Time', fontsize = 26)
    plt.ylabel(r'Variation', fontsize = 26)
    ax.tick_params(axis='both', which='major', labelsize=26)  # Adjust the size of major ticks


    ax.patch.set_alpha(0.8)



    ax.yaxis.get_offset_text().set_fontsize(24)

    plt.legend(fontsize = 24)
    plt.show()