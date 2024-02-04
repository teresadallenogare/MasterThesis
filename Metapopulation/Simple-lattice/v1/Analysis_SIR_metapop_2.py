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
ana_attack_rate = 1

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






