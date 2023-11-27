"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 23 November 2023

--------------------------------------------------------------------

Analysis of SIR data from simulations

"""

from functions_SIR_metapop_v1 import *
from functions_output_v1 import write_simulation_file
from functions_visualization_v1 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

datadir = os.getcwd()
plt.figure(figsize=(8, 8))


fixedR0 = 0
fixed_mu = 0
heatmap = 1

row = 50
col = 50

choice_bool = 0
c1 = 1

sim = 0

idx_node = 0

# Infection and recovery rate
beta_vals_3_5_10 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2, 0.23, 0.24, 0.3, 0.4, 0.6, 0.8, 0.345, 0.36, 0.45, 0.6, 0.9, 1.2]
mu_vals_3_5_10 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

beta_vals_30_50 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'

avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')

# ------------------------------------------------ Colors  -------------------------------------------------
grad_gray = []
grad_red = []
grad_blue = []
grad_green = []

for x in range(3):
    #                                dark           light
    grad_gray.append(colorFader('#505050', '#EAE9E9', x/3))
    grad_red.append(colorFader('#E51C00', '#FCE0DC', x/3))
    grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x/3))
    grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x/3))

### Fixed the dimensions of the lattice (consider idx_node = 0) and fixed the R0, show the different dynamics of
# the number of infected as a function of beta, mu

if fixedR0 == 1:
    beta_vals_R0 = [0.4, 0.8, 1.2]
    mu_vals_R0 = [0.1, 0.2, 0.3]

    i = 0
    for beta, mu in zip(beta_vals_R0, mu_vals_R0):
        T = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/T.npy')
        print('row:', row, 'col:', col, 'choice_bool:', choice_bool, 'c1:', c1, 'beta:', beta, 'mu:', mu, 'T:', T)
        T_sim = np.linspace(0, T - 1, T)

        node_NS_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NI_time.npy')
        node_NR_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NR_time.npy')

        density_NS_time = node_NS_time / avg_popPerNode
        density_NI_time = node_NI_time / avg_popPerNode
        density_NR_time = node_NR_time / avg_popPerNode

        plt.plot(T_sim[:120], density_NI_time[:120, idx_node], color = grad_red[i], label = f'beta = {beta}, mu = {mu}')

        i = i + 1
    plt.xlabel('t')
    plt.ylabel(r'$\rho_{I,0}(t)$', rotation = 0)
    plt.legend()
    plt.show()

### Fixed the dimension of the lattice (consider idx_node = 0) and fixed the mu parameter, show the different dynamics of
# the number of infected as a function of R0
#if fixed_mu == 1:


### Heatmap

if heatmap == 1:
    beta = 0.2
    mu = 0.1
    T = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/T.npy')
    print('row:', row, 'col:', col, 'choice_bool:', choice_bool, 'c1:', c1, 'beta:', beta, 'mu:', mu, 'T:', T)
    T_sim = np.linspace(0, T - 1, T)

    heatmap_time(row, col, choice_bool, c1, beta, mu, sim)



