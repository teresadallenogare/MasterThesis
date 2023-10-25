"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 25 October 2023

--------------------------------------------------------------------

Functions to use in the analysis of data from:
- network topology
- SIR simulation
- TDA pipeline

"""

from functions_SIR_metapop import colorFader
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_SIR_timeseries(N_row, N_col, choice_bool, c1, beta, mu, bool_density, idx_sims, idx_nodes, T_sim, avg_pop_node ):
    """ Compute plot of number of individuals (or density) in the S, I, R state for all simulations or for a specific one.


    :param bool_density: [bool] if 0 : compute the number of individuals
                                if 1 : compute the density
    :param idx_sims : [list] index of simulations to include in the plot
    :param idx_nodes : [list] index of nodes to include in the plot
    :param avg_pop_node : [scalar] value of average population per node

    :return: plot of SIR timeseries
    """

    # ------------------------------------------------ Colors  -------------------------------------------------

    grad_gray = []
    grad_red = []
    grad_blue = []
    grad_green = []

    for x in range(N_row * N_col):
        #                                dark           light
        grad_gray.append(colorFader('#505050', '#EAE9E9', x / (N_row * N_col)))
        grad_red.append(colorFader('#E51C00', '#FCE0DC', x / (N_row * N_col)))
        grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x / (N_row * N_col)))
        grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x / (N_row * N_col)))

    datadir = os.getcwd()
    # Folder
    folder_simulation = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'

    for sim in idx_sims:
        # Load data
        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

        if bool_density == 0:
            vals_population_time = node_population_time
            vals_NS_time = node_NS_time
            vals_NI_time = node_NI_time
            vals_NR_time = node_NR_time
        elif bool_density == 1:
            vals_population_time = node_population_time / np.mean(node_population_time, axis = 0)
            vals_NS_time = node_NS_time / np.mean(node_population_time, axis = 0)
            vals_NI_time = node_NI_time / np.mean(node_population_time, axis=0)
            vals_NR_time = node_NR_time / np.mean(node_population_time, axis = 0)
        else:
            print('Wrong value of bool_density')
        first = True
        for node in idx_nodes:
            if first == True:
                plt.plot(T_sim, vals_population_time[:, node], color=grad_gray[node], label = 'Population' if bool_density == 0 else 'Density')
                plt.plot(T_sim, vals_NS_time[:, node], color=grad_blue[node], label = 'S' if bool_density == 0 else 'S density')
                plt.plot(T_sim, vals_NI_time[:, node], color=grad_red[node], label = 'I' if bool_density == 0 else 'I density')
                plt.plot(T_sim, vals_NR_time[:, node], color=grad_green[node], label = 'R' if bool_density == 0 else 'R density')
                first = False
            else:
                plt.plot(T_sim, vals_population_time[:, node], color=grad_gray[node])
                plt.plot(T_sim, vals_NS_time[:, node], color=grad_blue[node])
                plt.plot(T_sim, vals_NI_time[:, node], color=grad_red[node])
                plt.plot(T_sim, vals_NR_time[:, node], color=grad_green[node])
    plt.xlabel('Timestep')
    plt.axhline(y = avg_pop_node if bool_density == 0 else 1, color='black', linestyle='--', label = 'Average population ' if bool_density == 0 else 'Average density')
    plt.ylabel('Node population' if bool_density == 0 else 'Node density')
    plt.legend()
    plt.show()
