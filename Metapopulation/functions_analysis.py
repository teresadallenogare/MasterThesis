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

# ---------------------------------------- Simulation SIR analysis ----------------------------------------
def plot_SIR_timeseries(N_row, N_col, choice_bool, c1, beta, mu, bool_density, idx_sims, idx_nodes, T_sim, avg_pop_node):
    """ Compute plot of number of individuals (or density) in the S, I, R state for all simulations or for a specific one.
    :param N_row: [scalar] number of rows of the lattice
    :param N_col: [scalar] number of columns of the lattice
    :param choice_bool: [bool] if 0: lattice is uniform populated
                               if 1: lattice has hubs of people in certain nodes
    :param c1: [scalar] accounts for the importance of self loops
    :param T_sim: [array] array with timesteps of the simulation

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

def mean_stdDev_repetitions(N_row, N_col, choice_bool, c1, T, beta, mu, bool_density, nbr_repetitions):
    """ Compute the mean and std deviation over repeated simulations with the same topology and parameters

    :param N_row: [scalar] number of rows of the lattice
    :param N_col: [scalar] number of columns of the lattice
    :param choice_bool: [bool] if 0: lattice is uniform populated
                               if 1: lattice has hubs of people in certain nodes
    :param c1: [scalar] accounts for the importance of self loops
    :param T: [scalar] length of the simulation
    :param bool_density: [bool] if 0 : compute the number of individuals
                                if 1 : compute the density
    :param nbr_repetitions: [scalar] number of repeated simulations given fixed set of parameters and topology

    :return: mean value and standard deviations of the repeated simulations for S, I and R states.
    """

    N = N_row * N_col

    datadir = os.getcwd()

    folder_simulation = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'


    node_population_time_repeat = np.zeros(shape=(T + 1, N, nbr_repetitions))
    node_NS_time_repeat = np.zeros(shape=(T + 1, N, nbr_repetitions))
    node_NI_time_repeat = np.zeros(shape=(T + 1, N, nbr_repetitions))
    node_NR_time_repeat = np.zeros(shape=(T + 1, N, nbr_repetitions))

    # 3D matrix that stores repetitions along axis = 2
    # To see repetition k : node_NI_time_repeat[:,:,k]
    for sim in range(nbr_repetitions):
        # Load data
        new_I_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')
        # ADD NODE STATE

        #  Store data in 3D matrix
        node_population_time_repeat[:, :, sim] = node_population_time
        node_NS_time_repeat[:, :, sim] = node_NS_time
        node_NI_time_repeat[:, :, sim] = node_NI_time
        node_NR_time_repeat[:, :, sim] = node_NR_time

    if bool_density == 0:
        vals_population_time_repeat = node_population_time_repeat
        vals_NS_time_repeat = node_NS_time_repeat
        vals_NI_time_repeat = node_NI_time_repeat
        vals_NR_time_repeat = node_NR_time_repeat

    elif bool_density == 1:
        vals_NS_time_repeat = node_NS_time_repeat / node_population_time_repeat
        vals_NI_time_repeat = node_NI_time_repeat / node_population_time_repeat
        vals_NR_time_repeat = node_NR_time_repeat / node_population_time_repeat

    else:
        print('Wrong value of bool density')


    # Mean value and stdDeviation over repetitions
    mean_vals_S_time = np.mean(vals_NS_time_repeat, axis=2)
    mean_vals_I_time = np.mean(vals_NI_time_repeat, axis=2)
    mean_vals_R_time = np.mean(vals_NR_time_repeat, axis=2)
    stdDev_vals_S_time = np.std(vals_NS_time_repeat, axis=2, ddof=1)
    stdDev_vals_I_time = np.std(vals_NI_time_repeat, axis=2, ddof=1)
    stdDev_vals_R_time = np.std(vals_NR_time_repeat, axis=2, ddof=1)

    return [mean_vals_S_time, mean_vals_I_time, mean_vals_R_time,stdDev_vals_S_time, stdDev_vals_I_time, stdDev_vals_R_time,
            vals_NS_time_repeat[0,0,0], vals_NI_time_repeat[0,0,0], vals_NR_time_repeat[0,0,0]]

def plot_phase_space(N_row, N_col, choice_bool, c1, beta, mu, sim):
    """ Plot data in space of variables

    :param N_row: [scalar] number of rows of the lattice
    :param N_col: [scalar] number of columns of the lattice
    :param choice_bool: [bool] if 0: lattice is uniform populated
                               if 1: lattice has hubs of people in certain nodes
    :param c1: [scalar] accounts for the importance of self loops
    :param T: [scalar] length of the simulation
    :param beta: [scalar] infection rate
    :param mu: [scalar] recovery rate
    :param sim: [scalar] index of the simulation

    :return: plot of the node's states in the space of variables
    """
    N = N_row * N_col

    datadir = os.getcwd()

    folder_simulation = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
    # Load data
    new_I_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
    node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
    node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
    node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
    node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')
    # ADD NODE STATE
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    color_map = plt.get_cmap('spring')
    for idx_node in range(N):
        x = node_NS_time[:, idx_node]
        y = node_NI_time[:, idx_node]
        z = node_NR_time[:, idx_node]
        sc = ax.scatter3D(x, y, z)

    ax.set_xlabel('S')
    ax.set_ylabel('I')
    ax.set_zlabel('R')
    ax.set_title(f'Network {N_row}x{N_col}, beta: {beta}, mu: {mu}')

    plt.show()