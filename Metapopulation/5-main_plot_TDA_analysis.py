"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 21 October 2023

--------------------------------------------------------------------
First analysis concerning the application of TDA to data obtained from the simulations

"""
from functions_SIR_metapop import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import ripser

from ripser import Rips

from persim import plot_diagrams
from persim.persistent_entropy import *
from scipy import stats

import matplotlib.animation as animation


# ------------------------------------------------ Parameters  -------------------------------------------------

N_row = [30]# ,30]
N_col = [30]# ,30]

choice_bool = 0
datadir = os.getcwd()
c1 = 0  # for now
beta_outbreak = [0.3]#, 0.4, 0.9]
beta_no_outbreak = [0.35, 0.75]
mu_outbreak = [0.1]#, 0.2, 0.1]
mu_no_outbreak = [0.3, 0.6]

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


########## Heatmap evolution ######################

def animate(t, img, grid, dict_vals, node_pop, N):
    mtrx_t = dict_vals[t]
    x_nodes = mtrx_t[:, 0]
    y_nodes = mtrx_t[:, 1]
    nbr_I_nodes = mtrx_t[:, 3]
    avg_pop_node = [np.mean(node_pop[:, idx_node] ) for idx_node in range(N)]
    # do that every row is averaged over the temporal average of the node population (of that node)
    density_I_nodes = nbr_I_nodes/ avg_pop_node


    density_I_nodes = nbr_I_nodes
    idx_row = 0
    for i, j in zip(x_nodes, y_nodes):
        grid[int(i), int(j)] = nbr_I_nodes[idx_row]
        #grid[int(i), int(j)] = density_I_nodes[idx_row]
        idx_row += 1
    img.set_data(grid)
    return img,

###### OUTBREAK CASE ######

# Plot heatmap with temporal evolution of the number of infected per node
for row, col in zip(N_row, N_col):
    N = row * col
    for beta, mu in zip(beta_outbreak, mu_outbreak):
        folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'
        folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
        T = np.load(folder_simulation + 'T.npy')
        #for sim in range(1):
        sim = 1
        # Initialize grid for later visualization at the beginning of every new simulation! That is my initial state
        grid = np.zeros(shape=(row, col))
        # Load dictionary that contains the information of every node (x_node, y_node, #S, #I, #R) at each timestep
        dict_outbreak = pickle.load(open(folder_dict + f'dict_data-{row}x{col}-sim{sim}.pickle', 'rb'))
        dict_outbreak_values = list(dict_outbreak.values())
        max_nbr_I = np.max(dict_outbreak_values)

        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        # Setup animation
        fig, ax = plt.subplots()
        img = ax.imshow(grid,  vmin = 0, vmax = max_nbr_I/3, cmap = 'coolwarm')
        ani = animation.FuncAnimation(fig, animate, fargs=(img, grid, dict_outbreak_values, node_population_time, N,  ), frames=dict_outbreak.keys(), blit=True)
        #writergif = animation.PillowWriter(fps=10)
        ani.save('animation.gif')
        plt.show()
        print('Done!')
# Do the code before better and also account for the value vmax properly.
# Look at differences between the outbreak and non outbreak cases


###### NO OUTBREAK CASE ######

# Plot heatmap with temporal evolution of the number of infected per node
for row, col in zip(N_row, N_col):
    N = row * col
    for beta, mu in zip(beta_no_outbreak, mu_no_outbreak):
        folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'
        folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
        T = np.load(folder_simulation + 'T.npy')
        #for sim in range(1):
        sim = 0
        # Initialize grid for later visualization at the beginning of every new simulation! That is my initial state
        grid = np.zeros(shape=(row, col))
        # Load dictionary that contains the information of every node (x_node, y_node, #S, #I, #R) at each timestep
        dict_no_outbreak = pickle.load(open(folder_dict + f'dict_data-{row}x{col}-sim{sim}.pickle', 'rb'))
        dict_no_outbreak_values = list(dict_no_outbreak.values())
        max_nbr_I = np.max(dict_no_outbreak_values)
        print(max_nbr_I)
        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        # Setup animation
        fig, ax = plt.subplots()
        img = ax.imshow(grid,  vmin = 0, vmax = max_nbr_I/10, cmap = 'coolwarm')
        ani = animation.FuncAnimation(fig, animate, fargs=(img, grid, dict_no_outbreak_values, node_population_time, N,  ), frames=dict_no_outbreak.keys(), blit=True)
        #writergif = animation.PillowWriter(fps=10)
        ani.save('animation.gif')
        plt.show()
        print('Done!')

########## Persistent entropy measure #############
# --------------------------------------------- Outbreak case ---------------------------------------------

for row, col in zip(N_row, N_col):

    for beta, mu in zip(beta_outbreak, mu_outbreak):
        fig, ax = plt.subplots()
        folder_analysis = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis/'
        folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
        folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'

        for sim in range(nbr_simulations):
            dict_outbreak = pickle.load(open(folder_dict + f'dict_data-{row}x{col}-sim{sim}.pickle', 'rb'))
            entropy_H0_outbreak = np.load(folder_analysis + f'entropy_H0_outbreak-sim{sim}.npy')
            entropy_H1_outbreak = np.load(folder_analysis + f'entropy_H1_outbreak-sim{sim}.npy')
            # Outbreak
            x_outbreak = range(0, len(dict_outbreak.keys()))
            y_H0_outbreak = entropy_H0_outbreak
            y_H1_outbreak = entropy_H1_outbreak

            ax.plot(x_outbreak, y_H0_outbreak, label=f'PE at H0, sim {sim}', color = grad_blue[sim])
            ax.plot(x_outbreak, y_H1_outbreak, label=f'PE at H1, sim {sim}', color = grad_blue[sim])
        ax.set_xlabel("Time")
        ax.set_ylabel("Persistent Entropy")
        ax.set_title(f"Persistent entropy for outbreak data, {N_row}x{N_col}, beta = {beta}, mu = {mu} ")
        #ax.legend()
        #plt.show()


# --------------------------------------------- No outbreak case ---------------------------------------------

for row, col in zip(N_row, N_col):
    for beta, mu in zip(beta_no_outbreak, mu_no_outbreak):
        fig, ax = plt.subplots()
        folder_analysis = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis/'
        folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
        folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'

        for sim in range(nbr_simulations):
            # Saved with same name because in folder with no-outbreak values of beta and mu
            dict_no_outbreak = pickle.load(open(folder_dict + f'dict_data-{row}x{col}-sim{sim}.pickle', 'rb'))
            entropy_H0_no_outbreak = np.load(folder_analysis + f'entropy_H0_no_outbreak-sim{sim}.npy')
            entropy_H1_no_outbreak = np.load(folder_analysis + f'entropy_H1_no_outbreak-sim{sim}.npy')
            # No outbreak
            x_no_outbreak = range(0, len(dict_no_outbreak.keys()))
            y_H0_no_outbreak = entropy_H0_no_outbreak
            y_H1_no_outbreak = entropy_H1_no_outbreak

            ax.plot(x_no_outbreak, y_H0_no_outbreak, label=f'PE at H0, sim {sim}', color = grad_green[sim])
            ax.plot(x_no_outbreak, y_H1_no_outbreak, label=f'PE at H1, sim {sim}', color = grad_green[sim])
        ax.set_xlabel("Time")
        ax.set_ylabel("Persistent Entropy")
        ax.set_title(f"Persistent entropy for no outbreak data, {N_row}x{N_col}, beta = {beta}, mu = {mu} ")
        #ax.legend()
        #plt.show()





