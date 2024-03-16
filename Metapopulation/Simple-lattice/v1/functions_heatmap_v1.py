"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 20 November 2023

--------------------------------------------------------------------

Functions to plot heatmaps

"""
import numpy as np
import matplotlib.pyplot as plt
import os

from IPython import display
import matplotlib.animation as animation
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def animate_avg_infecteds(t, img, grid, dict_vals, dict_norm_vals):
    mtrx_t = dict_vals[t]
    mtrx_norm_t = dict_norm_vals[t]
    # Extract node positions from the non-normalized dictionary
    x_nodes = mtrx_t[:, 0]
    y_nodes = mtrx_t[:, 1]
    # Extract the density of infected from the normalized dictionary
    density_I_nodes = mtrx_norm_t[:, 3]
    # print('t: ', t)
    # print('dI: ', density_I_nodes)
    idx_row = 0
    for i, j in zip(x_nodes, y_nodes):
        # grid[int(i), int(j)] = nbr_I_nodes[idx_row]
        grid[int(i), int(j)] = density_I_nodes[idx_row]
        idx_row += 1
    img.set_data(grid)
    return img,

def animate_avg_new_infecteds(t, img, grid, dict_vals, dict_norm_vals):
    mtrx_t = dict_vals[t]
    mtrx_norm_t = dict_norm_vals[t]
    # Extract node positions from the non-normalized dictionary
    x_nodes = mtrx_t[:, 0]
    y_nodes = mtrx_t[:, 1]
    # Extract the density of infected from the normalized dictionary
    density_I_nodes = mtrx_norm_t[:, 5]
    # print('t: ', t)
    # print('dI: ', density_I_nodes)
    idx_row = 0
    for i, j in zip(x_nodes, y_nodes):
        # grid[int(i), int(j)] = nbr_I_nodes[idx_row]
        grid[int(i), int(j)] = density_I_nodes[idx_row]
        idx_row += 1
    img.set_data(grid)
    return img,


def heatmap_time_avg_infecteds(N_row, N_col, choice_bool, c1, beta, mu, bool_static, bool_Inew, time):
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

    datadir = os.getcwd()
    sim = 0
    folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'

    folder_animations = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Animations/'

    # I extract the position of nodes in the non-normalized dictionary and the value of density in the normalized one
    # Initialize grid for later visualization at the beginning of every new simulation! That is my initial state
    grid = np.zeros(shape=(N_row, N_col))
    # Load dictionary that contains the information of every node (x_node, y_node, #S, #I, #R) at each timestep
    dict_load = pickle.load(open(folder_dict_noNorm + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    dict_load_values = list(dict_load.values())
    # Load normalized dictionary to have the density of individuals
    dict_load_normalized = pickle.load(
        open(folder_dict_normHand + f'dict_avg_data_beta{beta}-mu{mu}.pickle', 'rb'))
    dict_load_normalized_values = list(dict_load_normalized.values())
    # Brute force : maximum value of density of I in whole dictionary
    max_densityI_time = []
    max_densityInew_time = []
    # Determination of the maximum density of infected
    for t in dict_load.keys():
        mtrx_t_normalized = dict_load_normalized[t]
        density_I = mtrx_t_normalized[:, 3]
        density_Inew = mtrx_t_normalized[:, 5]
        max_densityI_time.append(max(density_I))
        max_densityInew_time.append(max(density_Inew))
    max_densityI_time = np.array(max_densityI_time)
    max_densityInew_time = np.array(max_densityInew_time)
    max_densityI = max(max_densityI_time)
    max_densityInew = max(max_densityInew_time)
    print('max-densityI', max_densityI)
    if bool_static == 0:
        # Setup animation
        Writer = animation.FFMpegWriter(fps=1)

        fig, ax = plt.subplots()
        if bool_Inew == 0:
            img = ax.imshow(grid, vmin=0, vmax=max_densityI, cmap='coolwarm')
        else:
            img = ax.imshow(grid, vmin=0, vmax=max_densityInew, cmap='coolwarm')
        ax.invert_yaxis()
        fig.colorbar(img, cmap='coolwarm')
        ax.set_xlabel('Node index')
        ax.set_ylabel('Node index')
        ax.set_title(f'Heatmap {N_row}x{N_col} : beta = {beta}, mu = {mu}, sim = {sim}')
        ax.grid(True, linestyle='-', linewidth=0.01, alpha=0.1, color='gray')
        if bool_Inew == 0:
            ani = animation.FuncAnimation(fig, animate_avg_infecteds, fargs=(img, grid, dict_load_values, dict_load_normalized_values,),
                                        frames=dict_load.keys())
        else:
            ani = animation.FuncAnimation(fig, animate_avg_new_infecteds, fargs=(img, grid, dict_load_values, dict_load_normalized_values,),
                                        frames=dict_load.keys())
        # converting to a html5 video
        video = ani.to_html5_video()
        if bool_Inew == 0:
            ani.save(folder_animations + f'animation-avgData-beta{beta}-mu{mu}.mp4', writer=Writer)
        else:
            ani.save(folder_animations + f'animation-avgData-Inew-beta{beta}-mu{mu}.mp4', writer=Writer)
        # embedding for the video
        html = display.HTML(video)
        # draw the animation
        display.display(html)
        plt.close()
        plt.show()
        print('Done!')
    elif bool_static == 1:
        t = time
        mtrx_t_normalized = dict_load_normalized[t]
        density_I = mtrx_t_normalized[:, 3]
        density_Inew = mtrx_t_normalized[:, 5]
        density_I_grid = density_I.reshape((30,30))
        density_Inew_grid = density_Inew.reshape((30,30))
        fig, ax = plt.subplots(figsize = (9,7))
        if bool_Inew == 0:
            img = ax.imshow(density_I_grid, vmin=0, vmax=max_densityI, cmap='coolwarm')
        else:
            img = ax.imshow(density_Inew_grid, vmin=0, vmax=max_densityInew, cmap='coolwarm')
        ax.invert_yaxis()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(img, cmap='coolwarm', cax=cax)
        cbar.set_label(r'$I/\langle n\rangle$', fontsize = 20)
        cbar.ax.tick_params(labelsize=20)
        ax.set_xlabel('Node index', fontsize = 20)
        ax.set_ylabel('Node index', fontsize = 20)
        #ax.set_title(f'Heatmap {N_row}x{N_col} : beta = {beta}, mu = {mu}, sim = {sim}')
        ax.grid(True, linestyle='-', linewidth=0.01, alpha=0.1, color='white')
        # Minor ticks
        ax.set_xticks(np.arange(-0.5, 29.5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 29.5, 1), minor=True)
        ax.grid(which='minor', color='whitesmoke', linestyle='-', linewidth=0.4)

        ax.tick_params(axis='both', which='major', labelsize=20)

        plt.tight_layout()
        plt.show()


