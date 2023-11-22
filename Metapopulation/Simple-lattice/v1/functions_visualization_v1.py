"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 20 November 2023

--------------------------------------------------------------------

Functions to plot and visualize

"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import os
from palettable.cubehelix import Cubehelix
import pandas as pd
import seaborn as sns
from IPython import display
import matplotlib.animation as animation
import pickle

def colorFader(c1 ,c2, mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    """ Color gradient between c1 - darker - and c2 - lighter.

    :param c1: [scalar] dark value of color
    :param c2: [scalar] light value of color
    :param mix: 0
    :return:
    """
    c1 = np.array(matplotlib.colors.to_rgb(c1))
    c2 = np.array(matplotlib.colors.to_rgb(c2))

    return matplotlib.colors.to_hex((1-mix)*c1 + mix*c2)

#######################################################################################################################
#                                                                                                                     #
#                                            NETWORK PROPERTIES                                                       #
#                                                                                                                     #
#######################################################################################################################

def plot_static_network(G, pop_nodes, dict_nodes, weight, N_row, N_col, choice_bool, c1):
    """ Plot a static version of the network, in which the size of nodes is proportional to the node population

    :param G: [networkx.class] graph structure from networkx
    :param pos: [list] position of nodes
    :param pop_nodes: [list] population inside every node
    :param dict_nodes: [dict] dictionary of nodes attributing each key a position
    :param weight: [list] weight to attribute to edges
    :param state_nodes: [list] state of the node
    """
    plt.figure(figsize=(8, 8))
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'

    size_map = [pop_nodes[i] / 10. for i in G.nodes]
    nx.draw_networkx_nodes(G, pos=dict_nodes, node_color = '#B7C8C4', edgecolors = '#374845', linewidths= 1.5, node_size=size_map)
    nx.draw_networkx_edges(G, pos=dict_nodes, width=weight, arrows = True, min_source_margin = 20,
                           min_target_margin = 20, connectionstyle= "arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos=dict_nodes, font_size=10)
    #plt.savefig(folder_topology + f'net-topol.pdf')
    plt.show()

def plot_TransitionMatrix(T, N_row, N_col, choice_bool, c1):
    """ Plot the transition matrix both with annotations and without annotations

    :param T: [matrix] transition matrix
    :param N_row, N_col: [scalar] number of rows and columns of the network
    :param choice_bool: [bool] 0 if uniform population, 1 if non-uniform population
    :param c1: [scalar] strenght of the connection
    """
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    T_df = pd.DataFrame(T)
    labels = T_df.applymap(lambda v: str(np.round(v, 2)) if v != 0 else '')
    palette = sns.cubehelix_palette(n_colors = 100, start=2,  dark = 0.1, reverse = True)
    annotation_list = [True, False] if N_row == 3 else [False]
    for annotation in annotation_list:
        if annotation == True:
            #plt.figure(figsize=(9, 8))
            ax = sns.heatmap(T, linewidth=0, square = True,  annot = labels, fmt = '', cmap= palette, cbar_kws={'label': 'weight'})
        else:
            #plt.figure(figsize=(9, 8))
            ax = sns.heatmap(T, linewidth=0, square=True, annot=annotation, fmt='', cmap= palette, cbar_kws={'label': 'weight'})

        ax.set_xlabel("Node index", fontsize = 12)
        ax.set_ylabel("Node index", fontsize=12)

        plt.savefig(folder_topology + f'TransMat_annot-{annotation}.pdf')
        plt.show()

def plot_degree_distribution(N_row, N_col, choice_bool, c1, k, pk, avg_k):
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    plt.bar(k, pk, color='#6F918A')
    plt.axvline(x=avg_k, color='k', label=r'$\langle k_{in} \rangle$', linestyle='--')
    plt.xlabel('$k_{in}$')
    plt.ylabel('$p_k$')
    # plt.title(f'Degree distribution of {row}x{col} network with choice_bool: {choice_bool}, c1: {c1}')
    plt.legend()
    #plt.savefig(folder_topology + f'degree_distribution.pdf')
    plt.show()


def plot_distance_distribution(N_row, N_col, choice_bool, c1, d, pd, avg_d):
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    plt.plot(d, pd, '-o', color='#93ACA7', mfc = '#536C67')
    plt.axvline(x=avg_d, color='k', label=r'$\langle d \rangle$', linestyle='--')
    plt.xlabel('$d$')
    plt.ylabel('$p_d$')
    # plt.title(f'Degree distribution of {row}x{col} network with choice_bool: {choice_bool}, c1: {c1}')
    plt.legend()
    plt.savefig(folder_topology + f'distance_distribution.pdf')
    plt.show()
#######################################################################################################################
#                                                                                                                     #
#                                            SIR SIMULATIONS                                                          #
#                                                                                                                     #
#######################################################################################################################
def plot_SIR_timeseries(N_row, N_col, choice_bool, c1, beta, mu, idx_sims, idx_nodes, T_sim, avg_pop_node, avg_pop_Nfix, avg_pop_Others):
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
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

    for sim in idx_sims:
        # Load data
        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

        first = True
        for node in idx_nodes:
            if first == True:
                plt.plot(T_sim, node_population_time[:, node], color=grad_gray[node], label = 'Population')
                plt.plot(T_sim, node_NS_time[:, node], color=grad_blue[node], label = 'S')
                plt.plot(T_sim, node_NI_time[:, node], color=grad_red[node], label = 'I')
                plt.plot(T_sim, node_NR_time[:, node], color=grad_green[node], label = 'R')
                first = False
            else:
                plt.plot(T_sim, node_population_time[:, node], color=grad_gray[node])
                plt.plot(T_sim, node_NS_time[:, node], color=grad_blue[node])
                plt.plot(T_sim, node_NI_time[:, node], color=grad_red[node])
                plt.plot(T_sim, node_NR_time[:, node], color=grad_green[node])
        plt.xlabel('Timestep')
        plt.ylabel('Node population')
        if choice_bool == 0:
            plt.axhline(y = avg_pop_node, color='black', linestyle='--', label = 'Average population ')
        elif choice_bool == 1:
            plt.axhline(y=avg_pop_Others, color='black', linestyle='--', label='Average population ')
            plt.axhline(y=avg_pop_Nfix, color='black', linestyle='--')
        else:
            print('Wrong choice_bool')
        plt.legend()
        plt.show()


def animate(t, img, grid, dict_vals, dict_norm_vals):
    mtrx_t = dict_vals[t]
    mtrx_norm_t = dict_norm_vals[t]
    # Extract node positions from the non-normalized dictionary
    x_nodes = mtrx_t[:, 0]
    y_nodes = mtrx_t[:, 1]
    # Extract the density of infected from the normalized dictionary
    density_I_nodes = mtrx_norm_t[:, 3]
    #print('t: ', t)
    #print('dI: ', density_I_nodes)
    idx_row = 0
    for i, j in zip(x_nodes, y_nodes):
        # grid[int(i), int(j)] = nbr_I_nodes[idx_row]
        grid[int(i), int(j)] = density_I_nodes[idx_row]
        idx_row += 1
    img.set_data(grid)
    return img,


def heatmap_time(N_row, N_col, choice_bool, c1, beta, mu, sim):
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
    folder_dict = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/No-normalized/'
    folder_dict_normalized = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/Normalized/'
    folder_densities = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Densities/'
    folder_animations = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Animations/'

    # I extract the position of nodes in the non-normalized dictionary and the value of density in the normalized one
    # Initialize grid for later visualization at the beginning of every new simulation! That is my initial state
    grid = np.zeros(shape=(N_row, N_col))
    # Load dictionary that contains the information of every node (x_node, y_node, #S, #I, #R) at each timestep
    dict_load = pickle.load(open(folder_dict + f'dict_data-{N_row}x{N_col}-sim{sim}.pickle', 'rb'))
    dict_load_values = list(dict_load.values())
    # Load normalized dictionary to have the density of individuals
    dict_load_normalized = pickle.load(open(folder_dict_normalized + f'dict_data_normalized-{N_row}x{N_col}-sim{sim}.pickle', 'rb'))
    dict_load_normalized_values = list(dict_load_normalized.values())
    # Brute force : maximum value of density of I in whole dictionary
    max_densityI_time = []
    f = open(folder_densities + f'densities-sim{sim}.txt', 'w')

    # Determination of the maximum density of infected
    for t in dict_load.keys():
        mtrx_t_normalized = dict_load_normalized[t]
        density_S = mtrx_t_normalized[:, 2]
        density_I = mtrx_t_normalized[:, 3]
        density_R = mtrx_t_normalized[:, 4]
        f.write('--------------------------------------------------------\n')
        f.write(f't: {t}\n')
        f.write(f'density_S: {density_S}\n\n')
        f.write(f'density_I: {density_I}\n\n')
        f.write(f'density_R: {density_R}\n\n\n')
        max_densityI_time.append(max(density_I))
    f.close()
    max_densityI_time = np.array(max_densityI_time)
    max_densityI = max(max_densityI_time)
    print('max-densityI', max_densityI)

    # Setup animation
    Writer = animation.FFMpegWriter(fps=1)

    fig, ax = plt.subplots()
    img = ax.imshow(grid, vmin=0, vmax=max_densityI, cmap='coolwarm')
    fig.colorbar(img, cmap='coolwarm')
    ax.set_xlabel('Node index')
    ax.set_ylabel('Node index')
    ax.set_title(f'Heatmap {N_row}x{N_col} : beta = {beta}, mu = {mu}, sim = {sim}')
    ani = animation.FuncAnimation(fig, animate, fargs=(img, grid, dict_load_values, dict_load_normalized_values, ),
                                  frames= dict_load.keys())
    # converting to a html5 video
    video = ani.to_html5_video()

    ani.save(folder_animations+f'animation-sim{sim}.mp4', writer=Writer)
    # embedding for the video
    html = display.HTML(video)
    # draw the animation
    display.display(html)
    plt.close()
    plt.show()
    print('Done!')


