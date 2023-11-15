"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 10 November 2023

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

def plot_static_network(G, pop_nodes, dict_nodes, weight, N_row, N_col, choice_bool, c1):
    """ Plot a static version of the network, in which the size of nodes is proportional to the node population

    :param G: [networkx.class] graph structure from networkx
    :param pos: [list] position of nodes
    :param pop_nodes: [list] population inside every node
    :param dict_nodes: [dict] dictionary of nodes attributing each key a position
    :param weight: [list] weight to attribute to edges
    :param state_nodes: [list] state of the node
    """
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'

    size_map = [pop_nodes[i] / 10. for i in G.nodes]
    nx.draw_networkx_nodes(G, pos=dict_nodes, node_color = '#B7C8C4', edgecolors = '#374845', linewidths= 1.5, node_size=size_map)
    nx.draw_networkx_edges(G, pos=dict_nodes, width=weight, arrows = True, min_source_margin = 20,
                           min_target_margin = 20, connectionstyle= "arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos=dict_nodes, font_size=10)
    plt.savefig(folder_topology + f'net-topol.pdf')
    plt.show()

def plot_TransitionMatrix(T, N_row, N_col, choice_bool, c1):
    """ Plot the transition matrix both with annotations and without annotations

    """
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    T_df = pd.DataFrame(T)
    labels = T_df.applymap(lambda v: str(np.round(v, 2)) if v != 0 else '')
    palette = sns.cubehelix_palette(n_colors = 100, start=2,  dark = 0.1, reverse = True)
    annotation_list = [True, False] if N_row == 3 else [False]
    for annotation in annotation_list:
        if annotation == True:
            ax = sns.heatmap(T, linewidth=0, square = True,  annot = labels, fmt = '', cmap= palette, cbar_kws={'label': 'weight'})
        else:
            ax = sns.heatmap(T, linewidth=0, square=True, annot=annotation, fmt='', cmap= palette, cbar_kws={'label': 'weight'})

        ax.set_xlabel("Node index", fontsize = 12)
        ax.set_ylabel("Node index", fontsize=12)

        plt.savefig(folder_topology + f'TransMat_annot-{annotation}.pdf')
        plt.show()

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
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

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