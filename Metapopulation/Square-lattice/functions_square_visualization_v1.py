"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 27 November 2023

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
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_squareLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'

    size_map = [pop_nodes[i] / 10. for i in G.nodes]
    nx.draw_networkx_nodes(G, pos=dict_nodes, node_color = '#B7C8C4', edgecolors = '#374845', linewidths= 1.5, node_size=size_map)
    nx.draw_networkx_edges(G, pos=dict_nodes, width=weight, arrows = True, min_source_margin = 20,
                           min_target_margin = 20, connectionstyle= "arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos=dict_nodes, font_size=12)
   #plt.savefig(folder_topology + f'net-topol.pdf')
    plt.show()

def plot_TransitionMatrix(T, N_row, N_col, choice_bool, c1):
    """ Plot the transition matrix both with annotations and without annotations

    """
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_squareLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
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

def plot_degree_distribution(N_row, N_col, choice_bool, c1, k, pk, avg_k):
    """ Plot degree distribution with Poisson fit

    :param N_row: [scalar] number of rows
    :param N_col: [scalar] number of columns
    :param choice_bool: [bool] 0 if uniform population, 1 if non-uniform population
    :param c1: [scalar] strength of the connection
    :param k: [array] values of degrees
    :param pk: [array] probability distribution of degrees
    :param avg_k: [scalar] average degree
    :param Poisson_funct: [function] fitting function
    :param param: [scalar] parameter of the fitting function
    """

    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_squareLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    plt.bar(k, pk, color='#6F918A', label = 'Data')
    plt.axvline(x=avg_k, color='k', label=r'$\langle k_{in} \rangle$', linestyle='--')
    plt.xlabel('$k_{in}$')
    plt.ylabel('$p_k$')
    # plt.title(f'Degree distribution of {row}x{col} network with choice_bool: {choice_bool}, c1: {c1}')
    plt.legend()
    plt.savefig(folder_topology + f'degree_distribution.pdf')
    plt.show()


def plot_distance_distribution(N_row, N_col, choice_bool, c1, d, pd, avg_d):
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_squareLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    plt.plot(d, pd, '-o', color='#93ACA7', mfc = '#536C67')
    plt.axvline(x=avg_d, color='k', label=r'$\langle d \rangle$', linestyle='--')
    plt.xlabel('$d$')
    plt.ylabel('$p_d$')
    # plt.title(f'Degree distribution of {row}x{col} network with choice_bool: {choice_bool}, c1: {c1}')
    plt.legend()
    plt.savefig(folder_topology + f'distance_distribution.pdf')
    plt.show()