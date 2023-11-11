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
    nx.draw_networkx_labels(G, pos=dict_nodes)
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