"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 06 September 2023

--------------------------------------------------------------------

Functions useful to visualize simulation results

"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_network(G, pop_nodes, dict_nodes, weight, state_nodes):
    """ Plot directed network. For the visualization, I distinguish between edges with weight equal to zero and non.
        In this way, I can attribute to edges with 0-weight the wight color to hide the arrowhead.
        It plots a single time step.
        Attribute a color to nodes depending on their state. Later on I can attribute it based on the density of
        infection inside each node.

    :param G: [networkx.class] graph structure from networkx
    :param pos: [list] position of nodes
    :param pop_nodes: [list] population inside every node
    :param dict_nodes: [dict] dictionary of nodes attributing each key a position
    :param weight: [list] weight to attribute to edges
    :param state_nodes: [list] state of the node
    """
    N = len(G.nodes)
    # Size of nodes
    size_map = [pop_nodes[i] for i in G.nodes]
    #        blue=S     red=I      green=R    violet=SI  vaqua=SR   orange=IR
    cmap = ['#1C86EE', '#FF3030', '#00C957', '#BF3EFF', '#458B74', '#FF7D40', '#CDC8B1' ]
    color_map = [''] * N
    i = 0
    idx = []
    while i < N:
        if 'S' == state_nodes[i]:
           # idx.append(i)
            color_map[i] = cmap[0]
        elif 'I' == state_nodes[i]:
            color_map[i] = cmap[1]
        elif 'R' == state_nodes[i]:
            color_map[i] = cmap[2]
        elif 'SI' == state_nodes[i]:
            color_map[i] = cmap[3]
        elif 'SR' == state_nodes[i]:
            color_map[i] = cmap[4]
        elif 'IR' == state_nodes[i]:
            color_map[i] = cmap[5]
        else:
            color_map[i] = cmap[6]
        i += 1

    # Draw nodes
    nx.draw_networkx_nodes(G, pos=dict_nodes, node_size=size_map, node_color = color_map)
    # Draw edges
    nx.draw_networkx_edges(G, pos=dict_nodes, width=weight, connectionstyle="arc3,rad=0.1")
    # Nodes with labels
    nx.draw_networkx_labels(G, pos=dict_nodes)
    # Edge with labels
   # nx.draw_networkx_edge_labels(G, pos=dict_nodes, edge_labels=dict_edges, label_pos=0.25, font_size=7)
    #plt.show()



