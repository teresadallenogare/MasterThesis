"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 06 September 2023

--------------------------------------------------------------------

Functions useful to visualize simulation results

"""
import networkx as nx
import matplotlib.pyplot as plt


def plot_network(G, pop_nodes, dict_nodes, dict_edges, weight):
    """ Plot directed network. For the visualization, I distinguish between edges with weight equal to zero and non.
        In this way, I can attribute to edges with 0-weight the wight color to hide the arrowhead.

    :param G: [networkx.class] graph structure from networkx
    :param pos: [list] position of nodes
    :param pop_nodes: [list] population inside every node
    :param dict_nodes: [dict] dictionary of nodes attributing each key a position
    :param T : [matrix] transition matrix
    """
    # Size of nodes
    size_map = [pop_nodes[i] for i in G.nodes]

    plt.figure(figsize=(8, 8))
    # Draw nodes
    nx.draw_networkx_nodes(G, pos=dict_nodes, node_size=size_map)
    # Draw edges
    nx.draw_networkx_edges(G, pos=dict_nodes, width=weight, connectionstyle="arc3,rad=0.1")
    # Nodes with labels
    nx.draw_networkx_labels(G, pos=dict_nodes)
    # Edge with labels
   # nx.draw_networkx_edge_labels(G, pos=dict_nodes, edge_labels=dict_edges, label_pos=0.25, font_size=7)
    plt.show()
