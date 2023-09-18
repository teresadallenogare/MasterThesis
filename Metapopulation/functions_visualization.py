"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 06 September 2023

--------------------------------------------------------------------

Functions useful to visualize simulation results

"""
import networkx as nx
import matplotlib.pyplot as plt


def plot_network(G, pos, pop_nodes, lab_nodes, weight):
    """ Plot directed network

    :param G: [networkx.class] graph structure from networkx
    :param pos: [list] position of nodes
    :param pop_nodes: [list] population inside every node
    :param lab_nodes: [dict] label of nodes
    :param T : [matrix] transition matrix
    """
    size_map = [pop_nodes[i] for i in G.nodes]

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos=pos, node_size=size_map)
    nx.draw_networkx_labels(G, pos=lab_nodes)
    nx.draw_networkx_edges(G, pos=lab_nodes, width=weight, connectionstyle="arc3,rad=0.1")



    plt.show()
