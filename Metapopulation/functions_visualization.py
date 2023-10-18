"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 06 September 2023

--------------------------------------------------------------------

Functions useful to visualize simulation results

"""
from functions_SIR_metapop import compute_centralities
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------ Characterize network ------------------------------------
def plot_centralities(G):
    """ Plot centrality measures to characterize the network I will work on.
    Note that the number of incoming nodes is equal to the number of outgoing ones by construction.

    :param G:
    :return:
    """
    #in_degrees = [G.in_degree(n) for n in G.nodes()]
    # Calculates the degree centrality of the network

    degree_cent, closeness_cent, betweenness_cent, eigenvalue_cent = compute_centralities(G)

    degree_cent_values = list(degree_cent.values())
    closeness_cent_values = list(closeness_cent.values())
    betweenness_cent_values = list(betweenness_cent.values())
    eigenvalue_cent_values = list(eigenvalue_cent.values())
    fig, axs = plt.subplots(2,2, figsize = (11,8))

    #plt.bar(*np.unique(in_degrees, return_counts=True))
    axs[0,0].bar(*np.unique(degree_cent_values, return_counts=True), width = 0.1)
    axs[0,0].set_title("Degree centrality input edges")# out degrees = in_degrees by construction
    axs[0,0].set_xlabel("Normalized input degree")
    axs[0,0].set_ylabel("Frequency")

    axs[0,1].bar(*np.unique(closeness_cent_values, return_counts=True), width = 0.1)
    axs[0, 1].set_title("Closeness centrality")  # out degrees = in_degrees by construction
    axs[0, 1].set_xlabel("Closeness")
    axs[0, 1].set_ylabel("Frequency")

    axs[1, 0].bar(*np.unique(betweenness_cent_values, return_counts=True), width = 0.1)
    axs[1, 0].set_title("Betweenness centrality")  # out degrees = in_degrees by construction
    axs[1, 0].set_xlabel("Betweenness")
    axs[1, 0].set_ylabel("Frequency")

    axs[1, 1].bar(*np.unique(eigenvalue_cent_values, return_counts=True), width = 0.1)
    axs[1, 1].set_title("Eigenvalue centrality")  # out degrees = in_degrees by construction
    axs[1, 1].set_xlabel("Eigenvalue")
    axs[1, 1].set_ylabel("Frequency")
    plt.show()



def plot_static_network(G, pop_nodes, dict_nodes, weight):
    """ Plot a static version of the network, in which the size of nodes is proportional to the node population

    :param G: [networkx.class] graph structure from networkx
    :param pos: [list] position of nodes
    :param pop_nodes: [list] population inside every node
    :param dict_nodes: [dict] dictionary of nodes attributing each key a position
    :param weight: [list] weight to attribute to edges
    :param state_nodes: [list] state of the node
    """
    size_map = [pop_nodes[i]/ 10. for i in G.nodes]
    nx.draw_networkx_nodes(G, pos=dict_nodes, node_size=size_map)
    nx.draw_networkx_edges(G, pos=dict_nodes, width=weight, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos=dict_nodes)
    plt.show()

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
    size_map = [pop_nodes[i] if pop_nodes[i] < 1e2 else 100 for i in G.nodes]
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

# ------------------------------------ Simulation characterization ------------------------------------

def plot_mean_std_singleNode(T_sim, meanS, meanI, meanR, stdDevS, stdDevI, stdDevR, detS, detI, detR, idx_node):
    """ Plot mean and standard deviation of repetitions for only 1 node, together with the deterministic model

    """
    plt.plot(T_sim, meanS[:, idx_node],  label = 'S')
    plt.plot(T_sim, meanI[:, idx_node],  label = 'I')
    plt.plot(T_sim, meanR[:, idx_node],  label = 'R')
    plt.fill_between(T_sim, meanS[:, idx_node] - stdDevS[:, idx_node], meanS[:, idx_node] + stdDevS[:, idx_node], facecolor='blue', alpha=0.25)
    plt.fill_between(T_sim, meanI[:, idx_node] - stdDevI[:, idx_node], meanI[:, idx_node] + stdDevI[:, idx_node], facecolor='red', alpha=0.25)
    plt.fill_between(T_sim, meanR[:, idx_node] - stdDevR[:, idx_node], meanR[:, idx_node] + stdDevR[:, idx_node], facecolor='green', alpha=0.25)
    plt.plot(T_sim, detS, 'b--')
    plt.plot(T_sim, detI, 'r--')
    plt.plot(T_sim, detR, 'g--')


    plt.title(f'Mean and standard deviation of density per node: {idx_node}')
    plt.xlabel('Timestep')
    plt.ylabel('Density')

    plt.show()


def plot_mean_allNodes(T_sim, meanS, meanI, meanR, detS, detI, detR, N):
    for idx_node in range(N):
        plt.plot(T_sim, meanS[:, idx_node])
        plt.plot(T_sim, meanI[:, idx_node])
        plt.plot(T_sim, meanR[:, idx_node])
    plt.plot(T_sim, detS, 'b--')
    plt.plot(T_sim, detI, 'r--')
    plt.plot(T_sim, detR, 'g--')
    plt.legend()
    plt.title(f'Mean of density for all nodes')
    plt.xlabel('Timestep')
    plt.ylabel('Density')

    plt.show()