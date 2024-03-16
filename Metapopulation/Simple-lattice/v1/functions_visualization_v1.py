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
import random
import matplotlib.gridspec as gridspec
import math
from scipy import interpolate
from brokenaxes import brokenaxes
from scipy.stats import norm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set Seaborn style with custom background and grid color
#sns.set_style("darkgrid", {"axes.facecolor": ".9", "grid.color": "white"})
#sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})
#sns.set(rc={"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    """ Color gradient between c1 - darker - and c2 - lighter.

    :param c1: [scalar] dark value of color
    :param c2: [scalar] light value of color
    :param mix: 0
    :return:
    """
    c1 = np.array(matplotlib.colors.to_rgb(c1))
    c2 = np.array(matplotlib.colors.to_rgb(c2))

    return matplotlib.colors.to_hex((1 - mix) * c1 + mix * c2)


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
    plt.figure(figsize=(13, 11))
    #sns.set(style="white")
    datadir = os.getcwd()

    size_map = [pop_nodes[i] / 2 for i in G.nodes]
    nx.draw_networkx_nodes(G, pos=dict_nodes, node_color='#B7C8C4', edgecolors='#374845', linewidths=1.5,
                           node_size=size_map)
    nx.draw_networkx_edges(G, pos=dict_nodes, width=0.7, arrows=True, min_source_margin=20,
                           min_target_margin=20, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(G, pos=dict_nodes, font_size=15)
    # plt.savefig(folder_topology + f'net-topol.pdf')
    plt.axis('off')
    plt.show()


def plot_TransitionMatrix(T, N_row, N_col, choice_bool, c1):
    """ Plot the transition matrix both with annotations and without annotations

    :param T: [matrix] transition matrix
    :param N_row, N_col: [scalar] number of rows and columns of the network
    :param choice_bool: [bool] 0 if uniform population, 1 if non-uniform population
    :param c1: [scalar] strength of the connection
    """
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    T_df = pd.DataFrame(T)
    labels = T_df.applymap(lambda v: str(np.round(v, 2)) if v != 0 else '')
    palette = sns.cubehelix_palette(n_colors=100, start=2, dark=0.1, reverse=True)
    annotation_list = [True, False] if N_row == 3 else [False]
    for annotation in annotation_list:
        if annotation == True:
            ax = sns.heatmap(T, linewidth=0, square=True, annot=labels, fmt='', cmap=palette,
                             cbar_kws={'label': 'weight'})
        else:
            ax = sns.heatmap(T, linewidth=0, square=True, annot=annotation, fmt='', cmap=palette,
                             cbar_kws={'label': 'weight'})

        ax.set_xlabel("Node index", fontsize=14)
        ax.set_ylabel("Node index", fontsize=14)

        plt.savefig(folder_topology + f'TransMat_annot-{annotation}.pdf')
        plt.show()




def fx(total_pop, data, data1, data2, p):
    total_pop_fact = math.factorial(int(total_pop))
    dim_data = int(len(data))
    prod = 1
    prod_p = 1
    f = []
    for i in range(dim_data):
        data_fact = math.factorial(int(data[i]))
        prod = prod * data_fact

    for i in range(dim_data):
        f_prod = total_pop_fact / prod * p**data[i]

        f.append(f_prod)

    return f


def plot_population_distribution(total_pop, p, data1, data2, data, avg_data1, stdDev_data1, avg_data2,
                                 stdDev_data2, choice_bool):

    x = np.linspace(min(data1), max(data1), 1000)
    mean = avg_data1
    var = stdDev_data1**2 * p #(I do * p because p is 1/N)

    gaussian_vals = norm.pdf(x, loc = mean, scale = np.sqrt(var))
    if choice_bool == 0:
        #f = fx(total_pop, data, data1, data2, p)
        plt.figure(figsize = (8, 6))

        num_bins = int(np.sqrt(len(data1)))  # int(1 + (len(node_population_0) // 2.0 ** 0.5))

        sns.histplot(data1, bins=num_bins, kde=False, color='g', stat = 'density', label=r'Population in $V$')

        #plt.plot(data, f)
        plt.axvline(x = avg_data1, color = 'k', linestyle = '--', label = 'Average subpopulation')
        plt.axvline(x=avg_data1 + stdDev_data1, color='red', linestyle=':', label='Std deviation')

        plt.axvline(x=avg_data1 - stdDev_data1, color='red', linestyle=':')
        plt.plot(x, gaussian_vals, color = 'r', linestyle = '-')
        plt.xlabel('Subpopulations', fontsize = 14)
        plt.ylabel('Counts', fontsize = 14)

        plt.legend(fontsize = 12)
        plt.show()
    elif choice_bool == 1:
        # Create 2x2 sub plots
        gs = gridspec.GridSpec(2, 2)

        #f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=False, figsize=((12, 6)))
        #fig = plt.subplots( figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1, 1]})
        plt.figure(figsize=(10, 8))

        num_bins1 = int(np.sqrt(len(data1)))
        num_bins2 = int(np.sqrt(len(data2)))
        num_bins = int(np.sqrt(len(data))) + 200

        # Plot histograms
        axs = plt.subplot(gs[0, :])
        sns.histplot(data, bins=num_bins, kde=False, color = 'orange',ax = axs, label = r'Population in $V$', linewidth = 0.5)
        axs.set_xlabel('Subpopulations', fontsize=14)
        axs.set_ylabel('Counts', fontsize=14)
        axs.legend(fontsize=10)

        # Plot histograms
        axs = plt.subplot(gs[1, 0])
        sns.histplot(data2, bins=num_bins2, kde=False, color='peru', label=r'Population in $V-V_{fix}$', ax = axs)
        axs.axvline(x = avg_data2, color = 'k', linestyle = '--', label = 'Average subpopulation')
        axs.axvline(x=avg_data2 + stdDev_data2, color='red', linestyle=':', label='Std deviation')
        axs.axvline(x=avg_data2 - stdDev_data2, color='red', linestyle=':')
        axs.set_xlim(min(data2) - 20, max(data2) + 20)
        axs.set_xlabel('Subpopulations', fontsize=14)
        axs.set_ylabel('Counts', fontsize=14)
        axs.legend(fontsize=10)


        axs = plt.subplot(gs[1, 1])
        sns.histplot(data1, bins=num_bins1, kde=False, color='goldenrod', label=r'Population in $V_{fix}$', ax = axs)
        axs.axvline(x=avg_data1, color='k', linestyle='--', label='Average subpopulation')
        axs.axvline(x=avg_data1 + stdDev_data1, color='red', linestyle=':', label='Std deviation')
        axs.axvline(x=avg_data1 - stdDev_data1, color='red', linestyle=':')
        axs.set_xlim(min(data1) - 20, max(data1) + 20)
        axs.set_xlabel('Subpopulations', fontsize=14)
        axs.set_ylabel('Counts', fontsize=14)
        axs.legend(fontsize=10)

        # Adjust layout for better spacing
        plt.tight_layout()

        plt.show()


def plot_degree_distribution(N_row, N_col, choice_bool, c1, k, pk, avg_k, Poisson_funct, param):
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
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    if choice_bool == 0:
        color = 'g'
    else:
        color = 'darkorange'
    plt.bar(k, pk, color=color, label='Data', alpha = 0.65)
    plt.axvline(x=avg_k, color='k', label=r'$\langle k^{in} \rangle$', linestyle='--')
    plt.plot(k, Poisson_funct(k, *param), marker='o', color='red', label='Poisson fit')
    plt.xlabel('$k^{in}$', fontsize = 14)
    plt.ylabel('$P(k^{in})$', fontsize = 14)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # plt.title(f'Degree distribution of {row}x{col} network with choice_bool: {choice_bool}, c1: {c1}')
    plt.legend(fontsize=12)
    plt.savefig(folder_topology + f'degree_distribution_poisson.pdf')
    plt.show()


def plot_distance_distribution(N_row, N_col, choice_bool, c1, d, pd, avg_d):
    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    plt.plot(d, pd, '-o', color='#93ACA7', mfc='#536C67')
    plt.axvline(x=avg_d, color='k', label=r'$\langle d \rangle$', linestyle='--')
    plt.xlabel('$d$')
    plt.ylabel('$p_d$')
    # plt.title(f'Degree distribution of {row}x{col} network with choice_bool: {choice_bool}, c1: {c1}')
    plt.legend(fontsize=12)
    plt.savefig(folder_topology + f'distance_distribution.pdf')
    plt.show()


def plot_node_population_0(N, N_fix, idx_Nfix, node_pop, mean_pop1, stdDev_pop1, mean_pop2, stdDev_pop2, homogeneous):
    idx_node = np.linspace(0, N - 1, N)
    x = np.linspace(0, N - 1, N * 1000)
    if homogeneous == 0:
        y_err1_up = (mean_pop1 + stdDev_pop1) * np.ones(len(x))
        y_err1_down = (mean_pop1 - stdDev_pop1) * np.ones(len(x))
        plt.scatter(idx_node, node_pop, marker='o', s=10, color='red')
        plt.axhline(y=mean_pop1, color='black', linestyle='--', label='Average population ')
        plt.fill_between(x, y_err1_down, y_err1_up, color='C0', alpha=0.3)
        plt.xlabel('Index node')
        plt.ylabel('Node population')
        plt.legend(fontsize=12)
        plt.show()

    elif homogeneous == 1:

        y_err1_up = (mean_pop1 + stdDev_pop1) * np.ones(len(x))
        y_err1_down = (mean_pop1 - stdDev_pop1) * np.ones(len(x))
        y_err2_up = (mean_pop2 + stdDev_pop2) * np.ones(len(x))
        y_err2_down = (mean_pop2 - stdDev_pop2) * np.ones(len(x))
        plt.scatter(idx_node, node_pop, marker='o', s=10, color='red')
        plt.axhline(y=mean_pop1, color='black', linestyle='--', label='Average population ')
        plt.fill_between(x, y_err1_down, y_err1_up, color='C0', alpha=0.5)
        plt.axhline(y=mean_pop2, color='black', linestyle='--')
        plt.fill_between(x, y_err2_down, y_err2_up, color='C0', alpha=0.5)
        plt.xlabel('Index node')
        plt.ylabel('Node population')
        plt.legend(fontsize=12)
        plt.show()

        print('hello')


def plot_network_final_size(G, row, pop_nodes, dict_nodes, weight, state_nodes):
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
    size_map = [pop_nodes[i] if pop_nodes[i] < 1e2 else 20 for i in G.nodes]
    #        blue=S     red=I      green=R    violet=SI  vaqua=SR   orange=IR
    cmap = ['#1C86EE', '#FF3030', '#00C957']
    color_map = [''] * N
    i = 0
    idx = []
    while i < N:
        if 0 == state_nodes[i]:
            # idx.append(i)
            color_map[i] = cmap[0]
        elif 1 == state_nodes[i]:
            color_map[i] = cmap[1]
        i += 1

    #plt.figure(figsize=(8, 8), frameon=True)  # Disable the figure frame
    ax = plt.Axes(plt.gcf(), [0., 0., 1., 1.], )
    ax.set_axis_off()
    plt.gcf().add_axes(ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos=dict_nodes, node_size=size_map, node_color=color_map)
    # Nodes with labels
    # nx.draw_networkx_labels(G, pos=dict_nodes)
    # Draw edges
    # Sort if I have a 10x10 or higher dimensional network
    if row == 10 or row == 30 or row == 50:
        random.seed(42)  # You can use any integer as the seed
        if row == 10:
            nbr_edges = 1000
        else:
            nbr_edges = 1000
        selected_nodes = [random.randint(0, N) for _ in range(nbr_edges)]
        # Create a subgraph containing only the selected nodes and their edges
        edges_to_draw = [(u, v) for u, v in G.edges() if u in selected_nodes and v in selected_nodes and u != v]
        nx.draw_networkx_edges(G, pos=dict_nodes, edgelist=edges_to_draw, edge_color='black', width=0.1, arrows=False,
                               min_source_margin=5,
                               min_target_margin=5, alpha=0.2)

    # Edge with labels


# nx.draw_networkx_edge_labels(G, pos=dict_nodes, edge_labels=dict_edges, label_pos=0.25, font_size=7)
#######################################################################################################################
#                                                                                                                     #
#                                            SIR REPEATED TRIALS                                                      #
#                                                                                                                     #
#######################################################################################################################
def plot_SIR_repeated_timeseries_single_node(N_row, N_col, choice_bool, c1, beta, mu, idx_sims, idx_node, T_sim, avg_pop_node,
                                            avg_pop_Nfix,
                                            avg_pop_Others):
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
    nbr_sims = int(len(idx_sims))
    for x in range(nbr_sims + 1):
        #                                dark           light
        grad_gray.append(colorFader('#505050', '#EAE9E9', x / nbr_sims))
        grad_red.append(colorFader('#E51C00', '#FCE0DC', x / nbr_sims))
        grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x / nbr_sims))
        grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x / nbr_sims))

    datadir = os.getcwd()
    # Folder
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
    i = 0
    first = True
    for sim in idx_sims:
        sim = int(sim)
        # Load data
        node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
        node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
        node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

        if first:
            plt.plot(T_sim, node_population_time[:, idx_node], color=grad_gray[i], label='Population')
            plt.plot(T_sim, node_NS_time[:, idx_node], color=grad_blue[i], label='S')
            plt.plot(T_sim, node_NI_time[:, idx_node], color=grad_red[i], label='I')
            plt.plot(T_sim, node_NR_time[:, idx_node], color=grad_green[i], label='R')
            # plt.plot(T_sim, node_NS_time[:, idx_node] + node_NI_time[:, idx_node] + node_NR_time[:, idx_node])
            first = False
        else:
            plt.plot(T_sim, node_population_time[:, idx_node], color=grad_gray[i])
            plt.plot(T_sim, node_NS_time[:, idx_node], color=grad_blue[i])
            plt.plot(T_sim, node_NI_time[:, idx_node], color=grad_red[i])
            plt.plot(T_sim, node_NR_time[:, idx_node], color=grad_green[i])
            # plt.plot(T_sim, node_NS_time[:, idx_node] + node_NI_time[:, idx_node] + node_NR_time[:, idx_node])
        i = i + 1
    if choice_bool == 0:
        plt.axhline(y=avg_pop_node, color='black', linestyle='--', label='Average population ')
    elif choice_bool == 1:
        plt.axhline(y=avg_pop_Others, color='black', linestyle='--', label='Average population ')
        plt.axhline(y=avg_pop_Nfix, color='black', linestyle='--')
    else:
        print('Wrong choice_bool')
    plt.xlabel('Timestep')
    plt.ylabel('Node population')
    plt.legend()
    plt.show()


def plot_SIR_repeated_timeseries_single_sim(N_row, N_col, choice_bool, c1, beta, mu, sim, idx_nodes, T_sim, avg_pop_node,
                                            avg_pop_Nfix,
                                            avg_pop_Others):
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

    for x in range(N_row * N_col + 1):
        #                                dark           light
        grad_gray.append(colorFader('#505050', '#EAE9E9', x / (N_row * N_col)))
        grad_red.append(colorFader('#E51C00', '#FCE0DC', x / (N_row * N_col)))
        grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x / (N_row * N_col)))
        grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x / (N_row * N_col)))

    datadir = os.getcwd()
    # Folder
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

    sim = int(sim)
    # Load data
    node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
    node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
    node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
    node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

    first = True
    for node in idx_nodes:
        node = int(node)
        if first:
            plt.plot(T_sim, node_population_time[:, node], color=grad_gray[node], label='Population')
            plt.plot(T_sim, node_NS_time[:, node], color=grad_blue[node], label='S')
            plt.plot(T_sim, node_NI_time[:, node], color=grad_red[node], label='I')
            plt.plot(T_sim, node_NR_time[:, node], color=grad_green[node], label='R')
            # plt.plot(T_sim, node_NS_time[:, node-1] + node_NI_time[:, node-1] + node_NR_time[:, node-1])
            first = False
        else:
            plt.plot(T_sim, node_population_time[:, node], color=grad_gray[node])
            plt.plot(T_sim, node_NS_time[:, node], color=grad_blue[node])
            plt.plot(T_sim, node_NI_time[:, node], color=grad_red[node])
            plt.plot(T_sim, node_NR_time[:, node], color=grad_green[node])
            # plt.plot(T_sim, node_NS_time[:, node-1] + node_NI_time[:, node-1] + node_NR_time[:, node-1])

    if choice_bool == 0:
        plt.axhline(y=avg_pop_node, color='black', linestyle='--', label='Average population ')
    elif choice_bool == 1:
        plt.axhline(y=avg_pop_Others, color='black', linestyle='--', label='Average population ')
        plt.axhline(y=avg_pop_Nfix, color='black', linestyle='--')
    else:
        print('Wrong choice_bool')

    plt.xlabel('Timestep')
    plt.ylabel('Node population')
    plt.legend()
    plt.show()


def plot_mean_std_singleNode(T_sim, meanS, meanI, meanR, stdDevS, stdDevI, stdDevR, detS, detI, detR, idx_node,
                             bool_density):
    """ Plot mean and standard deviation of repetitions for only 1 node, together with the deterministic model

    """

    plt.plot(T_sim, meanS[:, idx_node], label='S')
    plt.plot(T_sim, meanI[:, idx_node], label='I')
    plt.plot(T_sim, meanR[:, idx_node], label='R')

    plt.fill_between(T_sim, meanS[:, idx_node] - stdDevS[:, idx_node], meanS[:, idx_node] + stdDevS[:, idx_node],
                     facecolor='blue', alpha=0.25)
    plt.fill_between(T_sim, meanI[:, idx_node] - stdDevI[:, idx_node], meanI[:, idx_node] + stdDevI[:, idx_node],
                     facecolor='red', alpha=0.25)
    plt.fill_between(T_sim, meanR[:, idx_node] - stdDevR[:, idx_node], meanR[:, idx_node] + stdDevR[:, idx_node],
                     facecolor='green', alpha=0.25)
    if bool_density == 1:
        plt.plot(T_sim, detS, 'b--')
        plt.plot(T_sim, detI, 'r--')
        plt.plot(T_sim, detR, 'g--')

    plt.title(f'Mean and standard deviation of density per node: {idx_node}')
    plt.xlabel('Timestep')
    plt.ylabel('Density')

    plt.legend()
    plt.show()


def animate_infecteds(t, img, grid, dict_vals, dict_norm_vals):
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

def animate_new_infecteds(t, img, grid, dict_vals, dict_norm_vals):
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

from matplotlib.colors import LogNorm
def heatmap_time_infecteds(N_row, N_col, choice_bool, c1, beta, mu, sim, bool_static, bool_Inew, time):
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
    folder_images = '/Users/teresa/Desktop/Thesis/Images/Maps/'


    datadir = os.getcwd()
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
        open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
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
            #img = ax.imshow(grid, cmap='coolwarm', norm=LogNorm(vmin=0.001, vmax=1))
        else:
            img = ax.imshow(grid, vmin=0, vmax=max_densityInew, cmap='coolwarm')
            #img = ax.imshow(grid, cmap='coolwarm', norm=LogNorm(vmin=0.001, vmax=1))
        ax.invert_yaxis()
        fig.colorbar(img, cmap='coolwarm')
        ax.set_xlabel('Node index')
        ax.set_ylabel('Node index')
        ax.set_title(f'Heatmap {N_row}x{N_col} : beta = {beta}, mu = {mu}, sim = {sim}')
        ax.grid(True, linestyle='-', linewidth=0.01, alpha=0.1, color='gray')
        if bool_Inew == 0:
            ani = animation.FuncAnimation(fig, animate_infecteds, fargs=(img, grid, dict_load_values, dict_load_normalized_values,),
                                        frames=dict_load.keys())
        else:
            ani = animation.FuncAnimation(fig, animate_new_infecteds, fargs=(img, grid, dict_load_values, dict_load_normalized_values,),
                                        frames=dict_load.keys())
        # converting to a html5 video
        video = ani.to_html5_video()
        if bool_Inew == 0:
            ani.save(folder_animations + f'animation-beta{beta}-mu{mu}-sim{sim}.mp4', writer=Writer)
        else:
            ani.save(folder_animations + f'animation-Inew-beta{beta}-mu{mu}-sim{sim}.mp4', writer=Writer)
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
        fig, ax = plt.subplots(figsize = (10,8))
        if bool_Inew == 0:
            img = ax.imshow(density_I_grid, vmin=0, vmax=max_densityI, cmap='coolwarm')
            #img = ax.imshow(grid, cmap='coolwarm', norm=LogNorm(vmin=0.001, vmax=1))
        else:
            img = ax.imshow(density_Inew_grid, vmin=0, vmax=max_densityInew, cmap='coolwarm')
            #img = ax.imshow(grid, cmap='coolwarm', norm=LogNorm(vmin=0.001, vmax=1))
        ax.invert_yaxis()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(img, cmap='coolwarm', cax=cax)
        cbar.set_label(r'$I/\langle n\rangle$', fontsize = 36)
        cbar.ax.tick_params(labelsize=36)
        ax.set_xlabel('Node index', fontsize = 36)
        ax.set_ylabel('Node index', fontsize = 36)
        #ax.set_title(f'Heatmap {N_row}x{N_col} : beta = {beta}, mu = {mu}, sim = {sim}')
        ax.grid(True, linestyle='-', linewidth=0.01, alpha=0.1, color='white')
        # Minor ticks
        ax.set_xticks(np.arange(-0.5, 29.5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 29.5, 1), minor=True)
        ax.grid(which='minor', color='whitesmoke', linestyle='-', linewidth=0.4)

        ax.tick_params(axis='both', which='major', labelsize=36)

        plt.tight_layout()
        plt.savefig(folder_images + f'map-t{t}-R0{beta/mu}-sim{sim}.png')

        plt.show()


def animate_recovered(t, img, grid, dict_vals, dict_norm_vals):
    mtrx_t = dict_vals[t]
    mtrx_norm_t = dict_norm_vals[t]
    # Extract node positions from the non-normalized dictionary
    x_nodes = mtrx_t[:, 0]
    y_nodes = mtrx_t[:, 1]
    # Extract the density of infected from the normalized dictionary
    density_R_nodes = mtrx_norm_t[:, 4]
    # print('t: ', t)
    # print('dI: ', density_I_nodes)
    idx_row = 0
    for i, j in zip(x_nodes, y_nodes):
        # grid[int(i), int(j)] = nbr_I_nodes[idx_row]
        grid[int(i), int(j)] = density_R_nodes[idx_row]
        idx_row += 1
    img.set_data(grid)
    return img,
def heatmap_time_recovered(N_row, N_col, choice_bool, c1, beta, mu, sim, bool_static, time):
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
        open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    dict_load_normalized_values = list(dict_load_normalized.values())
    # Brute force : maximum value of density of I in whole dictionary
    max_densityR_time = []
    # Determination of the maximum density of infected
    for t in dict_load.keys():
        mtrx_t_normalized = dict_load_normalized[t]
        density_R = mtrx_t_normalized[:, 4]
        max_densityR_time.append(max(density_R))
    max_densityR_time = np.array(max_densityR_time)
    max_densityR = max(max_densityR_time)
    print('max-densityR', max_densityR)
    if bool_static == 0:
        # Setup animation
        Writer = animation.FFMpegWriter(fps=1)

        fig, ax = plt.subplots()

        img = ax.imshow(grid, vmin=0, vmax=max_densityR, cmap='RdYlGn')

        ax.invert_yaxis()
        fig.colorbar(img, cmap='RYlGn')
        ax.set_xlabel('Node index')
        ax.set_ylabel('Node index')
        ax.set_title(f'Heatmap {N_row}x{N_col} : beta = {beta}, mu = {mu}, sim = {sim}')
        ax.grid(True, linestyle='-', linewidth=0.01, alpha=0.1, color='gray')
        ani = animation.FuncAnimation(fig, animate_recovered, fargs=(img, grid, dict_load_values, dict_load_normalized_values,),
                                      frames=dict_load.keys())
        # converting to a html5 video
        video = ani.to_html5_video()

        ani.save(folder_animations + f'animation-R-beta{beta}-mu{mu}-sim{sim}.mp4', writer=Writer)

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
        density_R = mtrx_t_normalized[:, 4]
        density_R_grid = density_R.reshape((30, 30))
        fig, ax = plt.subplots(figsize=(9, 7))

        img = ax.imshow(density_R_grid, vmin=0, vmax=max_densityR, cmap='RdYlGn')

        ax.invert_yaxis()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(img, cmap='coolwarm', cax=cax)
        cbar.set_label(r'$R/\langle n\rangle$', fontsize=30)
        cbar.ax.tick_params(labelsize=30)
        ax.set_xlabel('Node index', fontsize=30)
        ax.set_ylabel('Node index', fontsize=30)
        # ax.set_title(f'Heatmap {N_row}x{N_col} : beta = {beta}, mu = {mu}, sim = {sim}')
        ax.grid(True, linestyle='-', linewidth=0.01, alpha=0.1, color='white')
        # Minor ticks
        ax.set_xticks(np.arange(-0.5, 29.5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 29.5, 1), minor=True)
        ax.grid(which='minor', color='whitesmoke', linestyle='-', linewidth=0.4)

        ax.tick_params(axis='both', which='major', labelsize=30)

        plt.tight_layout()
        plt.show()

def plot_nullcline(nodeNS, nodeNI, x, y, u, v, lineStyle, beta):
    plt.quiver(x, y, u, v, linewidth=0.5, color='k', capstyle='round',
               scale=1, scale_units='xy', angles='xy')
    plt.plot(nodeNS, nodeNI, linewidth=1, color='r', linestyle=lineStyle, label=rf'$\beta = {beta}$')
    plt.scatter(x, y, color='k')


def plot_phase_space(N_row, N_col, choice_bool, c1, beta, mu, sim):
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
    N = N_row * N_col

    datadir = os.getcwd()

    folder_simulation = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
    # Load data

    node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
    node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
    node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
    node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')
    # ADD NODE STATE
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    color_map = plt.get_cmap('spring')
    for idx_node in range(1):
        x = node_NS_time[:, idx_node]
        y = node_NI_time[:, idx_node]
        z = node_NR_time[:, idx_node]
        sc = ax.scatter3D(x, y, z)

    ax.set_xlabel('S')
    ax.set_ylabel('I')
    ax.set_zlabel('R')
    ax.set_title(f'Network {N_row}x{N_col}, beta: {beta}, mu: {mu}')

    plt.show()

#######################################################################################################################
#                                                                                                                     #
#                                                    SIR                                                              #
#                                                                                                                     #
#######################################################################################################################

def plot_SIR_time_node(N, T_sim, vals_pop, vals_S, vals_I, vals_R, det_S, det_I, det_R, beta, mu):
    f, ax = plt.subplots(figsize=(15, 8))
    ax.tick_params(axis='both', which='major', labelsize=30)
    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1e'))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(30)
    avg_vals_S = np.mean(vals_S, axis = 1)
    avg_vals_I = np.mean(vals_I, axis = 1)
    avg_vals_R = np.mean(vals_R, axis = 1)
    for i in range(N):
        if i == 0:
            plt.plot(T_sim, vals_S[:, i], color = '#6488ea', label = 'S')
            plt.plot(T_sim, vals_I[:, i], color = '#fc5a50', label = 'I')
            plt.plot(T_sim, vals_R[:, i], color = '#54ac68', label = 'R')
        else:
            plt.plot(T_sim, vals_S[:, i], color='#6488ea', alpha=0.2)
            plt.plot(T_sim, vals_I[:, i], color='#fc5a50', alpha=0.2)
            plt.plot(T_sim, vals_R[:, i], color='#54ac68', alpha=0.2)
    plt.plot(T_sim, avg_vals_S, color='k', linewidth = 0.9, label = 'Average')
    plt.plot(T_sim, avg_vals_I, color='k',  linewidth = 0.9)
    plt.plot(T_sim, avg_vals_R, color='k', linewidth = 0.9)

    plt.plot(T_sim, det_S, linestyle = ':', color = 'k', label = 'Deterministic')
    plt.plot(T_sim, det_I, linestyle=':', color='k')
    plt.plot(T_sim, det_R, linestyle=':', color='k')
    plt.text(110, 0.7, r'$R_0 =$' + str(np.round(beta / mu, 2)), fontsize=30)
    plt.xlabel('Time', fontsize = 30)
    plt.ylabel('Node density', fontsize = 30)

    plt.legend(fontsize = 30)
    plt.tight_layout()
    plt.show()

#######################################################################################################################
#                                                                                                                     #
#                                                    Barcodes                                                         #
#                                                                                                                     #
#######################################################################################################################

def plot_barcodes(birth0, end0, birth1, end1, y0, y1, t, R0, sim, clr):
    folder_images = '/Users/teresa/Desktop/Thesis/Images/New/'

    # Create a figure and axis
    #sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})
    fig = plt.figure(figsize=(16, 8))

    # Draw horizontal lines between elements of list1 and list2
    ax1 = plt.subplot(1, 2, 1)
    i = 0
    for x1, x2 in zip(birth0, end0):

        ax1.plot([x1, x2], [y0[i], y0[i]], color='r', linestyle='-', linewidth=2, markerfacecolor='white', markersize=8,
                 markeredgewidth=2)


        i = i + 1
    # Set labels and title

    ax1.set_xlabel('Persistence', fontsize=40)
    ax1.set_ylabel('Feature', fontsize=40)

    ax1.set_title(r'Persistence barcode $\mathcal{H}_0$', fontsize=40)
    ax1.tick_params(axis='both', which='major', labelsize=38)
    #ax1.tick_params(labelleft=False)
    ax2 = plt.subplot(1, 2, 2)
    i = 0
    for x1, x2 in zip(birth1, end1):
        ax2.plot([x1, x2], [y1[i], y1[i]],  color='b', linestyle='-', linewidth=2, markerfacecolor='white', markersize=8,
                 markeredgewidth=2) #clr

        i = i + 1
    ax2.set_xlabel('Persistence', fontsize=40)
    #ax2.tick_params(labelleft=False)

    ax2.set_ylabel('Feature', fontsize=40)
    ax2.set_title(r'Persistence barcode $\mathcal{H}_1$', fontsize=40)
    ax2.tick_params(axis='both', which='major', labelsize=38)
    plt.tight_layout()
    plt.savefig(folder_images + f'barcodes-t{t}-R0{R0}-sim{sim}.png')
    plt.show()



def plot_persistence_diagram(birth0, end0, birth1, end1):
    def diagonal(x):
        return x
    f, ax = plt.subplots(figsize = (10, 8))
    x = np.linspace(0, 0.25, 100)
    plt.scatter(birth0, end0, color = 'r', s = 8, label = r'$\mathcal{H}_0$')
    plt.scatter(birth1, end1, color = 'b', s = 8, label = r'$\mathcal{H}_1$')
    plt.plot(x, diagonal(x), color = 'k', linewidth = 1)
    plt.xlabel('Birth', fontsize = 16)
    plt.ylabel('Death', fontsize = 16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.legend(fontsize = 14)
    plt.show()

def plot_cc1_vs_time(time, pers0, pers1, N_cc0, N_cc1):
    # Plot how does the death time of the most important cc (after infty) variesas a function of time

    f, ax = plt.subplots(figsize = (13, 8))

    # Draw horizontal lines between elements of list1 and list2
    ax1 = plt.subplot(2, 1, 1)
    for cc in range(N_cc0):
        ax1.plot(time, pers0[cc], linewidth = 0.95, label=f'c.c. {cc + 1}')
    ax1.set_xlabel('Time', fontsize = 16)
    ax1.set_ylabel('Persistence', fontsize = 16)
    ax1.set_title(r'Persistence features $\mathcal{H}_0$', fontsize=16)

    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    for cc in range(N_cc1):
        ax2.plot(time, pers1[cc], linewidth = 0.95, label=f'c.c. {cc + 1}')
    ax2.set_xlabel('Time', fontsize=16)
    ax2.set_ylabel('Persistence', fontsize=16)
    ax2.set_title(r'Persistence features $\mathcal{H}_1$', fontsize=16)

    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_cc1_vs_time_H1(time, pers1, N_cc1):
    # Plot how does the death time of the most important cc (after infty) variesas a function of time

    f, ax = plt.subplots(figsize = (13, 7))


    for cc in range(N_cc1):
        ax.plot(time, pers1[cc], linewidth = 1.2, label=f'hole {cc + 1}')
    ax.set_xlabel('Time', fontsize = 26)
    ax.set_ylabel('Persistence', fontsize = 26)
    ax.set_title(r'Persistence of longest features in $\mathcal{H}_1$', fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25,
                     box.width, box.height * 0.75])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=5, fontsize = 26)

    #plt.tight_layout()
    plt.show()


def plot_cc1_vs_time_H0(time, pers0, N_cc0):
    # Plot how does the death time of the most important cc (after infty) variesas a function of time

    f, ax = plt.subplots(figsize = (13, 7))


    for cc in range(N_cc0):
        ax.plot(time, pers0[cc], linewidth = 1.2, label=f'c.c. {cc + 1}')
    ax.set_xlabel('Time', fontsize = 26)
    ax.set_ylabel('Persistence', fontsize = 26)
    ax.set_title(r'Persistence of longest features in $\mathcal{H}_0$', fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25,
                     box.width, box.height * 0.75])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=5, fontsize = 26)

    #plt.tight_layout()
    plt.show()