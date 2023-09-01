"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 30 August 2023

--------------------------------------------------------------------

Functions useful to implement simulations

"""

import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import LinearSegmentedColormap
import sys

np.set_printoptions(threshold=sys.maxsize)

def DefLattice(Gname, Nrow, Ncol):
    """

    ---- Description
    Definition of the topology of the lattice

    ---- Parameters
    @param Gname: key with the name of the lattice
    @param Nrow: number of rows for the lattice
    @param Ncol: number of columns for the lattice
    @return: G - graph in networkx, pos - positions of nodes in the graph

    """

    # dictionary = {<key>: <value>}
    dict = {'Lattice': nx.grid_2d_graph(Nrow, Ncol, periodic=True),
            'Triangle': nx.triangular_lattice_graph(Nrow, Ncol, with_positions=True, periodic=True),
            'Hexagon': nx.hexagonal_lattice_graph(Nrow, Ncol, with_positions=True, periodic=True)}
    G = dict[Gname]
    if Gname == 'Lattice':
        pos = {}
        for i in G.nodes:
            pos.update({i: (i[0], i[1])})
    else:
        pos = nx.get_node_attributes(G, 'pos')
    return G, pos


def DefNetwork(Gname, N, probEdge, k, m):
    """

    ---- Description
    Definition of the topology of the network

    ---- Parameters
    @param Gname: key with the name of the network
    @param N: number of nodes for the network
    @param probEdge:
    @param k:
    @param m:
    @return: G - graph in networkx, pos - positions of nodes in the graph

    """
    # dictionary = {<key>: <value>}
    dict = {'random': nx.fast_gnp_random_graph(N, probEdge),  # Erdos-Reni
            'small-world': nx.watts_strogatz_graph(N, k, probEdge),
            'barabasi': nx.barabasi_albert_graph(N, m)}
    G = dict[Gname]
    # Position nodes using Fruchterman-Reingold force-directed algorithm
    pos = nx.spring_layout(G)
    return G, pos


def CreateNeighbourList_lattice(A, N):
    """

    ---- Description
    Create list with first neighbours of nodes in the lattice

    ---- Parameters
    @param A: adjacency matrix
    @param N: number of nodes
    @return: Neig - list of first neighbours of nodes

    """
    # N = len(G.nodes())
    # print(list(G.edges))
    Neig = []
    # print(np.dot((np.dot(A,A)), A))
    # Cycle over N because the adjacency matrix for a graph of Nrow rows and
    # thus N = Nrow x Nrow nodes is N x N (accounts for the interactions of each node with all the others)
    for i in range(N):
        Neig_i = []
        for j in range(N):
            print('\r', math.floor(i / N * 100), '%', end='')
            if A[i, j] == 1: Neig_i.append(j)
        Neig.append(Neig_i)
    return Neig


def StoreTime_ControlNodes(N_row, rnd_nodes, X_prev, X_curr, timeSI, timeIR, t):
    """

    ---- Description
    Compute time at which SI and IR transitions occur in control node.
    Problem : times do not correspond to nodes. I still need to associate the correct time for transition
    at the correct control node

    ---- Parameters
    @param N_row: number of rows in the lattice
    @param rnd_nodes: list of control nodes taken at random
    @param X_prev: list with the state of the network at the previous time step
    @param X_curr: list with the state of the network at the current time step
    @param timeSI: list with times of transition from S to I of the control nodes
    @param timeIR: list with times of transition from I to R of the control nodes
    @param t: delta of time it occurs for a transition of a node
    @return: timeSI, timeIR

    """
    lattice_list = []
    for i in range(N_row):
        for j in range(N_row):
            if (i, j) in rnd_nodes:
                lattice_list.append(1)
            else:
                lattice_list.append(0)
    # print('ll: ',lattice_list)

    idx_rnd_nodes = []
    for idx_rnd_node in range(len(lattice_list)):
        if lattice_list[idx_rnd_node] == 1:
            idx_rnd_nodes.append(idx_rnd_node)
    # print(idx_rnd_nodes)  # index of control nodes

    for i in range(len(X_curr)):
        if i in idx_rnd_nodes:
            if X_curr[i] != X_prev[i]:
                if X_curr[i] == 'I' and X_prev[i] == 'S':
                    #print('dtSI: ', t)
                    timeSI.append(t)
                elif X_curr[i] == 'R' and X_prev[i] == 'I':
                   # print('dtIR: ', t)
                    timeIR.append(t)
    return [timeSI, timeIR]


#  ------------------------------- Deterministic simulations --------------------------    
def SimSIRDeterministic(variables, t, params):
    """

    ---- Description
    Compute the deterministic SIR equations

    ---- Parameters
    @param variables:
    @param t:
    @param params:
    @return:
    """
    S = variables[0]
    I = variables[1]
    R = variables[2]

    N = S + I + R
    beta = params[0]
    mu = params[1]

    alpha = beta * I / N

    dSdt = - alpha * S / N
    dIdt = alpha * S / N - mu * I / N
    dRdt = mu * I / N

    return [dSdt, dIdt, dRdt]


def InfectedBetaMu(y_init, time, beta_vals, mu_vals):
    """

    ---- Description
    Computes deterministic SIR model for different values of beta and mu

    ---- Parameters
    @param y_init: initial state of the deterministic model
    @param time:
    @param beta_vals: values of the
    @param mu_vals:

    """
    new_cmap = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6',
                '#6A3D9A', '#ECEC28', '#B15928']
    new_cmap = ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de", "#0571b0"]
    rtg_r = LinearSegmentedColormap.from_list("rtg", new_cmap)
    colors = rtg_r(np.linspace(0, 1, 12))
    # Fix mu and cycle on beta
    for mm in range(0, len(mu_vals)):
        mu = mu_vals[mm]
        for bb in range(0, len(beta_vals)):
            beta = beta_vals[bb]
            params = [beta, mu]
            y = odeint(SimSIRDeterministic, y_init, time, args=(params,))
            plt.plot(time, y[:, 1], label=f'beta={beta}', color=colors[bb])
        plt.xlabel('t')
        plt.ylabel('ProbI(t)')
        plt.title(f'Infected probability with mu={mu} and varying beta')
        plt.legend()
        plt.show()

    # Fix beta and cycle on mu 
    for bb in range(0, len(beta_vals)):
        beta = beta_vals[bb]
        for mm in range(0, len(mu_vals)):
            mu = mu_vals[mm]
            params = [beta, mu]
            y = odeint(SimSIRDeterministic, y_init, time, args=(params,))
            plt.plot(time, y[:, 1], label=f'beta={beta}, mu={mu}')
        plt.xlabel('t')
        plt.ylabel('I(t)')
        plt.title('I(t) with fixed beta and varying mu')
    # plt.legend()
    # plt.show()


# -------------------------------------- Plot functions -------------------------------- 
def plotSIR_line(time, Ptot_S, Ptot_I, Ptot_R, y, N, beta, mu):
    """

    ---- Description
    Plot time series with probability of S,I,R as a function of time,
    both stochastic and deterministic results are plotted

    ---- Parameters
    @param time:
    @param Ptot_S: total probability of being in state S
    @param Ptot_I: total probability of being in state I
    @param Ptot_R: total probability of being in state R
    @param y: contains solution to the deterministic equation. Is the result of
    the integration of the ODE
    @param N: number of nodes
    @param beta: rate of infection
    @param mu: rate of recovery

    """
    f, (ax1, ax2, ax3) = plt.subplots(3)
    line1, = ax1.plot(time, Ptot_S)
    line2, = ax2.plot(time, Ptot_I)
    line3, = ax3.plot(time, Ptot_R)
    line4, = ax1.plot(time, y[:, 0])
    line5, = ax2.plot(time, y[:, 1])
    line6, = ax3.plot(time, y[:, 2])

    ax1.set_ylabel('S')
    ax2.set_ylabel('I')
    ax3.set_ylabel('R')
    ax3.set_xlabel('time')

    plt.suptitle(f'N = {N}, beta={beta}, mu ={mu}')
    plt.show()


def plotSIR_lattice(T, N_row, N, pos, G, X, rnd_nodes):
    """

    ---- Description
    Plots the result of the simulation as function of time

    ---- Parameters
    @param T: end of the simulation
    @param N_row: number of rows in the lattice (square lattice)
    @param N: number of nodes of the lattice
    @param pos: dictionary with the position of each node in the lattice
    @param G: graph structure from networkx
    @param X: list containing the temporal evolution of the state of each node of the graph
    @param rnd_nodes: list of control nodes chosen randomly

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    size_map = []
    for i in range(N_row):
        for j in range(N_row):
            if (i, j) in rnd_nodes:
                size_map.append(150)
            else:
                size_map.append(50)

    for t in range(T):
        color_map = []
        for i in range(N):
            if X[t][i] == 'S':
                color_map.append('blue')
            elif X[t][i] == 'I':
                color_map.append('red')
            elif X[t][i] == 'R':
                color_map.append('green')

        plt.clf()

        nx.draw(G, pos=pos, node_color=color_map, edge_color='white', node_size=np.array(size_map))# , with_labels=True)
        fig.set_facecolor('black')

        plt.pause(0.1)  ###(10 figures per second) in second the time a figure lasts
    plt.close()
