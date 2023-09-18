"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 07 September 2023

--------------------------------------------------------------------

Functions useful to implement simulations using metapopulation approach

"""

import networkx as nx
import numpy as np
import random as rnd
import math
import scipy.sparse.linalg as sla


def initialize_lattice(N_row, N_col):
    """ Define a lattice with square topology with N_row rows and N_col columns, that is a
        networkx structure with positioning of nodes only.

    :param N_row: number of rows of the lattice
    :param N_col: number of columns of the lattice

    """
    # Directed graph
    G = nx.DiGraph()
    # Number of nodes in the graph
    N = int(N_row * N_col)
    # Label of nodes
    keys = range(N)
    # Position of nodes [tuple]
    values = []
    for i in range(N_row):
        for j in range(N_col):
            values.append((i, j))
            # values.append((j, -i))

    # Dictionary : {<keys>: <values>}
    vals = {i: values[i] for i in keys}

    # Set nodes in the graph
    G.add_nodes_from(vals.keys())
    # Assign to pos the position of nodes in the graph that were set in 'values'
    pos = list(vals.values())

    return G, pos, vals


def distance_matrix(G, pos):
    """ Compute Euclidean distance between every node and all the others
        in the lattice

    :param G : [networkx.class ] graph structure from networkx
    :param pos : [list] position of nodes

    :return D : [ndarray (N_row x N_col)] matrix with Euclidean distances between nodes

    """
    # Number of nodes in the lattice
    N = len(G.nodes)
    # Pedantic definition - to express dimensions of the matrix
    N_row = N
    N_col = N
    D = np.zeros(shape=(N_row, N_col))
    for i in range(N_row):
        for j in range(N_col):
            pos_i = np.asarray(pos[i])
            pos_j = np.array(pos[j])
            D[i, j] = np.linalg.norm(pos_i - pos_j)

    return D


def create_node_list(G, popTot, Nfix, percentage_FixNodes, choice_bool):
    """ Create a list with attributes associated to each node of the lattice.
        Population values are extracted from a multinomial distribution:
        0. with support equal to popTot, and probability 1/N equal for each of the N classes.
        1. in which the number Nfix of selected nodes contains the percentage percentage_FixNodes of population


    :param G: [networkx.class] graph structure from networkx
    :param popTot: [scalar] total population of the system
    :param Nfix: [scalar] number of selected nodes to set the percentage 'percentage_FixNodes' of the population
    :param percentage_FixNodes: [scalar] percentage of the population to set in Nfix selected nodes
    :param choice_bool: [0 or 1] boolean-like variable -
                        if 0 : populate nodes from a uniform probability distribution
                        if 1 : populate Nfix of nodes with 80% of population and the remaining 20% is
                               distributed among the remaining N-Nfix of nodes

    """
    # Number of nodes in the graph
    N = len(G.nodes)

    # Initialize node list
    class Node:
        def __init__(self, index):
            self.index = index
            self.Npart = 0
            self.N_S = 0
            self.N_I = 0
            self.N_R = 0
            self.state = 'S'
    # vector with population in each node. Multinomial distribution ensures that the sum of all elements is
    # equal to the whole population. Being probabilities all equal to 1/N, random values are sampled from a
    # uniform distribution (size = N).
    lst_nodes = []
    if choice_bool == 0:
        n = np.random.multinomial(popTot, [1 / N] * N)
        for i in G.nodes():
            lst_nodes.append(Node(i))
            lst_nodes[i].Npart = n[i]
    elif choice_bool == 1:
        pop_FixNodes = math.floor(percentage_FixNodes / 100 * popTot)
        pop_others = popTot - pop_FixNodes
        n_FixNodes = np.random.multinomial(pop_FixNodes, [1 / Nfix] * Nfix)
        n_others = np.random.multinomial(pop_others, [1 / (N-Nfix)] * (N-Nfix))
        for i in range(Nfix):
            lst_nodes.append(Node(i))
            lst_nodes[i].Npart = n_FixNodes[i]
        for i in range(Nfix, N):
            lst_nodes.append(Node(i))
            lst_nodes[i].Npart = n_others[i-Nfix]

    return lst_nodes


def transition_matrix(G, D, density):
    """ Compute weights of edges that correspond to the transition probability of people
        among nodes. Probability is proportional to the population of the destination node
        and inversely proportional to the distance between nodes (following a gravity law)

    :param G: [networkx.class] graph structure from networkx
    :param D: matrix of Euclidean distance
    :param density: [np.array] population density inside every node
    """
    N = len(G.nodes)

    N_row = N
    N_col = N
    T = np.zeros(shape=(N_row, N_col))
    # Calculate transition probabilities
    # To add proportionality term (and eventually population of the destination node)
    # NOTE : sum over i must be 1

    for i in range(N_row):
        for j in range(N_col):
            if i != j:  # implements the random condition (?)
                prob = density[j] / D[i, j]  # TO DO : 4 * pop_density /D
                rnd_ch = np.random.choice([1, 0], p=[prob, 1 - prob])
                if rnd_ch == 1:
                    T[i, j] = density[j] / D[i, j]
            # self loop
        T[i, i] = 1. - T[i, :].sum()

    return T





def perron_frobenius_theorem(TransMat):
    PFval, PFvec = sla.eigs(TransMat.T, k=1, which='LR')
    rho0 = abs(PFvec.T)[0]
    rho0 = rho0 / rho0.sum()
    rho0check = rho0.dot(TransMat)
    print('Check normalised PFvec is invarant under Transition matrix: \n', rho0 - rho0check)

    return rho0, rho0check