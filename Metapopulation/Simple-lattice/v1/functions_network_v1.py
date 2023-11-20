"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 10 November 2023

--------------------------------------------------------------------

Functions to initialize the network

"""

import networkx as nx
import numpy as np
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

    return G, vals


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


def initialize_node_population(G, popTot,Nfix, percentage_FixNodes, choice_bool, seed):
    """ Assign nodes with attributes:
        Npop : is the population assigned to each node. Values are extracted from a multinomial distribution:
               0. with support equal to popTot, and probability 1/N equal for each of the N classes.
               1. in which the number Nfix of selected nodes contains the percentage percentage_FixNodes of population
               Multinomial distribution ensures that the sum of all elements is
               equal to the whole population. Being probabilities all equal to 1/N, random values are sampled from a
               uniform distribution (size = N).

    :param G: [networkx.class] graph structure from networkx
    :param popTot: [scalar] total population of the system
    :param Nfix: [scalar] number of selected nodes to set the percentage 'percentage_FixNodes' of the population
    :param percentage_FixNodes: [scalar] percentage of the population to set in Nfix selected nodes
    :param choice_bool: [0 or 1] boolean-like variable -
                        if 0 : populate nodes from a uniform probability distribution
                        if 1 : populate Nfix of nodes with 80% of population and the remaining 20% is
                               distributed among the remaining N-Nfix of nodes

    """
    if seed is not None : np.random.seed(seed)

    N = len(G.nodes)
    # Populate nodes
    if choice_bool == 0:
        # Extract population of nodes from a multinomial distribution. it is a ndarray
        n = np.random.multinomial(popTot, [1 / N] * N)
        # Create dictionary with population values assigned to each node (necessary to assign nodes diverse populations)
        dict_Npop = {i: n[i] for i in G.nodes}
        # Assign attributes to nodes
        nx.set_node_attributes(G, dict_Npop, 'Npop')

    elif choice_bool == 1:
        # Whole population in fixed nodes
        pop_FixNodes = math.floor(percentage_FixNodes / 100 * popTot)
        # Whole population in all other nodes
        pop_others = popTot - pop_FixNodes
        # Distributed population among individual fixed nodes
        n_FixNodes = np.random.multinomial(pop_FixNodes, [1 / Nfix] * Nfix)
        # Distributed population among all other nodes
        n_others = np.random.multinomial(pop_others, [1 / (N - Nfix)] * (N - Nfix))
        n_FixNodes = list(n_FixNodes)
        n_others = list(n_others)
        n_AllNodes_final = n_FixNodes + n_others

        # List with index of all nodes
        idx_AllNodes = [i for i in range(0, N)]
        idx_FixNodes = []
        for i in range(Nfix):
            noInList = True
            idxN = np.random.randint(0, N - 1)
            if idxN not in idx_FixNodes:
                idx_FixNodes.append(idxN)
            else:
                noInList = False
                while noInList == False:
                    idxN = np.random.randint(0, N - 1)
                    if idxN not in idx_FixNodes:
                        idx_FixNodes.append(idxN)
                        noInList = True



        idx_others = list(set(idx_AllNodes) - set(idx_FixNodes))
        idx_AllNodes_final = idx_FixNodes + idx_others
        # Dictionary in which at nodes with index in idx_FixNodes I attribute the population
        dict_Npop = {idx_AllNodes_final[i]: n_AllNodes_final[i] for i in G.nodes}
        # Assign attributes to nodes
        nx.set_node_attributes(G, dict_Npop, 'Npop')
        print('idx fixed nodes: ', idx_FixNodes)
        print(len(idx_FixNodes))
        return idx_FixNodes
    else:
        print('Wrong value for choice_bool')

    return 0

def adjacency_matrix(G, D, density, a):
    """ Calculate the adjacency matrix that represent the connectivity of the network.

    :param G:
    :param D:
    :param density:
    :param a:
    :return:
    """
    N = len(G.nodes)

    # Parameter that quantifies the number of connections between nodes
    c = a / max(density)

    # Normalization condition ensured by the self loop value
    A = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            if j > i:
                # Probability to establish both the direct and forward edge in a pair of nodes
                prob = max(c * density[i] / D[i, j], c * density[j] / D[i, j])
                rnd_ch = np.random.choice([1, 0], p=[prob, 1 - prob])
                # This is the adjacency matrix built from the random choice
                A[i, j] = rnd_ch
                A[j, i] = rnd_ch

    return A

def transition_matrix(G, A, D, density, b):
    """ Compute probability to create an edge and its reversed one.
        Compute weights of edges that correspond to the transition probability of people
        among nodes.

    :param G: [networkx.class] graph structure from networkx
    :param D: matrix of Euclidean distance
    :param density: [np.array] population density inside every node
    :param a : parameter that accounts for the quantity of connections
    :param b : parameter that accounts for the strength of the self-loop
    """
    N = len(G.nodes)

    T = np.zeros(shape=(N, N))

    for i in range(N):
        for j in range(N):
            if j > i:
                if A[i,j] == 1:
                    T[i, j] = density[j] / D[i, j] # Tji (i->j)
                    T[j, i] = density[i] / D[i, j] # Tij (j->i)
    # sum over all the rows and take the maximum between these sums and call it Pmax.
    # axis = 1 sums over rows
    Pmax = T.sum(axis=1).max()
    c1 = b / Pmax
    T *= c1
    for i in range(N):
        # Self loop
        T[i, i] = 1. - T[i, :].sum()
        #print('T[i.i]: ', T[i,i])
        if T[i, i] < 0:
            print(f'ERROR : SELF LOOP WITH PROBABILITY < 0, i = {i}')

    return T, c1


def PF_convergence(A):
    # 1. Find the stationary probability distribution (that is the normalized left PF eigenvector)
    # Right PF eigvals and eigvect

    PF_eigval, PF_r = sla.eigs(A, k = 1, which = 'LR')
    print('PF_eigval: ',PF_eigval)
    print('PFr: ', PF_r)
    # Left PF eigvals and eigvect
    PF_eigval, PF_l = sla.eigs(A.T, k = 1, which = 'LR')
    print('PFl: ', PF_l)
    # Normalized stationary probability distribution (density of people)
    rho0 = np.abs(PF_l.T)[0]
    rho0 = rho0 / rho0.sum()

    # 2. Check long term convergence through the Perron-projection matrix
    # Normalization such that l^T * r = 1
    norm_factor = PF_l.T @ PF_r
    PF_l_norm = PF_l / norm_factor
    PF_r_norm = PF_r / norm_factor
    #Perron projection matrix
    P = PF_r_norm @ PF_l_norm.T
    # Verify that lim_{k-> infty} (A/eigval)^k = P
    k_list = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    diff_norm_lst = []
    for k in k_list:
        A_k = np.linalg.matrix_power(A/PF_eigval, k)
        diff = np.abs(A_k - P)
        # Calculate the norm of the difference matrix
        diff_norm = np.linalg.norm(diff, 'fro')
        diff_norm_lst.append(diff_norm)
        print(f"n = {k}, error = {diff_norm:.10f}")
    k_list = np.array(k_list)
    diff_norm_lst = np.array(diff_norm_lst)

    return rho0, k_list, diff_norm_lst
