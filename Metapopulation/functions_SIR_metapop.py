"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 27 September 2023

--------------------------------------------------------------------

Functions useful to implement simulations using metapopulation approach

"""

import networkx as nx
import numpy as np
import math
import scipy.sparse.linalg as sla
import statistics as stat


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


def initialize_nodes(G, popTot, Nfix, percentage_FixNodes, choice_bool, seed):
    """ Assign nodes with attributes:
        Npop : is the population assigned to each node. Values are extracted from a multinomial distribution:
               0. with support equal to popTot, and probability 1/N equal for each of the N classes.
               1. in which the number Nfix of selected nodes contains the percentage percentage_FixNodes of population
               Multinomial distribution ensures that the sum of all elements is
               equal to the whole population. Being probabilities all equal to 1/N, random values are sampled from a
               uniform distribution (size = N).
        N_S : initial number of susceptible individuals
        N_I : initial number of infected individuals
        N_R : initial number of recovered individuals
        state : initial state of individuals

    :param G: [networkx.class] graph structure from networkx
    :param popTot: [scalar] total population of the system
    :param Nfix: [scalar] number of selected nodes to set the percentage 'percentage_FixNodes' of the population
    :param percentage_FixNodes: [scalar] percentage of the population to set in Nfix selected nodes
    :param choice_bool: [0 or 1] boolean-like variable -
                        if 0 : populate nodes from a uniform probability distribution
                        if 1 : populate Nfix of nodes with 80% of population and the remaining 20% is
                               distributed among the remaining N-Nfix of nodes

    """
    if seed is not None: np.random.seed(seed)

    N = len(G.nodes)
    # Populate nodes
    if choice_bool == 0:
        # Extract population of nodes from a multinomial distribution. it is a ndarray
        n = np.random.multinomial(popTot, [1 / N] * N)
        nS = n
        nI = 0
        nR = 0
        state = 'S'
        # Create dictionary with population values assigned to each node (necessary to assign nodes diverse populations)
        dict_Npop = {i: n[i] for i in G.nodes}
        dict_S = {i: nS[i] for i in G.nodes}
        dict_I = {i: nI for i in G.nodes}
        dict_R = {i: nR for i in G.nodes}
        dict_state = {i: state for i in G.nodes}

        # Assign attributes to nodes
        nx.set_node_attributes(G, dict_Npop, 'Npop')
        nx.set_node_attributes(G, dict_S, 'N_S')
        nx.set_node_attributes(G, dict_I, 'N_I')
        nx.set_node_attributes(G, dict_R, 'N_R')
        nx.set_node_attributes(G, dict_state, 'state')

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

        n_S = n_AllNodes_final
        n_I = 0
        n_R = 0
        state = 'S'

        # List with index of all nodes
        idx_AllNodes = [i for i in range(0, N)]
        idx_FixNodes = []
        for i in range(Nfix + 1):
            idxN = np.random.randint(0, N - 1)
            if idxN not in idx_FixNodes:
                idx_FixNodes.append(idxN)
        idx_others = list(set(idx_AllNodes) - set(idx_FixNodes))
        idx_AllNodes_final = idx_FixNodes + idx_others
        print('idx:', idx_AllNodes_final)
        # Dictionary in which at nodes with index in idx_FixNodes I attribute the population
        dict_Npop = {idx_AllNodes_final[i]: n_AllNodes_final[i] for i in G.nodes}
        dict_S = {idx_AllNodes_final[i]: n_S[i] for i in G.nodes}
        dict_I = {idx_AllNodes_final[i]: n_I for i in G.nodes}
        dict_R = {idx_AllNodes_final[i]: n_R for i in G.nodes}
        dict_state = {idx_AllNodes_final[i]: state for i in G.nodes}
        print(dict_Npop)

        # Assign attributes to nodes
        nx.set_node_attributes(G, dict_Npop, 'Npop')
        nx.set_node_attributes(G, dict_S, 'N_S')
        nx.set_node_attributes(G, dict_I, 'N_I')
        nx.set_node_attributes(G, dict_R, 'N_R')
        nx.set_node_attributes(G, dict_state, 'state')
    else:
        print('Wrong value for choice_bool')


def transition_matrix(G, D, density):
    """ Compute probability to create an edge and its reversed one.
        Compute weights of edges that correspond to the transition probability of people
        among nodes.

    :param G: [networkx.class] graph structure from networkx
    :param D: matrix of Euclidean distance
    :param density: [np.array] population density inside every node
    """
    N = len(G.nodes)
    N_row = N
    N_col = N

    max_density = max(density)
    a = 0.2
    b = 0.9
    # Parameter that quantifies the number of connections between nodes
    c = a / max(density)
    T = np.zeros(shape=(N_row, N_col))
    # Normalization condition ensured by the self loop value
    for i in range(N_row):
        for j in range(i):
            if i != j:
                # Probability to establish both the direct and forward edge in a pair of nodes
                prob = max(c * density[i]/D[i,j], c * density[j]/D[i,j])
                #print('prob:', prob)
                #prob = min(c * maxDensity/ minDi - c * density[i] / D[i, j], c * maxDensity/ minDi - c * density[j] / D[i, j], 1.)
                #print('a- ', c * maxDensity/ minDi - c * density[i] / D[i, j],'b- ', c * maxDensity/ minDi - c * density[j] / D[i, j] )
                rnd_ch = np.random.choice([1, 0], p=[prob, 1 - prob])
                if rnd_ch == 1:
                    T[i, j] = density[i] / D[i, j]
                    T[j, i] = density[j] / D[i, j]
                #else:
                    #print('No edge!')
    # sum over all the rows and take the maximum between these sums and call it Pmax.
    # axis = 1 sums over rows
    Pmax = T.sum(axis=1).max()

    c1 = b / Pmax
    T *= c1
    for i in range(N_row):
        # Self loop
        T[i, i] = 1. - T[i, :].sum()
        if T[i, i] < 0:
            print(f'ERROR : SELF LOOP WITH PROBABILITY < 0, i = {i}')

    return T






def perron_frobenius_theorem(TransMat):
    PFval, PFvec = sla.eigs(TransMat.T, k=1, which='LR')
    rho0 = abs(PFvec.T)[0]
    rho0 = rho0 / rho0.sum()
    rho0check = rho0.dot(TransMat)
    print('Check normalised PFvec is invarant under Transition matrix: \n', rho0 - rho0check)

    return rho0, rho0check

from numpy.linalg import eig
def compute_perron_projection(M):
    # v : right eigenvector
    eigval, v = eig(M)
    # w : left eigenvector
    eigval, w = eig(M.T)

    # maximum eigenvalue (must be real)
    r = np.max(eigval)

    # Find the index of the dominant (Perron) eigenvalue
    i = np.argmax(eigval)

    # Get the Perron eigenvectors
    v_P = v[:, i].reshape(-1, 1)
    w_P = w[:, i].reshape(-1, 1)

    # Normalize the left and right eigenvectors
    norm_factor = w_P.T @ v_P
    v_norm = v_P / norm_factor
    w_norm = w_P / norm_factor

    # Compute the Perron projection matrix
    P = v_norm @ w_P.T

    return P, r

def check_convergence(M):
    P, r = compute_perron_projection(M)
    print("Perron projection:")
    print(P)
    # Define a list of values for n
    n_list = [1, 10, 100, 1000, 10000]

    for n in n_list:
        # Compute (A/r)^n
        M_n = np.linalg.matrix_power(M / r, n)

        # Compute the difference between A^n / r^n and the Perron projection
        diff = np.abs(M_n - P)

        # Calculate the norm of the difference matrix
        diff_norm = np.linalg.norm(diff, 'fro')
        print(f"n = {n}, error = {diff_norm:.10f}")