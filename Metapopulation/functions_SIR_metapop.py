"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 27 September 2023

--------------------------------------------------------------------

Functions useful to implement simulations using metapopulation approach

"""

import networkx as nx
import numpy as np
import random as rnd
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


def initialize_nodes(G, popTot, Nfix, percentage_FixNodes, choice_bool):
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
    N = len(G.nodes)
    # Populate nodes
    if choice_bool == 0:
        # Extract population of nodes from a multinomial distribution. it is a ndarray
        n = np.random.multinomial(popTot, [1 / N] * N)
        nS = n
        nI = 0
        nR = 0
        state = 'S'
        print('node populations', n)
        # Create dictionary with population values assigned to each node (necessary to assign nodes diverse populations)
        dict_Npop = {i: n[i] for i in G.nodes}
        dict_S = {i: nS[i] for i in G.nodes}
        dict_I = {i: nI for i in G.nodes}
        dict_R = {i: nR for i in G.nodes}
        dict_state = {i: state for i in G.nodes}
        print('dict Npop:', dict_Npop)

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
            idxN = rnd.randint(0, N - 1)
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


def transition_matrix(G, D, density, c):
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
                prob = c * density[i] * density[j] / D[i, j]
                rnd_ch = np.random.choice([1, 0], p=[prob, 1 - prob])
                if rnd_ch == 1:
                    T[i, j] = prob
            # self loop
        T[i, i] = 1. - T[i, :].sum()

    return T


def characterisation_network_SC(G, DistanceMatrix, node_density, search_max_number, c_min, c_max, max_trialsSC_fixed_c):
    """ Cycle over the c parameter of the gravity law to characterise which values guarantee the network
     is strongly connected
     :param G: [networkx.class] graph structure from networkx
     :param DistanceMatrix: matrix of Euclidean distance
     :param node_density: [scalar] density of the node
     :param search_max_number: [scalar] maximum number of iterations for a certain network topology to search
     for a strongly connected graph
     :param c_min: [scalar] minimum value for c to start searching for strong connection
     :param c_max: [scalar] maximum value for c to search for strong connection
     :param max_trialsSC_fixed_c: [scalar] limit number of repetition for search of strong connection using the same c


    """
    c_list = []
    N = len(G.nodes)
    for repeat_search in range(0, search_max_number):
        strongConnection = False
        for c in np.arange(c_min, c_max, 1):
            contFalse = 0
            while strongConnection == False and contFalse < max_trialsSC_fixed_c:
                contFalse = contFalse + 1
                # Transition matrix
                TransitionMatrix = transition_matrix(G, DistanceMatrix, node_density, c)
                # Add weighted edges to networks : only edges with weight != 0 are added
                for i in range(N):
                    for j in range(N):
                        if TransitionMatrix[i, j] != 0:
                            G.add_edge(i, j, weight=TransitionMatrix[i, j])
                # Control strongly connected graph
                strongConnection = nx.is_strongly_connected(G)
                # After controlling, remove edges
                for i in range(N):
                    for j in range(N):
                        if TransitionMatrix[i, j] != 0:
                            G.remove_edge(i, j)
            if strongConnection:
                c_list.append(c)
                print(f'{repeat_search} : Strong connection for c = {c} : ', strongConnection)
                # print(f'False iterations for c = {c}:', contFalse)
                break
    c_list = np.array(c_list)
    # Arithmetic mean of c_list reported as an integer value (use fmean() to return the float arithmetic mean)
    # Use mean in analogy to resolution of instrument: can't go
    avg_c = stat.fmean(c_list)
    err_c = stat.stdev(c_list, avg_c)
    return c_list, avg_c, err_c


def perron_frobenius_theorem(TransMat):
    PFval, PFvec = sla.eigs(TransMat.T, k=1, which='LR')
    rho0 = abs(PFvec.T)[0]
    rho0 = rho0 / rho0.sum()
    rho0check = rho0.dot(TransMat)
    print('Check normalised PFvec is invarant under Transition matrix: \n', rho0 - rho0check)

    return rho0, rho0check
