"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 07 October 2023

--------------------------------------------------------------------

Evolution of people and of infected individuals but no spreading of infection is implemented yet

"""

from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from math import gcd
from functools import reduce
import pickle
import time

################################################## FUNCTIONS #########################################################

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


def initialize_nodes(G, popTot, popI_init, idx_I_nodes ,Nfix, percentage_FixNodes, choice_bool, seed):
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

        In the initial state, I have only 'S' or 'I' individuals inside nodes. I assume to have no recovered ones at
        first stage.

    :param G: [networkx.class] graph structure from networkx
    :param popTot: [scalar] total population of the system
    :param popI_init: [scalar] number of individuals initially infected in nodes with index idx_I_nodes
    :param idx_I_nodes: [list] list of indices of nodes containing popI_init infected individuals
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
        nI = popI_init
        nR = 0
        nS = n - nI - nR

        # Create dictionary with population values assigned to each node (necessary to assign nodes diverse populations)
        dict_Npop = {i: n[i] for i in G.nodes}
        # If the node index is not in the list of nodes with infected individuals, I assign the population of
        # susceptible to be the total one. Otherwise, I assign nS = n - nI - nR
        dict_S = {i: n[i] if i not in idx_I_nodes else nS[i] for i in G.nodes}
        # If node index is not in the list of nodes with infected individuals, I assign the population of
        # infected to be 0. Otherwise, I assign nI (that is a constant value for now)
        dict_I = {i: 0 if i not in idx_I_nodes else nI for i in G.nodes}
        # Recovered people are 0 at the initial state
        dict_R = {i: nR for i in G.nodes}
        # Possible initial states of the node are 'S' if I have only susceptible individuals in it. Otherwise, is 'SI'
        dict_state = {i: 'S' if i not in idx_I_nodes else 'SI' for i in G.nodes}

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

# -------------------------------------------- Transition matrix  -----------------------------------------------------
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
                    T[i, j] = density[i] / D[i, j] # Tji (i->j)
                    T[j, i] = density[j] / D[i, j] # Tij (j->i)
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

# ---------------------------------------------- Perron-Frobeinus theorem ---------------------------------------------
from numpy.linalg import eig


def compute_perron_projection(M):
    """ Compute eigenvalues and eigenvectors ( left and right ) of the transition matrix.
        Finds the largest eigenvalue and calculates the Perron projection matrix.

    :param M: [matrix] will be the transition matrix - that is the stochastic matrix on which applying the PF theorem
    :return: P : Perron projection matrix
             r : Maximum eigenvalue (that for a stochastic matrix is 1)
    """
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
    """ Check convergence between the matrix (M/r)^n and the Perron projection matrix.
        (what does it mean ?)

    :param M:

    """
    P, r = compute_perron_projection(M)
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

#def perron_frobenius_theorem(TransMat):
#    PFval, PFvec = sla.eigs(TransMat.T, k=1, which='LR')
#    rho0 = abs(PFvec.T)[0]
#    rho0 = rho0 / rho0.sum()
#    rho0check = rho0.dot(TransMat)
#    print('Check normalised PFvec is invarant under Transition matrix: \n', rho0 - rho0check)

#    return rho0, rho0check

# --------------------------------------------------- Dynamics --------------------------------------------------------

def choice_particle_to_move(G, T):
    """ Stochastic choice of the number of particles to move inside a certain node i.
    Probabilities are given by the transition matrix array i (describes the interaction of node i with nodes j)
    The number of individuals is given by N... .

    :param T: [matrix] transition matrix (for now it is time independent)
    :return: Nij: [matrix] matrix of people going out of node i towards node j (row) and going into node i
                  from node j (col)
    """
    N = len(G.nodes)
    dict_Npop = nx.get_node_attributes(G, 'Npop')
    dict_N_S = nx.get_node_attributes(G, 'N_S')
    dict_N_I = nx.get_node_attributes(G, 'N_I')
    dict_N_R = nx.get_node_attributes(G, 'N_R')

    # Extract value of nodes
    NS_nodes = list(dict_N_S.values())
    NI_nodes = list(dict_N_I.values())
    NR_nodes = list(dict_N_R.values())

    Nij = np.zeros(shape = (N,N))
    Nij_S = np.zeros(shape = (N,N))
    Nij_I = np.zeros(shape = (N,N))
    Nij_R = np.zeros(shape = (N,N))
    for i in range(N):
        # Ti : ith row of the transition matrix
        Ti = np.array(T[i, :])
        Nij_S[i, :] = np.random.multinomial(NS_nodes[i], Ti)
        Nij_I[i, :] = np.random.multinomial(NI_nodes[i], Ti)
        Nij_R[i, :] = np.random.multinomial(NR_nodes[i], Ti)
        # The number of individuals that move is given by the sum of people in the three possible compartments
        Nij[i, :] = Nij_S[i, :] + Nij_I[i, :] + Nij_R[i, :]
    return Nij, Nij_S, Nij_I, Nij_R

def move_particle(G, Nij, Nij_S, Nij_I, Nij_R):
    N = len(G.nodes)
    # Dictionary with total population in each node
    dict_Npop = nx.get_node_attributes(G, 'Npop')
    dict_N_S = nx.get_node_attributes(G, 'N_S')
    dict_N_I = nx.get_node_attributes(G, 'N_I')
    dict_N_R = nx.get_node_attributes(G, 'N_R')
    # Extract Npop value of nodes
    lab_nodes = list(dict_Npop.keys())
    Npop_nodes = list(dict_Npop.values())
    # dictionary with other labels ???
    NS_nodes = list(dict_N_S.values())
    NI_nodes = list(dict_N_I.values())
    NR_nodes = list(dict_N_R.values())
    for i in range(N):
        for j in range(N):
            if i!=j:
                # Population going out from node i towards node j
                NS_nodes[i] -= Nij_S[i,j]
                NI_nodes[i] -= Nij_I[i,j]
                NR_nodes[i] -= Nij_R[i,j]
                Npop_nodes[i] -= Nij[i,j]
                # Population coming into node i from node j
                NS_nodes[i] += Nij_S[j,i]
                NI_nodes[i] += Nij_I[j,i]
                NR_nodes[i] += Nij_R[j,i]
                Npop_nodes[i] += Nij[j,i]

    dict_Npop = {lab_nodes[i]: Npop_nodes[i] for i in G.nodes}
    dict_N_S = {lab_nodes[i]: NS_nodes[i] for i in G.nodes}
    dict_N_I = {lab_nodes[i]: NI_nodes[i] for i in G.nodes}
    dict_N_R = {lab_nodes[i]: NR_nodes[i] for i in G.nodes}
    # Assign attributes to nodes
    nx.set_node_attributes(G, dict_Npop, 'Npop')
    nx.set_node_attributes(G, dict_N_S, 'N_S')
    nx.set_node_attributes(G, dict_N_I, 'N_I')
    nx.set_node_attributes(G, dict_N_R, 'N_R')

################################################################ MAIN ##################################################

start = time.time()

# --------------------------------------------------- Colors ----------------------------------------------------------

new_cmap = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6',
            '#6A3D9A', '#ECEC28', '#B15928', '#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6',
            '#6A3D9A', '#ECEC28', '#B15928', '#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6',
            '#6A3D9A', '#ECEC28', '#B15928']
#new_cmap = ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de", "#0571b0"]
rtg_r = LinearSegmentedColormap.from_list("rtg", new_cmap)
colors = rtg_r(np.linspace(0, 1, 100))

# --------------------------------------------- Parameter initialization ----------------------------------------------

seed = 66
np.random.seed(seed)

# Number of rows and columns in the lattice
N_row = 3
N_col = 3

# Average population per node (fixed)
avg_popPerNode = 1e3

# number of infected individuals in one node
popI_node = 1
# list of index of nodes initially containing popI_node infected individuals
idx_nodes_I_init = [0, 1]

# Number of fixed nodes containing the percentage percentage_FixNodes of population
Nfix = 3
percentage_FixNodes = 60

# choice_bool = 0 : uniform distribution
# choice_bool = 1 : Nfix nodes have percentage of population equal to percentage_FixNodes %
choice_bool = 0

# ------------------------------------------------ Network definition -------------------------------------------------
# Define node position in the lattice with a square topology
G, dict_nodes = initialize_lattice(N_row, N_col)
lab_nodes = list(dict_nodes.keys())
pos_nodes = list(dict_nodes.values())
# Number of nodes
N = len(G.nodes)
# Total population
populationTot = N * avg_popPerNode

# Compute distance matrix of every node with all the others
DistanceMatrix = distance_matrix(G, pos_nodes)
# Populate nodes and set initial conditions for infection
initialize_nodes(G, populationTot, popI_node, idx_nodes_I_init, Nfix, percentage_FixNodes, choice_bool, seed)
node_population = nx.get_node_attributes(G, name = 'Npop')
node_NS = nx.get_node_attributes(G, name = 'N_S')
node_NI = nx.get_node_attributes(G, name = 'N_I')
node_NR = nx.get_node_attributes(G, name = 'N_R')
node_population = np.array(list(node_population.values()))
node_NS = np.array(list(node_NS.values()))
node_NI = np.array(list(node_NI.values()))
node_NR = np.array(list(node_NR.values()))
print('node population: ', node_population)
print('node NS : ', node_NS)
print('node NI : ', node_NI)
print('node NR : ', node_NR)
node_density = node_population / populationTot  # population density vector
print('node density: ', node_density)

# Calculate transition matrix
TransitionMatrix = transition_matrix(G, DistanceMatrix, node_density)
weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0 ]
# Add weighted edges to networks : only edges with weight != 0 are added
for i in range(N):
    for j in range(N):
        if TransitionMatrix[i,j] != 0:
            G.add_edge(i, j, weight=TransitionMatrix[i, j])


# save graph object to file
#pickle.dump(G, open('G55.pickle', 'wb'))
# load graph object from file
#G = pickle.load(open('G55.pickle', 'rb'))

# Edge dictionary
dict_edges = nx.get_edge_attributes(G, name = 'weight')
# Control periodicity (the graph should be aperiodic)
#cycles = list(nx.algorithms.cycles.simple_cycles(G))
stopT1 = time.time()
durationT1 = stopT1 - start
print('duration T1:', durationT1)
#cycles_sizes = [len(c) for c in cycles]
#cycles_gcd = reduce(gcd, cycles_sizes)
#is_periodic = cycles_gcd > 1
#print("is_periodic: {}".format(is_periodic))
# Control strongly connected graph
strongConnection = nx.is_strongly_connected(G)
print('Strong connection : ', strongConnection)
stop1 = time.time()
duration1 = stop1 - start
print('Duration up to computation Transition matrix: ', duration1)
# Plot network
plot_network(G, node_population, dict_nodes, weightNonZero)
stop2 = time.time()
duration2 = stop2 - start
print('Duration up to plot of network: ', duration2)
check_convergence(TransitionMatrix)
stop3 = time.time()
duration3 = stop3 - start
print('Duration up to check convergence: ', duration3)
#rho0, rho0check = perron_frobenius_theorem(TransitionMatrix)

plot_network(G, node_population, dict_nodes, weightNonZero)

# ------------------------------------------------ Dynamics -------------------------------------------------
# total simulation length
T = 10
T_sim = np.linspace(0, T, T)

idx_node = 0

#fig2 = plt.figure(figsize=(10,10))
for idx_node in range(1):
    popNode_idx = []
    popDensity_idx = []
    # t starts from 1 because t = 0 is the initial condition
    for t in range(1, T):
        # Plot temporal evolution of network
        plt.clf()
        plot_network(G, node_NI, dict_nodes, weightNonZero)
        plt.pause(1)  ###(10 figures per second) in second the time a figure lasts
        Nij, Nij_S, Nij_I, Nij_R = choice_particle_to_move(G, TransitionMatrix)
        move_particle(G, Nij, Nij_S, Nij_I, Nij_R)
        node_population = nx.get_node_attributes(G, name = 'Npop')
        node_NS = nx.get_node_attributes(G, name='N_S')
        node_NI = nx.get_node_attributes(G, name='N_I')
        node_NR = nx.get_node_attributes(G, name='N_R')
        node_population = np.array(list(node_population.values()))
        node_NS = np.array(list(node_NS.values()))
        node_NI = np.array(list(node_NI.values()))
        node_NR = np.array(list(node_NR.values()))
        print('t: ', t, 'Npop:', node_population)
        print('t: ', t, 'NS: ', node_NS)
        print('t: ', t, 'NI: ', node_NI)
        print('t: ', t, 'NR: ', node_NR)
        node_density = node_population/populationTot
        popNode_idx.append(node_population[idx_node])
        popDensity_idx.append(node_density[idx_node])
        #print('node_pop after:', node_population)
        # Control that the total population is exactly the same as the initial one
        print('total pop: before -> ', populationTot, 'after ->', node_population.sum())
    #plt.close()
    #plt.plot(T_sim, popDensity_idx, color = colors[idx_node])
#plt.axhline(y = avg_popPerNode/populationTot, color = 'black', linestyle = '--', label = 'Fixed average density per node')
#plt.legend()
#plt.xlabel('Timestep')
#plt.ylabel('Node density')
plt.show()