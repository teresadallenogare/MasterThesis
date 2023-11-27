"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 27 November 2023

--------------------------------------------------------------------

Functions to perform SIR metapopulation infection process

"""
import networkx as nx
import numpy as np

# --------------------------------------------------- Simulation --------------------------------------------------------
def initial_configuration_SIR(G, node_pop0, popI_init, idx_I_nodes ,Nfix, percentage_FixNodes, choice_bool, seed):
    """ Assign nodes with attributes:

            N_S : initial number of susceptible individuals
            N_I : initial number of infected individuals
            N_R : initial number of recovered individuals
            state : initial state of individuals

            In the initial state, I have only 'S' or 'I' individuals inside nodes. I assume to have no recovered ones at
            first stage.

        :param G: [networkx.class] graph structure from networkx
        :param node_pop0: [ndarray] initial population in every node
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
    nI = popI_init
    nR = 0
    nS = node_pop0 - nI - nR
    # Populate nodes
    if choice_bool == 0 or choice_bool == 1:
        # If the node index is not in the list of nodes with infected individuals, I assign the population of
        # susceptible to be the total one. Otherwise, I assign nS = n - nI - nR
        dict_S = {i: node_pop0[i] if i not in idx_I_nodes else nS[i] for i in G.nodes}
        # If node index is not in the list of nodes with infected individuals, I assign the population of
        # infected to be 0. Otherwise, I assign nI (that is a constant value for now)
        dict_I = {i: 0 if i not in idx_I_nodes else nI for i in G.nodes}
        # Recovered people are 0 at the initial state
        dict_R = {i: nR for i in G.nodes}
        # Possible initial states of the node are 'S' if I have only susceptible individuals in it. Otherwise, is 'SI'
        dict_state = {i: 'S' if i not in idx_I_nodes else 'SI' for i in G.nodes}

        # Assign attributes to nodes
        nx.set_node_attributes(G, dict_S, 'N_S')
        nx.set_node_attributes(G, dict_I, 'N_I')
        nx.set_node_attributes(G, dict_R, 'N_R')
        nx.set_node_attributes(G, dict_state, 'state')



def choice_particle_to_move(G, T):
    """ Stochastic choice of the number of particles to move inside a certain node i.
    Probabilities are given by the transition matrix array i (describes the interaction of node i with nodes j)
    The number of individuals is given by N... .

    :param G: [networkx.class] graph structure from networkx
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
    # Get attributes before the motion
    # Dictionary with total population in each node
    dict_Npop = nx.get_node_attributes(G, 'Npop')
    dict_N_S = nx.get_node_attributes(G, 'N_S')
    dict_N_I = nx.get_node_attributes(G, 'N_I')
    dict_N_R = nx.get_node_attributes(G, 'N_R')
    dict_state = nx.get_node_attributes(G, 'state')

    # Extract Npop value of nodes
    lab_nodes = list(dict_Npop.keys())
    Npop_nodes = list(dict_Npop.values())
    NS_nodes = list(dict_N_S.values())
    NI_nodes = list(dict_N_I.values())
    NR_nodes = list(dict_N_R.values())
    state_nodes = list(dict_state.values())

    # Perform motion of particles and update number of particles
    for i in range(N):
        for j in range(N):
            if i != j:
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

    # Set new attributes after the motion
    dict_Npop = {lab_nodes[i]: Npop_nodes[i] for i in G.nodes}
    dict_N_S = {lab_nodes[i]: NS_nodes[i] for i in G.nodes}
    dict_N_I = {lab_nodes[i]: NI_nodes[i] for i in G.nodes}
    dict_N_R = {lab_nodes[i]: NR_nodes[i] for i in G.nodes}

    # Update the state of each node AFTER the move
    for i in range(N):
        if NS_nodes[i] == Npop_nodes[i]:
            state_nodes[i] = 'S'
        elif NI_nodes[i] == Npop_nodes[i]:
            state_nodes[i] = 'I'
        elif NR_nodes[i] == Npop_nodes[i]:
            state_nodes[i] = 'R'
        elif NS_nodes[i] + NI_nodes[i] == Npop_nodes[i]:
            state_nodes[i] = 'SI'
        elif NS_nodes[i] + NR_nodes[i] == Npop_nodes[i]:
            state_nodes[i] = 'SR'
        elif NI_nodes[i] + NR_nodes[i] == Npop_nodes[i]:
            state_nodes[i] = 'IR'
        else:
            state_nodes[i] = 'SIR'

    dict_state = {i: state_nodes[i] for i in G.nodes}

    # Assign attributes to nodes
    nx.set_node_attributes(G, dict_Npop, 'Npop')
    nx.set_node_attributes(G, dict_N_S, 'N_S')
    nx.set_node_attributes(G, dict_N_I, 'N_I')
    nx.set_node_attributes(G, dict_N_R, 'N_R')
    nx.set_node_attributes(G, dict_state, 'state')


# ---------------------------------- Infection spreading simulation ----------------------------------------------

def infection_step_node(G, beta, mu):
    """ Infect people inside the node according to a binomial distribution because an individual has 2 possibilities:
        or it changes its state or it remains in the same one.
        The number of new infected individuals is counted by taking each susceptible and, with probability given by the
        force of infection alpha, test it to and count the number of susceptible that turned into an infected person.
        The same is done for the transition from infected to recovered but, this time, with probability mu.

    :param G: [networkx.class] graph structure from networkx
    :param beta: [scalar] rate of infection, kept constant
    :param mu: [scalar] rate of recovery, kept constant
    :return:
    """

    N = len(G.nodes)
    # now I keep beta and mu fixed

    # Dictionary with total population in each node
    dict_Npop = nx.get_node_attributes(G, 'Npop')
    dict_N_S = nx.get_node_attributes(G, 'N_S')
    dict_N_I = nx.get_node_attributes(G, 'N_I')
    dict_N_R = nx.get_node_attributes(G, 'N_R')
    dict_state = nx.get_node_attributes(G, 'state')

    # Extract Npop value of nodes
    lab_nodes = list(dict_Npop.keys())
    Npop_nodes = list(dict_Npop.values())
    NS_nodes = list(dict_N_S.values())
    NI_nodes = list(dict_N_I.values())
    NR_nodes = list(dict_N_R.values())
    state_nodes = list(dict_state.values())

    alpha = np.zeros(N)
    NI_nodes_new = np.zeros(N)
    NR_nodes_new = np.zeros(N)

    for i in range(N):
        # force of infection : rate of infection * (infected population in node i / total population in node i)
        alpha[i] = beta * NI_nodes[i]/Npop_nodes[i]
        #print('NI', NI_nodes[i]/Npop_nodes[i])
        #print('alpha: ', alpha[i])
        # if the node contains infected ('I', 'SI', 'RI', 'SIR')
        if 'I' in state_nodes[i]:
            # Generate from a binomial distribution new infected and recovered individuals
            NI_nodes_new[i] = np.random.binomial(NS_nodes[i], alpha[i], 1)
            NR_nodes_new[i] = np.random.binomial(NI_nodes[i], mu, 1)
            NS_nodes[i] -= NI_nodes_new[i]
            NI_nodes[i] += NI_nodes_new[i]
            NI_nodes[i] -= NR_nodes_new[i]
            NR_nodes[i] += NR_nodes_new[i]
            # Change state after the infection has taken place in the node
            if NS_nodes[i] == Npop_nodes[i]:
                state_nodes[i] = 'S'
            elif NI_nodes[i] == Npop_nodes[i]:
                state_nodes[i] = 'I'
            elif NR_nodes[i] == Npop_nodes[i]:
                state_nodes[i] = 'R'
            elif NS_nodes[i] + NI_nodes[i] == Npop_nodes[i]:
                state_nodes[i] = 'SI'
            elif NS_nodes[i] + NR_nodes[i] == Npop_nodes[i]:
                state_nodes[i] = 'SR'
            elif NI_nodes[i] + NR_nodes[i] == Npop_nodes[i]:
                state_nodes[i] = 'IR'
            else:
                state_nodes[i] = 'SIR'

    # Set new attributes after the motion
    dict_N_S = {lab_nodes[i]: NS_nodes[i] for i in G.nodes}
    dict_N_I = {lab_nodes[i]: NI_nodes[i] for i in G.nodes}
    dict_N_R = {lab_nodes[i]: NR_nodes[i] for i in G.nodes}
    dict_state = {i: state_nodes[i] for i in G.nodes}

    # Assign attributes to nodes
    nx.set_node_attributes(G, dict_N_S, 'N_S')
    nx.set_node_attributes(G, dict_N_I, 'N_I')
    nx.set_node_attributes(G, dict_N_R, 'N_R')
    nx.set_node_attributes(G, dict_state, 'state')

    return NI_nodes_new

# ---------------------------------- Deterministic SIR  ----------------------------------------------

def SIRDeterministic_equations(variables, t, params):
  """ Determinisitc ODE for the SIR model. I consider equaitons for densities:
  ds/dt = - beta * i * s = - alpha * s
  di/dt = beta * i * s - mu * i = alpha * s - mu * i
  dr/dt = mu * i

  :param variables: [s, i, r] : densities
  :param t:
  :param params: [beta, mu] : infection rate and recovery rate
  :return:
  """
  s = variables[0]
  i = variables[1]

  beta = params[0]
  mu = params[1]

  alpha = beta * i

  dsdt = - alpha * s
  didt = alpha * s - mu * i
  drdt = mu * i

  return [dsdt, didt, drdt]
