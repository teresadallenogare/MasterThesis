"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 27 September 2023

--------------------------------------------------------------------

Study how the value of the total population and the number of nodes in the network influence the value of c for which I
first have a strongly connected graph.

"""
from functions_SIR_metapop import *
import numpy as np
import matplotlib.pyplot as plt



# Characterisation of the network : study of how the c parameter varies as a function of the total population
# and of the dimension of the network

## Second trial :
# --------------------------------------------- Parameter initialization ----------------------------------------------

# Number of rows and columns in the lattice
N_row = 3
N_col = 3
# Total population
populationTot = 1e2
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
# Compute distance matrix of every node with all the others
DistanceMatrix = distance_matrix(G, pos_nodes)

# Populate nodes and set initial conditions for infection
initialize_nodes(G, populationTot, Nfix, percentage_FixNodes, choice_bool)
node_population = nx.get_node_attributes(G, name = 'Npop')
node_population = np.array(list(node_population.values()))
print(type(node_population))
print('node population: ', node_population)
node_density = node_population / populationTot  # population density vector
print('node density: ', node_density)

# Add edges modeling human mobility through a gravity law
# c is the parameter for the gravity law. I stop as soon as I find the first strongly connected graph
c_min = 0
c_max = 200
# Perform search_max_number repeated trials
search_max_number = 100
# Limit of repetition for a given c
max_trialsSC_fixed_c = 1000

populationTot = np.array([1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])

c_list, avg_c, err_c = characterisation_network_SC(G, DistanceMatrix, node_density, search_max_number, c_min, c_max, max_trialsSC_fixed_c )
print('avg_c: ', avg_c)
print('err_c: ', err_c)

# Nrow = 2, Ncol = 2, N = 4 (only 1 number after the . is enough - I think -)
c2_avg = np.array([1.36, 1.30, 1.35, 1.34, 1.32, 1.35, 1.25, 1.35, 1.33, 1.31, 1.32, 1.34, 1.39 ])
c2_stdDev = np.array([0.31, 0.31, 0.25, 0.27, 0.28, 0.26, 0.32, 0.26, 0.29, 0.29, 0.3, 0.27, 0.29 ])

# TAKES A LOT TO RUN - NEED TO CONTINUE WITH CHARACTERISATION ONCE I HAVE ACCESS TO THE CLUSTER
# DECIDE THE STEP BETWEEN 2 CONSECUTIVE c
# Nrow = 3, Ncol = 3, N = 9
c3_avg = np.array([])
c3_stdDev = np.array([])

#plt.errorbar(populationTot, c2_avg, yerr=c2_stdDev, color = 'gray', marker = 'o', linewidth = 1, markersize = 6, label = 'N=4')
#plt.xlabel('Total population')
#plt.ylabel('c')
#plt.legend()
#plt.show()

## First trial : take the first value of c for which I have a strongly connected graph
# Nrow = 2, Ncol = 2, N = 4
c2 = np.array([2, 1, 2, 3, 2, 2, 2, 3, 2, 1, 3, 3, 2])

# Nrow = 3, Ncol = 3, N = 9
c3 = np.array([15, 14, 14, 14, 12, 12, 14, 12, 12, 8, 12, 9, 10 ])

# Nrow = 4, Ncol = 4, N = 16
c4 = np.array([53, 47, 41, 41, 40, 40, 41, 39, 37, 35, 33, 37, 42])

# Nrow = 5, Ncol = 5, N = 25
c5 = np.array([150, 117, 106, 91, 94, 90, 90, 83, 83, 80, 95, 89, 84])

plt.plot(populationTot, c2, '--',color = 'gray', marker = 'o', linewidth = 1, markersize = 6, label = 'N=4')
plt.plot(populationTot, c3, '--',color = 'blue', marker = 'o', linewidth = 1, markersize = 6, label = 'N=9')
plt.plot(populationTot, c4, '--',color = 'red', marker = 'o', linewidth = 1, markersize = 6, label = 'N=16')
plt.plot(populationTot, c5, '--',color = 'green', marker = 'o', linewidth = 1, markersize = 6, label = 'N=25')
plt.xlabel('Total population')
plt.ylabel('c')
plt.legend()
plt.show()

