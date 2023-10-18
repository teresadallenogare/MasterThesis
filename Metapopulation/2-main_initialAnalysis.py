"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 16 October 2023

--------------------------------------------------------------------


"""
from functions_SIR_metapop import *
from functions_visualization import *
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import sklearn
import kmapper as km
import networkx as nx
import pandas as pd


# Consider the case of an epidemic outbreak on a 3x3 network
N_row = 3
N_col = 3
N = N_row * N_col
choice_bool = 0
datadir = os.getcwd()
c1 = 0  # for now
beta = 0.4
mu = 0.2

# --------------------------------------------- Load data ---------------------------------------------
folder_topology = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Topology/'
folder_simulation = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'

pos_nodes = np.load(folder_topology + 'pos_nodes.npy')

avg_popPerNode = np.load(folder_topology + '/avg_popPerNode.npy')
populationTot = N * avg_popPerNode # from this then I use multinomial
if choice_bool == 1:
    Nfix = np.load(folder_topology + '/Nfix.npy')
    percentage_FixNodes = np.load(folder_topology + '/percentage_FixNodes.npy')
else:
    Nfix = 0
    percentage_FixNodes = 0

T = np.load(folder_simulation + 'T.npy')
T_sim = np.linspace(0, T, T+1)

nbr_repetitions = np.load(folder_simulation + '/nbr_repetitions.npy')

# 3D matrix that stores repetitions along axis = 2
node_population_time_repeat = np.zeros(shape = (T+1,N, nbr_repetitions))
node_NS_time_repeat = np.zeros(shape = (T+1, N, nbr_repetitions))
node_NI_time_repeat = np.zeros(shape = (T+1, N, nbr_repetitions))
node_NR_time_repeat = np.zeros(shape = (T+1, N, nbr_repetitions))
density_node_NS_time_repeat = np.zeros(shape = (T+1, N, nbr_repetitions))
density_node_NI_time_repeat = np.zeros(shape = (T+1, N, nbr_repetitions))
density_node_NR_time_repeat = np.zeros(shape = (T+1, N, nbr_repetitions))
# To see repetition k : node_NI_time_repeat[:,:,k]
for sim in range(nbr_repetitions):
    # Load data
    new_I_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
    node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
    node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
    node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
    node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')
    #  Store data in 3D matrix
    node_population_time_repeat[:, :, sim] = node_population_time
    node_NS_time_repeat[:, :, sim] = node_NS_time
    node_NI_time_repeat[:, :, sim] = node_NI_time
    node_NR_time_repeat[:, :, sim] = node_NR_time

# Compute densities
density_node_NS_time_repeat = node_NS_time_repeat / node_population_time_repeat
density_node_NI_time_repeat = node_NI_time_repeat / node_population_time_repeat
density_node_NR_time_repeat = node_NR_time_repeat / node_population_time_repeat

# Mean value of number of individuals over repetitions
mean_NS_time = np.mean(node_NS_time_repeat, axis = 2)
mean_NI_time = np.mean(node_NI_time_repeat, axis = 2)
mean_NR_time = np.mean(node_NR_time_repeat, axis = 2)
stdDev_NS_time = np.std(node_NS_time_repeat, axis = 2, ddof = 1)
stdDev_NI_time = np.std(node_NI_time_repeat, axis = 2, ddof = 1)
stdDev_NR_time = np.std(node_NR_time_repeat, axis = 2, ddof = 1)

# Mean value of densities over repetitions
mean_density_NS_time = np.mean(density_node_NS_time_repeat, axis = 2)
mean_density_NI_time = np.mean(density_node_NI_time_repeat, axis = 2)
mean_density_NR_time = np.mean(density_node_NR_time_repeat, axis = 2)
stdDev_density_NS_time = np.std(density_node_NS_time_repeat, axis = 2, ddof = 1)
stdDev_density_NI_time = np.std(density_node_NI_time_repeat, axis = 2, ddof = 1)
stdDev_density_NR_time = np.std(density_node_NR_time_repeat, axis = 2, ddof = 1)

# Deterministic SIR
# Initial conditions : densities (is the same at every repetition!)
y_init = [density_node_NS_time_repeat[0, 0, 0], density_node_NI_time_repeat[0, 0, 0], density_node_NR_time_repeat[0, 0, 0]]
print('y_init: ', y_init)
params = [beta, mu]
# Sole equation for densities
y = odeint(SIRDeterministic_equations, y_init, T_sim, args=(params,))

# Deterministic densities in time: solutions of SIR deterministic ODEs
det_s = y[:, 0]
det_i = y[:, 1]
det_r = y[:, 2]


idx_node = 0

plot_mean_std_singleNode(T_sim, mean_density_NS_time, mean_density_NI_time, mean_density_NR_time, stdDev_density_NS_time,
                         stdDev_density_NI_time, stdDev_density_NR_time, det_s, det_i, det_r, idx_node)

plot_mean_allNodes(T_sim, mean_density_NS_time, mean_density_NI_time, mean_density_NR_time,det_s, det_i, det_r, N)
#df_mean_NI_time = pd.DataFrame(mean_NI_time, columns=[f'node {i}' for i in range(N)])
#df_stdDev_NI_time = pd.DataFrame(stdDev_NI_time, columns=[f'node {i}' for i in range(N)], index=[f'time {i}' for i in range(T+1)])


# Quantify the distance of the mean simulated to the deterministic curve for the infection population inside a node.

# Pointwise difference of the mean simulated curve to the deterministic one for the node with index idx_node
diff_meanI_detI_node0 = mean_density_NI_time[:, idx_node] - det_i
mean_diff_meanI_detI_node0 = np.mean(diff_meanI_detI_node0)
mean_std_dev_diff_meanI_detI_node0 = np.mean(stdDev_density_NI_time[:, idx_node])

plt.errorbar(avg_popPerNode, mean_diff_meanI_detI_node0, yerr = mean_std_dev_diff_meanI_detI_node0, marker = 'o' )
plt.show()
print('hello')








