"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 25 October 2023

--------------------------------------------------------------------
Analysis on the topology of the network and the simulation od SIR

"""

from functions_SIR_metapop import *
from functions_visualization import *
from functions_analysis import *
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import pickle

# ------------------------------------------------ Parameters  -------------------------------------------------

N_row = 10
N_col = 10
N = N_row * N_col

choice_bool = 0
datadir = os.getcwd()

c1 = 0  # for now

beta = 0.3
mu = 0.1


# --------------------------------------------- Load data ---------------------------------------------

folder_topology = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Topology/'
folder_simulation = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'

G = pickle.load(open(folder_topology + 'G.pickle', 'rb'))
dict_nodes = pickle.load(open(folder_topology + 'dict_nodes.pickle', 'rb'))
pos_nodes = np.load(folder_topology + 'pos_nodes.npy')

TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0]

avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')
populationTot = N * avg_popPerNode # from this then I use multinomial
if choice_bool == 1:
    Nfix = np.load(folder_topology + 'Nfix.npy')
    percentage_FixNodes = np.load(folder_topology + 'percentage_FixNodes.npy')
else:
    Nfix = 0
    percentage_FixNodes = 0

T = np.load(folder_simulation + 'T.npy')
T_sim = np.linspace(0, T, T+1)

node_population0 = nx.get_node_attributes(G, name='Npop')
node_population0 = np.array(list(node_population0.values()))

# ----------------------------------------------  Network analysis  ----------------------------------------------

#plot_centralities(G)

in_degrees = [G.in_degree(n) for n in G.nodes()]
plt.bar(*np.unique(in_degrees, return_counts=True))
plt.xlabel('Degree of in-edges')
plt.ylabel('Frequency')
plt.show()
plt.figure()

#plot_static_network(G, node_population0, dict_nodes, weightNonZero)

# ----------------------------------------------  Simulation analysis  ----------------------------------------------
bool_density = 1

idx_sims = [0]
idx_nodes = [item for item in range(0, N)]

# 1. Plot number of individuals of densities in SIR for a certain simulation and for certain nodes
plot_SIR_timeseries(N_row, N_col, choice_bool, c1, beta, mu, bool_density, idx_sims, idx_nodes, T_sim, avg_popPerNode)

# 2. Mean and average over different simulations having the same topology
nbr_repetitions = np.load(folder_simulation + 'nbr_repetitions.npy')

y_mean_std = mean_stdDev_repetitions(N_row, N_col, choice_bool, c1, T, beta, mu, bool_density, nbr_repetitions)
mean_S_time = y_mean_std[0]
mean_I_time = y_mean_std[1]
mean_R_time = y_mean_std[2]
stdDev_S_time = y_mean_std[3]
stdDev_I_time = y_mean_std[4]
stdDev_R_time = y_mean_std[5]

print('Hello')

# 3. Plot deterministic SIR
if bool_density == 1:
    density_node_NS_time_repeat = y_mean_std[6]
    density_node_NI_time_repeat = y_mean_std[7]
    density_node_NR_time_repeat = y_mean_std[8]
    # Initial conditions : densities (is the same at every repetition!)
    y_init = [density_node_NS_time_repeat, density_node_NI_time_repeat, density_node_NR_time_repeat]
    print('y_init: ', y_init)
    params = [beta, mu]
    # Sole equation for densities
    y = odeint(SIRDeterministic_equations, y_init, T_sim, args=(params,))

    # Deterministic densities in time: solutions of SIR deterministic ODEs
    det_s = y[:, 0]
    det_i = y[:, 1]
    det_r = y[:, 2]
elif bool_density == 0:
    det_s = [0 for i in range(0, T+1)]
    det_i = [0 for i in range(0, T+1)]
    det_r = [0 for i in range(0, T+1)]

idx_node = 0

plot_mean_std_singleNode(T_sim, mean_S_time, mean_I_time, mean_R_time, stdDev_S_time,
                         stdDev_I_time, stdDev_R_time, det_s, det_i, det_r, idx_node)

plot_mean_allNodes(T_sim, mean_S_time, mean_I_time, mean_R_time,det_s, det_i, det_r, N)

# 4. Quantify the distance of the mean simulated to the deterministic curve for the infection population inside a node.

# Pointwise difference of the mean simulated curve to the deterministic one for the node with index idx_node
diff_meanI_detI_node0 = mean_I_time[:, idx_node] - det_i
mean_diff_meanI_detI_node0 = np.mean(diff_meanI_detI_node0)
mean_std_dev_diff_meanI_detI_node0 = np.mean(stdDev_I_time[:, idx_node])

plt.errorbar(avg_popPerNode, mean_diff_meanI_detI_node0, yerr = mean_std_dev_diff_meanI_detI_node0, marker = 'o')
plt.show()
print('hello')

# 5. See data in phase space
sim = 0
plot_phase_space(N_row, N_col, choice_bool, c1, beta, mu, sim)

# 6. Temporal heatmap
heatmap_time(N_row, N_col, choice_bool, c1, beta, mu, sim)


