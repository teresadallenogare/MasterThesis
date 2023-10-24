"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 22 October 2023

--------------------------------------------------------------------
Analysis on the topology of the network and the simulation od SIR

"""

from functions_SIR_metapop import *
from functions_visualization import *
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import pickle

# ------------------------------------------------ Parameters  -------------------------------------------------

N_row = 30
N_col = 30
N = N_row * N_col
choice_bool = 0
datadir = os.getcwd()
c1 = 0  # for now
beta = 0.9
mu = 0.1

# ------------------------------------------------ Colors  -------------------------------------------------
grad_gray = []
grad_red = []
grad_blue = []
grad_green = []

for x in range(N_row*N_col):
    #                                dark           light
    grad_gray.append(colorFader('#505050', '#EAE9E9', x/(N_row * N_col)))
    grad_red.append(colorFader('#E51C00', '#FCE0DC', x/(N_row * N_col)))
    grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x/(N_row * N_col)))
    grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x/(N_row * N_col)))



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

nbr_repetitions = np.load(folder_simulation + 'nbr_repetitions.npy')

sim = 0
node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')

# ########################################  Network analysis  ########################################

#plot_centralities(G)

in_degrees = [G.in_degree(n) for n in G.nodes()]
plt.bar(*np.unique(in_degrees, return_counts=True))
plt.xlabel('Degree of in-edges')
plt.ylabel('Frequency')
plt.show()
plt.figure()

#plot_static_network(G, node_population0, dict_nodes, weightNonZero)

# ########################################  Simulation analysis  ########################################

# 1. Density of S, I, R as a function of time.
# / np.mean(node_population) : I divide by a constant number so I keep fluctuations


for idx_node in range(1):
    if idx_node == 0:
        plt.plot(T_sim, node_population_time[:, idx_node] / np.mean(node_population_time[:, idx_node]), color=grad_gray[idx_node], label='population density')
        plt.plot(T_sim, node_NS_time[:, idx_node] / np.mean(node_population_time[:, idx_node]), color=grad_blue[idx_node], label='S density')
        plt.plot(T_sim, node_NI_time[:, idx_node] / np.mean(node_population_time[:, idx_node]), color=grad_red[idx_node], label='I density')
        plt.plot(T_sim, node_NR_time[:, idx_node] / np.mean(node_population_time[:, idx_node]), color=grad_green[idx_node], label='R density')
    else:
        plt.plot(T_sim, node_population_time[:, idx_node] / np.mean(node_population_time[:, idx_node]), color=grad_gray[idx_node])
        plt.plot(T_sim, node_NS_time[:, idx_node] / np.mean(node_population_time[:, idx_node]), color=grad_blue[idx_node])
        plt.plot(T_sim, node_NI_time[:, idx_node] / np.mean(node_population_time[:, idx_node]), color=grad_red[idx_node])
        plt.plot(T_sim, node_NR_time[:, idx_node] / np.mean(node_population_time[:, idx_node]), color=grad_green[idx_node])
plt.axhline(y=avg_popPerNode / avg_popPerNode, color='black', linestyle='--', label='Fixed average density per node')
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Node density')
# plt.title(f'SIR density node {idx_node}')
plt.show()

# 2. Mean and average over different simulations having the same topology
# 3D matrix that stores repetitions along axis = 2
node_population_time_repeat = np.zeros(shape = (T+1, N, nbr_repetitions))
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
    #ADD NODE STATE


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

# 3. Plot deterministic SIR
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

# 4. Quantify the distance of the mean simulated to the deterministic curve for the infection population inside a node.

# Pointwise difference of the mean simulated curve to the deterministic one for the node with index idx_node
diff_meanI_detI_node0 = mean_density_NI_time[:, idx_node] - det_i
mean_diff_meanI_detI_node0 = np.mean(diff_meanI_detI_node0)
mean_std_dev_diff_meanI_detI_node0 = np.mean(stdDev_density_NI_time[:, idx_node])

plt.errorbar(avg_popPerNode, mean_diff_meanI_detI_node0, yerr = mean_std_dev_diff_meanI_detI_node0, marker = 'o')
plt.show()
print('hello')

# 5. See data in phase space
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
color_map = plt.get_cmap('spring')
for idx_node in range(N):
    x = node_NS_time[:, idx_node]
    y = node_NI_time[:, idx_node]
    z = node_NR_time[:, idx_node]
    sc = ax.scatter3D(x, y, z)

ax.set_xlabel('S')
ax.set_ylabel('I')
ax.set_zlabel('R')
ax.set_title(f'Network {N_row}x{N_col}, beta: {beta}, mu: {mu}')

plt.show()




