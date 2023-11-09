"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 05 November 2023

--------------------------------------------------------------------
Analysis of persistent entropy data obtained in 4-TDA-note4book.ipynb
Single simulations are considered in this analysis file.

In the notebook there are 2 normalizations:
1. NORMALIZATION OF DATA
    - no normalized data : original dictionary
    - normalized data according to scaler

2. NORMALIZATION OF ENTROPY
    - no normalized persistent entropy
    - normalized persistent entropy according to the function
"""

from functions_SIR_metapop import *
from functions_analysis import min_PE
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# ------------------------------------------------ Parameters  -------------------------------------------------
choice_bool = 0
datadir = os.getcwd()
c1 = 0  # for now
beta_outbreak = [0.4, 0.3, 0.9]
mu_outbreak = [0.2, 0.1, 0.1]

beta_no_outbreak = [0.35, 0.75]
mu_no_outbreak = [0.3, 0.6]

nbr_simulations = 10

row = 30
col = 30
N = row * col
beta = 0.35
mu = 0.3
normalization_by_hand = 2 # normalization of data 'by hand'
scaler = 0
normalization_entropy = 0
outbreak = 1
sim = 7



id = ('XYSIR')
# ------------------------------------------------ Colors  -------------------------------------------------
grad_gray = []
grad_red = []
grad_blue = []
grad_green = []

for x in range(nbr_simulations):
    #                                dark           light
    grad_gray.append(colorFader('#505050', '#EAE9E9', x/nbr_simulations))
    grad_red.append(colorFader('#E51C00', '#FCE0DC', x/nbr_simulations))
    grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x/nbr_simulations))
    grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x/nbr_simulations))

# ------------------------------ Flux of new I per node ------------------------------
folder_topology = datadir+f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Topology/'
folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'

# Length simulation
T = np.load(folder_simulation + 'T.npy')
T_sim = np.linspace(0, T, T + 1)

# Transition matrix
TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
weight = [TransitionMatrix[i, j] for i in range(N) for j in range(N)]
weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0]

# Number of new infected in every node in time for a fixed simulation
new_I_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
# Number of infected in node in time
node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')

# Plot new
plt.figure()
plt.plot(T_sim, node_NI_time[:, 0], color = 'k', label = 'I at node 0') # node 0
plt.plot(T_sim, new_I_time[:, 0], color = 'b', label = 'new I at node 0')
plt.plot(T_sim, node_NI_time[:, 0] - new_I_time[:, 0], color = 'r', label = '#I - #newI node 0')
plt.plot(T_sim, node_NI_time[:, 899], color = 'k') # node 0
plt.plot(T_sim, new_I_time[:, 899], color = 'b')
plt.plot(T_sim, node_NI_time[:, 899] - new_I_time[:, 899], color = 'r')
#plt.legend()
#plt.show()

plt.figure()
plt.plot(T_sim, node_population_time)
#plt.show()

# trial 1 . fro rho_kt = pop_k(t)/np.mean(pop_k_time, over time)
# sum of weights from node k to node j ( k -> j) : fix k and go to any node j (considers exiting fluxes)
sum_weights_kj = []
for k in range(N): # rows : fixed node
    sum_Tfrom_k_to_j = 0
    for j in range(N): # cols : exiting nodes
        if j > k:
            sum_Tfrom_k_to_j += TransitionMatrix[k, j]
    sum_weights_kj.append(sum_Tfrom_k_to_j)
# sum of weights from node j to node k (j -> k) : fix j and go to k (considers exiting fluxes from j bout towards node k that is the entering flux in k)
#sum_weigths_jk = []
#for j in range(N):
#    sum_Tfrom_j_to_k = 0
#    for k in range(N):
#        if j < k:
#            sum_Tfrom_j_to_k += TransitionMatrix[k, j]
#    sum_weigths_jk.append(sum_Tfrom_j_to_k)

# flux of new I for node
flux_newI_kt = np.zeros(shape = (T+1, N))
DeltaI_kt_mtrx = np.zeros(shape = (T+1, N))
for k in range(N):
    sum_weights_k = sum_weights_kj[k]
    for t in range(T):
        DeltaI_kt = new_I_time[t, k]/node_population_time[t, k]
        rho_kt = node_NI_time[t, k] / np.mean(node_population_time[:, k], axis=0)
        # sum contains only the weights where node k is the starting node, towards any node j
        flux_newI_kt[t, k] = DeltaI_kt * rho_kt * sum_weights_k
        DeltaI_kt_mtrx[t, k] = DeltaI_kt
# sum over the whole network
flux_t = []
for t in range(T+1):
    flux_t.append(np.sum(flux_newI_kt[t, :]))
flux_t = np.array(flux_t)

DeltaI_t = []
for t in range(T+1):
    DeltaI_t.append(np.sum(DeltaI_kt_mtrx[t, :]))
DeltaI_t = np.array(DeltaI_t)
plt.figure()
plt.plot(T_sim, DeltaI_t, color = 'b')
plt.plot(T_sim, flux_t, color = 'r')
plt.show()
# --------------------------- PE -------------------------------------

fig, ax = plt.subplots()
folder_dict = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Dictionaries/'
# Fix folder containing non-normalized (0) or normalized (1) entropy data
if normalization_entropy == 0:
    folder_entropy = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis-notebook-TDA/No-normalized-entropy/'
elif normalization_entropy == 1:
    folder_entropy = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis-notebook-TDA/Normalized-entropy/'


print('row: ', row, 'col: ', col)
print('beta: ', beta, 'mu: ', mu)
print('sim: ', sim)
if scaler == 0:
    entropy_H0 = np.load(folder_entropy + f'/Entropy-{id}/entropy_H0-sim{sim}.npy')
    entropy_H1 = np.load(folder_entropy + f'/Entropy-{id}/entropy_H1-sim{sim}.npy')
elif scaler == 1:
    entropy_H0 = np.load(folder_entropy + f'/Entropy-scaler-{id}/entropy_H0-sim{sim}.npy')
    entropy_H1 = np.load(folder_entropy + f'/Entropy-scaler-{id}/entropy_H1-sim{sim}.npy')

if normalization_by_hand == 1: # ONLY XYSIR
    folder_entropy = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/Analysis/Normalized/'
    entropy_H0 = np.load(folder_entropy + f'entropy_H0-sim{sim}.npy')
    entropy_H1 = np.load(folder_entropy + f'entropy_H1-sim{sim}.npy')
x = range(0, len(entropy_H0))
y_H0 = entropy_H0
y_H1 = entropy_H1

min_yH0, t_min_yH0 = min_PE(y_H0, x)
min_yH1, t_min_yH1 = min_PE(y_H1, x)
print('t_min_H0: ', t_min_yH0, 'min_H0: ', min_yH0)
print('t_min_H1: ', t_min_yH1, 'min_H1: ', min_yH1)
ax.plot(x, y_H0, label=f'PE at H0', color=grad_green[sim] if outbreak == 0 else grad_blue[sim])
ax.plot(x, y_H1, label=f'PE at H1', color=grad_red[sim] if outbreak == 0 else grad_red[sim])
const = np.log(N) * np.ones((T+1, 1))
#diff = np.array([const[t] - np.log(flux_t[t]) for t in range(T)])
diff = np.array([const[t] - np.log(DeltaI_t[t]) for t in range(T)])

ax.plot(T_sim[1:], diff)
ax.scatter(t_min_yH0, min_yH0)
ax.scatter(t_min_yH1, min_yH1)
ax.set_xlabel("Time")
ax.set_ylabel("Persistent Entropy")
ax.set_title(f"PE for {row}x{col}, beta = {beta}, mu = {mu}, R0 = {round(beta/mu, 2)}")
ax.legend()
plt.show()


plt.figure()
plt.plot(T_sim, flux_newI_kt)
plt.xlabel('time')
plt.ylabel('Flux new infected per node')
plt.title(f'Outbreak: beta = {beta}, mu = {mu}')
plt.show()
#T_sim = np.linspace(0, T, T)
plt.figure()
plt.plot(T_sim, N - flux_t)
plt.plot(T_sim, N - DeltaI_t, color = 'k')
plt.xlabel('time')
plt.ylabel('Flux new infected network')
plt.title(f'Outbreak: beta = {beta}, mu = {mu}')
plt.show()


