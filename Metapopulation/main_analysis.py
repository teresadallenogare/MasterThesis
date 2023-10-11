"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 10 October 2023

--------------------------------------------------------------------

File to analyse data produced.

"""

from functions_SIR_metapop import *
from functions_visualization import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# Parameters I have control on : give in input to load data of the lattice I want
N_row = input('N_row: ')
N_col = input('N_col: ')
choice_bool = input('choice_bool: ')
datadir = os.getcwd()
c1 = 0  # for now
beta = input('beta: ')
mu = input('mu: ')


# -------------------------------------- Load data --------------------------------------
folder_data = f'/choice_bool-{choice_bool}/c1-{int(c1)}/beta-{beta}mu-{mu}'
folder_topology = datadir+f'/Data-simpleLattice/{N_row}x{N_col}'+folder_data+'/Topology'
folder_dynamics = datadir+f'/Data-simpleLattice/{N_row}x{N_col}'+folder_data+'/Dynamics'

seed = np.load(folder_topology + '/seed.npy')
avg_popPerNode = np.load(folder_topology + '/avg_popPerNode.npy')
if choice_bool == 1:
    Nfix = np.load(folder_topology + '/Nfix.npy')
    percentage_FixNodes = np.load(folder_topology + '/percentage_FixNodes.npy')


node_population_time = np.load(folder_dynamics + '/node_population_time.npy')
node_NS_time = np.load(folder_dynamics + '/node_NS_time.npy')
node_NI_time = np.load(folder_dynamics + '/node_NI_time.npy')
node_NR_time = np.load(folder_dynamics + '/node_NR_time.npy')



fig, ax = plt.subplots(3, 1)
pos0 = ax[0].imshow(node_NS_time.T, cmap = 'coolwarm')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('node idx')
ax[0].set_title('Number susceptible - infected - recovered over time')
pos1 = ax[1].imshow(node_NI_time.T, cmap = 'coolwarm')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('node idx')
pos2 = ax[2].imshow(node_NR_time.T, cmap = 'coolwarm')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('node idx')
fig.colorbar(pos0, ax=ax[0])
fig.colorbar(pos1, ax=ax[1])
fig.colorbar(pos2, ax=ax[2])
print(node_NI_time)
plt.show()
