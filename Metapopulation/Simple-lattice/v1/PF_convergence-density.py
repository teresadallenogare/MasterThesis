"""

--------------------------------------------------------------------

Author  :   Teresa Dalle Nogare
Version :   01 February 2024

--------------------------------------------------------------------

Heatmap of the difference between PF eigenvector and simulated density in one case of R0

"""

from functions_network_v1 import path_analysis
from functions_output_v1 import write_network_file
from functions_visualization_v1 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

datadir = os.getcwd()
plt.figure(figsize=(8, 6))
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})

row = 30
col = 30

N = row * col

choice_bool = 0
c1 = 0

beta = 0.115
mu = 0.1

folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
avg_population = np.load(folder_topology + f'avg_popPerNode.npy')

## Density obtained as eigenvector of PF
rho0 = np.load(folder_topology + f'/rho0.npy')
# Maybe due this because I have previously normalized (?)
rho0 = rho0 * N

## Density simulated (population density at the end of the simulation)
node_population_time = np.load(folder_simulation + 'sim_0_node_population_time.npy')



for i in range(3):
    if i == 0:
        node_population = node_population_time[1, :]
    elif i == 1:
        node_population = node_population_time[5, :]
    else:
        node_population = node_population_time[-1, :]

    node_density = node_population / avg_population
    diff_density = node_density - rho0

    diff_density_matrix = diff_density.reshape(30, 30)

    # Create a heatmap
    sns.heatmap(diff_density_matrix, center=0, cmap="coolwarm", annot=False)

    # Add labels and title
    plt.xlabel("Index node")
    plt.ylabel("Index node")
    plt.title("Heatmap difference rho0 - rho infty")

    # Show the plot
    plt.show()