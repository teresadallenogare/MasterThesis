"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 05 February 2023

--------------------------------------------------------------------

Extract generators of topological features.
I do it only at the time step of the minimum of entropy (that is when I want to focus)

"""
from functions_TDA_v1 import *
from functions_visualization_v1 import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import re
from scipy.spatial.distance import cdist

datadir = os.getcwd()

sim = 0

row = 30
col = 30

choice_bool = 0
c1 = 0

N = row * col

beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  0.1]

beta_vals = [1.2]
mu_vals = [0.1]

id = 'XYSIR'
columns = ['X', 'Y', 'S', 'I', 'R']

# Run over beta and mu
for beta, mu in zip(beta_vals, mu_vals):
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'
    folder_ripser = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Ripser_analysis/'

    T = np.load(folder_simulation + 'T.npy')
    if beta == 0.115 or beta == 0.12:
        T = 1000
    T_sim = np.linspace(0, T - 1, T)
    ## Import dataset with dictionary of normalized data by hand
    dict_data = pickle.load(
        open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    df_dict_data = data_2_pandas(dict_data)

    ## Extract time of minimum entropy (mostly vailid for high R0)
    idx_sims_not_start = np.load(folder_simulation + f'idx_sim_not_start.npy')

    min_H0_sims = np.load(folder_entropy + f'min_H0_sims-mu{mu}-beta{beta}.npy')
    min_H1_sims = np.load(folder_entropy + f'min_H1_sims-mu{mu}-beta{beta}.npy')

    column_index = np.where(min_H0_sims[0] == sim)[0]

    if len(column_index) > 0:
        min_H0_sim = min_H0_sims[:, column_index[0]]
        min_H1_sim = min_H1_sims[:, column_index[0]]
        print("Extracted H0:", min_H0_sim)
        print("Extracted H1:", min_H1_sim)
        t0 = min_H0_sim[1]
        t1 = min_H1_sim[1]
        #t0 = 16
        print('t0 : ', t0)
        print('t1 : ', t1)

    ## Repeat for H0 and H1 at 2 different times
    for t_star in [t0, t1]:
        with open(folder_ripser + f'1-ripser_localization-beta{beta}-mu{mu}-id{id}.txt', 'a') as file:
            # Dataframe at fixed time value
            S_t = df_dict_data.loc[df_dict_data['Time'] == t_star]
            # Select only data indexed by "column"
            S_data_t = S_t[columns]
            ## NEED DISTANCE MATRIX BECAUSE I GIVE IN INPUT TO RIPSER
            # Euclidean distance between pairs of points (Each point corresponds to a row. The dimension of the space is given by the number of columns.)
            D_t = cdist(S_data_t, S_data_t, 'euclid')
            # Write distance matrix to file (over-write at each t)
            np.savetxt(folder_ripser + 'D_t.csv', D_t, delimiter=',')
            subprocess.run([folder_ripser + 'ripser-representatives', folder_ripser + 'D_t.csv'], stdout=file)
            # Write result to screen
            result = subprocess.run([folder_ripser + 'ripser-representatives', folder_ripser + 'D_t.csv'],
                                    capture_output=True,
                                    text=True)
            print(result.stdout)
            #### Generators of persistence intervals in dimension 0
            # Extract nodes generating the 0-dim persistence intervals (are the integers inside the [])

            # 0 after nodes stands for the dimension of the space to which nodes contribute
            idx_nodes0_t0 = re.findall(r"\[\s*\+?(-?\d+)\s*\]", result.stdout)
            # Convert the elements of the list from strings to integers
            idx_nodes0_t0 = string_2_int(idx_nodes0_t0)

# Create a 30x30 lattice graph
G = nx.grid_2d_graph(30, 30)

# Assign different colors to nodes
node_colors = np.random.rand(len(G))

# Set up the figure and axis
fig, ax = plt.subplots()

# Draw nodes with custom colors
pos = {(i, j): (i, -j) for i, j in G.nodes()}
nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.Blues,  node_size=20)

# Manually draw connections between nodes
for edge in G.edges():
    x, y = pos[edge[0]], pos[edge[1]]
    #plt.plot([x[0], y[0]], [x[1], y[1]], color='black', linewidth=0.5)

# Hide the axis
ax.set_axis_off()

# Show the plot
plt.show()