"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 15 February 2024

--------------------------------------------------------------------

3D plot with filtration

"""
from functions_SIR_metapop_v1 import *
from functions_output_v1 import write_simulation_file
from functions_visualization_v1 import *
from functions_TDA_v1 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.integrate import odeint
import seaborn as sns
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist
import subprocess
import re
import csv
import pandas as pd

datadir = os.getcwd()
#plt.figure(figsize=(10, 8))
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})

tred_plot = 1
multiple_tred_plot = 0
dued_plot = 0


if tred_plot == 1:
    row = 30
    col = 30

    N = row * col
    choice_bool = 0
    c1 = 0

    sim = 0
    beta = 0.12
    mu = 0.1

    datadir = os.getcwd()

    folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'

    folder_animations = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Animations/'

    # Load normalized dictionary to have the density of individuals
    dict_load_normalized = pickle.load(
        open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    dict_load_normalized_values = list(dict_load_normalized.values())
    # Brute force : maximum value of density of I in whole dictionary
    max_densityI_time = []
    max_densityInew_time = []
    # Determination of the maximum density of infected

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    t = 500

    mtrx_t_normalized = dict_load_normalized[t]
    x_nodes = mtrx_t_normalized[:, 0]
    y_nodes = mtrx_t_normalized[:, 1]
    density_Inew = mtrx_t_normalized[:, 3]
    # Scatter plot
    sc = ax.scatter(x_nodes, y_nodes, density_Inew, c=density_Inew, cmap='gnuplot', marker='o')
    # Add color bar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label(r'Values $I/\langle n \rangle$')
    ax.set_xlabel('X node')
    ax.set_ylabel('Y node')
    #ax.set_zlabel(r'$\Delta I$')
    ax.set_zlabel(r'$I/\langle n \rangle$')

    plt.tight_layout()
    plt.show()


    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    # Function to update the plot for each frame
    def update(frame):
        #ax.clear()
        # Calculate sphere coordinates
        theta = np.linspace(0, 2 * np.pi, 100)
        phi = np.linspace(0, np.pi, 100)

        # Radius increases from 0 to 1 with a step of 0.1
        radius = frame / 100.0
        for i in range(N):
            x = x_nodes[i] + radius * np.outer(np.cos(theta), np.sin(phi))
            y = y_nodes[i] + radius * np.outer(np.sin(theta), np.sin(phi))
            z = density_Inew[i] + radius * np.outer(np.ones(np.size(theta)), np.cos(phi))

            ax.plot_surface(x, y, z, color='b', alpha=0.4, linewidth=0.2)


            ax.set_title(f'Sphere with Radius {radius:.1f}')

    # Create a figure and 3D axis
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the points at the center of the sphere
    sc = ax.scatter(x_nodes, y_nodes, density_Inew, c=density_Inew, cmap='gnuplot', marker='o')
    # Add color bar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label(r'Values $I/\langle n \rangle$')
    ax.set_xlabel('X node')
    ax.set_ylabel('Y node')
    # ax.set_zlabel(r'$\Delta I$')
    ax.set_zlabel(r'$I/\langle n \rangle$')
    # Set plot limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])


    # Create the animation
    animation = FuncAnimation(fig, update, frames=np.arange(0, 20), interval=500)

    # Show the animation
    plt.tight_layout()
    plt.show()



if dued_plot == 1:
    row = 30
    col = 30
    N = row * col
    choice_bool = 0
    c1 = 0
    beta = 1.2
    mu = 0.1

    columns = ['X', 'Y', 'S', 'I', 'R']
    id = 'XYSIR'

    sim = 0

    generate_Dt = 0

    folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'
    folder_ripser = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Ripser_analysis/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized/{id}/'
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

    dict_data_no_norm = pickle.load(
        open(folder_dict_noNorm + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    df_dict_data_no_norm = data_2_pandas(dict_data_no_norm)


    dict_data = pickle.load(
        open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    df_dict_data = data_2_pandas(dict_data)

    if beta == 0.115 or beta == 0.12:
        T = 1000
    else:
        T = df_dict_data['Time'].max()
        T_sim = np.linspace(0, T - 1, T)
    if generate_Dt == 1:
        with open(folder_ripser + f'ripser_localization-beta{beta}-mu{mu}-id{id}.txt', 'a') as file:
            for t in range(T):
                # Dataframe at fixed time value
                S_t = df_dict_data.loc[df_dict_data['Time'] == t]
                # Select only data indexed by "column"
                S_data_t = S_t[columns]
                # Euclidean distance between pairs of points (Each point corresponds to a row. The dimension of the space is given by the number of columns.)
                D_t = cdist(S_data_t, S_data_t, 'euclid')
                # Write distance matrix to file (over-write at each t)
                np.savetxt(folder_ripser + f'/Distance_matrices/D_t{t}-beta{beta}-mu{mu}-id{id}.csv', D_t, delimiter = ',')
                #### Compute persistent homology and cycle representatives with Ripser
                # Write result to file
                #subprocess.run([folder_ripser + 'ripser-representatives', folder_ripser + 'D_t.csv'], stdout=file)
                # Write result to screen
                #result = subprocess.run([folder_ripser + 'ripser-representatives', folder_ripser + 'D_t.csv'],
                #                        capture_output=True,
                #                        text= True)
                #print(result.stdout)
                #### Generators of persistence intervals in dimension 0
                # Extract nodes generating the 0-dim persistence intervals (are the integers inside the [])
    for t in range(T):
        df = pd.read_csv(folder_ripser + f'/Distance_matrices/D_t{t}-beta{beta}-mu{mu}-id{id}.csv')
        # Save column names
        existing_columns = df.columns.tolist()
        # Replace them
        df.columns = range(N)
        # Add a new row at the beginning with the saved column names
        df.loc[-1] = existing_columns
        df.index = df.index + 1
        df.sort_index(inplace=True)


        print('hello')


if multiple_tred_plot == 1:

    row = 30
    col = 30

    N = row * col
    choice_bool = 0
    c1 = 0

    sim = 0
    beta1 = 1.2
    beta2 = 0.4
    beta3 = 1.2
    mu = 0.1
    id = 'XYSIR'

    datadir = os.getcwd()

    folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'

    folder_animations = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Animations/'

    # Load normalized dictionary to have the density of individuals
    dict1 = pickle.load(
        open(folder_dict_normHand + f'dict_data_beta{beta1}-mu{mu}-sim{sim}.pickle', 'rb'))
    dict1_values = list(dict1.values())

    dict2 = pickle.load(
        open(folder_dict_normHand + f'dict_data_beta{beta2}-mu{mu}-sim{sim}.pickle', 'rb'))
    dict2_values = list(dict2.values())

    dict3 = pickle.load(
        open(folder_dict_normHand + f'dict_data_beta{beta3}-mu{mu}-sim{sim}.pickle', 'rb'))
    dict3_values = list(dict3.values())
    # Determination of the maximum density of infected
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')

    t1_sims = np.load(folder_entropy + f'min_H0_sims-mu{mu}-beta{beta1}.npy')
    t1 = int(t1_sims[sim,1])
    t2 = np.load(folder_entropy + f'min_H0_sims-mu{mu}-beta{beta2}.npy')
    t3 = np.load(folder_entropy + f'min_H0_sims-mu{mu}-beta{beta3}.npy')

    mtrx1 = dict1_values[t1]
    x1 = mtrx1[:, 0]
    y1 = mtrx1[:, 1]
    dS1 = mtrx1[:,2]
    dI1 = mtrx1[:, 3]
    dR1 = mtrx1[:, 4]
    # Scatter plot
    sc = ax.scatter(dS1,dI1,dR1, c=dI1, cmap='gnuplot', marker='o')
    # Add color bar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label(r'Values $I/\langle n \rangle$')
    ax.set_xlabel(r' $S/\langle n \rangle$')
    ax.set_ylabel(r' $I/\langle n \rangle$')
    #ax.set_zlabel(r'$\Delta I$')
    ax.set_zlabel(r' $I/\langle n \rangle$')

    plt.tight_layout()
    plt.show()



    plt.tight_layout()
    plt.show()

