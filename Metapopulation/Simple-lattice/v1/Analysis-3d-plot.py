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
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.integrate import odeint
import seaborn as sns
from matplotlib.animation import FuncAnimation
datadir = os.getcwd()
#plt.figure(figsize=(10, 8))
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})


tred_plot = 1

if tred_plot == 1:
    row = 10
    col = 10

    N = row * col
    choice_bool = 0
    c1 = 1

    sim = 0
    beta = 0.9
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

    t = 24

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