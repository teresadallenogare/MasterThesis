"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 08 November 2023

--------------------------------------------------------------------

Calculation of persistent entropy for a pair of nodes in the network.
Reasoning follows the paper 'TDA of synchronization of a network...'.
Consider the time series of new infected.

I fix a pair of nodes. I consider the 2D space of the new infected of the two nodes.
A plot of it corresponds to x: newI_node1 and y: newI_node2.
Then, I calculate the PE in this space as a function of the R0

"""

import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from ripser import ripser
from persim.persistent_entropy import *


datadir = os.getcwd()
datadir = datadir
# Parameters
row = 30
col = 30

c1 = 0
choice_bool = 0

beta = 0.9
mu = 0.1
sim = 7

idx_node1 = 0
idx_node2 = 99

normalized = 0

beta_vals = [0.35, 0.75, 0.4, 0.3, 0.9]
mu_vals = [0.3, 0.6, 0.2, 0.1, 0.1]

R0 = np.array([1.17, 1.25, 2, 3, 9 ])
def Euclidean_distance(x, y):
    """ Computes the pairwise Euclidean distance between point x and y that, in principle, can be embedded in any dimensional
    space. The dimension of the space is indeed given by len(x) or len(y) that represents the number of coordinates
    of each point in the space

    :param x: point 1 of point cloud dataset of dimension len(x)
    :param y: point 2 of point cloud dataset of dimension len(y)
    :return: Euclidean distance (scalar) between points x and y.
    """
    sum = 0
    for i in range(len(x)):
        sum = sum + (x[i]-y[i])**2


    return np.sqrt(sum)

entropy_H0_R0 = []
entropy_H1_R0 = []
for beta,mu in zip(beta_vals, mu_vals):
    folder_simulation = datadir + f'/Data-simpleLattice/{row}x{col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'
    # Length simulation
    T = np.load(folder_simulation + 'T.npy')
    T_sim = np.linspace(0, T, T + 1)

    # Number of new infected in every node in time for a fixed simulation
    new_I_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')

    new_I_time_node1 = new_I_time[:, idx_node1]
    new_I_time_node2 = new_I_time[:, idx_node2]

    plt.figure()
    plt.plot(T_sim, new_I_time_node1, color = 'b', label = f'node {idx_node1}')
    plt.plot(T_sim, new_I_time_node2, color = 'r', label = f'node {idx_node2}')
    plt.show()

    plt.figure()
    plt.plot(new_I_time_node1, new_I_time_node2, color ='r')
    plt.xlabel(f'newI node {idx_node1}')
    plt.ylabel(f'newI_node {idx_node2}')
    plt.show()

    new_I_12 = np.column_stack((new_I_time_node1, new_I_time_node2))

    # Persistent barcodes
    S = new_I_12
#    print('Set of points:')
#    for x in S:
#        print(x)
#    print('\n')

    # Compute number of points

    n = len(S)
#   print('Number of points:')
#   print(n)
#   print('\n')
    # Compute distance matrix

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = Euclidean_distance(S[i], S[j])

#   print('Distance matrix:')
#   print(D)
#   print('\n')

    # Write distance matrix to file
#   np.savetxt("distance_matrix_pair.csv", D, delimiter=",")

# Compute persistent homology and cycle representatives with Ripser
# Write result to screen
# subprocess.run(['./ripser-representatives', 'distance_matrix.csv'])

# Write result to file
# with open('ripser_localization.txt', 'w') as file:
#     subprocess.run(['./ripser-representatives', 'distance_matrix.csv'], stdout=file)

# Compute persistent homology with Ripser
    # Exactly the same result if I use the distance matrix or the system S
    data_phom = ripser(D, distance_matrix=True, maxdim = 1)
    data_2 = ripser(S)

# Write output on file
#    with open('data_phom.txt', 'w') as f:
#        for deg in range(len(data_phom['dgms'])):
#            f.write('Persistence intervals in degree ' + str(deg) + ':')
#            f.write(str(data_phom['dgms'][deg]))

# Write output on screen
#for deg in range(len(data_phom['dgms'])):
#    print('Persistence intervals in degree ' + str(deg) + ':')
#    print(data_phom['dgms'][deg])

# Persistent entropy

    entropy = persistent_entropy(data_phom['dgms'], normalize=True)
    entropy_H0 = entropy[0]
    entropy_H1 = entropy[1]
    entropy_H0_R0.append(entropy_H0)
    entropy_H1_R0.append(entropy_H1)
entropy_H0_R0 = np.array(entropy_H0_R0)
entropy_H1_R0 = np.array(entropy_H1_R0)

plt.plot(R0, entropy_H0_R0, color = 'r', label = 'PE at H0')
plt.plot(R0, entropy_H1_R0, color = 'b', label = 'PE at H1')
plt.xlabel('R0')
plt.ylabel('PE')
plt.legend()
plt.title(f'PE for pair of nodes {idx_node1}-{idx_node2} for sim {sim}')
plt.show()


