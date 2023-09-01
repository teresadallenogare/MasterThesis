"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 31 August 2023

--------------------------------------------------------------------

Simulation of stochastic and deterministic SIR models on a lattice.
Functions to run the code are in 'functions_SIR.py'

"""

from functions_SIR import *
from simulation_SIR import *

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import os
from scipy.integrate import odeint

rnd.seed(5)

new_cmap = ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b",
            "#1a1a1a", "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c",
            "#08306b"]
rtg_r = LinearSegmentedColormap.from_list("rtg", new_cmap)
colors = rtg_r(np.linspace(0, 1, 20))

datadir = os.getcwd()

# ======================== Parameters initialization ==============================
# Number of repetitions of the stochastic simulation
M = 1

# Length of the simulation
T = 200  # int(50/beta)

# Number of rows in lattice for network
N_row = np.array([10])

# Values of transition rates
# beta_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# mu_vals = np.array ([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
beta_vals = np.array([0.2])
mu_vals = np.array([0.1])

# ======================== Graph creation ==============================
# Cycle on dimensions of the lattice
for i in range(0, len(N_row)):
    # Definition of the graph : Lattice, Triangle, Hexagon
    G, pos = DefLattice('Lattice', Nrow=N_row[i], Ncol=N_row[i])
    N = len(G.nodes())
    # Adjacency matrix (N x N)
    A = nx.to_numpy_array(G)
    # Neighbour list
    neig_list = CreateNeighbourList_lattice(A, N)

    # Definition of the initial state of the network 
    I_init = 1.
    R_init = 0.
    S_init = N - R_init - I_init
    # X_init = random.choices( ['S', 'I', 'R'], [1. - I_init / N, I_init / N, R_init / N], k=N)
    X_init = ['I']
    for j in range(N - int(I_init)):
        X_init.append('S')
    # X_init.append('I')

    # Choice of random nodes (that is one-tenth) fixed in time
    rnd_nodes = []
    for j in range(int(N / 10)):
        rnd_node = rnd.choice(list(G.nodes))
        if rnd_node in rnd_nodes:
            # print('already exists')
            j = j - 1
            # print('j', j)
        else:
            rnd_nodes.append(rnd_node)
    # print('rnd_nodes: ', rnd_nodes)

    y_init = [S_init / N, I_init / N, R_init / N]

    # ==================================== Simulation ====================================
    for bb in range(0, len(beta_vals)):
        for mm in range(0, len(mu_vals)):
            beta = beta_vals[bb]
            mu = mu_vals[mm]
            print('tau_beta = ', 1 / beta, 'tau_mu = ', 1 / mu, 'R0 = ', beta / mu)

            # Repeat stochastic simulation M times
            tot_S = 0
            tot_I = 0
            tot_R = 0
            for ii in range(M):
                print('\r ii: ', ii, end='')
                X, Ptot_S, Ptot_I, Ptot_R, time = SimSIRAgentBased(N, neig_list, I_init, X_init, T,
                                                                   beta=beta,
                                                                   mu=mu,
                                                                   N_row=int(N_row),
                                                                   rnd_nodes=rnd_nodes)

                # np.save(datadir+f'/X_b={beta_vals[bb]}m={mu_vals[mm]}',X)
                tot_S = tot_S + np.asarray(Ptot_S)
                tot_I = tot_I + np.asarray(Ptot_I)
                tot_R = tot_R + np.asarray(Ptot_R)
                plt.plot(Ptot_S, Ptot_R, linewidth=0.75)
                plt.plot([0, 1], [1, 0], linewidth=0.55, color='black')
            avg_PtotS = tot_S / M
            avg_PtotI = tot_I / M
            avg_PtotR = tot_R / M
            plt.plot(avg_PtotS, avg_PtotR, linewidth=1, color='black')

            params = [beta, mu]
            y = odeint(SimSIRDeterministic, y_init, time, args=(params,))

            plt.plot(y[:, 0], y[:, 2], linewidth=1, color='black', linestyle='dashed')
            plt.xlabel('Prob(S)')
            plt.ylabel('Prob(R)')
            plt.title(f' beta={beta}, mu={mu}')
            plt.show()
            plotSIR_lattice(T, int(N_row), N, pos, G, X, rnd_nodes)
            # plotSIR_line(time, Ptot_S, Ptot_I, Ptot_R, y, N, beta_vals[bb], mu_vals[mm])

# InfectedBetaMu(y_init, time, beta_vals, mu_vals)
