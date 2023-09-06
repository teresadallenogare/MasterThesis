"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 31 August 2023

--------------------------------------------------------------------

Functions implementing the simulation of the SIR stochastic model

"""

from functions_SIR import *

import numpy as np
import random

from time import time


#  ------------------------------- Stochastic simulations --------------------------
def SimSIRAgentBased( N, neig_list, I_init, X_init, T, beta, mu, N_row, rnd_nodes):
    """

    ---- Description
    Simulation of the stochastic SIR model

    ---- Parameters
    @param A:
    @param N: number of nodes
    @param neig_list: list with first neighbours of each node
    @param I_init:
    @param X_init: list with initial state of the lattice
    @param T: end of the simulation
    @param beta: rate of infection
    @param mu: rate of recovery
    @param N_row: number of rows of the lattice (for square lattice)
    @param rnd_nodes: list with control nodes chosen randomly
    @return:

    """

    start = time()
    # -------------------- Initialization -------------------- 
    # Initialize probability rates (same probability for each node)
    ProbS = np.full(N, 1. - I_init / N)
    ProbI = np.full(N, I_init / N)
    ProbR = np.full(N, 0.)

    # Initialize list X containing the state of each node
    X = []
    X.append(X_init)

    # Initialize delta probability vectors (are 0 or 1)
    deltaS = np.zeros(N)
    deltaI = np.zeros(N)
    deltaR = np.zeros(N)
    prod = np.full(N, 1.)

    # Initialize densities of each compartment 
    rhoS = np.zeros(T)
    rhoI = np.zeros(T)
    rhoR = np.zeros(T)

    # -------------------- Main loop -------------------- 
    Ptot_S = []
    Ptot_I = []
    Ptot_R = []
    times = []

    X_prev = []
    X_prev = X_init

    # S = []
    # I = []
    # R = []

    timesSI = []
    timesIR = []

    for t in range(T):
        print('\r', t, end='')
        l = []

        for i in range(N):
            deltaS[i] = int(X[t][i] == 'S')
            deltaI[i] = int(X[t][i] == 'I')
            deltaR[i] = int(X[t][i] == 'R')
            if X[t][i] == 'S':
                rhoS[t] += 1. / N
            elif X[t][i] == 'I':
                rhoI[t] += 1. / N
            else: #X[t][i] == 'R':
                rhoR[t] += 1. / N

        # Compute a separate loop to account for all updated deltas
        for i in range(N):
            prod[i] = 1
            for j in neig_list[i]:  # among nearest neighbours
                prod[i] = prod[i] * (1. - beta * deltaI[j])
                if i == 80:
                    print('---- i:', i)
                    print('\nj:', j, '\n deltaS[j]:', deltaS[j], '\n deltaI[j]: ', deltaI[j], '\n deltaR[j]: ', deltaR[j], '\nstatus', X[t][j])

            # transition probabilities
            ProbS[i] = deltaS[i] * prod[i]
            ProbI[i] = (1. - mu) * deltaI[i] + deltaS[i] * (1. - prod[i])
            ProbR[i] = deltaR[i] + mu * deltaI[i]

            l.append(random.choices(['S', 'I', 'R'], np.array([ProbS[i], ProbI[i], ProbR[i]]), k=1)[0])
            # Control that total probability of each node at each iteration is 1
            Ptot = ProbS[i] + ProbI[i] + ProbR[i]
            if Ptot - 1 != 0:
                print(Ptot, i)

        X.append(l)
        Ptot_S.append(sum(ProbS) / N)
        Ptot_I.append(sum(ProbI) / N)
        Ptot_R.append(sum(ProbR) / N)
        times.append(t)
        # s, i, r = GeneratetimesSeries_SIR(t, N, X_prev, X[t])
        # timesSI, timesIR= Storetimes_ControlNodes( N_row, rnd_nodes, X_prev, X[t], timesSI=timesSI, timesIR=timesIR, t = t)
        X_prev = X[t]
        # S.append(s)
        # I.append(i)
        # R.append(r)
    print('timesSI: ', timesSI)
    print('timesIR', timesIR)

    # plt.plot(times,R, linewidth = 0.75, label = 'R')
    # plt.plot(times,I, linewidth = 0.75, label = 'I')

    # plt.legend()
    # plt.show()
    return np.array([np.array(xi) for xi in X]), Ptot_S, Ptot_I, Ptot_R, times
