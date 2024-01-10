"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 29 December 2023

--------------------------------------------------------------------

Stochastic simulations with HMA and deterministic SIR model
- Deterministic SIR model
- Stochastic SIR model : Reed-Frost
- Stochastic SIR model : DTMC

"""
from functions_stochastic_SIR_HMA_v1 import *
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os

datadir = os.getcwd()
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})

# initial conditions
total_population = 1e4

nbr_I0 = 1
nbr_R0 = 0

T = 200
nbr_steps = T
T_sim = np.linspace(0, T - 1, nbr_steps)

beta = 0.3
mu = 0.1

compare_DT_CT_SIR = 0
stochastic_SIR = 1
stochastic_SIR_due = 0

if compare_DT_CT_SIR == 1:
    # Deterministic SIR model (discrete-time)
    nbr_St, nbr_It, nbr_Rt = deterministic_SIR_discrete_time(total_population, nbr_I0, nbr_R0, T, nbr_steps, beta, mu)

    rho_St = nbr_St / total_population
    rho_It = nbr_It / total_population
    rho_Rt = nbr_Rt / total_population

    f, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.plot(T_sim, rho_St, color='k', linestyle='--', label=r'Discrete-time: $\Delta t=1$ ')
    plt.plot(T_sim, rho_It, color='k', linestyle='--')
    plt.plot(T_sim, rho_Rt, color='k', linestyle='--')

    # Deterministic SIR model (discrete-time)
    nbr_steps = T * 10
    T_sim = np.linspace(0, T - 1, nbr_steps)
    nbr_St, nbr_It, nbr_Rt = deterministic_SIR_discrete_time(total_population, nbr_I0, nbr_R0, T, nbr_steps, beta, mu)

    rho_St = nbr_St / total_population
    rho_It = nbr_It / total_population
    rho_Rt = nbr_Rt / total_population

    plt.plot(T_sim, rho_St, color='k', linestyle=':', label=r'Discrete-time: $\Delta t=0.1$ ')
    plt.plot(T_sim, rho_It, color='k', linestyle=':')
    plt.plot(T_sim, rho_Rt, color='k', linestyle=':')

    # Deterministic SIR model (continuous-time)
    y_init = [(total_population - nbr_I0 - nbr_I0) / total_population, nbr_I0 / total_population,
              nbr_R0 / total_population]
    params = [beta, mu]
    y = odeint(deterministic_SIR_continuous_time, y_init, T_sim, args=(params,))

    plt.plot(T_sim, y[:, 0], linestyle='-', color='k', label='Continuous-time')
    plt.plot(T_sim, y[:, 1], linestyle='-', color='k')
    plt.plot(T_sim, y[:, 2], linestyle='-', color='k')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    plt.legend(fontsize=12)
    plt.show()

if stochastic_SIR == 1:

    total_population = 1e5

    nbr_I0 = 5
    nbr_R0 = 0

    # Probability of an adequate contact (for Reed-Frost)
    p = 2e-5

    T = 30

    nbr_steps = T
    T_sim = np.linspace(0, T - 1, nbr_steps)

    nbr_repetitions = 100

    # Deterministic Reed-Frost (does not work)
    nbr_St_det = np.zeros(T)
    nbr_It_det = np.zeros(T)
    nbr_Rt_det = np.zeros(T)

    nbr_Rt_det[0] = nbr_R0
    nbr_It_det[0] = nbr_I0
    nbr_St_det[0] = total_population - nbr_R0 - nbr_I0

    for t in range(1, T):
        # Deterministic Reed-Frost evolution
        nbr_St_det[t], nbr_It_det[t], nbr_Rt_det[t] = deterministic_Reed_Frost(p, nbr_St_det[t - 1], nbr_It_det[t - 1],
                                                                               nbr_Rt_det[t - 1])

    rho_St_det = nbr_St_det / total_population
    rho_It_det = nbr_It_det / total_population
    rho_Rt_det = nbr_Rt_det / total_population

    # Stochastic
    nbr_St = np.zeros(shape=(nbr_repetitions, T))
    nbr_It = np.zeros(shape=(nbr_repetitions, T))
    nbr_Rt = np.zeros(shape=(nbr_repetitions, T))
    rho_St = np.zeros(shape=(nbr_repetitions, T))
    rho_It = np.zeros(shape=(nbr_repetitions, T))
    rho_Rt = np.zeros(shape=(nbr_repetitions, T))
    for sim in range(nbr_repetitions):
        # in this case nbr_steps = T, so that dt = 1 (fixed by the model)
        # Initial conditions
        nbr_Rt[sim, 0] = nbr_R0
        nbr_It[sim, 0] = nbr_I0
        nbr_St[sim, 0] = total_population - nbr_R0 - nbr_I0

        for t in range(1, T):
            # Stochastic Reed-Frost evolution
            nbr_St[sim, t], nbr_It[sim, t], nbr_Rt[sim, t] = stochastic_Reed_Frost(p, nbr_It[sim, t - 1],
                                                                                   nbr_St[sim, t - 1],
                                                                                   nbr_Rt[sim, t - 1])

        rho_St[sim, :] = nbr_St[sim, :] / total_population
        rho_It[sim, :] = nbr_It[sim, :] / total_population
        rho_Rt[sim, :] = nbr_Rt[sim, :] / total_population

    avg_St = rho_St.mean(axis=0)
    avg_It = rho_It.mean(axis=0)
    avg_Rt = rho_Rt.mean(axis=0)

    stdDev_St = rho_St.std(axis=0, ddof=1)
    stdDev_It = rho_It.std(axis=0, ddof=1)
    stdDev_Rt = rho_Rt.std(axis=0, ddof=1)

    #plt.plot(T_sim, rho_St_det, color='k', linestyle='--', label=r'Deterministic ')
    #plt.plot(T_sim, rho_It_det, color='k', linestyle='--')
    #plt.plot(T_sim, rho_Rt_det, color='k', linestyle='--')

    plt.plot(T_sim, avg_St, color='b', label='S')
    plt.plot(T_sim, avg_It, color='r', label='I')
    plt.plot(T_sim, avg_Rt, color='green', label='R')

    plt.fill_between(T_sim, avg_St - stdDev_St, avg_St + stdDev_St,
                     facecolor='b', alpha=0.25)
    plt.fill_between(T_sim, avg_It - stdDev_It, avg_It + stdDev_It,
                     facecolor='r', alpha=0.25)
    plt.fill_between(T_sim, avg_Rt - stdDev_Rt, avg_Rt + stdDev_Rt,
                     facecolor='g', alpha=0.25)

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Densities', fontsize=14)

    plt.legend(fontsize=12)

    plt.show()

if stochastic_SIR_due == 1:
    total_population = 1e3

    nbr_I0 = 5
    nbr_R0 = 0
    beta = 0.2
    mu = 0.1
    T = 150
    nbr_steps = T * 100
    T_sim = np.linspace(0, T - 1, nbr_steps)

    nbr_repetitions = 3

    nbr_St_det = np.zeros(T)
    nbr_It_det = np.zeros(T)
    nbr_Rt_det = np.zeros(T)

    nbr_Rt_det[0] = nbr_R0
    nbr_It_det[0] = nbr_I0
    nbr_St_det[0] = total_population - nbr_R0 - nbr_I0

    # Deterministic timestep dt
    nbr_St_det, nbr_It_det, nbr_Rt_det = deterministic_SIR_discrete_time(total_population, nbr_I0, nbr_R0, T, nbr_steps, beta, mu)

    rho_St_det = nbr_St_det / total_population
    rho_It_det = nbr_It_det / total_population
    rho_Rt_det = nbr_Rt_det / total_population

    nbr_St = np.zeros(shape=(nbr_repetitions, nbr_steps))
    nbr_It = np.zeros(shape=(nbr_repetitions, nbr_steps))
    nbr_Rt = np.zeros(shape=(nbr_repetitions, nbr_steps))
    rho_St = np.zeros(shape=(nbr_repetitions, nbr_steps))
    rho_It = np.zeros(shape=(nbr_repetitions, nbr_steps))
    rho_Rt = np.zeros(shape=(nbr_repetitions, nbr_steps))
    for sim in range(nbr_repetitions):
        nbr_St[sim, :], nbr_It[sim, :], nbr_Rt[sim, :] = stochastic_SIR_discrete_time(total_population, nbr_I0, nbr_R0, T, nbr_steps, beta, mu)

        rho_St[sim, :] = nbr_St[sim, :] / total_population
        rho_It[sim, :] = nbr_It[sim, :] / total_population
        rho_Rt[sim, :] = nbr_Rt[sim, :] / total_population


    avg_St = rho_St.mean(axis=0)
    avg_It = rho_It.mean(axis=0)
    avg_Rt = rho_Rt.mean(axis=0)

    stdDev_St = rho_St.std(axis=0, ddof=1)
    stdDev_It = rho_It.std(axis=0, ddof=1)
    stdDev_Rt = rho_Rt.std(axis=0, ddof=1)

    f, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.plot(T_sim, rho_St_det, color='k', linestyle='--', label=r'Deterministic ')
    plt.plot(T_sim, rho_It_det, color='k', linestyle='--')
    plt.plot(T_sim, rho_Rt_det, color='k', linestyle='--')


    for sim in range(nbr_repetitions):
        plt.plot(T_sim, rho_St[sim, :], color='b', label='S')
        plt.plot(T_sim, rho_It[sim, :], color='r', label='I')
        plt.plot(T_sim, rho_Rt[sim, :], color='green', label='R')

    #plt.plot(T_sim, avg_St, color='b', label='S')
    #plt.plot(T_sim, avg_It, color='r', label='I')
    #plt.plot(T_sim, avg_Rt, color='green', label='R')

    #plt.fill_between(T_sim, avg_St - stdDev_St, avg_St + stdDev_St,
    #                 facecolor='b', alpha=0.25)
    #plt.fill_between(T_sim, avg_It - stdDev_It, avg_It + stdDev_It,
    #                 facecolor='r', alpha=0.25)
    #plt.fill_between(T_sim, avg_Rt - stdDev_Rt, avg_Rt + stdDev_Rt,
    #                 facecolor='g', alpha=0.25)


    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Density', fontsize=14)

    plt.legend(fontsize=12)
    plt.show()
