"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 29 December 2023

--------------------------------------------------------------------

Functions for the stochastic SIR model

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
from scipy.integrate import odeint


def deterministic_SIR_discrete_time(total_population, nbr_I0, nbr_R0, T, nbr_steps, beta, mu):
    nbr_St = np.zeros(nbr_steps)
    nbr_It = np.zeros(nbr_steps)
    nbr_Rt = np.zeros(nbr_steps)
    alpha_t = np.zeros(nbr_steps)

    dt = T / nbr_steps

    # set initial conditions (t = 0)
    nbr_St[0] = total_population - nbr_I0 - nbr_R0
    nbr_It[0] = nbr_I0
    nbr_Rt[0] = nbr_R0

    # force of infection
    alpha_t[0] = beta * nbr_I0 / total_population

    # Here assuming dt = 1
    for t in range(1, nbr_steps):
        alpha_t[t] = beta * nbr_It[t - 1] / total_population
        nbr_St[t] = nbr_St[t - 1] * (1. - alpha_t[t - 1] * dt)
        nbr_It[t] = nbr_It[t - 1] * (1. - mu * dt) + alpha_t[t - 1] * nbr_St[t - 1] * dt
        nbr_Rt[t] = nbr_Rt[t - 1] + mu * nbr_It[t - 1] * dt

    return nbr_St, nbr_It, nbr_Rt


def deterministic_SIR_continuous_time(variables, t, params):
    """

    ---- Description
    Compute the deterministic SIR equations

    ---- Parameters
    @param variables:
    @param t:
    @param params:
    @return:

    """
    S = variables[0]
    I = variables[1]
    R = variables[2]

    N = S + I + R
    beta = params[0]
    mu = params[1]

    alpha = beta * I / N

    dSdt = - alpha * S
    dIdt = alpha * S - mu * I
    dRdt = mu * I

    return [dSdt, dIdt, dRdt]


def deterministic_Reed_Frost(p, nbr_It, nbr_St, nbr_Rt):
    nbr_It_plus_1 = nbr_St * (1. - (1 - p)**nbr_It)
    nbr_St_plus_1 = nbr_St - nbr_It_plus_1
    nbr_Rt_plus_1 = nbr_Rt + nbr_It

    return nbr_St_plus_1, nbr_It_plus_1, nbr_Rt_plus_1

def stochastic_Reed_Frost(p, nbr_It, nbr_St, nbr_Rt):

    # Probability of S -> I
    p_SI = 1. - (1 - p)**nbr_It

    # Sample from binomial distribution (number of infected at t+1)
    nbr_It_plus_1 = np.random.binomial(nbr_St, p_SI)
    # Number of susceptibles at t+1
    nbr_St_plus_1 = nbr_St - nbr_It_plus_1
    # Number of recovered at t+1
    nbr_Rt_plus_1 = nbr_Rt + nbr_It

    return nbr_St_plus_1, nbr_It_plus_1, nbr_Rt_plus_1


def stochastic_SIR_discrete_time(total_population, nbr_I0, nbr_R0, T, nbr_steps, beta, mu):
    nbr_St = np.zeros(nbr_steps)
    nbr_It = np.zeros(nbr_steps)
    nbr_Rt = np.zeros(nbr_steps)

    dt = T / nbr_steps

    # set initial conditions (t = 0)
    nbr_St[0] = total_population - nbr_I0 - nbr_R0
    nbr_It[0] = nbr_I0
    nbr_Rt[0] = nbr_R0

    for t in range(1, nbr_steps):
        rnd_SI = np.random.uniform(0, 1)
        rnd_IR = np.random.uniform(0, 1)

        norm = beta * nbr_St[t - 1] / total_population + mu

        # Probability of S->I over the probability of having any transition
        prob_SI = beta * nbr_St[t - 1] /(beta * nbr_St[t-1] + mu * total_population)
        # Probability of I->R over the probability of having any transition
        prob_IR = mu / norm


        if rnd_SI <= prob_SI and nbr_St[t]>0:
            nbr_St[t] = nbr_St[t - 1] - 1
            nbr_It[t] = nbr_It[t - 1] + 1
            nbr_Rt[t] = nbr_Rt[t - 1]
        else:
            nbr_St[t] = nbr_St[t - 1]
            nbr_Rt[t] = nbr_Rt[t - 1]
            nbr_It[t] = nbr_It[t - 1]

        if rnd_IR <= prob_IR and nbr_It[t] > 0:
            nbr_St[t] = nbr_St[t - 1]
            nbr_It[t] = nbr_It[t - 1] - 1
            nbr_Rt[t] = nbr_Rt[t - 1] + 1
        else:
            nbr_St[t] = nbr_St[t - 1]
            nbr_Rt[t] = nbr_Rt[t - 1]
            nbr_It[t] = nbr_It[t - 1]

    return nbr_St, nbr_It, nbr_Rt
