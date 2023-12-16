"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 15 December 2023

--------------------------------------------------------------------

Analysis of repeated trials of SIR data from simulations

"""

from functions_SIR_metapop_v1 import *
from functions_output_v1 import write_simulation_file
from functions_visualization_v1 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.integrate import odeint

datadir = os.getcwd()
plt.figure(figsize=(8, 8))

repeat_trials = 1

# ------------------------------------------------ Colors  -------------------------------------------------
grad_gray = []
grad_red = []
grad_blue = []
grad_green = []

for x in range(3):
    #                                dark           light
    grad_gray.append(colorFader('#505050', '#EAE9E9', x / 3))
    grad_red.append(colorFader('#E51C00', '#FCE0DC', x / 3))
    grad_blue.append(colorFader('#1D3ACE', '#C5CEFF', x / 3))
    grad_green.append(colorFader('#0A8E1A', '#DAF7A6', x / 3))

######################################################################################################################
if repeat_trials == 1:
    row = 30
    col = 30
    N = row * col

    choice_bool = 0
    c1 = 0

    bool_density = 1

    # Infection and recovery rate
    beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
    mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


    def exp_growth(x, y0, beta, mu):
        return y0 * np.exp((beta - mu) * x)

    for beta, mu in zip(beta_vals, mu_vals):
        x = np.linspace(0, 10, 100)
        folder_simulation = datadir + f'/Data_simpleLattice_v1/Repeated_trials/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

        nbr_repetitions = np.load(folder_simulation + f'nbr_repetitions.npy')
        idx_repetitions = np.linspace(0, nbr_repetitions - 1, nbr_repetitions)
        idx_sim_not_start = np.load(folder_simulation + 'idx_sim_not_start.npy')
        idx_sim_start = list((set(idx_repetitions) - set(idx_sim_not_start)))
        nbr_sim_start = len(idx_sim_start)
        print('nbr sim start: ', nbr_sim_start)
        T = np.load(folder_simulation + 'T.npy')
        print('row:', row, 'col:', col, 'choice_bool:', choice_bool, 'c1:', c1, 'beta:', beta, 'mu:', mu, 'T:', T)
        T_sim = np.linspace(0, T - 1, T)
        # Change with index simulation started
        y_mean_std = mean_stdDev_repetitions(row, col, choice_bool, c1, T, beta, mu, bool_density, idx_sim_start)
        mean_S_time = y_mean_std[0]
        mean_I_time = y_mean_std[1]
        mean_R_time = y_mean_std[2]
        stdDev_S_time = y_mean_std[3]
        stdDev_I_time = y_mean_std[4]
        stdDev_R_time = y_mean_std[5]

        idx_node = 0

        #### Deterministic model (only if the density is considered)
        if bool_density == 1:
            # Set initial conditions (is the same at every repetition!)
            val_NS0 = y_mean_std[6]
            val_NI0 = y_mean_std[7]
            val_NR0 = y_mean_std[8]

            y_init = [val_NS0, val_NI0, val_NR0]
            print('y_init: ', y_init)
            params = [beta, mu]
            # Sole equation for densities
            y = odeint(SIRDeterministic_equations, y_init, T_sim, args=(params,))

            # Deterministic densities in time: solutions of SIR deterministic ODEs
            det_s = y[:, 0]
            det_i = y[:, 1]
            det_r = y[:, 2]
        else:
            det_s = [0 for i in range(0, T)]
            det_i = [0 for i in range(0, T)]
            det_r = [0 for i in range(0, T)]

        # Appendix_A()

        plot_mean_std_singleNode(T_sim, mean_S_time, mean_I_time, mean_R_time, stdDev_S_time,
                                 stdDev_I_time, stdDev_R_time, det_s, det_i, det_r, idx_node, bool_density)
        #if bool_density == 1:
        #    plt.plot(x, exp_growth(x, val_NI0, beta, mu), 'k--')





