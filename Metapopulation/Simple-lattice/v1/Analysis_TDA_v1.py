"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 23 November 2023

--------------------------------------------------------------------

Analysis of data according to the Topological Data Analysis pipeline

"""

from functions_TDA_v1 import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import subprocess
import re
import seaborn as sns

datadir = os.getcwd()

generator_HomLoops = 0
plot_histogram = 1
PE_beta_mu = 0

sim = 0

# normalization = 0 -> no normalized data
#               = 1 -> standard scaler normalization
#               = 2 -> normalization by hand
normalization = 1
id = 'XYSIR'
columns = ['X', 'Y', 'S', 'I', 'R']
nrm_entropy = False

# Dimension
row = 30
col = 30
N = row * col

# Population method
choice_bool_vals = [0, 1]

# Strength self loops
c1_vals = [0, 1]

# Infection and recovery rate
beta_vals_3_5_10 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2, 0.23, 0.24, 0.3, 0.4, 0.6, 0.8, 0.345, 0.36, 0.45, 0.6, 0.9, 1.2]
mu_vals_3_5_10 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

beta_vals_30_50 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

########################################################################################################################
# Fix configuration (dim, population, strength loops) and extract generators of homological loops.
########################################################################################################################
# Generate matrix with index of nodes contributing to H0
#-----------------------------------------------------------------------------------------------------------------------

if generator_HomLoops == 1:
    if row == 3 or row == 5 or row == 10:
        beta_vals = beta_vals_3_5_10
        mu_vals = mu_vals_3_5_10
    else:
        beta_vals = beta_vals_30_50
        mu_vals = mu_vals_30_50
    for choice_bool in choice_bool_vals:
        print('choice bool: ', choice_bool)
        for c1 in c1_vals:
            print('c1: ', c1)
            folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
            folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'
            folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'
            folder_ripser = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Ripser_analysis/'

            folder_entropy_normScaler = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-scaler/{id}/'

            for beta, mu in zip(beta_vals, mu_vals):
                print('beta: ', beta, 'mu: ', mu)
                folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
                T = np.load(folder_simulation + 'T.npy')
                T_sim = np.linspace(0, T - 1, T)
                if normalization != 2:
                    # Import no normalized dictionary
                    dict_data = pickle.load(
                        open(folder_dict_noNorm + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
                    # 1. No normalized DataFrame
                    df_dict_data = data_2_pandas(dict_data)
                    if normalization == 1:
                        # 2. Scaled with StdScaler DataFrame
                        df_dict_data = scaler_df_data_dict(df_dict_data)
                elif normalization == 2:
                    # 3. Normalized by hand DataFrame
                    # Import normalized by hand dictionary
                    dict_data = pickle.load(
                        open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
                    df_dict_data = data_2_pandas(dict_data)

                with open(folder_ripser + f'ripser_localization-beta{beta}-mu{mu}-id{id}.txt', 'a') as file:
                    for t in range(T):
                        # Dataframe at fixed time value
                        S_t = df_dict_data.loc[df_dict_data['Time'] == t]
                        # Select only data indexed by "column"
                        S_data_t = S_t[columns]
                        # Euclidean distance between pairs of points (Each point corresponds to a row. The dimension of the space is given by the number of columns.)
                        D_t = cdist(S_data_t, S_data_t, 'euclid')
                        # Write distance matrix to file (over-write at each t)
                        np.savetxt(folder_ripser + 'D_t.csv', D_t, delimiter = ',')
                        #### Compute persistent homology and cycle representatives with Ripser
                        # Write result to file
                        subprocess.run([folder_ripser + 'ripser-representatives', folder_ripser + 'D_t.csv'], stdout=file)
                        # Write result to screen
                        result = subprocess.run([folder_ripser + 'ripser-representatives', folder_ripser + 'D_t.csv'],
                                                capture_output=True,
                                                text= True)
                        #print(result.stdout)
                        #### Generators of persistence intervals in dimension 0
                        # Extract nodes generating the 0-dim persistence intervals (are the integers inside the [])
                        if t == 0:
                            # 0 after nodes stands for the dimension of the space to which nodes contribute
                            idx_nodes0_t0 = re.findall(r"\[\s*\+?(-?\d+)\s*\]", result.stdout)
                            # Convert the elements of the list from strings to integers
                            idx_nodes0_t0 = string_2_int(idx_nodes0_t0)
                        elif t == 1:
                            idx_nodes0_t1 = re.findall(r"\[\s*\+?(-?\d+)\s*\]", result.stdout)
                            # Convert the elements of the list from strings to integers
                            idx_nodes0_t1 = string_2_int(idx_nodes0_t1)

                            idx_nodes0_time = np.vstack((idx_nodes0_t0, idx_nodes0_t1))
                        else:
                            idx_nodes0_t = re.findall(r"\[\s*\+?(-?\d+)\s*\]", result.stdout)
                            idx_nodes0_t = string_2_int(idx_nodes0_t)
                            idx_nodes0_time = np.vstack((idx_nodes0_time, idx_nodes0_t))

                        #print(re.findall(r"\[\s*[-+]?(?:\d*\.*\d+)\s*\]", result.stdout))
                    if normalization == 0:
                        np.save(folder_ripser+f'No-normalized/idx_nodes0_time-beta{beta}-mu{mu}-id{id}', idx_nodes0_time)
                    elif normalization == 1:
                        np.save(folder_ripser+f'Normalized-scaler/idx_nodes0_time-beta{beta}-mu{mu}-id{id}', idx_nodes0_time)
                    elif normalization == 2:
                        np.save(folder_ripser+f'Normalized-hand/idx_nodes0_time-beta{beta}-mu{mu}-id{id}', idx_nodes0_time)

                    file.close()

#-----------------------------------------------------------------------------------------------------------------------
# Plot histogram showing the frequency with which each node appears in time
#-----------------------------------------------------------------------------------------------------------------------

if plot_histogram == 1:
    if row == 3 or row == 5 or row == 10:
        beta_vals = beta_vals_3_5_10
        mu_vals = mu_vals_3_5_10
    else:
        beta_vals = beta_vals_30_50
        mu_vals = mu_vals_30_50
    for choice_bool in choice_bool_vals:
        plt.figure(figsize=(15, 8))
        print('choice bool: ', choice_bool)
        for c1 in c1_vals:
            print('c1: ', c1)
            folder_ripser = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Ripser_analysis/'
            for beta, mu in zip(beta_vals, mu_vals):
                folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
                T = np.load(folder_simulation + 'T.npy')
                T_sim = np.linspace(0, T - 1, T)
                if normalization == 0:
                    idx_nodes0_time = np.load(folder_ripser + f'No-normalized/idx_nodes0_time-beta{beta}-mu{mu}-id{id}.npy')
                elif normalization == 1:
                    idx_nodes0_time = np.load(folder_ripser + f'Normalized-scaler/idx_nodes0_time-beta{beta}-mu{mu}-id{id}.npy')
                elif normalization == 2:
                    idx_nodes0_time = np.load(folder_ripser + f'Normalized-hand/idx_nodes0_time-beta{beta}-mu{mu}-id{id}.npy')

                mtrx_hist_nodes0_time = np.zeros(shape = (T, N))

                for t in range(T):
                    for i in range(N):
                        cont = (idx_nodes0_time[t, :] == i).sum()
                        mtrx_hist_nodes0_time[t, i] = cont
                array_hist_nodes0 = np.sum(mtrx_hist_nodes0_time, axis = 0)
                top_indices = np.argsort(array_hist_nodes0)[-3:]
                print('top indeces: ', top_indices)
                #array_hist_nodes0 = array_hist_nodes0.reshape((N, 1))

                # Try heatmap + histogram
                # Create a subplot grid
                fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 0.2]}, figsize=(10, 5))
                idx_nodes = np.linspace(0, N - 1, N)
                mtrx_hist_nodes0_time_T = mtrx_hist_nodes0_time.T
                vmin = mtrx_hist_nodes0_time_T.min()
                vmax = mtrx_hist_nodes0_time_T.max()
                heatm = sns.heatmap(mtrx_hist_nodes0_time_T, ax=ax1, cmap='viridis', cbar=True)
                heatm.invert_yaxis()
                #ax.set_xlabel('Time')
                #ax.set_ylabel('Node index')
                #ax.set_title(f'{row}x{col} ch_bool={choice_bool} c1={c1} beta={beta} mu={mu}')
                bars = ax2.barh(idx_nodes, array_hist_nodes0, color = 'gray' )
                # Color the top three bars differently
                for i in top_indices:
                    bars[i].set_color('red')
                ax1.set_title(f'{row}x{col} ch_bool={choice_bool} c1={c1} beta={beta} mu={mu}')
                plt.show()

                # Try 2D
                #idx_nodes = np.linspace(0, N-1, N)
                #plt.bar(idx_nodes, array_hist_nodes0, color='crimson', edgecolor='white')
                #plt.xlabel('index node')
                #plt.ylabel('frequency')
                #plt.title(f'{row}x{col} ch_bool={choice_bool} c1={c1} beta={beta} mu={mu}')
                #plt.show()

########################################################################################################################
# Fix configuration (dim, population, strength loops) and study PE as a function of beta and mu
########################################################################################################################
if PE_beta_mu == 1:
    if row == 3 or row == 5 or row == 10:
        beta_vals = beta_vals_3_5_10
        mu_vals = mu_vals_3_5_10
    else:
        beta_vals = beta_vals_30_50
        mu_vals = mu_vals_30_50

    print('hello')





