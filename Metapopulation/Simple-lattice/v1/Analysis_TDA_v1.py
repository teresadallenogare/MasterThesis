"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 23 November 2023

--------------------------------------------------------------------

Analysis of data according to the Topological Data Analysis pipeline

"""

from functions_TDA_v1 import *
from functions_visualization_v1 import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import subprocess
import re
import seaborn as sns
from matplotlib.animation import FuncAnimation

datadir = os.getcwd()

generator_HomLoops = 1
plot_histogram = 0
find_ends_barcodes = 0
PE_beta_mu = 0
trial = 0


sim = 0

# normalization = 0 -> no normalized data
#               = 1 -> standard scaler normalization
#               = 2 -> normalization by hand
normalization = 1
id = 'XYSIR'
columns = ['X', 'Y', 'S', 'I', 'R']
nrm_entropy = [False]

# Dimension
row = 3
col = 3
N = row * col

# Population method
choice_bool_vals = [0]

# Strength self loops
c1_vals = [0]

# Infection and recovery rate
beta_vals_3_5_10 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2, 0.23, 0.24, 0.3, 0.4, 0.6, 0.8, 0.345, 0.36, 0.45, 0.6, 0.9, 1.2]
mu_vals_3_5_10 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

beta_vals_30_50 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


beta_vals_3_5_10 = [0.9]
mu_vals_3_5_10 = [0.1]
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

                with open(folder_ripser + f'1-ripser_localization-beta{beta}-mu{mu}-id{id}.txt', 'a') as file:
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
                        print('\n-----------t: ', t, ' ------------\n')
                        print(result.stdout)
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
                    #if normalization == 0:
                    #    np.save(folder_ripser+f'No-normalized/1-idx_nodes0_time-beta{beta}-mu{mu}-id{id}', idx_nodes0_time)
                    #elif normalization == 1:
                    #    np.save(folder_ripser+f'Normalized-scaler/1-idx_nodes0_time-beta{beta}-mu{mu}-id{id}', idx_nodes0_time)
                    #elif normalization == 2:
                    #    np.save(folder_ripser+f'Normalized-hand/1-idx_nodes0_time-beta{beta}-mu{mu}-id{id}', idx_nodes0_time)

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

#-----------------------------------------------------------------------------------------------------------------------
# Find endpoints of persistence barcodes
#-----------------------------------------------------------------------------------------------------------------------
beta = 0.9
mu = 0.1

if find_ends_barcodes == 1:
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
            folder_ripser = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Ripser_analysis/'
            folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'
            folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'

            folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
            T = np.load(folder_simulation + 'T.npy')
            T_sim = np.linspace(0, T - 1, T)
            if normalization == 0:
                idx_nodes0_time = np.load(folder_ripser + f'No-normalized/idx_nodes0_time-beta{beta}-mu{mu}-id{id}.npy')
                dict_data = pickle.load(
                open(folder_dict_noNorm + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
                # 1. No normalized DataFrame
                df_dict_data = data_2_pandas(dict_data)
            elif normalization == 1:
                idx_nodes0_time = np.load(folder_ripser + f'Normalized-scaler/idx_nodes0_time-beta{beta}-mu{mu}-id{id}.npy')
                dict_data = pickle.load(
                open(folder_dict_noNorm + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
                # 1. No normalized DataFrame
                df_dict_data = data_2_pandas(dict_data)
                # 2. Scaled with StdScaler DataFrame
                df_dict_data = scaler_df_data_dict(df_dict_data)
            elif normalization == 2:
                idx_nodes0_time = np.load(folder_ripser + f'Normalized-hand/idx_nodes0_time-beta{beta}-mu{mu}-id{id}.npy')
                # 3. Normalized by hand DataFrame
                # Import normalized by hand dictionary
                dict_data = pickle.load(
                    open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
                df_dict_data = data_2_pandas(dict_data)
            # Create dataframe in which I have the relationship between the position and the node ID
            df_pos_label_node = df_dict_data[df_dict_data['Time'] == 0][['X', 'Y', 'Node ID']]
            # Get the number of columns in the matrix
            num_columns = idx_nodes0_time.shape[1]

            # Create masks for odd and even columns
            even_columns = np.arange(num_columns) % 2 == 0
            odd_columns = np.arange(num_columns) % 2 == 1

            # Use array slicing to create submatrices
            # mtrx_start0_time: each column contains the index of the node corresponding to the starting point of a new barcode
            # while each column shows the index of the node corresponding to the starting point of the barcode at a different timestep
            # the same for the end matrix but with the endpoint
            # start = even
            mtrx_start0_time = idx_nodes0_time[:, even_columns]
            mtrx_start0_time_infinite_bar = mtrx_start0_time[:, -1] # take the last colum that is the index of the nodes for the starting point of the infinite barcode
            mtrx_start0_time = mtrx_start0_time[:, :-1] # take all the other indeces corresponding to the starting point of an ending barcode

            # end = odd
            mtrx_end0_time = idx_nodes0_time[:, odd_columns]

            # Barcodes with the longest life are always the last ones. Thus, I can take the indexes of the nodes
            # that contribute to the longest barcodes by selecting the last columns
            num_cols_start = mtrx_start0_time.shape[1]
            num_cols_end = mtrx_end0_time.shape[1]

            # Indexes of the nodes contributing to the starting point of the 3 most relevant barcodes in time
            mtrx_start0_time_last3 = mtrx_start0_time[:, num_cols_start - 3:]
            # Indexes of the nodes contributing to the ending point of the 3 most relevant barcodes in time
            mtrx_end0_time_last3 = mtrx_end0_time[:, num_cols_end - 3:]


            # Function to update the plot for each time step
            def update(frame):
                plt.clf()
                plt.scatter(df_pos_label_node['X'], df_pos_label_node['Y'], c='blue', marker='o')
                # 3rd most important barcode
                df_filt_start3 = df_pos_label_node[df_pos_label_node['Node ID'] == mtrx_start0_time_last3[frame, 0]]
                df_filt_end3 = df_pos_label_node[df_pos_label_node['Node ID'] == mtrx_end0_time_last3[frame, 0]]
                x_val_start3 = df_filt_start3['X'].values[0]
                y_val_start3 = df_filt_start3['Y'].values[0]
                x_val_end3 = df_filt_end3['X'].values[0]
                y_val_end3 = df_filt_end3['Y'].values[0]

                # 2nd most important barcode
                df_filt_start2 = df_pos_label_node[df_pos_label_node['Node ID'] == mtrx_start0_time_last3[frame, 1]]
                df_filt_end2 = df_pos_label_node[df_pos_label_node['Node ID'] == mtrx_end0_time_last3[frame, 1]]
                x_val_start2 = df_filt_start2['X'].values[0]
                y_val_start2 = df_filt_start2['Y'].values[0]
                x_val_end2 = df_filt_end2['X'].values[0]
                y_val_end2 = df_filt_end2['Y'].values[0]

                # 1st most important barcode
                df_filt_start1 = df_pos_label_node[df_pos_label_node['Node ID'] == mtrx_start0_time_last3[frame, 2]]
                df_filt_end1 = df_pos_label_node[df_pos_label_node['Node ID'] == mtrx_end0_time_last3[frame, 2]]
                x_val_start1 = df_filt_start1['X'].values[0]
                y_val_start1 = df_filt_start1['Y'].values[0]
                x_val_end1 = df_filt_end1['X'].values[0]
                y_val_end1 = df_filt_end1['Y'].values[0]

                # Plot line connecting nodes
                plt.plot([x_val_start3, x_val_end3],  # X
                         [y_val_start3, y_val_end3],        # Y
                         color = 'k', label = '3rd barcode')
                plt.plot([x_val_start2, x_val_end2],  # X
                         [y_val_start2, y_val_end2],        # Y
                         color = 'g', label = '2nd barcode')
                plt.plot([x_val_start1, x_val_end1],  # X
                         [y_val_start1, y_val_end1],        # Y
                         color = 'r', label = '1st barcode')
                plt.title(f'Time Step: {frame}- beta: {beta}, mu: {mu}')
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')

            # Create an animation
            plt.figure(figsize=(8, 8))
            animation = FuncAnimation(plt.gcf(), update, frames= df_dict_data['Time'].max(), repeat=False)
            plt.show()


            print('hello')

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
    for choice_bool in choice_bool_vals:
        for c1 in c1_vals:
            folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'
            if normalization == 0:
                folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/No-normalized/{id}/'
            elif normalization == 1:
                folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'
            elif normalization == 2:
                folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-scaler/{id}/'
            for beta, mu in zip(beta_vals, mu_vals):
                T = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/T.npy')
                print('row: ', row, 'col: ', col, 'choice_bool: ', choice_bool, 'c1: ', c1)
                print('beta: ', beta, 'mu: ', mu)

                for normalize_entropy in nrm_entropy:
                    plt.figure(figsize=(8, 8))
                    entropy_H0 = np.load(folder_entropy + f'entropy_H0-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}.npy')
                    entropy_H1 = np.load(folder_entropy + f'entropy_H1-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}.npy')

                    x = range(0, len(entropy_H0))
                    min_yH0, t_min_yH0 = min_PE(entropy_H0, x)
                    min_yH1, t_min_yH1 = min_PE(entropy_H1, x)

                    print('min H0: ', min_yH0, 't min: ', t_min_yH0)
                    print('min H1: ', min_yH1, 't min: ', t_min_yH1)

                    plt.plot(x, entropy_H0, color = 'r', label = 'entropy at H0')
                    plt.plot(x, entropy_H1, color = 'b', label = 'entropy at H1')
                    plt.scatter(t_min_yH0, min_yH0)
                    plt.scatter(t_min_yH1, min_yH1)
                    plt.xlabel('Time')
                    plt.ylabel('Persistent Entropy')
                    plt.title(f'PE choice_bool:{choice_bool}, c1:{c1}, beta:{beta}, mu:{mu}, normE:{normalize_entropy}, normDat:{normalization}')
                    plt.legend()
                    plt.show()









