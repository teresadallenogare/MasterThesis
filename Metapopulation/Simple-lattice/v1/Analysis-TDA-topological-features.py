"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 04 February 2023

--------------------------------------------------------------------

Extract topological features from data of single realizations at time step corresponding to the time at minimum of entropy
Idea: Compute PE/NPE with all the features both for H0 and H1. Then, I want to separate noise from data.
This can be done for a fixed simulation and at a fixed time. Moreover, the number of topological features that are not noise
changes as a function of time.
First, I fix one simulation from the repeated trials.
To find the time at which studying topological features, I consider the time at which the entropy is minimum.
This is different for H0 and H1, so I need to extract topological features separately, as the number and nature of the
intervals that are topological features changes with time.
Extraction of topological features is done following Rucco's proceeding that relies on PE.
Once I extracted the topological features, I study their evolution in time!
All the previous steps were necessary to find which are the topological features.

"""
from functions_TDA_v1 import *
from functions_visualization_v1 import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

acquire_data = 0

datadir = os.getcwd()

nbr_repetitions = 10

choice_bool = 1
c1 = 1

row = 30
col = 30

N = row * col

beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

id = 'XYSIR'
columns = ['X', 'Y', 'S', 'I', 'R']
normalize_entropy = True

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})
sns.set(rc={"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})

## Fix simulation (sim = 2 for choice_bool = 1, c1 = 1)
sim = 5

# Run over beta and mu
for beta, mu in zip(beta_vals, mu_vals):
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'

    ## Extract time of minimum entropy (mostly vailid for high R0)
    idx_sims_not_start = np.load(folder_simulation + f'idx_sim_not_start.npy')

    min_H0_sims = np.load(folder_entropy + f'min_H0_sims-mu{mu}-beta{beta}.npy')
    min_H1_sims = np.load(folder_entropy + f'min_H1_sims-mu{mu}-beta{beta}.npy')

    column_index = np.where(min_H0_sims[0] == sim)[0]

    if len(column_index) > 0:
        min_H0_sim = min_H0_sims[:, column_index[0]]
        min_H1_sim = min_H1_sims[:, column_index[0]]
        print("Extracted H0:", min_H0_sim)
        print("Extracted H1:", min_H1_sim)
        t0 = min_H0_sim[1]
        t1 = min_H1_sim[1]
        print('t0 : ', t0)
        print('t1 : ', t1)

    else:
        print("No column found for the condition.")
    T = np.load(folder_simulation + 'T.npy')
    if beta == 0.115 or beta == 0.12:
        T = 1000
    T_sim = np.linspace(0, T - 1, T)

    if acquire_data == 1:
        ## Import dataset with dictionary of normalized data by hand
        dict_data = pickle.load(
            open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
        df_dict_data = data_2_pandas(dict_data)

        ## Compute topological features for the 2 times separated
        kk = 0
        for t_star in [t0, t1]:
            # Dataframe at fixed time value
            S_t = df_dict_data.loc[df_dict_data['Time'] == t_star]   ##### ATTENTION HERE : t0 or t1??
            # Select only data indexed by "column"
            S_data_t = S_t[columns]
            # Euclidean distance between pairs of points (Each point corresponds to a row. The dimension of the space is given by the number of columns.)
            pers_homology = ripser.ripser(S_data_t)
            dgms = pers_homology['dgms']

            # Persistence diagrams for H0
            dgms0 = dgms[0]
            print('type dgms', type(dgms0))
            print('len dgms', len(dgms0))
            # Sort features in decreasing order
            brith0_upDown = dgms0[:, 0]
            birth0 = brith0_upDown[::-1]
            end0_upDown = dgms0[:, 1]
            end0 = end0_upDown[::-1]

            # Persistence diagrams for H1 (not sorted)
            dgms1 = dgms[1]
            birth1_upDown = dgms1[:, 0]
            end1_upDown = dgms1[:, 1]
            length_1 = end1_upDown - birth1_upDown
            # Get the indices that would sort the difference vector in decreasing order
            sorted_indices = np.argsort(length_1)[::-1]

            # Sort both vectors based on the sorted indices
            birth1 = birth1_upDown[sorted_indices]
            end1 = end1_upDown[sorted_indices]

            y0 = np.linspace(len(dgms0), 0, len(dgms0) + 1)
            y1 = np.linspace(len(dgms1), 0, len(dgms1) + 1)

            # 1. Sort barcodes for decreasing length
            # H0
            dgms0_sorted = np.vstack((np.array(birth0), np.array(end0))).transpose()
            # H1
            dgms1_sorted = np.vstack((np.array(birth1), np.array(end1))).transpose()
            dgms_sorted = [dgms0_sorted, dgms1_sorted]

            # 2. Compute PE (H_L)
            PE_sorted = persistent_entropy(dgms_sorted, normalize=False)
            PE_H0_sorted = PE_sorted[0]
            PE_H1_sorted = PE_sorted[1]

            if kk == 0:
                #########################################
                # Necessary to extract the number of topological features to consider!
                # 3a. Cycle H0
                n0 = len(dgms0_sorted)
                print('TF: H0')
                top_features0 = topological_features(dgms0_sorted, PE_H0_sorted, birth0)
                print(top_features0)
                #########################################
                nbr_cc = len(top_features0)
                persistence_time = np.zeros((nbr_cc, T))

                for t in T_sim:
                    # Dataframe at fixed time value
                    S_t = df_dict_data.loc[df_dict_data['Time'] == t]
                    # Select only data indexed by "column"
                    S_data_t = S_t[columns]
                    # print('t:', t, 'S_t:', S_data_t)
                    # Euclidean distance between pairs of points (Each point corresponds to a row. The dimension of the space is given by the number of columns.)
                    pers_homology = ripser.ripser(S_data_t)
                    dgms = pers_homology['dgms']

                    # Persistence diagrams for H0
                    dgms0 = dgms[0]
                    # Sort features in decreasing order
                    brith0_upDown = dgms0[:, 0]
                    birth0 = brith0_upDown[::-1]
                    end0_upDown = dgms0[:, 1]
                    end0 = end0_upDown[::-1]

                    birth0_3cc = birth0[1:nbr_cc + 1]
                    end0_3cc = end0[1:nbr_cc + 1]
                    for cc in range(nbr_cc):
                        t = int(t)
                        persistence_time[cc, t] = end0_3cc[cc] - birth0_3cc[cc]

                np.save(folder_entropy + f'nbr_cc0-mu{mu}-beta{beta}-sim{sim}', nbr_cc)
                np.save(folder_entropy + f'pers_time0-mu{mu}-beta{beta}-sim{sim}', persistence_time)

            else:
                #########################################
                # Necessary to extract the number of topological features to consider!
                # 3b. Cycle H1
                n1 = len(dgms1_sorted)
                print('TF: H1')
                top_features1 = topological_features(dgms1_sorted, PE_H1_sorted, birth1)
                print(top_features1)
                #########################################
                nbr_cc = len(top_features1)
                #persistence_time = np.zeros((nbr_cc, T))

                for t in T_sim:
                    # Dataframe at fixed time value
                    S_t = df_dict_data.loc[df_dict_data['Time'] == t]

                    # Select only data indexed by "column"
                    S_data_t = S_t[columns]
                    # print('t:', t, 'S_t:', S_data_t)
                    # Euclidean distance between pairs of points (Each point corresponds to a row. The dimension of the space is given by the number of columns.)
                    pers_homology = ripser.ripser(S_data_t)
                    dgms = pers_homology['dgms']

                    dgms1 = dgms[1]
                    birth1_upDown = dgms1[:, 0]
                    end1_upDown = dgms1[:, 1]
                    length_1 = end1_upDown - birth1_upDown
                    # Get the indices that would sort the difference vector in decreasing order
                    sorted_indices = np.argsort(length_1)[::-1]

                    # Sort both vectors based on the sorted indices (by length)
                    birth1 = birth1_upDown[sorted_indices]
                    end1 = end1_upDown[sorted_indices]

                    # Select only births and deaths of longest cc
                    birth1_3cc = birth1[1:nbr_cc + 1]
                    end1_3cc = end1[1:nbr_cc + 1]
                    for cc in range(nbr_cc):
                        t = int(t)
                        persistence_time[cc, t] = end1_3cc[cc] - birth1_3cc[cc]
                print('hello2')
                np.save(folder_entropy + f'nbr_cc1-mu{mu}-beta{beta}-sim{sim}', nbr_cc)
                np.save(folder_entropy + f'pers_time1-mu{mu}-beta{beta}-sim{sim}', persistence_time)

            kk = kk + 1
    else:
        ## Load data H0
        nbr_cc0 = np.load(folder_entropy + f'nbr_cc0-mu{mu}-beta{beta}-sim{sim}.npy')
        persistence_time0 = np.load(folder_entropy + f'pers_time0-mu{mu}-beta{beta}-sim{sim}.npy')
        ## Load data H1
        nbr_cc1 = np.load(folder_entropy + f'nbr_cc1-mu{mu}-beta{beta}-sim{sim}.npy')
        persistence_time1 = np.load(folder_entropy + f'pers_time1-mu{mu}-beta{beta}-sim{sim}.npy')
        #f, ax = plt.subplots(figsize=(10, 8))
        #for cc in range(nbr_cc1):
        #    plt.plot(T_sim, persistence_time1[cc])
        #plt.show()

        # Plot
        plot_cc1_vs_time(T_sim, persistence_time0, persistence_time1, nbr_cc0, nbr_cc1)


