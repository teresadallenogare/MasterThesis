"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 23 November 2023

--------------------------------------------------------------------

Scheleton of persistent entropy calculation (done in cluster so versions in cluster should be updated)

"""
from functions_TDA_v1 import *
from functions_visualization_v1 import *
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


noise_from_feature = 1
varying_time = 0
persistence_diagrams = 1

datadir = os.getcwd()

if persistence_diagrams == 1:
    nbr_repetitions = 1

    choice_bool = 0
    c1 = 0

    row = 30
    col = 30

    N = row * col

    beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
    mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    id = 'XYSIR'
    columns = ['X', 'Y', 'S', 'I', 'R']

    folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'
    folder_ripser = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Ripser_analysis/'
    folder_entropy = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'

    # Consider data with normalization by hand
    for beta, mu in zip(beta_vals, mu_vals):
        for sim in range(nbr_repetitions):
            folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
            T = np.load(folder_simulation + 'T.npy')
            if beta == 0.115 or beta == 0.12:
                T = 1000
            T_sim = np.linspace(0, T - 1, T)

            dict_data = pickle.load(
                open(folder_dict_normHand + f'dict_data_beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
            df_dict_data = data_2_pandas(dict_data)

            ##### Fixed value of time
            if varying_time == 0:
                t = 25
                # Dataframe at fixed time value
                S_t = df_dict_data.loc[df_dict_data['Time'] == t]
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


                y0 = np.linspace(len(dgms0), 0, len(dgms0)+1)
                y1 = np.linspace(len(dgms1), 0, len(dgms1)+1)

                #plot_barcodes(birth0, end0, birth1, end1, y0, y1)

                if noise_from_feature == 1:
                    # 1. Sort barcodes for decreasing length
                    # H0
                    dgms0_sorted = np.vstack((np.array(birth0), np.array(end0))).transpose()
                    # H1
                    dgms1_sorted = np.vstack((np.array(birth1), np.array(end1))).transpose()
                    dgms_sorted = [dgms0_sorted, dgms1_sorted]

                    # 2. Compute PE (H_L)
                    PE_sorted = persistent_entropy(dgms_sorted, normalize = False)
                    PE_H0_sorted = PE_sorted[0]
                    PE_H1_sorted = PE_sorted[1]

                    # 3a. Cycle H0
                    n0 = len(dgms0_sorted)
                    n1 = len(dgms1_sorted)
                    print('TF: H0')
                    top_features0 = topological_features(dgms0_sorted, PE_H0_sorted, birth0)

                    # 3b. Cycle H1
                    print('TF: H1')
                    top_features1 = topological_features(dgms1_sorted, PE_H1_sorted, birth1)


            else:

                # Extract the first 10 cc and see in time
                nbr_cc = 10
                persistence_time = np.zeros((nbr_cc, T))

                for t in T_sim:
                    # Dataframe at fixed time value
                    S_t = df_dict_data.loc[df_dict_data['Time'] == t]

                    # Select only data indexed by "column"
                    S_data_t = S_t[columns]
                    #print('t:', t, 'S_t:', S_data_t)
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

                    birth0_3cc = birth0[1:nbr_cc+1]
                    end0_3cc = end0[1:nbr_cc+1]
                    for cc in range(nbr_cc):
                        t = int(t)
                        persistence_time[cc, t] = end0_3cc[cc] - birth0_3cc[cc]

                np.save(folder_entropy + f'pers_time-mu{mu}-beta{beta}-sim{sim}', persistence_time)
