"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 23 November 2023

--------------------------------------------------------------------

Topological data analysis pipeline.

"""
from functions_TDA_v1 import *
import os
import pickle
import numpy as np


# ------------------------------------------------ Parameters  -------------------------------------------------

N_row = [50]
N_col = [50]

choice_bool_lst = [0]
c1_lst = [0]

# Infection and recovery rate
beta_vals_3_5_10 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2, 0.23, 0.24, 0.3, 0.4, 0.6, 0.8, 0.345, 0.36, 0.45, 0.6, 0.9, 1.2]
mu_vals_3_5_10 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

beta_vals_30_50 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

sim = 0

# normalization = 0 -> no normalized data
#               = 1 -> standard scaler normalization
#               = 2 -> normalization by hand
normalization_data_vals = [0, 1, 2]

columns = ['X', 'Y', 'S', 'I', 'R']
id = 'XYSIR'
datadir = os.getcwd()

for row, col in zip(N_row, N_col):
    N = row * col
    if row == 3 or row == 5 or row == 10:
        beta_vals = beta_vals_3_5_10
        mu_vals = mu_vals_3_5_10

    else:
        beta_vals = beta_vals_30_50
        mu_vals = mu_vals_30_50

    for choice_bool in choice_bool_lst:
        for c1 in c1_lst:
            folder_dict_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/No-normalized/'
            folder_dict_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Dictionaries/Normalized-hand/'
            folder_entropy_noNorm = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/No-normalized/{id}/'
            folder_entropy_normHand = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-hand/{id}/'
            folder_entropy_normScaler = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-scaler/{id}/'

            for beta, mu in zip(beta_vals, mu_vals):
                print('row: ', row, 'col: ', col, 'choice_bool: ', choice_bool, 'c1: ', c1)
                print('beta: ', beta, 'mu: ', mu)
                for normalization in normalization_data_vals:
                    #### Load data and transform in DataFrame
                    if normalization != 2 :
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

                    #### PE calculation
                    for normalize_entropy in [ False, True]:
                        print('normalize_entropy: ', normalize_entropy)
                        ph, dgms, entropy_H0, entropy_H1 = entropy_calculation(df_dict_data, columns, normalize_entropy)
                        if normalization == 0:
                            pickle.dump(ph, open(folder_entropy_noNorm + f'ph-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', 'wb'))
                            pickle.dump(dgms, open(folder_entropy_noNorm + f'dgms-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', 'wb'))
                            np.save(folder_entropy_noNorm + f'entropy_H0-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', entropy_H0)
                            np.save(folder_entropy_noNorm + f'entropy_H1-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', entropy_H1)
                        elif normalization == 1:
                            pickle.dump(ph, open(folder_entropy_normScaler + f'ph-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', 'wb'))
                            pickle.dump(dgms, open(folder_entropy_normScaler + f'dgms-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', 'wb'))
                            np.save(folder_entropy_normScaler + f'entropy_H0-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', entropy_H0)
                            np.save(folder_entropy_normScaler + f'entropy_H1-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', entropy_H1)
                        elif normalization == 2:
                            pickle.dump(ph, open(folder_entropy_normHand + f'ph-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', 'wb'))
                            pickle.dump(dgms,open(folder_entropy_normHand + f'dgms-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', 'wb'))
                            np.save(folder_entropy_normHand + f'entropy_H0-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', entropy_H0)
                            np.save(folder_entropy_normHand + f'entropy_H1-nrm{normalize_entropy}-beta{beta}-mu{mu}-sim{sim}', entropy_H1)






