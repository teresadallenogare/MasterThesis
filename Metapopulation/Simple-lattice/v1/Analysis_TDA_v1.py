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

datadir = os.getcwd()

sim = 0

# normalization = 0 -> no normalized data
#               = 1 -> standard scaler normalization
#               = 2 -> normalization by hand
normalization = 1
id = 'XYSIR'
nrm_entropy = False

########################################################################################################################
# Fix configuration (dim, population, strength loops) and study PE as a function of beta and mu
########################################################################################################################
# Dimension
row = 3
col = 3

# Population method
choice_bool = 0

# Strength self loops
c1 = 0

# Infection and recovery rate
beta_vals_30_50 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

folder_entropy_normScaler = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Entropy/Normalized-scaler/{id}/'

for beta, mu in zip(beta_vals_30_50, mu_vals_30_50):
    dgms = pickle.load(open(folder_entropy_normScaler + f'dgms-nrm{nrm_entropy}-beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    ph = pickle.load(open(folder_entropy_normScaler + f'ph-nrm{nrm_entropy}-beta{beta}-mu{mu}-sim{sim}.pickle', 'rb'))
    print('hello')





