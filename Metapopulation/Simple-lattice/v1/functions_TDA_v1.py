"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 23 November 2023

--------------------------------------------------------------------
Functions useful for the TDA pipeline

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import ripser
from persim.persistent_entropy import *

def data_2_pandas(data_dict):
    """ Convert the dictionary of data in a DataFrame

    :param data_dict: [dictionary] input dictionary containing data
    :return: data_dict_df: [DataFrame] DataFrame of the input dictionary
    """
    frames = [] # List to hold individual dataframes

    for time, array in data_dict.items():
        N = len(array) # Get the number of rows in the numpy array
        node_id = np.arange(0, N)  # Generate Node ID sequence from 0 to N-1
        time_column = np.full((N,), time)  # Create a column filled with the current time value
        # Create a DataFrame
        df = pd.DataFrame({
            'Time': time_column,
            'Node ID': node_id,
            'X': array[:, 0],
            'Y': array[:, 1],
            'S': array[:, 2],
            'I': array[:, 3],
            'R': array[:, 4]
        })

        frames.append(df)

    final_df = pd.concat(frames, ignore_index=True)

    return final_df


def scaler_df_data_dict(df_data):
    """ Scale data using standard scaler

    :param df_data: [DataFrame] DataFrame of data
    :return: df_data_scaled: [DataFrame] DataFrame of data scaled according to standard scaler
    """
    scaler = StandardScaler()
    df_data_scaled = pd.DataFrame()

    df_data_scaled = pd.DataFrame(scaler.fit_transform(df_data[['X', 'Y', 'S', 'I', 'R']]), columns = ['X', 'Y', 'S', 'I', 'R'])
    # Time and Node ID are preserved attributes, not subject to rescaling
    df_data_scaled['Time'] = df_data['Time']
    df_data_scaled['Node ID'] = df_data['Node ID']

    return df_data_scaled

def entropy_calculation(df_data, columns, normalize_entropy):

    time_interval = range(df_data['Time']. min(), df_data['Time'].max())

    entropy_H0 = []
    entropy_H1 = []

    for t in time_interval:
        mask = df_data['Time'] == t
        # Extract data to analyse from DataFrame
        extracted_df_data = df_data.loc[mask, columns]. values

        # Calculate persistent diagrams
        pers_homology = ripser.ripser(extracted_df_data)
        dgms = pers_homology['dgms']

        # Calculate persistent entropy
        entropy = persistent_entropy(dgms, normalize = normalize_entropy)
        entropy_H0.append(entropy[0])
        entropy_H1.append(entropy[1])

    return pers_homology, dgms, entropy_H0, entropy_H1