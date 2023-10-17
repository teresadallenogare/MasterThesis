"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 16 October 2023

--------------------------------------------------------------------


"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn
import kmapper as km
import networkx as nx
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Consider the case of an epidemic outbreak on a 3x3 network
N_row = 3
N_col = 3
N = N_row * N_col
choice_bool = 0
datadir = os.getcwd()
c1 = 0  # for now
beta = 0.4
mu = 0.2

# --------------------------------------------- Load data ---------------------------------------------
folder_topology = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Topology/'
folder_simulation = datadir + f'/Data-simpleLattice/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{int(np.floor(c1))}/Simulations/beta-{beta}mu-{mu}/'

pos_nodes = np.load(folder_topology + 'pos_nodes.npy')

avg_popPerNode = np.load(folder_topology + '/avg_popPerNode.npy')
populationTot = N * avg_popPerNode # from this then I use multinomial
if choice_bool == 1:
    Nfix = np.load(folder_topology + '/Nfix.npy')
    percentage_FixNodes = np.load(folder_topology + '/percentage_FixNodes.npy')
else:
    Nfix = 0
    percentage_FixNodes = 0

T = np.load(folder_simulation + 'T.npy')
T_sim = np.linspace(0, T, T+1)

nbr_repetitions = np.load(folder_simulation + '/nbr_repetitions.npy')

# 3D matrix that stores repetitions along axis = 2
node_NI_time_repeat = np.zeros(shape = (T+1, N, nbr_repetitions))

# To see repetition k : node_NI_time_repeat[:,:,k]
for sim in range(nbr_repetitions):
    new_I_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
    node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
    node_NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
    node_NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
    density_node_NI_time = node_NI_time / populationTot # normalisation (density of infected as a global property?)
    node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')
    node_NI_time_repeat[:, :, sim] = node_NI_time
node_NI_time_repeat = np.array(node_NI_time_repeat)

# Mean value over repetitions
mean_NI_time = np.mean(node_NI_time_repeat, axis = 2)
stdDev_NI_time = np.std(node_NI_time_repeat, axis = 2, ddof = 1)

# For plot : convert matrices in dataframe
df_mean_NI_time = pd.DataFrame(mean_NI_time, columns=[f'node {i}' for i in range(N)])
df_stdDev_NI_time = pd.DataFrame(stdDev_NI_time, columns=[f'node {i}' for i in range(N)], index=[f'time {i}' for i in range(T+1)])

# convert plotly hex colors to rgba to enable transparency adjustments
def hex_rgba(hex, transparency):
    col_hex = hex.lstrip('#')
    col_rgb = list(int(col_hex[i:i+2], 16) for i in (0, 2, 4))
    col_rgb.extend([transparency])
    areacol = tuple(col_rgb)
    return areacol

# define colors as a list
colors = px.colors.qualitative.Plotly
rgba = [hex_rgba(c, transparency=0.2) for c in colors]
colCycle = ['rgba'+str(elem) for elem in rgba]
# Make sure the colors run in cycles if there are more lines than colors
def next_col(cols):
    while True:
        for col in cols:
            yield col
line_color=next_col(cols=colCycle)


# plotly  figure
fig = go.Figure()

# Add line and 1 std_deviation above and 1 below
for i, col in enumerate(df_mean_NI_time):
    new_col = next(line_color)
    x = list(df_mean_NI_time.index.values + 1)
    # Select column
    y1 = df_mean_NI_time[col]
    y1_upper = [(y + stdDev_NI_time[col]) for y in df_mean_NI_time[col]]
    y1_lower = [(y - stdDev_NI_time[col]) for y in df_mean_NI_time[col]]
    y1_lower = y1_lower[::-1] # subtracts -1 (?)

    # trace adds multiple plots in 1 figure

    # std deviation
    fig.add_traces(go.Scatter(x=x+x[::-1],
                    y=y1_upper+y1_lower,
                    fill='tozerox',
                    fillcolor=new_col,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name=col))
    # line trace
    fig.add_traces(go.Scatter(x=x,
                              y=y1,
                              line=dict(color=new_col, width=2.5),
                              mode='lines',
                              name=col)
                   )
# set x-axis
fig.update_layout(xaxis=dict(range=[1, len(df_mean_NI_time)]))


#fig = px.line(df_mean_NI_time.iloc[:, 1])
fig.show()

print('Hello')








