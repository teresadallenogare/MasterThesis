"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 15 December 2023

--------------------------------------------------------------------

Analysis of SIR data from simulations

"""

from functions_SIR_metapop_v1 import *
from functions_output_v1 import write_simulation_file
from functions_visualization_v1 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

datadir = os.getcwd()
plt.figure(figsize=(8, 8))

SIR_time = 0
fixedR0 = 0
fixed_mu = 0
duration_analysis = 0
heatmap = 0
final_size_analysis = 0
network_infected = 0
phase_space = 1

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

### SIR time
if SIR_time == 1:
    row = 3
    col = 3
    N = row * col
    choice_bool = 0
    c1 = 0

    idx_nodes = np.linspace(0, N, N - 1)

    # Infection and recovery rate
    beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
    mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    for beta, mu in zip(beta_vals, mu_vals):
        folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
        folder_simulation = datadir + f'/Data_simpleLattice_v1/Repeated_trials/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

        T = np.load(folder_simulation + f'T.npy')
        print('row:', row, 'col:', col, 'choice_bool:', choice_bool, 'c1:', c1, 'beta:', beta, 'mu:', mu, 'T:', T)
        T_sim = np.linspace(0, T - 1, T)
        nbr_repetitions = np.load(folder_simulation + f'nbr_repetitions.npy')
        idx_repetitions = np.linspace(0, nbr_repetitions - 1, nbr_repetitions)
        idx_sim_not_start = np.load(folder_simulation + 'idx_sim_not_start.npy')
        idx_sim_start = list((set(idx_repetitions) - set(idx_sim_not_start)))
        avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')
        if choice_bool == 1:
            avg_pop_Nfix = np.load(folder_topology + 'avg_pop_Nfix.npy')
            avg_pop_Others = np.load(folder_topology + 'avg_pop_Others.npy')
        else:
            avg_pop_Nfix = 0
            avg_pop_Others = 0
        plot_SIR_timeseries(row, col, choice_bool, c1, beta, mu, idx_sim_start, idx_nodes, T_sim, avg_popPerNode,
                        avg_pop_Nfix, avg_pop_Others)



### Fixed R0
# The dimensions of the lattice (consider idx_node = 0 and whole network) and fixed the R0, show the different dynamics of
# the number of infected as a function of beta, mu
if fixedR0 == 1:

    row = 10
    col = 10
    N = row * col

    choice_bool = 0
    c1 = 0

    sim = 0

    idx_node = 0

    folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'

    avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')

    beta_vals_R0 = [0.4, 0.8, 1.2]
    mu_vals_R0 = [0.1, 0.2, 0.3]

    i = 0
    for beta, mu in zip(beta_vals_R0, mu_vals_R0):
        T = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/T.npy')
        print('row:', row, 'col:', col, 'choice_bool:', choice_bool, 'c1:', c1, 'beta:', beta, 'mu:', mu, 'T:', T)
        T_sim = np.linspace(0, T - 1, T)

        node_NS_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NI_time.npy')
        node_NR_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NR_time.npy')

        NI_time = node_NI_time.sum(axis=1)

        density_node_NI_time = node_NI_time / avg_popPerNode
        density_NI_time = NI_time / (N * avg_popPerNode)

        plt.plot(T_sim[:120], density_node_NI_time[:120, idx_node], color=grad_red[i],
                 label=f'beta = {beta}, mu = {mu}')
        # plt.plot(T_sim[:120], density_NI_time[:120], color = grad_red[i], label = f'beta = {beta}, mu = {mu}')

        i = i + 1
    plt.xlabel('t')
    plt.ylabel(r'$\rho_{I}(t)$', rotation=0)
    plt.legend()
    plt.show()

### Fixed the dimension of the lattice (consider idx_node = 0) and fixed the mu parameter, show the different dynamics of
# the number of infected as a function of R0
if fixed_mu == 1:
    row = 30
    col = 30
    N = row * col

    choice_bool = 0
    c1 = 0

    sim = 0

    idx_node = 0

    folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'

    avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')

    beta_vals_mu = [0.2, 0.3, 0.4]
    mu_vals_mu = [0.1, 0.1, 0.1]

    i = 0
    for beta, mu in zip(beta_vals_mu, mu_vals_mu):
        T = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/T.npy')
        print('row:', row, 'col:', col, 'choice_bool:', choice_bool, 'c1:', c1, 'beta:', beta, 'mu:', mu, 'T:', T)
        T_sim = np.linspace(0, T - 1, T)

        node_NS_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NS_time.npy')
        node_NI_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NI_time.npy')
        node_NR_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NR_time.npy')

        NI_time = node_NI_time.sum(axis=1)

        density_node_NI_time = node_NI_time / avg_popPerNode
        density_NI_time = NI_time / (N * avg_popPerNode)

        plt.plot(T_sim, density_node_NI_time[:, idx_node], color=grad_red[i], label=f'R0 = {np.round(beta / mu, 2)}')
        # plt.plot(T_sim, density_NI_time, color=grad_red[i], label=f'R0 = {np.round(beta/mu, 2)}')

        i = i + 1
    plt.xlabel('t')
    plt.ylabel(r'$\rho_{I}(t)$', rotation=0)
    plt.legend()
    plt.show()

######################################################################################################################

### Epidemic duration
if duration_analysis == 1:
    N_row = [3, 5, 10, 30]
    N_col = [3, 5, 10, 30]

    # Infection and recovery rate
    beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
    mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    choice_bool = 0
    c1 = 0

    sim = 0

    # starting point of the epidemics : it occurs as soon as the infectious agent is inserted in the population, that
    # is at t = 0 for us
    start = 0
    x = np.linspace(1, 12, 100)



    for row, col in zip(N_row, N_col):

        folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'

        avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')

        R0_vals = []
        epidemic_duration = []
        for beta, mu in zip(beta_vals, mu_vals):
            R0_vals.append(beta / mu)
            node_NI_time = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/sim_{sim}_node_NI_time.npy')
            NI_time = node_NI_time.sum(axis=1)
            # Set end after the pick has occurred (at network level)
            t_peak = np.argmax(NI_time)
            # End of the epidemics is set to be 3 timesteps after the peak value. (How to justify? When it starts to decrease)
            end = t_peak + 3
            duration = end - start

            epidemic_duration.append(duration)

        plt.plot(R0_vals, epidemic_duration, marker='o', label=f'Dim : {row}x{col}')
        # plt.plot(x, funct(x), 'k--')
        # plt.plot(x, funct2(x), 'k-.')

    plt.xlabel('R0')
    plt.ylabel('Epidemic duration (timesteps)')
    plt.legend()
    plt.show()

######################################################################################################################

### Heatmap

if heatmap == 1:
    row = 10
    col = 10

    choice_bool = 0
    c1 = 0

    beta = 0.9
    mu = 0.1

    sim = 0

    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'

    T = np.load(folder_simulation + f'mu-{mu}/beta-{beta}/T.npy')
    print('row:', row, 'col:', col, 'choice_bool:', choice_bool, 'c1:', c1, 'beta:', beta, 'mu:', mu, 'T:', T)
    T_sim = np.linspace(0, T - 1, T)

    heatmap_time(row, col, choice_bool, c1, beta, mu, sim)

######################################################################################################################

### Final size of the epidemics : with final size I intend the number of recovered people at the end of the epidemics

if final_size_analysis == 1:
    # In the case of 3, 5, 10 I have both mu = 0.1, 0.2 and 0.3
    N_row = [10, 30, 50]  # , 5, 10, 30, 50]
    N_col = [10, 30, 50]  # , 5, 10, 30, 50]

    # Infection and recovery rate
    beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
    mu_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    choice_bool_lst = [0, 1]
    c1_lst = [0, 1]

    sim = 0

    # Threshold for outbreak in node : if the node has a percentage  of infected greater than the threshold, then
    threshold_perc_R_outbreak = 30
    # Threshold major outbreak : if the % of nodes considered as infected is greater than threshold_major_outbreak,
    # than the outbreak occurs. Otherwise no.
    threshold_major_outbreak = 80

    for row, col in zip(N_row, N_col):
        N = row * col
        for choice_bool in choice_bool_lst:
            for c1 in c1_lst:
                folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
                folder_analysis = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Analysis/'

                avg_population = np.load(folder_topology + 'avg_popPerNode.npy')
                total_population = avg_population * N
                final_size_beta = []
                R0_vals = []
                final_size_node = np.zeros(shape=(len(mu_vals), N))
                final_population_node = np.zeros(shape=(len(mu_vals), N))
                mtrx_nodes_infected = np.zeros(shape=(len(mu_vals), N)) # is 1 if node was infected, is 0 otherwise
                nbr_nodes_infected = []
                count_R0 = 0
                for beta, mu in zip(beta_vals, mu_vals):
                    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'

                    node_NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')
                    node_population_time = np.load(folder_simulation + f'sim_{sim}_node_population_time.npy')
                    # Number of recovered per node (one node per column and one R0 per row)
                    for i in range(N):
                        final_size_node[count_R0, i] = node_NR_time[-1, i]
                        final_population_node[count_R0, i] = node_population_time[-1, i]
                    np.save(folder_analysis + f'final_population_node_sim{sim}_mu{mu}_beta{beta}', final_population_node)
                    final_size = node_NR_time[-1].sum()
                    final_size_beta.append(final_size)
                    R0 = beta / mu
                    R0_vals.append(R0)
                    perc_final_size_node = final_size_node / final_population_node * 100
                    # Establish if the single node is infected
                    # 1 if %R in node and at fixed R0 is greater than threshold, else 0
                    # perc_final_size_node tells me in which nodes I have a local outbreak
                    for i in range(N):
                        if perc_final_size_node[count_R0, i] > threshold_perc_R_outbreak:
                            mtrx_nodes_infected[count_R0, i] = 1
                        else:
                            mtrx_nodes_infected[count_R0, i] = 0
                    sum_nodes_infected = mtrx_nodes_infected[count_R0, :].sum()
                    nbr_nodes_infected.append(sum_nodes_infected)

                    count_R0 = count_R0 + 1

                np.save(folder_analysis + f'mtrx_nodes_infected_sim{sim}', mtrx_nodes_infected)
                nbr_nodes_infected = np.array(nbr_nodes_infected)
                perc_nodes_infected = nbr_nodes_infected / N * 100
                outbreak = []
                # Establish if a major outbreak has occurred in the network
                for j in range(len(mu_vals)):
                    if perc_nodes_infected[j] > threshold_major_outbreak:
                        out = 1
                    else:
                        out = 0
                    outbreak.append(out)

                final_size_beta = np.array(final_size_beta)
                R0_vals = np.array(R0_vals)
                if choice_bool == 0 and c1 == 0:
                    plt.plot(R0_vals, final_size_beta, marker='o', linestyle='-')
                    plt.axhline(y=total_population, linestyle='--', color='k')
                    plt.xlabel(r'$R_0$')
                    plt.ylabel('Final size')
                    plt.title(f'dim: {row}x{col}')
                    plt.show()
                print('------------------------------------------------------------------------')
                print('dim: ', row, 'ch_bool: ', choice_bool, 'c1: ', c1)
                print('------------------------------------------------------------------------')
                print('outbreak: ', outbreak)
                print('perc_nodes_I', perc_nodes_infected)

######################################################################################################################

### Network infected

if network_infected == 1:
    # In the case of 3, 5, 10 I have both mu = 0.1, 0.2 and 0.3
    N_row = [10, 30, 50]
    N_col = [10, 30, 50]

    # Infection and recovery rate
    beta = 0.2
    mu = 0.1
    beta_vals = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
    choice_bool_lst = [0, 1]
    c1_lst = [0, 1]

    sim = 0
    for row, col in zip(N_row, N_col):
        N = row * col
        for choice_bool in choice_bool_lst:
            for c1 in c1_lst:
                folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
                folder_analysis = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Analysis/'

                avg_population = np.load(folder_topology + 'avg_popPerNode.npy')
                total_population = avg_population * N
                G = pickle.load(open(folder_topology + 'G.pickle', 'rb'))
                TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
                weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if
                                 TransitionMatrix[i, j] != 0]
                dict_nodes = pickle.load(open(folder_topology + 'dict_nodes.pickle', 'rb'))
                final_population_node = np.load(folder_analysis + f'final_population_node_sim{sim}_mu{mu}_beta{beta}.npy')
                mtrx_nodes_infected = np.load(folder_analysis + f'mtrx_nodes_infected_sim{sim}.npy')

                idx_beta = beta_vals.index(beta)

                plot_network_final_size(G, row, final_population_node[idx_beta, :], dict_nodes, weightNonZero,
                                        mtrx_nodes_infected[idx_beta, :])
                plt.title(f'choice_bool: {choice_bool}, c1: {c1}')
                plt.show()

######################################################################################################################
### Phase space
if phase_space == 1:
    row = 30
    col = 30
    N = row * col
    choice_bool = 0
    c1 = 0

    sim = 0

    beta_vals = [0.2, 0.3, 0.4]
    mu_vals = [0.1, 0.1, 0.1]

    bool_network = 1

    folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    avg_popPerNode = np.load(folder_topology + 'avg_popPerNode.npy')
    i = 0
    for beta, mu in zip(beta_vals, mu_vals):

        folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
        T = np.load(folder_simulation + f'T.npy')
        T_sim = np.linspace(0, T - 1, T)

        NS_time = np.load(folder_simulation + f'sim_{sim}_node_NS_time.npy')
        NI_time = np.load(folder_simulation + f'sim_{sim}_node_NI_time.npy')
        NR_time = np.load(folder_simulation + f'sim_{sim}_node_NR_time.npy')
        lineStyle = ['-', '--', ':']
        phase_space_flux(row, col, bool_network, NS_time, NI_time, NR_time, avg_popPerNode, T_sim, lineStyle[i])
        i = i+1
    plt.xlabel(r'$\rho_S$')
    plt.ylabel(r'$\rho_I$')
    plt.show()






