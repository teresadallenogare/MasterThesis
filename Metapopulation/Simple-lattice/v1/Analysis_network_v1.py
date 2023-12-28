"""

--------------------------------------------------------------------

Author  :   Teresa Dalle Nogare
Version :   22 November 2023

--------------------------------------------------------------------

Analysis of network topology computing the key characteristic properties of graphs.

"""
from functions_network_v1 import path_analysis
from functions_output_v1 import write_network_file
from functions_visualization_v1 import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import scipy
from scipy.optimize import curve_fit
from scipy.stats import poisson, kstest, probplot
import scipy.linalg as linalg
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import statsmodels.api as sm
from scipy.stats import linregress

datadir = os.getcwd()
plt.figure(figsize=(8, 6))
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})

# ------------------------------------------------ Parameters  -------------------------------------------------
N_row = [30]
N_col = [30]

choice_bool_lst = [0]
c1_lst = [0]

beta_vals_3_5_10 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2, 0.23, 0.24, 0.3, 0.4, 0.6, 0.8, 0.345, 0.36, 0.45, 0.6,
                    0.9, 1.2]
mu_vals_3_5_10 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

beta_vals_30_50 = [0.115, 0.12, 0.15, 0.2, 0.3, 0.4, 0.9, 1.2]
mu_vals_30_50 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

sim = 0

population_analysis = 0
degree_analysis = 1
distance_analysis = 0
clustering_analysis = 0
weight_analysis = 0
PF_convergence = 0
Rstar_def = 0
outbreak = 0

log_dependence = 1

write_file = 0

plt.figure(figsize=(8, 6))
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#ebebeb"})


# --------------------------------------------------------------------------------------------------------------

def Poisson_funct(k, lamb):
    # poisson probability mass function
    return poisson.pmf(k, lamb)


def nth_moment_v2(g, n):
    degree_np = np.array(list(dict(g.in_degree).values()))
    return (sum(degree_np ** n) / len(g))


######################################################################################################################
avg_distance_N = []
N_vals = []
for row, col in zip(N_row, N_col):
    N = row * col
    for choice_bool in choice_bool_lst:
        for c1 in c1_lst:
            folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'
            folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
            folder_analysis = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Analysis/'

            G = pickle.load(open(folder_topology + 'G.pickle', 'rb'))
            dict_nodes = pickle.load(open(folder_topology + 'dict_nodes.pickle', 'rb'))
            TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
            AdjacencyMatrix = np.load(folder_topology + 'AdjacencyMatrix.npy')
            D = np.load(folder_topology + 'DistanceMatrix.npy')
            avg_population = np.load(folder_topology + 'avg_popPerNode.npy')
            total_population = N * avg_population
            weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if TransitionMatrix[i, j] != 0]

            plt.show()
            ######################################################################################################################

            if population_analysis == 1:
                # Node population
                node_population_0 = nx.get_node_attributes(G, name='Npop')
                node_population_0 = np.array(list(node_population_0.values()))

                #plot_static_network(G, node_population_0, dict_nodes, weightNonZero, N_row, N_col, choice_bool, c1)
                if choice_bool == 0:
                    # Mean and average from the multinomial distribution
                    prob_sample1 = 1 / N
                    mean_population_multi1 = total_population * prob_sample1
                    stdDev_population_multi1 = np.sqrt(total_population * prob_sample1 * (1 - prob_sample1))

                    N_fix = 0
                    mean_population_multi2 = 0
                    stdDev_population_multi2 = 0
                    idx_Nfix = 0
                    # Separate data in the two distributions
                    data1 = node_population_0
                    data2 = 0
                elif choice_bool == 1:
                    # Number of fixed nodes
                    N_fix = np.load(folder_topology + 'Nfix.npy')
                    # Index of the fixed nodes
                    idx_Nfix = np.load(folder_topology + 'idxNfix.npy')
                    percentage_Nfix = np.load(folder_topology + 'percentage_FixNodes.npy')
                    # Population inside the fixed nodes (and other)
                    N_fix_population = total_population * percentage_Nfix / 100
                    N_other_population = total_population - N_fix_population

                    prob_sample1 = 1 / N_fix
                    prob_sample2 = 1 / (N - N_fix)

                    mean_population_multi1 = N_fix_population * prob_sample1
                    stdDev_population_multi1 = np.sqrt(N_fix_population * prob_sample1 * (1. - prob_sample1))

                    mean_population_multi2 = N_other_population * prob_sample2
                    stdDev_population_multi2 = np.sqrt(N_other_population * prob_sample2 * (1. - prob_sample2))

                    # Separate data in the two distributions
                    data1 = node_population_0[node_population_0 > 13000]
                    data2 = node_population_0[node_population_0 < 13000]

                #plot_node_population_0(N, N_fix, idx_Nfix, node_population_0, mean_population_multi1,
                #                       stdDev_population_multi1, mean_population_multi2, stdDev_population_multi2,
                #                       choice_bool)

                plot_population_distribution(data1, data2, mean_population_multi1, stdDev_population_multi1,
                                             mean_population_multi2, stdDev_population_multi2, choice_bool)
            ######################################################################################################################

            if degree_analysis == 1:

                # Node population
                node_population_0 = nx.get_node_attributes(G, name='Npop')
                node_population_0 = np.array(list(node_population_0.values()))
                ########################################################################################################
                density_population_0 = node_population_0/avg_population
                a = 0.2
                c = a / max(density_population_0)
                # matrix of probabilities
                P = np.zeros(shape=(N, N))
                for i in range(N):
                    for j in range(N):
                        if j > i:
                            # Probability to establish both the direct and forward edge in a pair of nodes
                            prob = max(c * density_population_0[i] / D[i, j], c * density_population_0[j] / D[i, j])

                            P[i, j] = prob
                            P[j, i] = prob
                P_unique = np.unique(P)
                P_sum = P_unique.sum()
                ########################################################################################################

                # [Degree properties]
                # Input degrees
                in_degrees = np.array([G.in_degree(n) for n in G.nodes()])
                # Total number of links
                L = in_degrees.sum()
                L_max = N * (N - 1) / 2
                # Average input degree
                avg_in_degree = L / N
                second_moment = nth_moment_v2(G, 2)
                # print('avg1:', avg_in_degree)
                # print('second moment: ', second_moment)
                # In-degree distribution
                # No normalized
                Pk_noNorm = np.unique(in_degrees, return_counts=True)
                k_vals = Pk_noNorm[0]
                N_k = Pk_noNorm[1]
                # Normalization : the Pk divided by the total number of nodes st sum(pk) = 1
                Pk_norm = N_k / N

                # Fit with Poisson distribution
                guess = avg_in_degree
                param, cov_matrix = curve_fit(Poisson_funct, k_vals, Pk_norm, p0=guess)
                print('param:', param)
                SE = np.sqrt(np.diag(cov_matrix))
                print('SE: ', SE)


                ### KS test : better -> before I used the normalized Pk. Now I am using the non-normalized ones.
                ks_statistic, ks_p_value = kstest(N_k, 'poisson', N=len(k_vals), args=(param,))
                plot_degree_distribution(row, col, choice_bool, c1, k_vals, Pk_norm, avg_in_degree, Poisson_funct,
                                         param)
                # Display the KS test results
                print(f'KS Statistic: {ks_statistic}')
                print(f'P-value: {ks_p_value}')

                # Interpret the results
                KS_threshold = 0.05
                if ks_p_value < KS_threshold:
                    print("Reject the null hypothesis: The sample does not follow a poisson distribution.")
                else:
                    print("Fail to reject the null hypothesis: The sample follows a poisson distribution.")

                # Matrix of connections: total number of edges between vertices of degree k and vertices of degree k′ N_kk'

                ContactMatrix = np.zeros(shape=(max(k_vals) + 1, max(k_vals) + 1))

                for i in range(N):
                    for j in range(N):  # self-loops??
                        if TransitionMatrix[i, j] != 0:
                            k = G.in_degree(i)
                            k_prime = G.in_degree(j)
                            # Cont the number of edges from k to k' and viceversa
                            ContactMatrix[k, k_prime] = ContactMatrix[k, k_prime] + 1
                # Delete if both row and col are zeros
                max_dim = max(k_vals) + 1
                idx_delete = []
                for i in range(max_dim):
                    sum_row = np.sum(ContactMatrix[i, :])
                    sum_col = np.sum(ContactMatrix[:, i])
                    if sum_row == 0 and sum_col == 0:
                        idx_delete.append(i)

                ContactMatrix = np.delete(ContactMatrix, idx_delete, 0)  # delete i-th row
                ContactMatrix = np.delete(ContactMatrix, idx_delete, 1)  # delete i-th col

                # Joint probability P(k, k')
                P_joint = ContactMatrix / (avg_in_degree * N)

                # Conditional probability P(k|k')
                P_cond = np.zeros(shape=(len(k_vals), len(k_vals)))
                for k in range(len(k_vals)):
                    P_cond[k, :] = ContactMatrix[k, :] / (k_vals[k] * N_k[k])

                # ANND k_barn_nn
                k_bar_nn = []
                for k in range(len(k_vals)):
                    sum_k = 0
                    for k_prime in range(len(k_vals)):
                        sum_k = sum_k + k_vals[k_prime] * P_cond[k, k_prime]
                    k_bar_nn.append(sum_k)

                k_bar_nn_non_corr = second_moment / avg_in_degree
                np.save(folder_analysis + 'k_vals', k_vals)
                np.save(folder_analysis + f'k_bar_nn', k_bar_nn)
                np.save(folder_analysis + f'k_bar_nn_non_corr', k_bar_nn_non_corr)

                # print(f'ch_bool: {choice_bool}, c1: {c1}, {row}x{col}, avg_k:', avg_in_degree, 'L_in: ', L, 'L_max: ',
                #      L_max, 'Perc. link: ', np.round(L / L_max * 100, 2), '%')

                # Attempt to find an epidemic threshold by diagonalization of the C_matrix (ref. Epidemic spreading in complex netwokrs)
                C_matrix = np.zeros(shape=(len(k_vals), len(k_vals)))
                for k in range(len(k_vals)):
                    for k_prime in range(len(k_vals)):
                        C_matrix[k, k_prime] = k_vals[k] * (k_vals[k_prime] - 1.) / k_vals[k_prime] * P_cond[k, k_prime]
                eigval, eigvect = linalg.eig(C_matrix, left=False, right=True)
                max_eigval = max(eigval)
                print('eigval:', eigval)
                print('max eigval: ', max_eigval)
                beta_threshold = 1. / max_eigval

                print('beta_threshold:', beta_threshold)
                print('hello')

                ## QQ plot

                # Generate theoretical quantiles for a Poisson distribution with the same mean
                theoretical_quantiles = poisson.ppf(np.linspace(0.01, 0.99, 100), mu=param)

                probplot(N_k, dist=poisson, sparams=(param,), plot=plt)
                # Add a line representing the theoretical quantiles
                plt.plot(theoretical_quantiles, theoretical_quantiles, color='red', linestyle='--')

                # Customize the plot
                plt.title('QQ Plot of Sampled Poisson Data')
                plt.xlabel('Theoretical Quantiles')
                plt.ylabel('Sample Quantiles')


                print('Pk-norm:', Pk_norm)
                plt.show()


            ######################################################################################################################

            if clustering_analysis == 1:
                # [Clustering coefficient]
                print('TO DO')

            ######################################################################################################################

            if PF_convergence == 1:
                # Error as the average over repeated generations of the topology, reported with the associated standard deviation
                nbr_repetitions = 10
                folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
                k_list = np.load(folder_topology + f'k_list.npy')

                diff_list_repeat = np.zeros(shape=(nbr_repetitions, int(len(k_list))))
                avg_diff_list = []
                stdDev_diff_list = []

                for repeat in range(nbr_repetitions):
                    idx_node = np.linspace(0, N - 1, N)
                    folder_topology_repeat = datadir + f'/Data_simpleLattice_v1/Repeated_topologies/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
                    # [PF convergence]
                    # Plot the error as a function of the dimension
                    diff_list = np.load(folder_topology_repeat + f'diff_list_rep{repeat}.npy')
                    diff_list_repeat[repeat, :] = diff_list
                avg_diff_list = np.mean(diff_list_repeat, axis=0)
                stdDev_diff_list = np.std(diff_list_repeat, axis=0, ddof=1)
                fig, ax = plt.subplots(figsize = (8, 6))
                ax.errorbar(k_list, avg_diff_list, stdDev_diff_list,
                            linestyle='-', fmt='o',
                            ecolor='darkgreen' if choice_bool == 0 else 'orangered',
                            elinewidth=1,
                            capsize=1.4,
                            color='g' if choice_bool == 0 else 'darkorange')
                #ax.fill_between(k_list, avg_diff_list - stdDev_diff_list,
                #                   avg_diff_list + stdDev_diff_list,
                #                   facecolor='green' if choice_bool == 0 else 'darkorange', alpha=0.25)
                axins = inset_axes(ax, width='40%', height='40%', loc='upper right')
                axins.errorbar(k_list, avg_diff_list, stdDev_diff_list,
                               linestyle='-', fmt = 'o',
                               ecolor = 'darkgreen' if choice_bool == 0 else 'orangered',
                               elinewidth = 1,
                               capsize = 1.4,
                               color='g' if choice_bool == 0 else 'darkorange')
                #axins.fill_between(k_list, avg_diff_list - stdDev_diff_list,
                #                avg_diff_list + stdDev_diff_list,
                #                facecolor= 'green' if choice_bool == 0 else 'darkorange', alpha=0.25)
                if choice_bool == 0:
                    xlim1 = k_list[3]-10
                    xlim2 = k_list[5]+50
                    ylim1 = avg_diff_list[5]-0.5
                    ylim2 = avg_diff_list[3]+0.75
                else:
                    xlim1 = k_list[3] - 10
                    xlim2 = k_list[6] + 50
                    ylim1 = avg_diff_list[6]-0.5
                    ylim2 = avg_diff_list[3]+0.8
                axins.set_xlim(xlim1, xlim2)
                axins.set_ylim(ylim1, ylim2)
                # Set white background for the inset plot
                axins.set_facecolor('white')

                # Mark the region in the main plot
                # Mark the region in the main plot and draw connecting lines
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw = 0.5)
                #mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")
                ax.indicate_inset_zoom(axins)
                #axins.set_xticklabels('')
                #axins.set_yticklabels('')
                # add labels and plot multiple dimensions in one to see how the decay of the error to zero changes as
                # a function of the network dimension.
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.tick_params(axis='both', which='minor', labelsize=12)
                ax.set_xlabel('Timestep', fontsize = 14)
                ax.set_ylabel('Error', fontsize = 14)

                plt.show()

                # Plot the difference between rho0 and the density of people in the final time
                if row == 3 or row == 5 or row == 10:
                    beta_vals = beta_vals_3_5_10
                    mu_vals = mu_vals_3_5_10
                else:
                    beta_vals = beta_vals_30_50
                    mu_vals = mu_vals_30_50
                # Consider the simulation done on my selected topology
                for beta, mu in zip(beta_vals, mu_vals):
                    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
                    folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
                    avg_population = np.load(folder_topology + f'avg_popPerNode_rep{repeat}.npy')

                    rho0 = np.load(folder_topology + f'/rho0.npy')
                    rho0 = rho0 * N
                    node_population_time = np.load(folder_simulation + 'sim_0_node_population_time.npy')
                    node_population_final = node_population_time[-1, :]
                    node_density_final = node_population_final / avg_population

                    diff_density = node_density_final - rho0

                    sns.histplot(node_density_final, bins = int(np.sqrt(len(node_density_final))))
                    sns.histplot(rho0, bins = int(np.sqrt(len(rho0))))
                    #plt.scatter(idx_node, diff_density, color='k')
                    #plt.axhline(y=0, linestyle='--', color='k')
                    #plt.xlabel('Index node')
                    #plt.ylabel(r'$\rho_{\infty} - \rho_0$')
                    plt.show()
                    print('hello')
                ######################################################################################################################

                if weight_analysis == 1:
                    TransitionMatrix = np.load(folder_topology + 'TransitionMatrix.npy')
                    TM_ravel = TransitionMatrix.ravel()
                    TM_round = np.round(TM_ravel, 2)  # keep 2 decimals for plot
                    TM_removed = TM_round[TM_round != 0.00]

                    plt.hist(TM_removed, color='#0504aa', alpha=0.7, align='mid', bins=100)
                    plt.yscale('log')
                    plt.xlabel('weight')
                    plt.ylabel('Frequency')
                    plt.title(f'dim = {row}x{col}, choice_bool = {choice_bool}, c1 = {c1}')
                    plt.show()
            ######################################################################################################################

            if Rstar_def == 1:
                if row == 3 or row == 5 or row == 10:
                    beta_vals = beta_vals_3_5_10
                    mu_vals = mu_vals_3_5_10
                else:
                    beta_vals = beta_vals_30_50
                    mu_vals = mu_vals_30_50

                # Input degrees
                in_degrees = np.array([G.in_degree(n) for n in G.nodes()])
                # Total number of links
                L = in_degrees.sum()
                L_max = N * (N - 1) / 2
                # Average input degree and second moment
                avg_in_degree = L / N
                second_moment = nth_moment_v2(G, 2)
                weightNonZero = [TransitionMatrix[i, j] for i in range(N) for j in range(N) if
                                 TransitionMatrix[i, j] != 0]
                # Since the transmission rate is not constant, I take the average value
                avg_transmission = np.mean(weightNonZero)
                # avg_population = avg_population / total_population
                for beta, mu in zip(beta_vals, mu_vals):
                    R0 = beta / mu
                    alpha = 2. * (R0 - 1.) / R0 ** 2
                    # Global invasion threshold (don't really understand the meaning)
                    R_star = (R0 - 1.) * (
                                second_moment - avg_in_degree) / avg_in_degree ** 2. * avg_transmission * avg_population * alpha / mu
                    print('row: ', row, 'col: ', col, 'choice-bool: ', choice_bool, 'c1: ', c1, 'beta: ', beta, 'mu: ',
                          mu,
                          'R0: ', R0, 'R_star: ', R_star, 'avg_transmisison: ', avg_transmission)

            ######################################################################################################################
            if outbreak == 1:

                # Plot the difference between rho0 and the density of people in the final time
                if row == 3 or row == 5 or row == 10:
                    beta_vals = beta_vals_3_5_10
                    mu_vals = mu_vals_3_5_10
                else:
                    beta_vals = beta_vals_30_50
                    mu_vals = mu_vals_30_50

                for beta, mu in zip(beta_vals, mu_vals):
                    folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
                    T = np.load(folder_simulation + 'T.npy')
                    new_I_time = np.load(folder_simulation + f'sim_{sim}_new_I_time.npy')
                    print('--------- outbreak trial 1 ---------')
                    # Calculate the cumulative number of NEW infected individuals per node over the whole period of time
                    cumulat_newI_perNode = np.zeros(shape=(T, N))
                    for i in range(N):
                        cumulat_newI_perNode[:, i] = np.cumsum(new_I_time[:, i])
                    # Cumulative new infected in the whole network per time step
                    cumulat_newI = cumulat_newI_perNode.sum(axis=1)
                    # The last number is the cumulative number of new infected in the whole network that is the
                    # number of individuals who got infected in the whole network during the total duration of the epidemics
                    total_newI = cumulat_newI[-1]
                    perc_newI = total_newI / total_population * 100
                    print('N: ', N, 'choice_bool: ', choice_bool, 'c1: ', c1, 'beta: ', beta, 'mu:', mu)
                    print('percentage new I: ', perc_newI, '%')
                    # Threshold outbreak : lim_t->infinity R_N(t) > N^1/4
                    threshold_outbreak = pow(total_population, 1. / 4.)
                    print('total newI: ', total_newI, 'threshold_outbreak: ', threshold_outbreak)
                    print('--------- outbreak trial 2 ---------')
                    threshold_outbreak = pow(total_population, 7. / 8.)
                    print('total newI: ', total_newI, 'threshold_outbreak: ', threshold_outbreak)
                    # + done eigenvalue on the degree_analysis part
            ######################################################################################################################

            if write_file == 1:
                write_network_file(row, col, choice_bool, c1, in_degrees, avg_in_degree, L, L_max,
                                   np.round(L / L_max * 100, 2),
                                   k_vals, Pk_norm, Pk_noNorm, diameter, avg_distance, param)

if log_dependence == 1:
    N_row_L = [3, 5, 10, 30, 50]
    N_col_L = [3, 5, 10, 30, 50]
    choice_bool_lst_L = [0, 1]

    dim_network = [9, 25, 100, 900, 2500]
    nbr_repetitions_L = 10
    perc_L_dim_homo = np.zeros(shape=(nbr_repetitions_L, len(N_row_L)))
    perc_L_dim_hetero = np.zeros(shape=(nbr_repetitions_L, len(N_row_L)))
    for choice_bool in choice_bool_lst_L:

        for rep in range(nbr_repetitions_L):
            i = 0
            for row, col in zip(N_row_L, N_col_L):
                N = row * col
                folder_simulation = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/'
                folder_topology_repeat = datadir + f'/Data_simpleLattice_v1/Repeated_topologies/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
                G = pickle.load(open(folder_topology_repeat + f'G_rep{rep}.pickle', 'rb'))
                in_degrees = np.array([G.in_degree(n) for n in G.nodes()])
                # Total number of links
                L = in_degrees.sum()
                L_max = N * (N - 1) / 2
                if choice_bool == 0:
                    perc_L_dim_homo[rep, i] = L/L_max * 100
                else:
                    perc_L_dim_hetero[rep, i] = L/L_max * 100

                i = i + 1

    avg_perc_L_dim_homo = np.mean(perc_L_dim_homo, axis=0)
    avg_perc_L_dim_hetero = np.mean(perc_L_dim_hetero, axis = 0)

    stdDev_L_dim_homo = np.std(perc_L_dim_homo, axis=0, ddof = 1)
    stdDev_L_dim__hetero = np.std(perc_L_dim_hetero, axis = 0, ddof = 1)
    logx = np.log10(dim_network)
    logy_homo = np.log10(avg_perc_L_dim_homo)
    logy_hetero = np.log10(avg_perc_L_dim_hetero)
    # Perform linear regression on the log-transformed data
    slope_homo, intercept_homo, r_value_homo, p_value_homo, std_err_homo = linregress(logx, logy_homo)
    slope_hetero, intercept_hetero, r_value_hetero, p_value_hetero, std_err_hetero = linregress(logx, logy_hetero)
    # Plot the linear fit in log-log space
    x_fit = np.linspace(1, dim_network[-1]+100, 1000)
    y_fit_homo = 10**(intercept_homo + slope_homo * np.log10(x_fit))
    y_fit_hetero = 10**(intercept_hetero + slope_hetero * np.log10(x_fit))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.errorbar(dim_network, avg_perc_L_dim_homo, stdDev_L_dim_homo,
                color = 'g', label = r'$RN_{HOM}$',
                #linestyle = '-',
                fmt = 'o',
                ecolor = 'darkgreen',
                elinewidth = 1,
                capsize = 1.6)
    ax.errorbar(dim_network, avg_perc_L_dim_hetero, stdDev_L_dim__hetero,
                color = 'orange', label = r'$RN_{HET}$',
                #linestyle = '-',
                fmt = 'o',
                ecolor = 'orangered',
                elinewidth = 1,
                capsize = 1.6)
    plt.loglog(x_fit, y_fit_homo, label=f'Linear Fit (Slope={slope_homo:.2f})', color='darkgreen')
    plt.loglog(x_fit, y_fit_hetero, label=f'Linear Fit (Slope={slope_hetero:.2f})', color='darkorange')
    plt.legend(fontsize=12)
    plt.xlabel('Network size', fontsize = 14)
    plt.ylabel(r'$L/L_{max}$%', fontsize = 14)
    axins = inset_axes(ax, width='40%', height='40%', loc='lower left')
    axins.errorbar(dim_network, avg_perc_L_dim_hetero, stdDev_L_dim__hetero,
                   fmt='o',
                   ecolor='orangered',
                   color = 'orange',
                   elinewidth=1,
                   capsize=1.4)

    xlim1 = dim_network[0]
    xlim2 = dim_network[0]
    ylim1 = avg_perc_L_dim_hetero[0]
    ylim2 = avg_perc_L_dim_hetero[0]

    axins.set_xlim(xlim1, xlim2)
    axins.set_ylim(ylim1, ylim2)
    # Set white background for the inset plot
    axins.set_facecolor('white')

    # Mark the region in the main plot
    # Mark the region in the main plot and draw connecting lines
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.5)
    # mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")
    ax.indicate_inset_zoom(axins)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    # add labels and plot multiple dimensions in one to see how the decay of the error to zero changes as
    # a function of the network dimension.
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)



    plt.show()



######################################################################################################################
row = 30
col = 30
choice_bool_lst_d = [0, 1]
c1 = 0

if distance_analysis == 1:
    for choice_bool in choice_bool_lst_d:
        folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'

        G = pickle.load(open(folder_topology + 'G.pickle', 'rb'))

        # [Paths and distances] (referred to the number of edges composing a path not to the Euclidan distance)
        max_dist = 10
        d_vals = np.linspace(0, max_dist, max_dist + 1)
        if choice_bool == 0:
            diameter_hom, avg_distance_hom, d_dist_hom = path_analysis(G, max_dist)
        else:
            diameter_het, avg_distance_het, d_dist_het = path_analysis(G, max_dist)

        # plot_distance_distribution(row, col, choice_bool, c1, d_vals, pd_norm, avg_distance)
        print('diameter:', diameter_hom, 'avg_distance', np.round(avg_distance_hom, 3))
        print('d_dist_hom_sum:', d_dist_hom.sum())
        #print('Expected value', (d_dist_hom * d_vals).sum())
    fig, ax = plt.subplots(figsize = (8,6))
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.plot(d_vals, d_dist_hom, 'o-', color='g', label = r'$RN_{HOM}$')
    plt.plot(d_vals, d_dist_het, 'o-', color='darkorange', label = r'$RN_{HET}$')
    plt.axvline(x=avg_distance_hom, color='darkgreen', label=r'$\langle d_{HOM} \rangle$', linestyle='--')
    plt.axvline(x=avg_distance_het, color='orangered', label=r'$\langle d_{HET} \rangle$', linestyle='--')
    plt.xlabel('$d$', fontsize = 14)
    plt.ylabel('$P(d)$', fontsize = 14)
    # plt.title(f'Degree distribution of {row}x{col} network with choice_bool: {choice_bool}, c1: {c1}')
    plt.legend(fontsize=12)
    plt.show()



if degree_analysis == 1:
    for choice_bool in choice_bool_lst_d:
        folder_topology = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
        folder_analysis = datadir + f'/Data_simpleLattice_v1/{row}x{col}/choice_bool-{choice_bool}/c1-{c1}/Analysis/'
        if choice_bool == 0:
            k_vals_hom = np.load(folder_analysis + 'k_vals.npy')
            k_bar_nn_hom = np.load(folder_analysis + 'k_bar_nn.npy')
            k_bar_nn_non_corr_hom = np.load(folder_analysis + 'k_bar_nn_non_corr.npy')
        else:
            k_vals_het = np.load(folder_analysis + 'k_vals.npy')
            k_bar_nn_het = np.load(folder_analysis + 'k_bar_nn.npy')
            k_bar_nn_non_corr_het = np.load(folder_analysis + 'k_bar_nn_non_corr.npy')
    fig, ax = plt.subplots(figsize = (8,6))

    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.plot(k_vals_hom, k_bar_nn_hom, 'o-', color='g', label=r'$RN_{HOM}$')
    plt.plot(k_vals_het, k_bar_nn_het, 'o-', color='darkorange', label=r'$RN_{HET}$')
    plt.axhline(y=k_bar_nn_non_corr_hom, color='darkgreen', label=r'$\bar{k}_{nn, nc}^{HOM}$', linestyle='--')
    plt.axhline(y=k_bar_nn_non_corr_het, color='orangered', label=r'$\bar{k}_{nn, nc }^{HET}$', linestyle='--')
    plt.xlabel('$k$', fontsize=14)
    plt.ylabel(r'$\bar{k}_{nn}(k)$', fontsize=14)
    # plt.title(f'Degree distribution of {row}x{col} network with choice_bool: {choice_bool}, c1: {c1}')
    plt.legend(fontsize=12)
    plt.show()
