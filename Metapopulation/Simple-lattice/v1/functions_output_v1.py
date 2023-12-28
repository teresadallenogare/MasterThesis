"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 21 November 2023

--------------------------------------------------------------------

Functions to output data and write output files

"""
import numpy as np
import os
from datetime import datetime

def write_topology_file(N_row, N_col, N, pos_nodes, avg_pop_node, populationTot, choice_bool, Nfix, idxNfix, percentPopNfix, c1,
                        node_pop0, strongConnection, a, b, rho0, k_list, diff_list, in_degrees, repeat):

    datadir = os.getcwd()

    folder_topology = datadir + f'/Data_simpleLattice_v1/Repeated_topologies/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    f = open(folder_topology + f'topologyFile_rep{repeat}.txt', 'w')
    f.write(' ---------------------------------------------\n\n')
    f.write('              TOPOLOGY FILE - v1      \n\n')
    f.write('      Author : Teresa Dalle Nogare\n')
    f.write('      Date :' + datetime.today().strftime('%Y-%m-%d')+'\n\n')
    f.write(' ---------------------------------------------\n\n')
    f.write('[Nodes] \n')
    f.write(f'N_row = {N_row}, N_col = {N_col}, N = N_row x N_col = {N} \n')
    f.write(f'Position of nodes = {pos_nodes}\n')
    f.write(f'[Population data] \n')
    f.write(f'Average population per node = {avg_pop_node} \n')
    f.write(f'Total population = {populationTot}\n\n')
    if choice_bool == 0:
        f.write(f'choice bool = {choice_bool} - uniform population (homogeneous network) \n\n')
    elif choice_bool == 1:
        f.write(f'choice bool = {choice_bool} - non uniform population (heterogeneous network) \n')
        f.write(f'number of fixed nodes = {Nfix}\n')
        f.write(f'Index fixed nodes = {idxNfix}\n')
        f.write(f'percentage of population in fixed nodes = {percentPopNfix}\n\n')
    else:
        f.write(f'Wrong choice bool value \n\n')
    f.write(f'[Initial configuration]\n')
    f.write(f'Node population 0: {node_pop0}\n\n')
    f.write('[Connectivity]\n')
    f.write(f'Strong connection = {strongConnection}\n')
    f.write(f'a = {a}\n')
    f.write(f'b = {b}\n\n')
    f.write(f'[PF convergence]\n')
    f.write(f'rho0 = {rho0}\n')
    f.write(f'k_list = {k_list}\n')
    f.write(f'diff_list = {diff_list}\n\n')
    f.write('[Input degrees]\n')
    f.write(f'in_degrees = {in_degrees}')
    f.close()

def write_simulation_file(N_row, N_col, choice_bool, c1, node_pop0, node_S0, node_I0, node_R0, node_state0, T, beta, mu, nbr_repetitions,
                          nbr_sim_not_start, idx_sim_not_start):
    datadir = os.getcwd()
    folder_simulation = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Simulations/mu-{mu}/beta-{beta}/'
    f = open(folder_simulation + f'simulationFile.txt', 'w')
    f.write(' ---------------------------------------------\n\n')
    f.write('              SIMULATION FILE - v1      \n\n')
    f.write('      Author : Teresa Dalle Nogare\n')
    f.write('      Date :' + datetime.today().strftime('%Y-%m-%d')+'\n\n')
    f.write(' ---------------------------------------------\n\n')
    f.write(f'[Simulation parameters]\n')
    f.write(f'Length of simulation, T : {T}\n')
    f.write(f'Infection rate, beta: {beta}\n')
    f.write(f'Recovery rate, mu: {mu}\n')
    f.write(f'Nuber of repetitions: {nbr_repetitions}\n')
    f.write(f'[Initial configuration]\n')
    f.write(f'Node population 0: {node_pop0}\n')
    f.write(f'Node S0: {node_S0}\n')
    f.write(f'Node I0: {node_I0}\n')
    f.write(f'Node R0: {node_R0}\n')
    f.write(f'Node state0: {node_state0}\n\n')
    f.write('[Not started simulations]\n')
    f.write(f'Number of not started simulations: {nbr_sim_not_start}\n')
    f.write(f'Index of not started simulations: {idx_sim_not_start}')

def write_network_file(N_row, N_col, choice_bool, c1, in_degrees, kin_avg, L, Lmax, percL, k, pk_noNorm, pk_norm,
                       diameter, avg_distance, param):

    datadir = os.getcwd()
    folder_topology = datadir + f'/Data_simpleLattice_v1/{N_row}x{N_col}/choice_bool-{choice_bool}/c1-{c1}/Topology/'
    f = open(folder_topology + f'networkFile.txt', 'w')
    f.write(' ---------------------------------------------\n\n')
    f.write('              NETWORK PROPERTIES FILE - v1      \n\n')
    f.write('      Author : Teresa Dalle Nogare\n')
    f.write('      Date : ' + datetime.today().strftime('%Y-%m-%d') + '\n\n')
    f.write(' ---------------------------------------------\n\n')
    f.write('[Degree properties]\n')
    f.write(f'Input degree: {in_degrees}\n')
    f.write(f'Total number of input links: {L}\n')
    f.write(f'Maximum number of input links: {Lmax}\n')
    f.write(f'Average input degree: {kin_avg}\n')
    f.write(f'Fraction of links compared to complete graph: {percL}\n\n')
    f.write('[Degree distribution]\n')
    f.write(f'In-degree values: {k}\n')
    f.write(f'pk no normalized: {pk_noNorm}\n')
    f.write(f'pk normalized: {pk_norm}\n\n')
    f.write('[Paths and distances]\n')
    f.write(f'Diameter: {diameter}\n')
    f.write(f'Average distance: {np.round(avg_distance, 3)}\n\n')
    f.write('[Posson fit]\n')
    f.write(f'param: {param}')

