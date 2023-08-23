from FunctAgentBasedSIR import *
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy.integrate import odeint


new_cmap = ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b", "#1a1a1a", "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]
rtg_r = LinearSegmentedColormap.from_list("rtg", new_cmap)
colors = rtg_r(np.linspace(0,1,20))

datadir = os.getcwd()

# ================== Parameters initialization ==================
M = 10 # number of repetitions of the stochastic simulation

N_row = np.array([32]) # number of rows in lattice for network

#beta_vals = np.array([0.2, 0.5])
#beta_vals = np.array([0.03, 0.04, 0.05, 0.06, 0.08])
beta_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
mu_vals = np.array([0.1])
#mu_vals = np.array([0.02, 0.025])
#mu_vals = np.array ([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
for i in range(0,len(N_row)):
    # Definition of the graph : Lattice, Triangle, Hexagon
    G,pos = DefPlanarGraph('Lattice', Nraw = N_row[i], Ncol = N_row[i])
    N = len(G.nodes())
    I_init = 1.
    R_init = 0.
    S_init = N - R_init - I_init
    X_init =['I']
    for i in range(N-int(I_init)):
        X_init.append('S')
    #X_init.append('I')
    print(len(X_init))
    #X_init = random.choices( ['S', 'I', 'R'], [1. - I_init / N, I_init / N, R_init / N], k=N)
    y_init = [S_init/N, I_init/N, R_init/N]
    

    # Simulation
    for bb in range(0,len(beta_vals)):
        for mm in range(0, len(mu_vals)):
            beta = beta_vals[bb]
            mu = mu_vals[mm]
            print('tau_beta = ', 1/beta, 'tau_mu = ', 1/mu, 'R0 = ', beta/mu)
            T = 200#int(50/beta)
            time = np.linspace(0, T, num = 1000)
            #A = nx.to_numpy_array(G)  
            # Repeat simulation M times
            tot_S = 0
            tot_I = 0
            tot_R = 0
            for ii in range(0,M):  
                print('\r ii: ', ii, end='')
                X ,Ptot_S, Ptot_I, Ptot_R, time = SimSIRAgentBased( G, I_init, X_init, T, 
                                                                    beta = beta, 
                                                                    mu = mu)      
                #np.save(datadir+f'/X_b={beta_vals[bb]}m={mu_vals[mm]}',X)
                tot_S = tot_S + np.asarray(Ptot_S)
                tot_I = tot_I + np.asarray(Ptot_I)
                tot_R = tot_R + np.asarray(Ptot_R)
                plt.plot(Ptot_S, Ptot_R, linewidth = 0.75)
                plt.plot([0, 1], [1, 0], linewidth = 0.55, color = 'black')
            avg_PtotS = tot_S/M
            avg_PtotI = tot_I/M
            avg_PtotR = tot_R/M
            plt.plot(avg_PtotS, avg_PtotR,linewidth = 0.85, color = 'black' )
            plt.xlabel('Prob(S)')
            plt.ylabel('Prob(R)')
            plt.title(f' beta={beta}, mu={mu}')
            plt.show()


            params = [beta, mu]
            y = odeint(SimSIRDeterministic, y_init, time, args = (params, ) )
            plotSIR_lattice(T, N, pos, G, X)
            #plotSIR_line(time, Ptot_S, Ptot_I, Ptot_R, y, N, beta_vals[bb], mu_vals[mm])
        
        
InfectedBetaMu(y_init, time, beta_vals, mu_vals)