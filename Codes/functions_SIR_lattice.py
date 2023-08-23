import networkx as nx 
import numpy as np
import random 
import math
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def DefPlanarGraph(Gname, Nraw, Ncol):
    '''
    Determines the topology of the graph to work on
    '''
    # dictionary = {<key>: <value>}
    dict = {'Lattice' : nx.grid_2d_graph(Nraw,Ncol, periodic = True),
            'Triangle' : nx.triangular_lattice_graph(Nraw, Ncol, with_positions=True, periodic=True),
            'Hexagon' : nx.hexagonal_lattice_graph(Nraw, Ncol, with_positions=True, periodic=True)}
    G = dict[Gname]
    if Gname == 'Lattice':
        pos = {}
        for i in G.nodes:
            pos.update({i: (i[0], i[1])})
    else:
        pos = nx.get_node_attributes(G,'pos')
    return G,pos


def CreateNeighbourList(G):
    '''
    Creates neighbour list : set of particles with which each particle interacts
    '''
    datadir = os.getcwd()
    A = nx.to_numpy_array(G) # returns adjecency matrix of G
    N = len(G.nodes())
    #print(list(G.edges))
    Neig = []
   #print(np.dot((np.dot(A,A)), A))
    for i in range(N):
        Neig_i = []
        for j in range(N):
            print('\r', math.floor(i/N*100), '%', end='')
            if A[i,j] == 1 : Neig_i.append(j)
        Neig.append(Neig_i)
    return Neig    

def SimSIRAgentBased(G, I_init, X_init, T, beta, mu):
    '''
    Simulation of SIR agent based 
    beta = infection rate
    mu = recovery rate
    '''
    N = len(G.nodes())
    # initialize probability rates (same probability for each node)
    ProbS = np.full(N,1. -I_init/N)
    ProbI = np.full(N, I_init/N)
    ProbR = np.full(N,0.)
    # initialize random vector X containing the state of each node
    X = []
    X.append(X_init)
    # initialize 'delta' probability vectors
    deltaS = np.zeros(N)
    deltaI = np.zeros(N)
    deltaR = np.zeros(N)
    prod = np.full(N,1.)
    # create neighbour list
    nlist = CreateNeighbourList(G)
    # --- main loop ---
    # densities of each compartment
    rhoS = np.zeros(T)
    rhoI = np.zeros(T)
    rhoR = np.zeros(T)
    
    Ptot_S = []
    Ptot_I = []
    Ptot_R = []
    time = []
    
    X_prev = []
    S = []
    I = []
    R = []
    
    X_prev = X_init

    for t in range(T-1):
        print('\r',t,end='')
        l = []
        for i in range(N):
            if X[t][i]== 'S':
                deltaS[i] = 1.
                rhoS[t] +=1. / N
                
            else:
                deltaS[i] = 0.
            if X[t][i] == 'I':
                deltaI[i] = 1.
                rhoI[t] += 1./N
            else:
                deltaI[i] = 0.
            if X[t][i] == 'R':
                deltaR[i] = 1.
                rhoR[t] += 1./N
            else:
                deltaR[i] = 0.
            prod[i] = 1
            # transition probabilities
            for j in nlist[i]: # among nearest neighbours
                prod[i] = prod[i] * (1. - beta * deltaI[j])
            ProbS[i] = deltaS[i] * prod[i]
            ProbI[i] = (1. - mu) * deltaI[i] + deltaS[i] * (1. - prod[i])
            ProbR[i] = deltaR[i] + mu * deltaI[i]
            #print(i, ProbS[i], ProbR[i], ProbI[i])
            l.append(random.choices(['S', 'I', 'R'], [ProbS[i], ProbI[i], ProbR[i]], k=1)[0])     
        X.append(l)
        # X[t] contains the time evolved graph
        Ptot_S.append(sum(ProbS)/N)
        Ptot_I.append(sum(ProbI)/N)
        Ptot_R.append(sum(ProbR)/N)
        time.append(t)
        s, i, r = GenerateTimeSeries_SIR(t, N, X_prev, X[t])
        X_prev = X[t]
        S.append(s)
        I.append(i)
        R.append(r)
        
    #plt.plot(time,R, linewidth = 0.75, label = 'R')
    #plt.plot(time,I, linewidth = 0.75, label = 'I')

    #plt.legend()
    #plt.show()
    return np.array([np.array(xi) for xi in X]), Ptot_S, Ptot_I, Ptot_R, time



def SimSIRDeterministic(variables, t, params):
    '''
    Simulation of deterministic SIR model
    '''
    S = variables[0]
    I = variables[1]
    R = variables[2]
    
    N = S + I + R
    beta = params[0]
    gamma = params[1]
    
    dSdt = - beta * I/N * S/N
    dIdt = beta * I/N * S/N - gamma * I
    dRdt = gamma * I/N 

    return [dSdt, dIdt, dRdt]

    
def InfectedBetaMu(y_init, time, beta_vals, mu_vals):
    new_cmap = ['#A6CEE3','#1F78B4','#B2DF8A','#33A02C','#FB9A99','#E31A1C','#FDBF6F','#FF7F00','#CAB2D6','#6A3D9A','#ECEC28','#B15928']
    new_cmap = ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de", "#0571b0"]
    rtg_r = LinearSegmentedColormap.from_list("rtg", new_cmap)
    colors = rtg_r(np.linspace(0,1,12))
    # Fix mu and cycle on beta
    for mm in range(0, len(mu_vals)):
        mu = mu_vals[mm]
        for bb in range(0, len(beta_vals)):
            beta = beta_vals[bb]
            params = [beta, mu]
            y = odeint(SimSIRDeterministic, y_init, time, args = (params, ) )
            plt.plot(time, y[:,1], label = f'beta={beta}', color = colors[bb])
        plt.xlabel('t')
        plt.ylabel('ProbI(t)')
        plt.title(f'Infected probability with mu={mu} and varying beta')
        plt.legend()
        plt.show()
        
    # Fix beta and cycle on mu 
    for bb in range(0, len(beta_vals)):
        beta = beta_vals[bb]
        for mm in range(0, len(mu_vals)):
            mu = mu_vals[mm]
            params = [beta,mu]
            y = odeint(SimSIRDeterministic, y_init, time, args = (params,) )
            plt.plot(time, y[:,1], label = f'beta={beta}, mu={mu}')
        plt.xlabel('t')
        plt.ylabel('I(t)')
        plt.title('I(t) with fixed beta and varying mu')
       # plt.legend()
       # plt.show()

def GenerateTimeSeries_SIR(t, N, X_prev, X_curr):
    '''
    Number of new cases per time step
    '''
    S = N
    I = 0
    R = 0
    # index variable
    idx = 0
    # Result list
    res = []
    # Detect index of different elements between the 2 lists and store in variable res
    for i in X_curr: 
        if i != X_prev[idx]:
            res.append(idx)
            if i == 'I' and X_prev[idx] == 'S':
                I = I + 1
            elif i == 'R' and X_prev[idx] == 'I':
                R = R + 1
        idx = idx + 1
    # Result
    #print("The index positions with mismatched values:\n",res)
    return [S, I ,R]

# -------------------------------------- Plot functions -------------------------------- 
def plotSIR_line(time, Ptot_S, Ptot_I, Ptot_R, y, N, beta, mu):
    '''
    plot time evolution of S, I and R compartments
    '''
    f,(ax1,ax2,ax3) = plt.subplots(3)
    line1, = ax1.plot(time,Ptot_S)
    line2, = ax2.plot(time,Ptot_I)
    line3, = ax3.plot(time,Ptot_R)
    line4, = ax1.plot(time,y[:,0])
    line5, = ax2.plot(time,y[:,1])
    line6, = ax3.plot(time,y[:,2])

    ax1.set_ylabel('S')
    ax2.set_ylabel('I')
    ax3.set_ylabel('R')
    ax3.set_xlabel('time')
    
    plt.suptitle(f'N = {N}, beta={beta}, mu ={mu}')
    plt.show()



def plotSIR_lattice(T, N, pos, G, X):
    ''''
    Movie of the simulation for the lattice
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for t in range(T):
        color_map = []
        for i in range(N):
            if X[t][i] == 'S':
                color_map.append('blue')
            elif X[t][i] == 'I':
                color_map.append('red')
            elif X[t][i] == 'R':
                color_map.append('green')
        plt.clf()

        nx.draw(G, pos=pos, node_color=color_map, edge_color='white', node_size=60)
        fig.set_facecolor('black')

        plt.pause(0.1)###in second the time a figure lasts
    plt.close()
