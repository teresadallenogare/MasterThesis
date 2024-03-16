# MasterThesis
Repository containing the code and progress of my master's thesis in Physics at the University of Trento
## Agent-Based
Contains code of agent-based simulation of an SIR epidemics on a lattice with variable connectivity:

- Triangular lattice
- Square lattice
- Hexagonal lattice

## Metapopilation
Contains code of metapopulation simulation of SIR epidemics on a lattice with variable topology

### Simple lattice
The repo contains two versions of the code:

- **v0**: exploratory version of the code
- **v1**: actual version of the code. The folder contains both the code to run on the local computer and to run on the cluster of UniTN. The code is organised in the following way:

    - _0_topology_v1_: generates a strongly connected network with arbitrary dimension. The location of nodes is fixed on the vertices of a square lattice and the connectivity is established through the gravity law. Two parameters `c` and `c1` control the connectivity and the strength of the self-loop respectively. 

    - _1_SIR-metapop_v1_: simulates SIR epidemics on the previously generated network with arbitrary values of the infection and recovery rates. 
        
