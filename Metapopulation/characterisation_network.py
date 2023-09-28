"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 27 September 2023

--------------------------------------------------------------------

Study how the value of the total population and the number of nodes in the network influence the value of c for which I
first have a strongly connected graph.

"""
import numpy as np
import matplotlib.pyplot as plt


populationTot = np.array([1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])

# Nrow = 2, Ncol = 2, N = 4
c2 = np.array([2, 1, 2, 3, 2, 2, 2, 3, 2, 1, 3, 3, 2])

# Nrow = 3, Ncol = 3, N = 9
c3 = np.array([15, 14, 14, 14, 12, 12, 14, 12, 12, 8, 12, 9, 10 ])

# Nrow = 4, Ncol = 4, N = 16
c4 = np.array([53, 47, 41, 41, 40, 40, 41, 39, 37, 35, 33, 37, 42])

# Nrow = 5, Ncol = 5, N = 25
c5 = np.array([150, 117, 106, 91, 94, 90, 90, 83, 83, 80, 95, 89, 84])

plt.plot(populationTot, c2, '--',color = 'gray', marker = 'o', linewidth = 1, markersize = 6, label = 'N=4')
plt.plot(populationTot, c3, '--',color = 'blue', marker = 'o', linewidth = 1, markersize = 6, label = 'N=9')
plt.plot(populationTot, c4, '--',color = 'red', marker = 'o', linewidth = 1, markersize = 6, label = 'N=16')
plt.plot(populationTot, c5, '--',color = 'green', marker = 'o', linewidth = 1, markersize = 6, label = 'N=25')
plt.xlabel('Total population')
plt.ylabel('c')
plt.legend()
plt.show()

