"""

--------------------------------------------------------------------

Author : Teresa Dalle Nogare
Version : 07 October 2023

--------------------------------------------------------------------

Implementation and resolution of determinisitc SIR equations

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIRDeterministic_equations(variables, t, params):
  """ Determinisitc ODE for the SIR model. I consider equaitons for densities:
  ds/dt = - beta * i * s = - alpha * s
  di/dt = beta * i * s - mu * i = alpha * s - mu * i
  dr/dt = mu * i

  :param variables: [s, i, r] : densities
  :param t:
  :param params: [beta, mu] : infection rate and recovery rate
  :return:
  """
  s = variables[0]
  i = variables[1]

  beta = params[0]
  mu = params[1]

  alpha = beta * i

  dsdt = - alpha * s
  didt = alpha * s - mu * i
  drdt = mu * i

  return [dsdt, didt, drdt]

# ------------------------------------------ Resolution ------------------------------------------

Tsim = 100
time = np.linspace(0, Tsim, Tsim* 10)

popTot = 1e5
popI_init = 1
popR_init = 0
popS_init = popTot - popR_init - popI_init

# initial conditions : densities
y_init = [popS_init/popTot, popI_init/popTot, popR_init/popTot]

# infection parameters
beta = 0.9
mu = 0.2

params = [beta, mu]

# sole equation for densities
y = odeint(SIRDeterministic_equations, y_init, time, args=(params,))

# densities in time: solutions of SIR deterministic ODEs
s = y[:, 0]
i = y[:, 1]
r = y[:, 2]

plt.plot(time, s, color = 'blue', label = 's')
plt.plot(time, i, color = 'red', label = 'i')
plt.plot(time, r, color = 'green', label = 'r')
plt.legend()
plt.xlabel('time')
plt.ylabel('densities')

plt.show()