!pip install netket

# Import netket library
import netket as nk

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import json
import time


## RBM ansatz

def maIsing(h,alpha):
  #define chain lattice
  g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

  #spin-based Hamiltonian
  hi = nk.hilbert.Spin(s=0.5, graph=g)
  ha = nk.operator.Ising(hilbert=hi, h=h, J=1.0)

  ma = nk.machine.RbmSpin(alpha=alpha, hilbert=nk.hilbert.Spin(s=0.5, graph=g))
  return ma

def RBMLearn(h, alpha, learning_rate, n_samples, n_iter):

    ma = maIsing(h=h, alpha=alpha)
    ma.init_random_parameters(seed=123, sigma=0.01)

  # Sampler
    sa = nk.sampler.MetropolisExchange(machine=ma)
    
  # Optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
  # Stochastic reconfiguration
    gs = nk.variational.Vmc(
      hamiltonian=ha,
      sampler=sa,
      optimizer=op,
      n_samples=n_samples,
      diag_shift=0.1,
      use_iterative=True,
      method='Sr')
    
    start = time.time()
    gs.run(out='Ising1D', n_iter=n_iter)
    end = time.time()

    return gs, start, end, n_iter


L = 16
h = [0.5,1,2]
a = [1,2,3,4]
color = ['orange', 'red', 'green']
E_NQS = [[0]*len(h)]*len(a)
EGS = [[],[],[]]
err_relative = [[],
                [],
                []]

for it in range(len(h)):

  for i in range(len(a)):

    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

    #spin-based Hamiltonian
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    ha = nk.operator.Ising(hilbert=hi, h=h[it], J=1.0)

    #compute the ground state energy with Lanczos ED
    def exact_gs_energy(ha): 
      exact_result = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=False)
      exact_gs_energy = exact_result.eigenvalues[0]
      
      return exact_gs_energy



#RUN RBMLearn

RBMLearn(h[it], a[i], 0.02, 600, 600)

# import the data from log file
data=json.load(open("Ising1D.log"))

iters= []
energy_RBM = []

EGS[it] = exact_gs_energy(ha)

for iteration in data["Output"]:
    iters.append(iteration["Iteration"])
    energy_RBM.append(iteration["Energy"]["Mean"])

E_NQS = energy_RBM[len(iters)-1]

err_relative[it].append(abs((E_NQS-EGS[it])/(EGS[it])))
