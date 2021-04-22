!pip install netket

# Import netket library
import netket as nk

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import json
import time

L = 4

def maHeis2D(alpha):
  #define chain lattice
  g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

  #spin-based Hamiltonian
  hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)

  ha = nk.operator.Heisenberg(hilbert=hi)
  ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi)
  return ma

def RBMHeis2D(alpha, learning_rate, n_samples, n_iter):

    ma = maHeis2D(alpha=alpha)
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
    gs.run(out='Heis2D', n_iter=n_iter)
    end = time.time()

    return gs, start, end, n_iter
  
##RUN RBMHeis2D
  
L = 4
a = [1,2,4]
E_NQS = []
EGS = []
err_relative = []


for i in range(len(a)):

    g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

    #spin-based Hamiltonian
    hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)
    ha = nk.operator.Heisenberg(hilbert=hi)

    #compute the ground state energy with Lanczos ED
    def exact_gs_energy(ha): 
      exact_result = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=False)
      exact_gs_energy = exact_result.eigenvalues[0]
      
      return exact_gs_energy

    EGS.append(exact_gs_energy(ha))

    #RBM machine learn
    RBMHeis2D(a[i], 0.07, 1000, 600)

    # import the data from log file
    data=json.load(open("Heis2D.log"))

    iters= []
    energy_RBM = []

    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy_RBM.append(iteration["Energy"]["Mean"])

    E_NQS.append(energy_RBM[len(iters)-1])

    err_relative.append(abs((E_NQS[i]-EGS[i])/(EGS[i])))
