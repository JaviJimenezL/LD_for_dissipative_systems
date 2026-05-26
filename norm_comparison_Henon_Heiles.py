"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

# Comparison of the results obtained for Delta L and Omega for different defintion of the LD (p-norm and euclidean)

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "axes.labelsize": 22,
    "axes.labelweight": "bold",
    "axes.titlesize": 22,
    "axes.titleweight": "bold",
    "legend.fontsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

"""
Load data from each simulation and calculate the two indicators
"""

# Euclidea norm:
data_euclidean_reg = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/euclidean/LD_evolution_x0_reg_euclidean.txt')
data_euclidean_reg_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/euclidean/LD_evolution_x0_perturbed_reg_euclidean.txt')

data_euclidean_chao = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/euclidean/LD_evolution_x0_chao_euclidean.txt')
data_euclidean_chao_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/euclidean/LD_evolution_x0_perturbed_chao_euclidean.txt')

delta_L_reg_euclidean = np.abs(data_euclidean_reg[:,1] - data_euclidean_reg_neigh[:,1])
delta_L_chao_euclidean = np.abs(data_euclidean_chao[:,1] - data_euclidean_chao_neigh[:,1])

delta_L_reg_euclidean_int = cumulative_trapezoid(delta_L_reg_euclidean, data_euclidean_reg[:,0], initial=0.0)
delta_L_chao_euclidean_int = cumulative_trapezoid(delta_L_chao_euclidean, data_euclidean_chao[:,0], initial=0.0)

Omega_reg_euclidean = data_euclidean_reg[:,1] / data_euclidean_reg[:,0]
Omega_chao_euclidean = data_euclidean_chao[:,1] / data_euclidean_chao[:,0]

###############################################################################

# p-norm with p = 0.1:
data_p_01_reg = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_01/LD_evolution_x0_reg_p_01.txt')
data_p_01_reg_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_01/LD_evolution_x0_perturbed_reg_p_01.txt')

data_p_01_chao = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_01/LD_evolution_x0_chao_p_01.txt')
data_p_01_chao_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_01/LD_evolution_x0_perturbed_chao_p_01.txt')

delta_L_reg_p_01 = np.abs(data_p_01_reg[:,1] - data_p_01_reg_neigh[:,1])
delta_L_chao_p_01 = np.abs(data_p_01_chao[:,1] - data_p_01_chao_neigh[:,1])

delta_L_reg_p_01_int = cumulative_trapezoid(delta_L_reg_p_01, data_p_01_reg[:,0], initial=0.0)
delta_L_chao_p_01_int = cumulative_trapezoid(delta_L_chao_p_01, data_p_01_chao[:,0], initial=0.0)

Omega_reg_p_01 = data_p_01_reg[:,1] / data_p_01_reg[:,0]
Omega_chao_p_01 = data_p_01_chao[:,1] / data_p_01_chao[:,0]

###############################################################################

# p-norm with p = 0.25:
data_p_025_reg = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_025/LD_evolution_x0_reg_p_025.txt')
data_p_025_reg_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_025/LD_evolution_x0_perturbed_reg_p_025.txt')

data_p_025_chao = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_025/LD_evolution_x0_chao_p_025.txt')
data_p_025_chao_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_025/LD_evolution_x0_perturbed_chao_p_025.txt')

delta_L_reg_p_025 = np.abs(data_p_025_reg[:,1] - data_p_025_reg_neigh[:,1])
delta_L_chao_p_025 = np.abs(data_p_025_chao[:,1] - data_p_025_chao_neigh[:,1])

delta_L_reg_p_025_int = cumulative_trapezoid(delta_L_reg_p_025, data_p_025_reg[:,0], initial=0.0)
delta_L_chao_p_025_int = cumulative_trapezoid(delta_L_chao_p_025, data_p_025_chao[:,0], initial=0.0)

Omega_reg_p_025 = data_p_025_reg[:,1] / data_p_025_reg[:,0]
Omega_chao_p_025 = data_p_025_chao[:,1] / data_p_025_chao[:,0]

###############################################################################

# p-norm with p = 0.5:
data_p_05_reg = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_05/LD_evolution_x0_reg_p_05.txt')
data_p_05_reg_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_05/LD_evolution_x0_perturbed_reg_p_05.txt')

data_p_05_chao = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_05/LD_evolution_x0_chao_p_05.txt')
data_p_05_chao_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_05/LD_evolution_x0_perturbed_chao_p_05.txt')

delta_L_reg_p_05 = np.abs(data_p_05_reg[:,1] - data_p_05_reg_neigh[:,1])
delta_L_chao_p_05 = np.abs(data_p_05_chao[:,1] - data_p_05_chao_neigh[:,1])

delta_L_reg_p_05_int = cumulative_trapezoid(delta_L_reg_p_05, data_p_05_reg[:,0], initial=0.0)
delta_L_chao_p_05_int = cumulative_trapezoid(delta_L_chao_p_05, data_p_05_chao[:,0], initial=0.0)

Omega_reg_p_05 = data_p_05_reg[:,1] / data_p_05_reg[:,0]
Omega_chao_p_05 = data_p_05_chao[:,1] / data_p_05_chao[:,0]

###############################################################################

# p-norm with p = 0.75:
data_p_075_reg = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_075/LD_evolution_x0_reg_p_075.txt')
data_p_075_reg_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_075/LD_evolution_x0_perturbed_reg_p_075.txt')

data_p_075_chao = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_075/LD_evolution_x0_chao_p_075.txt')
data_p_075_chao_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_075/LD_evolution_x0_perturbed_chao_p_075.txt')

delta_L_reg_p_075 = np.abs(data_p_075_reg[:,1] - data_p_075_reg_neigh[:,1])
delta_L_chao_p_075 = np.abs(data_p_075_chao[:,1] - data_p_075_chao_neigh[:,1])

delta_L_reg_p_075_int = cumulative_trapezoid(delta_L_reg_p_075, data_p_075_reg[:,0], initial=0.0)
delta_L_chao_p_075_int = cumulative_trapezoid(delta_L_chao_p_075, data_p_075_chao[:,0], initial=0.0)

Omega_reg_p_075 = data_p_075_reg[:,1] / data_p_075_reg[:,0]
Omega_chao_p_075 = data_p_075_chao[:,1] / data_p_075_chao[:,0]

###############################################################################

# p-norm with p = 0.9:
data_p_09_reg = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_09/LD_evolution_x0_reg_p_09.txt')
data_p_09_reg_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_09/LD_evolution_x0_perturbed_reg_p_09.txt')

data_p_09_chao = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_09/LD_evolution_x0_chao_p_09.txt')
data_p_09_chao_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_09/LD_evolution_x0_perturbed_chao_p_09.txt')

delta_L_reg_p_09 = np.abs(data_p_09_reg[:,1] - data_p_09_reg_neigh[:,1])
delta_L_chao_p_09 = np.abs(data_p_09_chao[:,1] - data_p_09_chao_neigh[:,1])

delta_L_reg_p_09_int = cumulative_trapezoid(delta_L_reg_p_09, data_p_09_reg[:,0], initial=0.0)
delta_L_chao_p_09_int = cumulative_trapezoid(delta_L_chao_p_09, data_p_09_chao[:,0], initial=0.0)

Omega_reg_p_09 = data_p_09_reg[:,1] / data_p_09_reg[:,0]
Omega_chao_p_09 = data_p_09_chao[:,1] / data_p_09_chao[:,0]

###############################################################################

# p-norm with p = 1:
data_p_1_reg = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_1/LD_evolution_x0_reg_p_1.txt')
data_p_1_reg_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_1/LD_evolution_x0_perturbed_reg_p_1.txt')

data_p_1_chao = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_1/LD_evolution_x0_chao_p_1.txt')
data_p_1_chao_neigh = np.loadtxt('/home/javier/dissipative_systems/henon_heiles_results/p_1/LD_evolution_x0_perturbed_chao_p_1.txt')

delta_L_reg_p_1 = np.abs(data_p_1_reg[:,1] - data_p_1_reg_neigh[:,1])
delta_L_chao_p_1 = np.abs(data_p_1_chao[:,1] - data_p_1_chao_neigh[:,1])

delta_L_reg_p_1_int = cumulative_trapezoid(delta_L_reg_p_1, data_p_1_reg[:,0], initial=0.0)
delta_L_chao_p_1_int = cumulative_trapezoid(delta_L_chao_p_1, data_p_1_chao[:,0], initial=0.0)

Omega_reg_p_1 = data_p_1_reg[:,1] / data_p_1_reg[:,0]
Omega_chao_p_1 = data_p_1_chao[:,1] / data_p_1_chao[:,0]

###############################################################################

"""
Structure of the plot
Delta_L_reg        Delta_L_chao
int_Delta_L_reg    int_Delta_L_chao
Omega_reg          Omega_chao
"""


plt.figure(figsize=(8, 6))
plt.plot(data_euclidean_reg[:,0], delta_L_reg_euclidean, label='Euclidean')
plt.plot(data_p_01_reg[:,0], delta_L_reg_p_01, label='p = 0.1')
plt.plot(data_p_025_reg[:,0], delta_L_reg_p_025, label='p = 0.25')
plt.plot(data_p_05_reg[:,0], delta_L_reg_p_05, label='p = 0.5')
plt.plot(data_p_075_reg[:,0], delta_L_reg_p_075, label='p = 0.75')
plt.plot(data_p_09_reg[:,0], delta_L_reg_p_09, label='p = 0.9')
plt.plot(data_p_1_reg[:,0], delta_L_reg_p_1, label='p = 1.0')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Delta \mathcal{L}$')
plt.xlim([3.0, 1e4])
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Delta_L_HH_reg.pdf", dpi = 300)
plt.show()

###############################################################################

plt.figure(figsize=(8, 6))
plt.plot(data_euclidean_chao[:,0], delta_L_chao_euclidean, label='Euclidean')
plt.plot(data_p_01_chao[:,0], delta_L_chao_p_01, label='p = 0.1')
plt.plot(data_p_025_chao[:,0], delta_L_chao_p_025, label='p = 0.25')
plt.plot(data_p_05_chao[:,0], delta_L_chao_p_05, label='p = 0.5')
plt.plot(data_p_075_chao[:,0], delta_L_chao_p_075, label='p = 0.75')
plt.plot(data_p_09_chao[:,0], delta_L_chao_p_09, label='p = 0.9')
plt.plot(data_p_1_chao[:,0], delta_L_chao_p_1, label='p = 1.0')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Delta \mathcal{L}$')
plt.xlim([3.0, 1e4])
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Delta_L_HH_chao.pdf", dpi = 300)
plt.show()

###############################################################################

plt.figure(figsize=(8, 6))
plt.plot(data_euclidean_reg[:,0], delta_L_reg_euclidean_int, label='Euclidean')
plt.plot(data_p_01_reg[:,0], delta_L_reg_p_01_int, label='p = 0.1')
plt.plot(data_p_025_reg[:,0], delta_L_reg_p_025_int, label='p = 0.25')
plt.plot(data_p_05_reg[:,0], delta_L_reg_p_05_int, label='p = 0.5')
plt.plot(data_p_075_reg[:,0], delta_L_reg_p_075_int, label='p = 0.75')
plt.plot(data_p_09_reg[:,0], delta_L_reg_p_09_int, label='p = 0.9')
plt.plot(data_p_1_reg[:,0], delta_L_reg_p_1_int, label='p = 1.0')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\int_{0}^{\tau}\Delta \mathcal{L} \, \mathrm{d}t$')
plt.xlim([3.0, 1e4])
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/int_Delta_L_HH_reg.pdf", dpi = 300)
plt.show()

###############################################################################

plt.figure(figsize=(8, 6))
plt.plot(data_euclidean_chao[:,0], delta_L_chao_euclidean_int, label='Euclidean')
plt.plot(data_p_01_chao[:,0], delta_L_chao_p_01_int, label='p = 0.1')
plt.plot(data_p_025_chao[:,0], delta_L_chao_p_025_int, label='p = 0.25')
plt.plot(data_p_05_chao[:,0], delta_L_chao_p_05_int, label='p = 0.5')
plt.plot(data_p_075_chao[:,0], delta_L_chao_p_075_int, label='p = 0.75')
plt.plot(data_p_09_chao[:,0], delta_L_chao_p_09_int, label='p = 0.9')
plt.plot(data_p_1_chao[:,0], delta_L_chao_p_1_int, label='p = 1.0')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\int_{0}^{\tau}\Delta \mathcal{L} \, \mathrm{d}t$')
plt.xlim([3.0, 1e4])
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/int_Delta_L_HH_chao.pdf", dpi = 300)
plt.show()

###############################################################################

plt.figure(figsize=(8, 6))
plt.plot(data_euclidean_reg[:,0], Omega_reg_euclidean, label='Euclidean')
plt.plot(data_p_01_reg[:,0], Omega_reg_p_01, label='p = 0.1')
plt.plot(data_p_025_reg[:,0], Omega_reg_p_025, label='p = 0.25')
plt.plot(data_p_05_reg[:,0], Omega_reg_p_05, label='p = 0.5')
plt.plot(data_p_075_reg[:,0], Omega_reg_p_075, label='p = 0.75')
plt.plot(data_p_09_reg[:,0], Omega_reg_p_09, label='p = 0.9')
plt.plot(data_p_1_reg[:,0], Omega_reg_p_1, label='p = 1.0')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Omega$')
plt.xlim([3.0, 1e4])
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Omega_HH_reg.pdf", dpi = 300)
plt.show()

###############################################################################

plt.figure(figsize=(8, 6))
plt.plot(data_euclidean_chao[:,0], Omega_chao_euclidean, label='Euclidean')
plt.plot(data_p_01_chao[:,0], Omega_chao_p_01, label='p = 0.1')
plt.plot(data_p_025_chao[:,0], Omega_chao_p_025, label='p = 0.25')
plt.plot(data_p_05_chao[:,0], Omega_chao_p_05, label='p = 0.5')
plt.plot(data_p_075_chao[:,0], Omega_chao_p_075, label='p = 0.75')
plt.plot(data_p_09_chao[:,0], Omega_chao_p_09, label='p = 0.9')
plt.plot(data_p_1_chao[:,0], Omega_chao_p_1, label='p = 1.0')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Omega$')
plt.xlim([3.0, 1e4])
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Omega_HH_chao.pdf", dpi = 300)
plt.show()