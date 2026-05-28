"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

import numpy as np
import matplotlib.pyplot as plt

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
Load data
"""
data_1 = np.loadtxt('/home/javier/dissipative_systems/extended_Lorenz_system_results/results_4d_system_stablefixed_point.txt')
data_2 = np.loadtxt('/home/javier/dissipative_systems/extended_Lorenz_system_results/results_4d_system_stable_limit_cycle.txt')
data_3 = np.loadtxt('/home/javier/dissipative_systems/extended_Lorenz_system_results/results_4d_system_chaotic_strange_attractor.txt')
data_4 = np.loadtxt('/home/javier/dissipative_systems/extended_Lorenz_system_results/results_4d_system_hyperchaotic_attractor.txt')


"""
Delta L plot
"""
plt.figure(figsize=(8, 6))
plt.plot(data_1[:,0], data_1[:,1], label=r'Stable fixed point')
plt.plot(data_2[:,0], data_2[:,1], label=r'Stable limit cycle')
plt.plot(data_3[:,0], data_3[:,1], label=r'Chaotic strange attractor')
plt.plot(data_4[:,0], data_4[:,1], label=r'Hyperchaotic attractor')
plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Delta \mathcal{L}$')
plt.xlim([3.0, 1e4])
plt.ylim([1e-16, 1e12])
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Delta_L_lorenz_extended.pdf", dpi = 300)
plt.show()

"""
Integral of Delta L
"""
plt.figure(figsize=(8, 6))
plt.plot(data_1[:,0], data_1[:,2], label=r'Stable fixed point')
plt.plot(data_2[:,0], data_2[:,2], label=r'Stable limit cycle')
plt.plot(data_3[:,0], data_3[:,2], label=r'Chaotic strange attractor')
plt.plot(data_4[:,0], data_4[:,2], label=r'Hyperchaotic attractor')
plt.legend(loc='upper left')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\int_{0}^{\tau}\Delta \mathcal{L} \, \mathrm{d}t$')
plt.xlim([3.0, 1e4])
plt.ylim([1e-11, 1e12])
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/int_Delta_L_lorenz_extended.pdf", dpi = 300)
plt.show()

"""
Omega
"""
plt.figure(figsize=(8, 6))
plt.plot(data_1[:,0], data_1[:,3], label=r'Stable fixed point')
plt.plot(data_2[:,0], data_2[:,3], label=r'Stable limit cycle')
plt.plot(data_3[:,0], data_3[:,3], label=r'Chaotic strange attractor')
plt.plot(data_4[:,0], data_4[:,3], label=r'Hyperchaotic attractor')
# plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Omega$')
plt.xlim([3.0, 1e4])
plt.ylim([-50.0, 1500.0])
# plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Omega_lorenz_extended.pdf", dpi = 300)
plt.show()


plt.figure(figsize=(8, 6))
# plt.plot(data_1[:,0], data_1[:,3], label=r'Stable fixed point')
plt.plot(data_2[:,0], data_2[:,3], label=r'Stable limit cycle')
plt.plot(data_3[:,0], data_3[:,3], label=r'Chaotic strange attractor')
plt.plot(data_4[:,0], data_4[:,3], label=r'Hyperchaotic attractor')
# plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Omega$')
plt.xlim([3.0, 1e4])
plt.ylim([950.0, 1200.0])
# plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Omega_lorenz_extended_2.pdf", dpi = 300)
plt.show()














