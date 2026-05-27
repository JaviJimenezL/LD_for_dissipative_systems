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

data = np.loadtxt('/home/javier/dissipative_systems/chirped_duffing_results/results_chirped_duffing_delta_0_alpha_1_beta_1_gamma_0p3_omega0_1p2_kappa_1em4.txt')


"""
Delta L plot
"""
plt.figure(figsize=(8, 6))
plt.plot(data[:,0], data[:,1], label=r'$\sigma = 1$ and $\beta = 0$')
plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Delta \mathcal{L}$')
plt.xlim([3.0, 1e4])
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
# plt.savefig("/home/javier/dissipative_systems/figures/Delta_L_hopf.pdf", dpi = 300)
plt.show()

"""
Integral of Delta L
"""
plt.figure(figsize=(8, 6))
plt.plot(data[:,0], data[:,2], label=r'$\sigma = 1$ and $\beta = 0$')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\int_{0}^{\tau}\Delta \mathcal{L} \, \mathrm{d}t$')
plt.xlim([3.0, 1e4])
# plt.ylim([1e-11, 1e-4])
# plt.yscale('log')
# plt.xscale('log')
plt.tight_layout()
# plt.savefig("/home/javier/dissipative_systems/figures/int_Delta_L_hopf.pdf", dpi = 300)
plt.show()

"""
º
"""
plt.figure(figsize=(8, 6))
plt.plot(data[:,0], data[:,3], label=r'$\sigma = 1$ and $\beta = 0$')
# plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Omega$')
plt.xlim([3.0, 1e4])
# plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
# plt.savefig("/home/javier/dissipative_systems/figures/Omega_hopf.pdf", dpi = 300)
plt.show()

"""
Trajectory
"""
plt.figure(figsize=(8, 6))
plt.plot(data[:,4], data[:,5], label=r'$\sigma = 1$ and $\beta = 0$')
plt.scatter(0.5, 0.5, marker='o', color='magenta', label='Initial condition')
plt.xlim([-2.0, 2.25])
plt.ylim([-1.5, 1.55])
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
plt.tight_layout()
# plt.savefig("/home/javier/dissipative_systems/figures/trajectories_hopf.pdf", dpi = 300)
plt.show()















