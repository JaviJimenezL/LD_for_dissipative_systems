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

data_sigma_1_beta_0 = np.loadtxt('/home/javier/dissipative_systems/hopf_results/sigma_1_beta_0/results_sigma_1_beta_0.txt')
data_sigma_1_beta_0p5 = np.loadtxt('/home/javier/dissipative_systems/hopf_results/sigma_1_beta_0p5/results_sigma_1_beta_0p5.txt')
data_sigma_1_beta_0p5_2 = np.loadtxt('/home/javier/dissipative_systems/hopf_results/sigma_1_beta_0p5/results_sigma_1_beta_0p5_2.txt')
data_sigma_1_beta_m0p5 = np.loadtxt('/home/javier/dissipative_systems/hopf_results/sigma_1_beta_m0p5/results_sigma_1_beta_m0p5.txt')


"""
Delta L plot
"""
plt.figure(figsize=(8, 6))
plt.plot(data_sigma_1_beta_0[:,0], data_sigma_1_beta_0[:,1], label=r'$\sigma = 1$ and $\beta = 0$')
plt.plot(data_sigma_1_beta_0p5[:,0], data_sigma_1_beta_0p5[:,1], label=r'$\sigma = 1$ and $\beta = 0.5$')
plt.plot(data_sigma_1_beta_0p5_2[:,0], data_sigma_1_beta_0p5_2[:,1], label=r'$\sigma = 1$ and $\beta = 0.5$ (2)')
plt.plot(data_sigma_1_beta_m0p5[:,0], data_sigma_1_beta_m0p5[:,1], label=r'$\sigma = 1$ and $\beta = -0.5$')
plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Delta \mathcal{L}$')
plt.xlim([3.0, 1e4])
# plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
# plt.savefig("/home/javier/dissipative_systems/figures/Delta_L_hopf.pdf", dpi = 300)
plt.show()

"""
Integral of Delta L
"""
plt.figure(figsize=(8, 6))
plt.plot(data_sigma_1_beta_0[:,0], data_sigma_1_beta_0[:,2], label=r'$\sigma = 1$ and $\beta = 0$')
plt.plot(data_sigma_1_beta_0p5[:,0], data_sigma_1_beta_0p5[:,2], label=r'$\sigma = 1$ and $\beta = 0.5$')
plt.plot(data_sigma_1_beta_0p5_2[:,0], data_sigma_1_beta_0p5_2[:,2], label=r'$\sigma = 1$ and $\beta = 0.5$ (2)')
plt.plot(data_sigma_1_beta_m0p5[:,0], data_sigma_1_beta_m0p5[:,2], label=r'$\sigma = 1$ and $\beta = -0.5$')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\int_{0}^{\tau}\Delta \mathcal{L} \, \mathrm{d}t$')
plt.xlim([3.0, 1e4])
plt.ylim([1e-11, 1e-4])
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
# plt.savefig("/home/javier/dissipative_systems/figures/int_Delta_L_hopf.pdf", dpi = 300)
plt.show()

"""
Omega
"""
plt.figure(figsize=(8, 6))
plt.plot(data_sigma_1_beta_0[:,0], data_sigma_1_beta_0[:,3], label=r'$\sigma = 1$ and $\beta = 0$')
plt.plot(data_sigma_1_beta_0p5[:,0], data_sigma_1_beta_0p5[:,3], label=r'$\sigma = 1$ and $\beta = 0.5$')
plt.plot(data_sigma_1_beta_0p5_2[:,0], data_sigma_1_beta_0p5_2[:,3], label=r'$\sigma = 1$ and $\beta = 0.5$ (2)')
plt.plot(data_sigma_1_beta_m0p5[:,0], data_sigma_1_beta_m0p5[:,3], label=r'$\sigma = 1$ and $\beta = -0.5$')
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
plt.plot(data_sigma_1_beta_0[:,4], data_sigma_1_beta_0[:,5], label=r'$\sigma = 1$ and $\beta = 0$')
plt.plot(data_sigma_1_beta_0p5[:,4], data_sigma_1_beta_0p5[:,5], label=r'$\sigma = 1$ and $\beta = 0.5$')
# plt.plot(data_sigma_1_beta_0p5_2[:,4], data_sigma_1_beta_0p5_2[:,5], label=r'$\sigma = 1$ and $\beta = 0.5$ (2)')
plt.plot(data_sigma_1_beta_m0p5[:,4], data_sigma_1_beta_m0p5[:,5], label=r'$\sigma = 1$ and $\beta = -0.5$')
plt.scatter(0.5, 0.1, marker='o', color='magenta', label='Initial condition')
plt.xlim([-1.0, 2.25])
plt.ylim([-1.25, 1.25])
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$', rotation=0)
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/trajectories_hopf.pdf", dpi = 300)
plt.show()













