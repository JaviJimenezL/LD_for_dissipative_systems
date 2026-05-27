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

data_1 = np.loadtxt('/home/javier/dissipative_systems/duffing_results/results_duffing_delta_0_alpha_1_beta_1_gamma_0_omega_1p2.txt')
data_2 = np.loadtxt('/home/javier/dissipative_systems/duffing_results/results_duffing_delta_0p3_alpha_1_beta_1_gamma_0_omega_1p2.txt')
data_3 = np.loadtxt('/home/javier/dissipative_systems/duffing_results/results_duffing_delta_0p7_alpha_1_beta_1_gamma_0_omega_1p2.txt')
data_4 = np.loadtxt('/home/javier/dissipative_systems/duffing_results/results_duffing_delta_1p0_alpha_1_beta_1_gamma_0_omega_1p2.txt')
data_5 = np.loadtxt('/home/javier/dissipative_systems/duffing_results/results_duffing_delta_1p5_alpha_1_beta_1_gamma_0_omega_1p2.txt')
data_6 = np.loadtxt('/home/javier/dissipative_systems/duffing_results/results_duffing_delta_2p5_alpha_1_beta_1_gamma_0_omega_1p2.txt')


"""
Delta L plot
"""
plt.figure(figsize=(8, 6))
plt.plot(data_1[:,0], data_1[:,1], label=r'$\delta = 0$')
plt.plot(data_2[:,0], data_2[:,1], label=r'$\delta = 0.3$')
plt.plot(data_3[:,0], data_3[:,1], label=r'$\delta = 0.7$')
plt.plot(data_4[:,0], data_4[:,1], label=r'$\delta = 1$')
plt.plot(data_5[:,0], data_5[:,1], label=r'$\delta = 1.5$')
plt.plot(data_6[:,0], data_6[:,1], label=r'$\delta = 2.5$')
plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Delta \mathcal{L}$')
plt.ylim([1e-15, 1])
plt.xlim([3.0, 1e4])
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Delta_L_Duffin.pdf", dpi = 300)
plt.show()

"""
Integral of Delta L
"""
plt.figure(figsize=(8, 6))
plt.plot(data_1[:,0], data_1[:,2], label=r'$\delta = 0$')
plt.plot(data_2[:,0], data_2[:,2], label=r'$\delta = 0.3$')
plt.plot(data_3[:,0], data_3[:,2], label=r'$\delta = 0.7$')
plt.plot(data_4[:,0], data_4[:,2], label=r'$\delta = 1$')
plt.plot(data_5[:,0], data_5[:,2], label=r'$\delta = 1.5$')
plt.plot(data_6[:,0], data_6[:,2], label=r'$\delta = 2.5$')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\int_{0}^{\tau}\Delta \mathcal{L} \, \mathrm{d}t$')
plt.xlim([3.0, 1e4])
# plt.ylim([1e-15, 1e-1])
# plt.yscale('log')
# plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/int_Delta_L_Duffin.pdf", dpi = 300)
plt.show()

"""
º
"""
plt.figure(figsize=(8, 6))
plt.plot(data_1[:,0], data_1[:,3], label=r'$\delta = 0$')
plt.plot(data_2[:,0], data_2[:,3], label=r'$\delta = 0.3$')
plt.plot(data_3[:,0], data_3[:,3], label=r'$\delta = 0.7$')
plt.plot(data_4[:,0], data_4[:,3], label=r'$\delta = 1$')
plt.plot(data_5[:,0], data_5[:,3], label=r'$\delta = 1.5$')
plt.plot(data_6[:,0], data_6[:,3], label=r'$\delta = 2.5$')
# plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Omega$')
plt.xlim([3.0, 1e4])
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Omega_Duffin.pdf", dpi = 300)
plt.show()

"""
Trajectory
"""
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
"""













