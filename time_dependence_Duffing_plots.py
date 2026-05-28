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

data_1 = np.loadtxt('/home/javier/dissipative_systems/duffing_results/results_duffing_delta_0_alpha_1_beta_1_gamma_0p5_omega_1p0.txt')
data_2 = np.loadtxt('/home/javier/dissipative_systems/duffing_results/results_duffing_delta_1_alpha_1_beta_1_gamma_0p5_omega_1p0.txt')
data_3 = np.loadtxt('/home/javier/dissipative_systems/quasiperiodic_duffing_results/results_quasiperiodic_duffing_delta_0_alpha_1_beta_1_gamma1_0p5_gamma2_0p5_omega1_1_omega2_sqrt2.txt')
data_4 = np.loadtxt('/home/javier/dissipative_systems/quasiperiodic_duffing_results/results_quasiperiodic_duffing_delta_1_alpha_1_beta_1_gamma1_0p5_gamma2_0p5_omega1_1_omega2_sqrt2.txt')
data_5 = np.loadtxt('/home/javier/dissipative_systems/chirped_duffing_results/results_chirped_duffing_delta_0_alpha_1_beta_1_gamma_0p5_omega0_1_kappa_1em4.txt')
data_6 = np.loadtxt('/home/javier/dissipative_systems/chirped_duffing_results/results_chirped_duffing_delta_1_alpha_1_beta_1_gamma_0p5_omega0_1_kappa_1em4.txt')


"""
Delta L plot
"""
plt.figure(figsize=(8, 6))
plt.plot(data_1[:,0], data_1[:,1], label=r'$\delta = 0$ and $\omega = 1$')
plt.plot(data_2[:,0], data_2[:,1], label=r'$\delta = 1$ and $\omega = 1$')
plt.plot(data_3[:,0], data_3[:,1], label=r'$\delta = 0$, $\omega_1 = 1$ and $\omega_2 = \sqrt{2}$')
plt.plot(data_4[:,0], data_4[:,1], label=r'$\delta = 1$, $\omega_1 = 1$ and $\omega_2 = \sqrt{2}$')
plt.plot(data_5[:,0], data_5[:,1], label=r'$\delta = 0$, $\omega_0 = 1$ and $\kappa = 10^{-4}$')
plt.plot(data_6[:,0], data_6[:,1], label=r'$\delta = 1$, $\omega_0 = 1$ and $\kappa = 10^{-4}$')
plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Delta \mathcal{L}$')
plt.ylim([1e-17, 1e8])
plt.xlim([0.01, 1e4])
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Delta_L_Duffing_time_dependence.pdf", dpi = 300)
plt.show()

"""
Integral of Delta L
"""
plt.figure(figsize=(8, 6))
plt.plot(data_1[:,0], data_1[:,2], label=r'$\delta = 0$ and $\omega = 1$')
plt.plot(data_2[:,0], data_2[:,2], label=r'$\delta = 1$ and $\omega = 1$')
plt.plot(data_3[:,0], data_3[:,2], label=r'$\delta = 0$, $\omega_1 = 1$ and $\omega_2 = \sqrt{2}$')
plt.plot(data_4[:,0], data_4[:,2], label=r'$\delta = 1$, $\omega_1 = 1$ and $\omega_2 = \sqrt{2}$')
plt.plot(data_5[:,0], data_5[:,2], label=r'$\delta = 0$, $\omega_0 = 1$ and $\kappa = 10^{-4}$')
plt.plot(data_6[:,0], data_6[:,2], label=r'$\delta = 1$, $\omega_0 = 1$ and $\kappa = 10^{-4}$')
plt.legend(loc='upper left')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\int_{0}^{\tau}\Delta \mathcal{L} \, \mathrm{d}t$')
plt.xlim([0.01, 1e4])
# plt.ylim([1e-15, 1e-1])
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/int_Delta_L_Duffing_time_dependence.pdf", dpi = 300)
plt.show()

"""
Omega
"""
plt.figure(figsize=(8, 6))
plt.plot(data_1[:,0], data_1[:,3], label=r'$\delta = 0$ and $\omega = 1$')
plt.plot(data_2[:,0], data_2[:,3], label=r'$\delta = 1$ and  $\omega = 1$')
plt.plot(data_3[:,0], data_3[:,3], label=r'$\delta = 0$, $\omega_1 = 1$ and $\omega_2 = \sqrt{2}$')
plt.plot(data_4[:,0], data_4[:,3], label=r'$\delta = 1$, $\omega_1 = 1$ and $\omega_2 = \sqrt{2}$')
plt.plot(data_5[:,0], data_5[:,3], label=r'$\delta = 0$, $\omega_0 = 1$ and $\kappa = 10^{-4}$')
plt.plot(data_6[:,0], data_6[:,3], label=r'$\delta = 1$, $\omega_0 = 1$ and $\kappa = 10^{-4}$')
# plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Omega$')
plt.xlim([0.01, 1e4])
# plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Omega_Duffing_time_dependence.pdf", dpi = 300)
plt.show()



"""
Omega for dissipative
"""
plt.figure(figsize=(8, 6))
plt.plot(data_2[:,0], data_2[:,3], label=r'$\delta = 1$, $\omega = 1$')
plt.plot(data_4[:,0], data_4[:,3], label=r'$\delta = 1$, $\omega_1 = 1$ and $\omega_2 = \sqrt{2}$')
plt.plot(data_6[:,0], data_6[:,3], label=r'$\delta = 1$, $\omega_0 = 1$ and $\kappa = 10^{-4}$')
# plt.axhline(y = 1e-10, linestyle='-', linewidth=2, color='red', label='Initial separation')
plt.legend()
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\Omega$')
plt.xlim([0.01, 1e4])
plt.ylim([0.4, 1.5])
plt.yscale('log')
plt.xscale('log')
plt.tight_layout()
plt.savefig("/home/javier/dissipative_systems/figures/Omega_Duffing_time_dependence_dissipative.pdf", dpi = 300)
plt.show()

"""
Trajectory
"""

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.5, 4.6),
    constrained_layout=True,
    gridspec_kw={"wspace": 0.1},
)

datasets = [data_2, data_4, data_6]


labels = [
    r"$\delta = 1$ and $\omega = 1$",
    r"$\delta = 1$, $\omega_1 = 1$ and $\omega_2 = \sqrt{2}$",
    r"$\delta = 1$, $\omega_0 = 1$ and $\kappa = 10^{-4}$",
]

xlim = (-2.0, 2.0)
ylim = (-1.5, 1.5)

x0, y0 = 0.5, 0.5

for ax, data, label in zip(axes, datasets, labels):

    ax.plot(
        data[:, 4],
        data[:, 5],
        lw=1.0,
        label=label,
    )

    ax.scatter(
        x0,
        y0,
        marker="o",
        color="magenta",
        s=45,
        label="Initial condition",
        zorder=5,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$", rotation=0)


    ax.legend(fontsize=12)


plt.savefig(
    "/home/javier/dissipative_systems/figures/trajectories_duffing_time_dependence_comparison.pdf",
    dpi=300,
)


plt.show()














