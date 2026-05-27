"""
Author: Javier Jiménez-López
UCM
25-5-2026
"""

# Simulation of the forced Duffing system to calculate the LD map
# both forward and backward in time.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
Functions
"""

def duffing_rhs(t, z, delta, alpha, beta_duffing, gamma, drive_omega):
    x, y = z

    xdot = y
    ydot = -delta * y + alpha * x - beta_duffing * x**3 + gamma * np.cos(drive_omega * t)

    return np.array([xdot, ydot], dtype=float)


###############################################################################

def rhs_aug(t, z, delta, alpha, beta_duffing, gamma, drive_omega, p):

    x, y, s = z

    xdot = y
    ydot = (
        -delta * y
        + alpha * x
        - beta_duffing * x**3
        + gamma * np.cos(drive_omega * t)
    )

    integrand = np.abs(xdot)**p + np.abs(ydot)**p
    # integrand = np.sqer(xdot**2 + ydot**2)

    sdot = integrand

    return np.array([xdot, ydot, sdot], dtype=float)

###############################################################################

def ld_forward_backward_one_ic(
    x0,
    y0,
    delta,
    alpha,
    beta_duffing,
    gamma,
    drive_omega,
    p,
    tau,
    t0=0.0,
    rtol=1e-8,
    atol=1e-11,
    max_step=np.inf,
):

    z0_f = np.array([x0, y0, 0.0], dtype=float)

    sol_f = solve_ivp(
        rhs_aug,
        (t0, t0 + tau),
        z0_f,
        method="DOP853",
        args=(delta, alpha, beta_duffing, gamma, drive_omega, p),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )

    Lf = sol_f.y[2, -1] if sol_f.success else np.nan

    z0_b = np.array([x0, y0, 0.0], dtype=float)

    sol_b = solve_ivp(
        rhs_aug,
        (t0, t0 - tau),
        z0_b,
        method="DOP853",
        args=(delta, alpha, beta_duffing, gamma, drive_omega, p),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )

    Lb = sol_b.y[2, -1] if sol_b.success else np.nan

    return Lf, Lb

###############################################################################

def compute_total_ld_grid(
    delta,
    alpha,
    beta_duffing,
    gamma,
    drive_omega,
    p=0.5,
    tau=8.0,
    t0=0.0,
    n_grid=301,
    xlim=(-2.0, 2.0),
    ylim=(-2.0, 2.0),
    rtol=1e-8,
    atol=1e-11,
    max_step=np.inf,
):

    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)

    X0, Y0 = np.meshgrid(x, y, indexing="xy")

    LD_tot = np.full_like(X0, np.nan, dtype=float)

    for j in range(n_grid):
        print(f"Row {j + 1}/{n_grid}")

        for i in range(n_grid):
            Lf, Lb = ld_forward_backward_one_ic(
                X0[j, i],
                Y0[j, i],
                delta=delta,
                alpha=alpha,
                beta_duffing=beta_duffing,
                gamma=gamma,
                drive_omega=drive_omega,
                p=p,
                tau=tau,
                t0=t0,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )

            LD_tot[j, i] = Lf + Lb

    return X0, Y0, LD_tot

###############################################################################

def normalize_minmax(A):
    amin = np.nanmin(A)
    amax = np.nanmax(A)

    if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin) == 0.0:
        return A * np.nan

    return (A - amin) / (amax - amin)

###############################################################################

def main():


    delta = 0.0
    alpha = 1.0
    beta_duffing = 1.0
    drive_omega = 1.2

    gamma_values = [0.0, 0.0, 0.0]
    delta_values = [0.0, 0.3, 0.7]

    p = 0.5
    tau = 8.0
    t0 = 0.0

    xlim = (-2.0, 2.0)
    ylim = (-2.0, 2.0)

    n_grid = 500

    rtol = 1e-8
    atol = 1e-11

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.5, 4.6),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.1},
    )

    for ax, gamma, delta in zip(axes, gamma_values, delta_values):

        print("Starting a new Duffing LD simulation...")
        print(f"gamma = {gamma}")

        X0, Y0, LD = compute_total_ld_grid(
            delta=delta,
            alpha=alpha,
            beta_duffing=beta_duffing,
            gamma=gamma,
            drive_omega=drive_omega,
            p=p,
            tau=tau,
            t0=t0,
            n_grid=n_grid,
            xlim=xlim,
            ylim=ylim,
            rtol=rtol,
            atol=atol,
        )

        LDn = normalize_minmax(LD)

        im = ax.imshow(
            LDn,
            origin="lower",
            extent=[X0.min(), X0.max(), Y0.min(), Y0.max()],
            aspect="equal",
            interpolation="nearest",
        )

        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$y_0$", rotation=0)


        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(
        "/home/javier/dissipative_systems/figures/duffing_LD_maps_p_0p5.pdf",
        dpi=300,
    )

    plt.show()


###############################################################################

if __name__ == "__main__":
    main()
