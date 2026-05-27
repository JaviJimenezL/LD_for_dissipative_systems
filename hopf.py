"""
Author: Javier Jiménez-López
UCM
25-5-2026
"""

# Simulation of the Hopf system to calculate the map of the LD both forward and backward

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
Define functions
"""

def hopf_rhs(t, z, beta, sigma):
    x, y = z
    r2 = x * x + y * y
    xdot = beta * x - y - sigma * x * r2
    ydot = x + beta * y - sigma * y * r2
    return np.array([xdot, ydot], dtype=float)

###############################################################################

def rhs_aug(t, z, beta, sigma, p):
    x, y, s = z
    r2 = x * x + y * y
    xdot = beta * x - y - sigma * x * r2
    ydot = x + beta * y - sigma * y * r2
    integrand = np.abs(xdot) ** p + np.abs(ydot) ** p
    # integrand = np.sqrt(xdot ** 2 + ydot ** 2)
    sdot = integrand
    return np.array([xdot, ydot, sdot], dtype=float)

###############################################################################

def make_escape_event(R):
    """
    Stop integration when sqrt(x^2+y^2) reaches R.
    """
    def event(t, z, beta, sigma, p):
        x, y = z[0], z[1]
        return np.sqrt(x * x + y * y) - R

    event.terminal = True
    event.direction = 1.0
    return event

###############################################################################

def ld_forward_backward_one_ic(
    x0, y0, beta, sigma, p, tau, R,
    rtol=1e-8, atol=1e-11, max_step=np.inf
):
    """
    Compute forward and backward LD contributions.
    """
    escape_event = make_escape_event(R)

    # Forward: t from 0 to +tau
    z0_f = np.array([x0, y0, 0.0], dtype=float)
    sol_f = solve_ivp(
        rhs_aug, (0.0, tau), z0_f,
        method="DOP853",
        args=(beta, sigma, p),
        events=escape_event,
        rtol=rtol, atol=atol, max_step=max_step,
        dense_output=False
    )
    Lf = sol_f.y[2, -1] if sol_f.success else np.nan

    # Backward: t from 0 to -tau
    z0_b = np.array([x0, y0, 0.0], dtype=float)
    sol_b = solve_ivp(
        rhs_aug, (0.0, -tau), z0_b,
        method="DOP853",
        args=(beta, sigma, p),
        events=escape_event,
        rtol=rtol, atol=atol, max_step=max_step,
        dense_output=False
    )
    Lb = sol_b.y[2, -1] if sol_b.success else np.nan

    return Lf, Lb

###############################################################################

def compute_total_ld_grid(
    beta, sigma=1.0, p=0.5, tau=8.0, R=4.0,
    n_grid=301, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
    rtol=1e-8, atol=1e-11, max_step=np.inf
):
    """
    Compute total LD (forward + backward) on a grid for a single beta value.
    """
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X0, Y0 = np.meshgrid(x, y, indexing="xy")

    LD_tot = np.full_like(X0, np.nan, dtype=float)

    for j in range(n_grid):
        for i in range(n_grid):
            Lf, Lb = ld_forward_backward_one_ic(
                X0[j, i], Y0[j, i],
                beta=beta, sigma=sigma, p=p,
                tau=tau, R=R,
                rtol=rtol, atol=atol, max_step=max_step
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

    sigma = 1.0
    p = 0.5
    tau = 8.0
    R = 4.0

    betas = [-0.5, 0.0, 0.5]

    xlim = (-1.5, 1.5)
    ylim = (-1.5, 1.5)


    n_grid = 500

    rtol = 1e-8
    atol = 1e-11

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True, 
                             gridspec_kw={"wspace": 0.1})

    for ax, beta in zip(axes, betas):
        print("Starting a new simulation...")
        X0, Y0, LD = compute_total_ld_grid(
            beta=beta, sigma=sigma, p=p, tau=tau, R=R,
            n_grid=n_grid, xlim=xlim, ylim=ylim,
            rtol=rtol, atol=atol
        )

        LDn = normalize_minmax(LD)

        im = ax.imshow(
            LDn,
            origin="lower",
            extent=[X0.min(), X0.max(), Y0.min(), Y0.max()],
            aspect="equal",
            interpolation="nearest",
        )
        # ax.set_title(rf"$\beta={beta}$, $\sigma={sigma}$, $p={p}$, $\tau={tau}$")
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$y_0$", rotation = 0)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label("Normalized total LD")
          
    plt.savefig("/home/javier/dissipative_systems/figures/hopf_LD_maps_p_0p5.pdf", dpi = 300)
    plt.show()


###############################################################################

if __name__ == "__main__":
    main()
























