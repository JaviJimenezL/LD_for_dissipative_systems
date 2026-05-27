"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid

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


# ============================================================
# Hopf normal form
# ============================================================

def rhs_aug_arclength(t, z, beta, sigma):
    x, y, s = z
    r2 = x * x + y * y
    xdot = beta * x - y - sigma * x * r2
    ydot = x + beta * y - sigma * y * r2
    speed = np.sqrt(xdot * xdot + ydot * ydot)
    return np.array([xdot, ydot, speed], dtype=float)

###############################################################################

def integrate_ld_curve(x0, y0, beta=0.0, sigma=1.0, T=8.0,
    n_eval=2000, rtol=1e-10, atol=1e-12):

    t_eval = np.linspace(0.0, T, n_eval)
    z0 = np.array([x0, y0, 0.0], dtype=float)

    sol = solve_ivp(
        rhs_aug_arclength,
        (0.0, T),
        z0,
        method="DOP853",
        t_eval=t_eval,
        args=(beta, sigma),
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y[2, :], sol.y[0, :], sol.y[1, :]

###############################################################################

def make_neighbor_ic(x0, y0, eps=1e-8, direction=(1.0, 1.0)):
    u = np.array(direction, dtype=float)
    u /= np.linalg.norm(u)
    return x0 + eps * u[0], y0 + eps * u[1]

###############################################################################

def main():

    beta = 0.5
    sigma = 1.0
    T = 10000.0

    eps = 1e-10
    direction = (1.0, 1.0)

    base_ics = [(0.5, 0.5)]

    n_eval = 1000000
    rtol = 1e-10
    atol = 1e-12

    for (x0, y0) in base_ics:

        # Integrate base orbit
        t, ld_base, x_base, y_base = integrate_ld_curve(
            x0, y0, beta, sigma, T, n_eval, rtol, atol
        )

        # Integrate neighbor orbit
        x0n, y0n = make_neighbor_ic(x0, y0, eps, direction)

        t2, ld_nei, x_nei, y_nei = integrate_ld_curve(
            x0n, y0n, beta, sigma, T, n_eval, rtol, atol
        )

        delta_ld = np.abs(ld_nei - ld_base)

        omega = ld_base / t

        # Numerical integration of Delta L over time: I(t) = ∫_0^t ΔL(τ) dτ
        integral_delta_ld = cumulative_trapezoid(delta_ld, t, initial=0.0)

        data = np.column_stack((t, delta_ld, integral_delta_ld, omega, x_base, y_base, x_nei, y_nei))
        
        np.savetxt("/home/javier/dissipative_systems/hopf_results/results_sigma_1_beta_0p5_2.txt", data)


if __name__ == "__main__":
    main()