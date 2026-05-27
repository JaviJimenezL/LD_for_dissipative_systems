"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

# This code simulates one trajectory in the Duffin system


import os
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


"""
Functions
"""

def rhs_aug_arclength(t, z, delta, alpha, beta, gamma, drive_omega):
    x, y, s = z

    xdot = y
    ydot = -delta * y + alpha * x - beta * x**3 + gamma * np.cos(drive_omega * t)

    speed = np.sqrt(xdot**2 + ydot**2)

    return np.array([xdot, ydot, speed], dtype=float)


###############################################################################

def integrate_ld_curve(
    x0,
    y0,
    delta=0.2,
    alpha=1.0,
    beta=1.0,
    gamma=0.3,
    drive_omega=1.2,
    t0=0.0,
    T=20.0,
    n_eval=10000,
    rtol=1e-10,
    atol=1e-12,
):
    t_eval = np.linspace(t0, t0 + T, n_eval)
    z0 = np.array([x0, y0, 0.0], dtype=float)

    sol = solve_ivp(
        rhs_aug_arclength,
        (t0, t0 + T),
        z0,
        method="DOP853",
        t_eval=t_eval,
        args=(delta, alpha, beta, gamma, drive_omega),
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    ld = sol.y[2, :]
    x = sol.y[0, :]
    y = sol.y[1, :]

    return t, ld, x, y

###############################################################################

def make_neighbor_ic(x0, y0, eps=1e-8, direction=(1.0, 1.0)):
    u = np.array(direction, dtype=float)
    u /= np.linalg.norm(u)

    x0_neighbor = x0 + eps * u[0]
    y0_neighbor = y0 + eps * u[1]

    return x0_neighbor, y0_neighbor

###############################################################################

def main():

    delta = 0.0
    alpha = 1.0
    beta = 1.0
    gamma = 0.0
    drive_omega = 1.2

    t0 = 0.0
    T = 10000.0

    eps = 1e-10
    direction = (1.0, 1.0)

    base_ics = [(0.5, 0.5)]

    n_eval = 1000000
    rtol = 1e-10
    atol = 1e-12

    output_dir = "/home/javier/dissipative_systems/duffing_results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "results_duffing_delta_0_alpha_1_beta_1_gamma_0_omega_1p2.txt"
    )

    for (x0, y0) in base_ics:

        # Integrate base orbit
        t, ld_base, x_base, y_base = integrate_ld_curve(
            x0=x0,
            y0=y0,
            delta=delta,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            drive_omega=drive_omega,
            t0=t0,
            T=T,
            n_eval=n_eval,
            rtol=rtol,
            atol=atol,
        )

        # Integrate neighboring orbit
        x0n, y0n = make_neighbor_ic(
            x0=x0,
            y0=y0,
            eps=eps,
            direction=direction,
        )

        t2, ld_nei, x_nei, y_nei = integrate_ld_curve(
            x0=x0n,
            y0=y0n,
            delta=delta,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            drive_omega=drive_omega,
            t0=t0,
            T=T,
            n_eval=n_eval,
            rtol=rtol,
            atol=atol,
        )

        if not np.allclose(t, t2):
            raise RuntimeError("The time grids of the base and neighboring orbit do not match.")

        delta_ld = np.abs(ld_nei - ld_base)

        tau = t - t[0]

        omega_ld = np.empty_like(ld_base)
        omega_ld[0] = np.nan
        omega_ld[1:] = ld_base[1:] / tau[1:]

        integral_delta_ld = cumulative_trapezoid(delta_ld, t, initial=0.0)

        data = np.column_stack((
            t,
            delta_ld,
            integral_delta_ld,
            omega_ld,
            x_base,
            y_base,
            x_nei,
            y_nei,
        ))

        header = (
            "t delta_ld integral_delta_ld omega_ld "
            "x_base y_base x_neighbor y_neighbor\n"
            f"Parameters: delta={delta}, alpha={alpha}, beta={beta}, "
            f"gamma={gamma}, drive_omega={drive_omega}, "
            f"x0={x0}, y0={y0}, eps={eps}, direction={direction}, "
            f"t0={t0}, T={T}, n_eval={n_eval}, rtol={rtol}, atol={atol}"
        )

        np.savetxt(output_file, data, header=header)

        print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()
