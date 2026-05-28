"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

# This code simulates one trajectory in the 4D system

import os
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid


"""
Functions
"""

def rhs_aug_arclength(t, state, a, b, c, r):
    x, y, z, w, s = state

    xdot = a * (y - x) + y * z
    ydot = c * x - y - x * z + w
    zdot = x * y - b * z
    wdot = -x * z + r * w

    speed = np.sqrt(xdot**2 + ydot**2 + zdot**2 + wdot**2)

    return np.array([xdot, ydot, zdot, wdot, speed], dtype=float)

###############################################################################

def integrate_ld_curve(
    x0,
    y0,
    z0,
    w0,
    a=10.0,
    b=8.0 / 3.0,
    c=28.0,
    r=1.0,
    t0=0.0,
    T=20.0,
    n_eval=10000,
    rtol=1e-10,
    atol=1e-12,
):
    t_eval = np.linspace(t0, t0 + T, n_eval)

    initial_state = np.array([x0, y0, z0, w0, 0.0], dtype=float)

    sol = solve_ivp(
        rhs_aug_arclength,
        (t0, t0 + T),
        initial_state,
        method="DOP853",
        t_eval=t_eval,
        args=(a, b, c, r),
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    t = sol.t
    ld = sol.y[4, :]

    x = sol.y[0, :]
    y = sol.y[1, :]
    z = sol.y[2, :]
    w = sol.y[3, :]

    return t, ld, x, y, z, w

###############################################################################

def make_neighbor_ic(x0, y0, z0, w0, eps=1e-8, direction=(1.0, 1.0, 1.0, 1.0)):
    direction_vector = np.array(direction, dtype=float)
    direction_vector /= np.linalg.norm(direction_vector)

    x0_neighbor = x0 + eps * direction_vector[0]
    y0_neighbor = y0 + eps * direction_vector[1]
    z0_neighbor = z0 + eps * direction_vector[2]
    w0_neighbor = w0 + eps * direction_vector[3]

    return x0_neighbor, y0_neighbor, z0_neighbor, w0_neighbor

###############################################################################

def main():

    # System parameters
    a = 35.0
    b = 8.0 / 3.0
    c = 55.0
    r = 1.5

    t0 = 0.0
    T = 10000.0

    eps = 1e-10
    direction = (1.0, 1.0, 1.0, 1.0)

    base_ics = [
        (3.0, 2.0, 10.0, 1.0),
    ]

    n_eval = 1000000
    rtol = 1e-10
    atol = 1e-12

    output_dir = "/home/javier/dissipative_systems/extended_Lorenz_system_results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "results_4d_system_hyperchaotic_attractor.txt"
    )

    for (x0, y0, z0, w0) in base_ics:

        t, ld_base, x_base, y_base, z_base, w_base = integrate_ld_curve(
            x0=x0,
            y0=y0,
            z0=z0,
            w0=w0,
            a=a,
            b=b,
            c=c,
            r=r,
            t0=t0,
            T=T,
            n_eval=n_eval,
            rtol=rtol,
            atol=atol,
        )

        x0n, y0n, z0n, w0n = make_neighbor_ic(
            x0=x0,
            y0=y0,
            z0=z0,
            w0=w0,
            eps=eps,
            direction=direction,
        )

        t2, ld_nei, x_nei, y_nei, z_nei, w_nei = integrate_ld_curve(
            x0=x0n,
            y0=y0n,
            z0=z0n,
            w0=w0n,
            a=a,
            b=b,
            c=c,
            r=r,
            t0=t0,
            T=T,
            n_eval=n_eval,
            rtol=rtol,
            atol=atol,
        )

        if not np.allclose(t, t2):
            raise RuntimeError("The time grids of the reference and neighboring orbit do not match.")

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
            z_base,
            w_base,
            x_nei,
            y_nei,
            z_nei,
            w_nei,
        ))

        header = (
            "t delta_ld integral_delta_ld omega_ld "
            "x_base y_base z_base w_base "
            "x_neighbor y_neighbor z_neighbor w_neighbor\n"
            f"Parameters: a={a}, b={b}, c={c}, r={r}, "
            f"x0={x0}, y0={y0}, z0={z0}, w0={w0}, "
            f"eps={eps}, direction={direction}, "
            f"t0={t0}, T={T}, n_eval={n_eval}, rtol={rtol}, atol={atol}"
        )

        np.savetxt(output_file, data, header=header)

        print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()