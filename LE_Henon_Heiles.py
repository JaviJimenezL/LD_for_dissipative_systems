"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

# Lyapunov spectrum for the Hénon-Heiles system

import numpy as np


"""
Functions
"""
def field(Y):
    x, y, px, py = Y
    return np.array([
        px,
        py,
        -x - 2.0 * x * y,
        -y - x * x + y * y,
    ])


def jacobian(Y):
    x, y, px, py = Y
    return np.array([
        [0.0,         0.0,        1.0, 0.0],
        [0.0,         0.0,        0.0, 1.0],
        [-1.0 - 2*y, -2.0 * x,    0.0, 0.0],
        [-2.0 * x,   -1.0 + 2*y,  0.0, 0.0],
    ])

###############################################################################

def energy(Y):
    x, y, px, py = Y
    return 0.5 * (px**2 + py**2) + 0.5 * (x**2 + y**2) + x**2 * y - y**3 / 3.0

###############################################################################

def section_ic(y, H=1.0/8.0):
    px = np.sqrt(2.0 * (H - 0.5 * y**2 + y**3 / 3.0))
    return np.array([0.0, y, px, 0.0])

###############################################################################

def _rk4_step(Y, Q, dt):
    def deriv(Y, Q):
        return field(Y), jacobian(Y) @ Q

    k1y, k1Q = deriv(Y, Q)
    k2y, k2Q = deriv(Y + 0.5 * dt * k1y, Q + 0.5 * dt * k1Q)
    k3y, k3Q = deriv(Y + 0.5 * dt * k2y, Q + 0.5 * dt * k2Q)
    k4y, k4Q = deriv(Y + dt * k3y,       Q + dt * k3Q)
    Y_new = Y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
    Q_new = Q + dt / 6.0 * (k1Q + 2 * k2Q + 2 * k3Q + k4Q)
    return Y_new, Q_new

###############################################################################

def lyapunov_spectrum(
    Y0,
    dt=0.01,           
    renorm_every=10,    
    T=10000.0,         
    T_transient=0.0,    
    history=False,     
):
    n = 4
    Y = np.asarray(Y0, dtype=float)
    E0 = energy(Y)

    for _ in range(int(T_transient / dt)):
        Y, _ = _rk4_step(Y, np.eye(n), dt)

    Q = np.eye(n)
    log_sum = np.zeros(n)
    n_steps = int(T / dt)
    t = 0.0
    max_dE = 0.0
    hist_t, hist_l1 = [], []

    for i in range(1, n_steps + 1):
        Y, Q = _rk4_step(Y, Q, dt)
        if i % renorm_every == 0:
            Q, R = np.linalg.qr(Q)
            sign = np.sign(np.diag(R))
            sign[sign == 0] = 1.0
            Q *= sign
            log_sum += np.log(np.abs(np.diag(R)) + 1e-300)
            t = i * dt
            max_dE = max(max_dE, abs(energy(Y) - E0))
            if history:
                hist_t.append(t)
                hist_l1.append(log_sum[0] / t)

    chi = np.sort(log_sum / t)[::-1]
    info = {
        "E0": E0,
        "max_energy_drift": max_dE,
        "pairing_residual": float(np.max(np.abs(chi + chi[::-1]))),  # |chi_i + chi_{n+1-i}|
        "history": (np.array(hist_t), np.array(hist_l1)) if history else None,
    }
    return chi, info

###############################################################################

def is_chaotic(chi, tol=1e-2):
    return chi[0] > tol

###############################################################################

if __name__ == "__main__":
    H = 1.0 / 8.0
    cases = [
        ("regular", section_ic(y=0.200,  H=H)),
        ("chaotic", section_ic(y=-0.175, H=H)),
    ]

    print(f"Henon-Heiles, H = {H:.6f}\n")
    for name, Y0 in cases:
        chi, info = lyapunov_spectrum(Y0, dt=0.01, renorm_every=10, T=10000.0)
        print(f"{name} orbit, IC = {np.round(Y0, 6)}")
        print(f"    chi = {np.round(chi, 5)}")
        print(f"    chi_1 (mLE)      = {chi[0]:.5f}   -> "
              f"{'CHAOTIC' if is_chaotic(chi) else 'regular'}")
        print(f"    sum chi          = {chi.sum():.2e}   (must be ~0: Hamiltonian)")
        print(f"    pairing residual = {info['pairing_residual']:.2e}   "
              f"(|chi_i + chi_(5-i)|, must be ~0)")
        print(f"    max energy drift = {info['max_energy_drift']:.2e}\n")