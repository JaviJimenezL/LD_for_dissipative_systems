"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

# Lyapunov spectrum of the periodically forced Duffing oscillator.


import numpy as np


"""
Functions
"""
def field(t, s, delta, alpha, beta, gamma, omega):
    x, y = s
    return np.array([
        y,
        -delta * y + alpha * x - beta * x**3 + gamma * np.cos(omega * t),
    ])


def jacobian(s, delta, alpha, beta, gamma, omega):
    x, _ = s
    return np.array([
        [0.0,                  1.0],
        [alpha - 3*beta*x*x,  -delta],
    ])

###############################################################################

def _rk4_step(t, s, Q, dt, p):
    def deriv(t, s, Q):
        return field(t, s, *p), jacobian(s, *p) @ Q

    k1s, k1Q = deriv(t,          s,                Q)
    k2s, k2Q = deriv(t + 0.5*dt, s + 0.5*dt*k1s,   Q + 0.5*dt*k1Q)
    k3s, k3Q = deriv(t + 0.5*dt, s + 0.5*dt*k2s,   Q + 0.5*dt*k2Q)
    k4s, k4Q = deriv(t + dt,     s + dt*k3s,       Q + dt*k3Q)
    s_new = s + dt/6.0 * (k1s + 2*k2s + 2*k3s + k4s)
    Q_new = Q + dt/6.0 * (k1Q + 2*k2Q + 2*k3Q + k4Q)
    return s_new, Q_new

###############################################################################

def lyapunov_spectrum(
    s0,
    delta, alpha, beta, gamma, omega,
    t0=0.0,
    dt=0.005,
    renorm_every=20,    
    T_transient=300.0,  
    T=10000.0,          
):
    p = (delta, alpha, beta, gamma, omega)
    n = 2
    s = np.asarray(s0, dtype=float)
    t = t0

    for _ in range(int(T_transient / dt)):
        s, _ = _rk4_step(t, s, np.eye(n), dt, p)
        t += dt

    Q = np.eye(n)
    log_sum = np.zeros(n)
    t_start = t
    n_steps = int(T / dt)
    for i in range(1, n_steps + 1):
        s, Q = _rk4_step(t, s, Q, dt, p)
        t += dt
        if i % renorm_every == 0:
            Q, R = np.linalg.qr(Q)
            sign = np.sign(np.diag(R))
            sign[sign == 0] = 1.0
            Q *= sign
            log_sum += np.log(np.abs(np.diag(R)) + 1e-300)

    return np.sort(log_sum / (t - t_start))[::-1]

###############################################################################

def classify(chi, tol=1e-2):
    if chi[0] > tol:
        return "chaotic (strange) attractor"
    if np.abs(chi[0]) <= tol:
        return "quasiperiodic (torus)"
    return "phase-locked periodic attractor"



if __name__ == "__main__":
    IC = (0.5, 0.5)
    # (delta, alpha, beta, gamma, omega)
    cases = [
        ("main_1        ", (0.0, 1.0, 1.0, 0.0, 1.0)),
        ("main_2        ", (0.3, 1.0, 1.0, 0.0, 1.0)),
        ("main_3        ", (0.7, 1.0, 1.0, 0.0, 1.0)),
        ("main_4        ", (1.0, 1.0, 1.0, 0.0, 1.0)),
        ("main_5        ", (1.5, 1.0, 1.0, 0.0, 1.0)),
        ("main_6        ", (2.5, 1.0, 1.0, 0.0, 1.0)),
        ("main_7        ", (0.0, 1.0, 1.0, 0.5, 1.0)),
        ("main_8        ", (1.0, 1.0, 1.0, 0.5, 1.0)),
    ]

    for name, (delta, alpha, beta, gamma, omega) in cases:
        chi = lyapunov_spectrum(IC, delta, alpha, beta, gamma, omega)
        print(f"{name}  delta={delta}, gamma={gamma}, omega={omega}, IC={IC}")
        print(f"    chi (nontrivial) = {np.round(chi, 4)}   "
              f"(+ implicit phase exponent 0)")
        print(f"    sum = {chi.sum():+.5f}   (must equal -delta = {-delta:+.2f})")
        print(f"    verdict: {classify(chi)}\n")
