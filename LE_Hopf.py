"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

# Lyapunov spectrum of the Hopf normal form.


import numpy as np

"""
Functions
"""
def field(s, beta, sigma):
    x, y = s
    r2 = x * x + y * y
    return np.array([
        beta * x - y - sigma * x * r2,
        x + beta * y - sigma * y * r2,
    ])


def jacobian(s, beta, sigma):
    x, y = s
    return np.array([
        [beta - sigma * (3 * x * x + y * y),  -1.0 - 2 * sigma * x * y],
        [1.0 - 2 * sigma * x * y,              beta - sigma * (x * x + 3 * y * y)],
    ])

###############################################################################

def _rk4_step(s, Q, dt, beta, sigma):
    def deriv(s, Q):
        return field(s, beta, sigma), jacobian(s, beta, sigma) @ Q

    k1s, k1Q = deriv(s, Q)
    k2s, k2Q = deriv(s + 0.5 * dt * k1s, Q + 0.5 * dt * k1Q)
    k3s, k3Q = deriv(s + 0.5 * dt * k2s, Q + 0.5 * dt * k2Q)
    k4s, k4Q = deriv(s + dt * k3s,       Q + dt * k3Q)
    s_new = s + dt / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s)
    Q_new = Q + dt / 6.0 * (k1Q + 2 * k2Q + 2 * k3Q + k4Q)
    return s_new, Q_new

###############################################################################

def lyapunov_spectrum(
    s0,
    beta,
    sigma,
    dt=0.005,          
    renorm_every=20,   
    T_transient=200.0, 
    T=10000.0,          
):
    n = 2
    s = np.asarray(s0, dtype=float)

    for _ in range(int(T_transient / dt)):
        s, _ = _rk4_step(s, np.eye(n), dt, beta, sigma)

    Q = np.eye(n)
    log_sum = np.zeros(n)
    n_steps = int(T / dt)
    t = 0.0
    for i in range(1, n_steps + 1):
        s, Q = _rk4_step(s, Q, dt, beta, sigma)
        if i % renorm_every == 0:
            Q, R = np.linalg.qr(Q)
            sign = np.sign(np.diag(R))
            sign[sign == 0] = 1.0
            Q *= sign
            log_sum += np.log(np.abs(np.diag(R)) + 1e-300)
            t = i * dt

    return np.sort(log_sum / t)[::-1], s

###############################################################################

def classify(chi, tol=2e-2):
    if np.abs(chi[0]) <= tol:
        return "stable limit cycle"
    if chi[0] < 0:
        return "stable fixed point"
    return "unbounded / repelling"


def analytic_spectrum(beta, sigma):
    if beta > 0:                    
        return np.array([0.0, -2.0 * beta])
    return np.array([beta, beta])   


# --------------------------------------------------------------------------- #
#  Reproduce the case in the script: beta = 0.5, sigma = 1, IC (0.5, 0.5)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sigma = 1.0
    cases = [
        ("limit cycle ", 0.5,  (0.5, 0.5)),
        ("stable focus", 0.0, (0.5, 0.5)),
    ]

    print(f"Hopf normal form, sigma = {sigma}\n")
    for name, beta, ic in cases:
        chi, s_end = lyapunov_spectrum(ic, beta, sigma)
        ana = analytic_spectrum(beta, sigma)
        print(f"{name}  (beta={beta}, IC={ic})")
        print(f"    chi  computed = {np.round(chi, 4)}")
        print(f"    chi  analytic = {np.round(ana, 4)}")
        print(f"    sum  = {chi.sum():.4f}   (divergence on attractor = {ana.sum():.4f})")
        print(f"    r_end = {np.hypot(*s_end):.4f}"
              f"{'   (expected r* = %.4f)' % np.sqrt(beta/sigma) if beta>0 else ''}")
        print(f"    verdict: {classify(chi)}\n")