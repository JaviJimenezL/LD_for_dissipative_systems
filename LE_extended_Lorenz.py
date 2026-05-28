"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

# Lyapunov exponent spectrum for the extended Lorenz system

import numpy as np

B = 8.0 / 3.0  # the b parameter is fixed to 8/3 throughout the paper

"""
Functions
"""
def field(s, a, c, r):
    x, y, z, w = s
    return np.array([
        a * (y - x) + y * z,
        c * x - y - x * z + w,
        x * y - B * z,
        -x * z + r * w,
    ])


def jacobian(s, a, c, r):
    x, y, z, w = s
    return np.array([
        [-a,      a + z,   y,    0.0],
        [c - z,  -1.0,    -x,    1.0],
        [y,       x,      -B,    0.0],
        [-z,      0.0,    -x,    r  ],
    ])

###############################################################################

def _rk4_step(s, Q, dt, a, c, r):
    def deriv(s, Q):
        return field(s, a, c, r), jacobian(s, a, c, r) @ Q

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
    a,
    c,
    r,
    dt=0.002,          
    renorm_every=25,   
    T_transient=300.0, 
    T=10000.0,         
    report_every=None,
):
    n = 4
    s = np.asarray(s0, dtype=float)

    for _ in range(int(T_transient / dt)):
        s, _ = _rk4_step(s, np.eye(n), dt, a, c, r)

    Q = np.eye(n)
    log_sum = np.zeros(n)
    n_steps = int(T / dt)
    t = 0.0
    for i in range(1, n_steps + 1):
        s, Q = _rk4_step(s, Q, dt, a, c, r)
        if i % renorm_every == 0:
            Q, R = np.linalg.qr(Q)

            sign = np.sign(np.diag(R))
            sign[sign == 0] = 1.0
            Q *= sign
            log_sum += np.log(np.abs(np.diag(R)) + 1e-300)
            t = i * dt
            if report_every and (i * dt) % report_every < dt:
                print(f"    t={t:8.1f}  chi={np.round(np.sort(log_sum / t)[::-1], 4)}")

    return np.sort(log_sum / t)[::-1], s

###############################################################################

def classify(chi, tol=2e-2):
    npos = int(np.sum(chi > tol))
    nzero = int(np.sum(np.abs(chi) <= tol))
    if npos >= 2:
        return "hyperchaotic attractor"
    if npos == 1:
        return "chaotic (strange) attractor"
    if nzero >= 1:
        return "stable limit cycle"
    return "stable fixed point"

###############################################################################

if __name__ == "__main__":
    IC = (3.0, 2.0, 10.0, 1.0)
    a = 35.0

    cases = [
        ("stable fixed point", 2.0,  -12.0),
        ("stable limit cycle", 55.0,  -5.0),
        ("chaotic attractor",  55.0,  -1.0),
        ("hyperchaotic",       55.0,   1.5),
    ]

    print(f"4D Lorenz system, a={a}, b=8/3, IC={IC}\n")
    for name, c, r in cases:
        chi, s_end = lyapunov_spectrum(IC, a, c, r)
        print(f"{name:22s} (c={c}, r={r})")
        print(f"    chi   = {np.round(chi, 3)}")
        print(f"    sum   = {chi.sum():.3f}   (divergence = trace J = {-a-1-B+r:.3f})")
        print(f"    verdict: {classify(chi)}\n")