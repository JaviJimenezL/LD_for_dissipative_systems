"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""

# Lyapunov spectrum of the quasi-periodically forced Duffing oscillator.


import numpy as np

"""
Functions
"""
def field(t, s, delta, alpha, beta, gamma1, gamma2, omega1, omega2):
    x, y = s
    forcing = gamma1 * np.cos(omega1 * t) + gamma2 * np.cos(omega2 * t)
    return np.array([
        y,
        -delta * y + alpha * x - beta * x**3 + forcing,
    ])

###############################################################################

def jacobian(s, delta, alpha, beta, gamma1, gamma2, omega1, omega2):
    x, _ = s
    return np.array([
        [0.0,                  1.0],
        [alpha - 3*beta*x*x,  -delta],
    ])


# --------------------------------------------------------------------------- #
#  One classical RK4 step (explicit time) for (state, deviation-matrix)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Lyapunov spectrum (two nontrivial exponents)
# --------------------------------------------------------------------------- #
def lyapunov_spectrum(
    s0,
    delta, alpha, beta, gamma1, gamma2, omega1, omega2,
    t0=0.0,
    dt=0.005,
    renorm_every=20,
    T_transient=300.0,
    T=10000.0,
):
    p = (delta, alpha, beta, gamma1, gamma2, omega1, omega2)
    n = 2
    s = np.asarray(s0, dtype=float)
    t = t0

    # settle on the attractor (time threaded continuously: forcing phases matter)
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


def classify(chi, delta, tol=1e-2):
    """Classification by the two nontrivial exponents.

    With strictly incommensurate forcing (omega1 / omega2 irrational), the
    attracting object always carries the full 2-torus of forcing phases, so
    the "regular" outcome is quasi-periodic motion on (or onto) that 2-torus
    -- there is no phase-locked periodic option. LEs alone cannot distinguish
    a smooth 2-torus from a strange non-chaotic attractor (SNA); both have
    chi_1 <= 0.
    """
    if chi[0] > tol:
        return "chaotic (strange) attractor"
    if delta == 0:
        return "regular motion on a torus (volume-preserving)"
    return "quasi-periodic 2-torus attractor (possibly SNA)"


# --------------------------------------------------------------------------- #
#  Reproduce delta = 0 and delta = 1 at the user's parameters
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Common parameters
    alpha  = 1.0
    beta   = 1.0
    gamma1 = 0.5
    gamma2 = 0.5
    omega1 = 1.0
    omega2 = np.sqrt(2.0)        # incommensurate with omega1
    IC     = (0.5, 0.5)

    print(f"Quasi-periodically forced Duffing,  alpha={alpha}, beta={beta},")
    print(f"  gamma1={gamma1}, gamma2={gamma2}, omega1={omega1}, omega2=sqrt(2)\n")

    for delta in (0.0, 1.0):
        chi = lyapunov_spectrum(IC, delta, alpha, beta, gamma1, gamma2, omega1, omega2)
        print(f"delta = {delta}")
        print(f"    chi (nontrivial) = {np.round(chi, 4)}   "
              f"(+ two implicit phase exponents 0)")
        print(f"    sum              = {chi.sum():+.5f}   "
              f"(must equal -delta = {-delta:+.2f})")
        print(f"    verdict: {classify(chi, delta)}\n")