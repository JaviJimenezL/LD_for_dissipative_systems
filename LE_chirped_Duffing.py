"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""
# Finite-time Lyapunov exponents (FTLEs) of the chirp-forced Duffing oscillator.


import numpy as np

"""
Functions
"""
def field(t, s, delta, alpha, beta, gamma, omega0, kappa):
    x, y = s
    phase = omega0 * t + 0.5 * kappa * t * t
    return np.array([
        y,
        -delta * y + alpha * x - beta * x**3 + gamma * np.cos(phase),
    ])


def jacobian(s, delta, alpha, beta, gamma, omega0, kappa):
    x, _ = s
    return np.array([
        [0.0,                  1.0],
        [alpha - 3*beta*x*x,  -delta],
    ])


def instantaneous_frequency(t, omega0, kappa):
    return omega0 + kappa * t

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

def ftle_spectrum(
    s0,
    delta, alpha, beta, gamma, omega0, kappa,
    t0=0.0,
    dt=0.005,            # RK4 time step
    renorm_every=20,     # QR reorthonormalization interval (in steps)
    T=10000.0,           # total integration time
    sample_dt=100.0,     # how often to record the running FTLE in history
):
    """Return (chi_final, t_hist, chi_hist) where chi_hist[k] is the
    sorted running FTLE pair at t_hist[k]."""
    p = (delta, alpha, beta, gamma, omega0, kappa)
    n = 2
    s = np.asarray(s0, dtype=float)
    t = t0
    Q = np.eye(n)
    log_sum = np.zeros(n)

    t_hist, chi_hist = [], []
    n_steps = int(T / dt)
    next_sample = t0 + sample_dt

    for i in range(1, n_steps + 1):
        s, Q = _rk4_step(t, s, Q, dt, p)
        t += dt
        if i % renorm_every == 0:
            Q, R = np.linalg.qr(Q)
            sign = np.sign(np.diag(R))
            sign[sign == 0] = 1.0
            Q *= sign
            log_sum += np.log(np.abs(np.diag(R)) + 1e-300)
            if t >= next_sample and (t - t0) > 0:
                chi_run = np.sort(log_sum / (t - t0))[::-1]
                t_hist.append(t)
                chi_hist.append(chi_run)
                next_sample += sample_dt

    chi_final = np.sort(log_sum / (t - t0))[::-1]
    return chi_final, np.asarray(t_hist), np.asarray(chi_hist)

###############################################################################

if __name__ == "__main__":
    alpha  = 1.0
    beta   = 1.0
    gamma  = 0.5
    omega0 = 1.0
    kappa  = 1e-4
    T      = 10000.0          # forcing freq sweeps from 1.0 to 2.0 over this
    IC     = (0.5, 0.5)

    print(f"Chirp-forced Duffing: alpha={alpha}, beta={beta}, gamma={gamma},")
    print(f"  omega(t) = omega0 + kappa t = {omega0} + {kappa} t   "
          f"-> sweeps {omega0:.2f} to {instantaneous_frequency(T, omega0, kappa):.2f}\n")

    for delta in (0.0, 1.0):
        chi, th, ch = ftle_spectrum(
            IC, delta, alpha, beta, gamma, omega0, kappa,
            T=T, sample_dt=1000.0,
        )
        print(f"delta = {delta}")
        print(f"    final FTLE pair = {np.round(chi, 4)}")
        print(f"    sum             = {chi.sum():+.5f}   (must equal -delta = {-delta:+.2f})")
        print(f"    history of chi_1(t) as the chirp sweeps:")
        print(f"        {'t':>8s}   {'omega(t)':>9s}   {'chi_1':>9s}   {'sum':>11s}")
        for tt, cc in zip(th, ch):
            print(f"        {tt:8.0f}   {instantaneous_frequency(tt, omega0, kappa):9.3f}"
                  f"   {cc[0]:+9.4f}   {cc.sum():+11.6f}")
        print()