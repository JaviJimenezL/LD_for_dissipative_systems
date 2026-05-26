"""
Author: Javier Jiménez-López
UCM
25-5-2026
"""

# Calculation of Delta L and Omega for a regular and a chaotic intial condition in the Henon-Heiles system

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

def ode_system(t, Y):

    dYdt = np.zeros_like(Y)

    x = Y[0]
    y = Y[1]
    px = Y[2]
    py = Y[3]

    dxdt = px
    dydt = py
    dpxdt = -x - 2.0 * x * y
    dpydt = -y - x * x + y * y

    dYdt[0] = dxdt
    dYdt[1] = dydt
    dYdt[2] = dpxdt
    dYdt[3] = dpydt

    # dLDdt = np.sqrt(dxdt**2 + dydt**2 + dpxdt**2 + dpydt**2)
    p = 1.0
    dLDdt = np.abs(dxdt)**p + np.abs(dydt)**p + np.abs(dpxdt)**p + np.abs(dpydt)**p

    dYdt[4] = dLDdt

    return dYdt

###############################################################################

def H_henon_heiles(x, y, px, py):
    T = 0.5 * (px**2 + py**2)
    V = 0.5 * (x**2 + y**2) + (x**2 * y - (y**3) / 3.0)
    return T + V

###############################################################################

def perturbed_vector(original_vec, offset_norm):

    orig = np.asarray(original_vec).reshape(4,)
    
    rand_dir = np.random.randn(4)          
    rand_dir /= np.linalg.norm(rand_dir)
    
    rand_offset = offset_norm * rand_dir
    
    new_vec = orig + rand_offset
    
    return new_vec

###############################################################################

def main():

    """
    Define a regular intial condition
    """
    y_reg = 0.2
    H = 1 / 8
    px_reg = np.sqrt(2 * (H - y_reg**2 / 2 + y_reg**3 / 3))
    x0_reg = np.array([0.0, y_reg, px_reg, 0.0])

    """
    Define a chaotic initial condition
    """
    y_chao = -0.175
    H = 1 / 8
    px_chao = np.sqrt(2 * (H - y_chao**2 / 2 + y_chao**3 / 3))
    x0_chao = np.array([0.0, y_chao, px_chao, 0.0])
    # x0_chao = np.array([-0.238388, -0.201509, -0.906243, 0.314226])

    """
    Define a perturbed initial condition
    """
    x0_perturbed_reg = perturbed_vector(x0_reg, 1e-8)
    x0_reg_perturbed = np.concatenate((x0_perturbed_reg, [0.0]))
    x0_perturbed_chao = perturbed_vector(x0_chao, 1e-8)
    x0_chao_perturbed = np.concatenate((x0_perturbed_chao, [0.0]))

    """
    Define the time span for the integration
    """
    t_span = (0, 10000)
    t_eval = np.linspace(1, t_span[1], 10000) # Use consistent t_eval

    """
    Solve the ODE for the regular initial condition
    """
    sol_reg = solve_ivp(ode_system, t_span, np.append(x0_reg, 0.0), method='DOP853', t_eval=t_eval, atol=1e-12, rtol=1e-12)

    LD_evolution_x0_reg = sol_reg.y[4, :]
    print(f"Final LD for x0_reg: {LD_evolution_x0_reg[-1]}")

    """
    Solve the ODE for the perturbed initial condition for the regular initial condition
    """
    sol_perturbed_reg = solve_ivp(ode_system, t_span, x0_reg_perturbed, method='DOP853', t_eval=t_eval, atol=1e-12, rtol=1e-12)

    LD_evolution_x0_perturbed_reg = sol_perturbed_reg.y[4, :]
    print(f"Final LD for perturbed x0_reg: {LD_evolution_x0_perturbed_reg[-1]}")

    """
    Solve the ODE for the chaotic initial condition
    """
    sol_chao = solve_ivp(ode_system, t_span, np.append(x0_chao, 0.0), method='DOP853', t_eval=t_eval, atol=1e-12, rtol=1e-12)

    LD_evolution_x0_chao = sol_chao.y[4, :]
    print(f"Final LD for x0_chao: {LD_evolution_x0_chao[-1]}")

    """
    Solve the ODE for the perturbed initial condition for the chaotic initial condition
    """
    sol_perturbed_chao = solve_ivp(ode_system, t_span, x0_chao_perturbed, method='DOP853', t_eval=t_eval, atol=1e-12, rtol=1e-12)

    LD_evolution_x0_perturbed_chao = sol_perturbed_chao.y[4, :]
    print(f"Final LD for perturbed x0_chao: {LD_evolution_x0_perturbed_chao[-1]}")

    """
    Test energy conservation regular initial condition
    """
    H_x0_reg = H_henon_heiles(sol_reg.y[0, :], sol_reg.y[1, :], sol_reg.y[2, :], sol_reg.y[3, :])
    H_x0_perturbed_reg = H_henon_heiles(sol_perturbed_reg.y[0, :], sol_perturbed_reg.y[1, :], sol_perturbed_reg.y[2, :], sol_perturbed_reg.y[3, :])
    print(f"Final Energy for x0_reg: {H_x0_reg[-1]}")
    print(f"The difference in energy is: {np.abs(H_x0_reg[-1] - H_x0_perturbed_reg[-1])}")
    print(f"Final Energy for perturbed x0_reg: {H_x0_perturbed_reg[-1]}")

    """
    Test energy conservation chaotic initial condition
    """
    H_x0_chao = H_henon_heiles(sol_chao.y[0, :], sol_chao.y[1, :], sol_chao.y[2, :], sol_chao.y[3, :])
    H_x0_perturbed_chao = H_henon_heiles(sol_perturbed_chao.y[0, :], sol_perturbed_chao.y[1, :], sol_perturbed_chao.y[2, :], sol_perturbed_chao.y[3, :])
    print(f"Final Energy for x0_chao: {H_x0_chao[-1]}")
    print(f"The difference in energy is: {np.abs(H_x0_chao[-1] - H_x0_perturbed_chao[-1])}")
    print(f"Final Energy for perturbed x0_chao: {H_x0_perturbed_chao[-1]}")

    """
    Define the difference in the LD evolution
    """
    Delta_LD_evolution_reg = np.abs(LD_evolution_x0_perturbed_reg - LD_evolution_x0_reg)
    Delta_LD_evolution_chao = np.abs(LD_evolution_x0_perturbed_chao - LD_evolution_x0_chao)
    
    # Use actual solution time points to ensure matching dimensions
    t_reg = sol_reg.t[1:]
    t_chao = sol_chao.t[1:]
    
    plt.figure(figsize=(9, 6))
    plt.plot(t_reg, Delta_LD_evolution_reg[1:], label='Regular')
    plt.plot(t_chao, Delta_LD_evolution_chao[1:], label='Chaotic')
    plt.legend()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\Delta \mathcal{L}$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(t_reg, Delta_LD_evolution_reg[1:] / t_reg, label='Regular')
    plt.plot(t_chao, Delta_LD_evolution_chao[1:] / t_chao, label='Chaotic')
    plt.legend()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\Delta \mathcal{L} / \tau$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(9, 6))
    plt.plot(t_reg, LD_evolution_x0_reg[1:] / t_reg, label='Regular')
    plt.plot(t_chao, LD_evolution_x0_chao[1:] / t_chao, label='Chaotic')
    plt.legend()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\mathcal{L} / \tau$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    """
    Calculate average of L/tau and average deviation from mean for both trajectories
    """
    # Calculate L/tau for both trajectories using actual solution time points
    L_over_tau_reg = LD_evolution_x0_reg[1:] / t_reg
    L_over_tau_chao = LD_evolution_x0_chao[1:] / t_chao
    
    # Calculate mean values
    mean_L_over_tau_reg = np.mean(L_over_tau_reg)
    mean_L_over_tau_chao = np.mean(L_over_tau_chao)
    
    # Calculate average deviation from mean (mean absolute deviation)
    avg_deviation_reg = np.mean(np.abs(L_over_tau_reg - mean_L_over_tau_reg))
    avg_deviation_chao = np.mean(np.abs(L_over_tau_chao - mean_L_over_tau_chao))

    max_deviation_reg = np.max(np.abs(L_over_tau_reg - mean_L_over_tau_reg))
    max_deviation_chao = np.max(np.abs(L_over_tau_chao - mean_L_over_tau_chao))

    sum_deviation_reg = np.sum(np.abs(L_over_tau_reg - mean_L_over_tau_reg))
    sum_deviation_chao = np.sum(np.abs(L_over_tau_chao - mean_L_over_tau_chao))

    sum_squared_deviation_reg = np.sum((L_over_tau_reg - mean_L_over_tau_reg)**2)
    sum_squared_deviation_chao = np.sum((L_over_tau_chao - mean_L_over_tau_chao)**2)

    std_deviation_reg = np.sqrt(sum_squared_deviation_reg / len(L_over_tau_reg))
    std_deviation_chao = np.sqrt(sum_squared_deviation_chao / len(L_over_tau_chao))
    
    # Print results
    print(f"\nRegular trajectory:")
    print(f"  Average L/tau: {mean_L_over_tau_reg:.6e}")
    print(f"  Average deviation from mean: {avg_deviation_reg:.6e}")
    print(f"  Max deviation from mean: {max_deviation_reg:.6e}")
    print(f"  Sum of deviations: {sum_deviation_reg:.6e}")
    print(f"  Sum of squared deviations: {sum_squared_deviation_reg:.6e}")
    print(f"  Standard deviation: {std_deviation_reg:.6e}")
    print(f"\nChaotic trajectory:")
    print(f"  Average L/tau: {mean_L_over_tau_chao:.6e}")
    print(f"  Average deviation from mean: {avg_deviation_chao:.6e}")
    print(f"  Max deviation from mean: {max_deviation_chao:.6e}")
    print(f"  Sum of deviations: {sum_deviation_chao:.6e}")
    print(f"  Sum of squared deviations: {sum_squared_deviation_chao:.6e}")
    print(f"  Standard deviation: {std_deviation_chao:.6e}")

    # Store the results in diferent .txt files (times and Ld for each intial condition)
    
    np.savetxt('LD_evolution_x0_reg_p_1.txt', np.column_stack((t_reg, LD_evolution_x0_reg[1:])))
    np.savetxt('LD_evolution_x0_chao_p_1.txt', np.column_stack((t_chao, LD_evolution_x0_chao[1:])))
    np.savetxt('LD_evolution_x0_perturbed_reg_p_1.txt', np.column_stack((sol_perturbed_reg.t[1:], LD_evolution_x0_perturbed_reg[1:])))
    np.savetxt('LD_evolution_x0_perturbed_chao_p_1.txt', np.column_stack((sol_perturbed_chao.t[1:], LD_evolution_x0_perturbed_chao[1:])))
    

if __name__ == "__main__":
    main()
