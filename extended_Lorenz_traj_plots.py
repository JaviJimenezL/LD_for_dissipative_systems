"""
Author: Javier Jiménez-López
UCM
26-5-2026
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
from matplotlib.ticker import MaxNLocator

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
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


base    = '/home/javier/dissipative_systems/extended_Lorenz_system_results'
fig_dir = '/home/javier/dissipative_systems/figures'

cases = {
    'stable_fixed_point':        ('a', f'{base}/results_4d_system_stablefixed_point.txt'),
    'stable_limit_cycle':        ('b', f'{base}/results_4d_system_stable_limit_cycle.txt'),
    'chaotic_strange_attractor': ('c', f'{base}/results_4d_system_chaotic_strange_attractor.txt'),
    'hyperchaotic_attractor':    ('d', f'{base}/results_4d_system_hyperchaotic_attractor.txt'),
}


def _lim(v, margin=0.05):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    d = hi - lo if hi > lo else 1.0
    return lo - margin * d, hi + margin * d


def _draw_wall_edges(ax, color='black', lw=0.6):
    """
    Close the three visible walls (floor + back-left + back-right) by drawing
    their 6 OUTER edges. The 3 inner edges where panes meet are already drawn
    by Matplotlib's pane.set_edgecolor.
    """
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    zlo, zhi = ax.get_zlim()

    edges = [
        # Floor (z = zlo) — outer edges
        [(xlo, ylo, zlo), (xhi, ylo, zlo)],   # front edge along x
        [(xhi, ylo, zlo), (xhi, yhi, zlo)],   # right edge along y
        # Back-left wall (x = xlo) — outer edges
        [(xlo, ylo, zlo), (xlo, ylo, zhi)],   # *** the missing left vertical ***
        [(xlo, ylo, zhi), (xlo, yhi, zhi)],   # top edge along y
        # Back-right wall (y = yhi) — outer edges
        [(xhi, yhi, zlo), (xhi, yhi, zhi)],   # right vertical
        [(xlo, yhi, zhi), (xhi, yhi, zhi)],   # top edge along x
    ]
    for (x1, y1, z1), (x2, y2, z2) in edges:
        ax.plot([x1, x2], [y1, y2], [z1, z2],
                color=color, linewidth=lw, zorder=0)


def plot_4d_projections(data, letter='a', fname=None):
    x, y, z, w = data[:, 4], data[:, 5], data[:, 6], data[:, 7]

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(w)
    dropped = mask.size - mask.sum()
    if dropped:
        print(f'[{letter}] dropped {dropped} non-finite samples '
              f'({dropped / mask.size:.2%})')
    x, y, z, w = x[mask], y[mask], z[mask], w[mask]
    if x.size == 0:
        raise ValueError(f'[{letter}] no finite samples to plot')

    panels = [
        ((x, y, z), ('x', 'y', 'z')),
        ((x, y, w), ('x', 'y', 'w')),
        ((w, y, z), ('w', 'y', 'z')),
    ]

    # Wider figure -> the rightmost z label has somewhere to go
    fig = plt.figure(figsize=(14.5, 3.6))

    for i, ((a, b, c), (la, lb, lc)) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        a_lo, a_hi = _lim(a); b_lo, b_hi = _lim(b); c_lo, c_hi = _lim(c)
        ax.set_xlim(a_lo, a_hi); ax.set_ylim(b_lo, b_hi); ax.set_zlim(c_lo, c_hi)
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

        # Trajectory + wall shadows
        
        ax.plot(a, b, np.full_like(c, c_lo), color='red',   linewidth=0.3, rasterized=True)
        ax.plot(np.full_like(a, a_lo), b, c, color='blue',  linewidth=0.3, rasterized=True)
        ax.plot(a, np.full_like(b, b_hi), c, color='green', linewidth=0.3, rasterized=True)
        ax.plot(a, b, c, color='black', linewidth=0.3, rasterized=True)

        # Initial condition
        ax.scatter([a[0]], [b[0]], [c[0]],
                   color='orange', s=55,
                   edgecolors='k', linewidths=0.5, zorder=10)

        # White panes + visible inner edges (where panes meet) + no grid
        white = (1.0, 1.0, 1.0, 1.0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(white)
            axis.pane.set_edgecolor('black')
            axis.pane.set_linewidth(0.6)
        ax.set_facecolor('white')
        ax.grid(False)

        # The 6 outer edges that close the three walls
        _draw_wall_edges(ax)

        # Labels (no panel tag)
        ax.set_xlabel(fr'${la}$', labelpad=3)
        ax.set_ylabel(fr'${lb}$', labelpad=3)
        ax.set_zlabel(fr'${lc}$', labelpad=3)
        
        ax.set_box_aspect((1, 1, 1), zoom=0.9)   # 0.85 = 15% smaller cube

        ax.view_init(elev=22, azim=-60)
    
    fig.patch.set_facecolor('white')

    # Big right margin so panel 3's z label is never clipped.
    # DO NOT pass bbox_inches='tight' to savefig — it under-measures 3D zlabels.
    plt.subplots_adjust(left=0.02, right=0.95,
                        bottom=0.01, top=1.00, wspace=0.05)

    if fname:
        plt.savefig(fname, dpi=300, facecolor='white', format='pdf')
    plt.show()


if __name__ == '__main__':
    for name, (letter, path) in cases.items():
        print(f'\n--- {name} ---')
        data = np.loadtxt(path)
        plot_4d_projections(
            data,
            letter=letter,
            fname=f'{fig_dir}/4d_projections_{name}.pdf',
        )