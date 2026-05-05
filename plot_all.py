#!/usr/bin/env python3
"""
plot_all.py  —  unified plotting for the FVM heat-conduction project

Usage:
    python3 plot_all.py [--steady] [--unsteady] [--scaling] [--all]

Reads:
    steady_result.txt       (or steady_result_mpi.txt)
    Usteady.txt             (or Usteady_mpi.txt)
    results/scaling_omp.csv
    results/scaling_mpi.csv

CSV columns:
    study,solver,paradigm,p,lc,elements,setup_time,solver_time,total_time,extra
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

os.makedirs("plots", exist_ok=True)

# Dirichlet boundary conditions (must match solver source)
T_LEFT, T_RIGHT, T_TOP, T_BOTTOM = 50.0, 200.0, 100.0, 300.0
GRID_N = 200        # regular grid resolution for pcolormesh
N_BOUNDARY = 80     # boundary sample points per edge

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def save(fig, name):
    path = f"plots/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved  {path}")


def load_csv(path):
    if not os.path.exists(path):
        return None
    return np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)


def inject_boundary(x, y, T):
    """Append known Dirichlet boundary values before interpolation."""
    lin = np.linspace(0, 1, N_BOUNDARY + 2)   # N_BOUNDARY interior + 2 corners

    xa, ya, Ta = [], [], []

    # left x=0, T_LEFT
    for yv in lin:
        xa.append(0.0); ya.append(yv)
        if   yv == 0.0: Ta.append((T_LEFT + T_BOTTOM) / 2)
        elif yv == 1.0: Ta.append((T_LEFT  + T_TOP)   / 2)
        else:            Ta.append(T_LEFT)

    # right x=1, T_RIGHT
    for yv in lin:
        xa.append(1.0); ya.append(yv)
        if   yv == 0.0: Ta.append((T_RIGHT + T_BOTTOM) / 2)
        elif yv == 1.0: Ta.append((T_RIGHT  + T_TOP)   / 2)
        else:            Ta.append(T_RIGHT)

    # bottom y=0, T_BOTTOM (interior only — corners already added)
    for xv in lin[1:-1]:
        xa.append(xv); ya.append(0.0); Ta.append(T_BOTTOM)

    # top y=1, T_TOP (interior only)
    for xv in lin[1:-1]:
        xa.append(xv); ya.append(1.0); Ta.append(T_TOP)

    return (np.append(x, xa), np.append(y, ya), np.append(T, Ta))


def inject_boundary_zeros(x, y, T):
    """Inject zeros at boundary (for error / diff fields)."""
    lin = np.linspace(0, 1, N_BOUNDARY + 2)
    xa, ya, Ta = [], [], []
    for xv in [0.0, 1.0]:
        for yv in lin:
            xa.append(xv); ya.append(yv); Ta.append(0.0)
    for yv in [0.0, 1.0]:
        for xv in lin[1:-1]:
            xa.append(xv); ya.append(yv); Ta.append(0.0)
    return (np.append(x, xa), np.append(y, ya), np.append(T, Ta))


def field_plot(x, y, T, title, cmap='inferno', clabel='Temperature'):
    """Scatter → regular grid → pcolormesh."""
    xi = np.linspace(0, 1, GRID_N)
    yi = np.linspace(0, 1, GRID_N)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), T, (xi, yi), method='linear')

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cf = ax.pcolormesh(xi, yi, zi, shading='auto', cmap=cmap)
    fig.colorbar(cf, ax=ax, label=clabel)
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# 1.  Steady-state contour plots                                      #
# ------------------------------------------------------------------ #

def plot_steady(fname='steady_result.txt', tag='omp'):
    if not os.path.exists(fname):
        print(f"  [skip] {fname} not found")
        return

    data = np.loadtxt(fname, skiprows=1)
    x, y, T, T_exact = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    xi, yi, Ti     = inject_boundary(x, y, T)
    xe, ye, Te     = inject_boundary(x, y, T_exact)
    err            = np.abs(T - T_exact)
    xr, yr, erri   = inject_boundary_zeros(x, y, err)

    fig = field_plot(xi, yi, Ti, f'Steady FVM ({tag})', cmap='inferno')
    save(fig, f'steady_T_{tag}')

    fig = field_plot(xe, ye, Te, 'Steady analytical', cmap='inferno')
    save(fig, 'steady_T_exact')

    # linear version
    fig = field_plot(xr, yr, erri, f'|T_FVM − T_exact| ({tag}) — linear',
                     cmap='inferno', clabel='|ΔT| (°C)')
    save(fig, f'steady_error_{tag}_linear')

    # log version
    err_pos = err[err > 0]
    if err_pos.size > 0:
        emin = max(err_pos.min(), 1e-6)
        emax = err.max()
        xi_g = np.linspace(0, 1, GRID_N)
        yi_g = np.linspace(0, 1, GRID_N)
        xi_g, yi_g = np.meshgrid(xi_g, yi_g)
        zi_g = griddata((xr, yr), erri, (xi_g, yi_g), method='linear')
        zi_g = np.clip(zi_g, emin, emax)
        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        cf2 = ax2.pcolormesh(xi_g, yi_g, zi_g, shading='auto', cmap='inferno',
                             norm=LogNorm(vmin=emin, vmax=emax))
        fig2.colorbar(cf2, ax=ax2, label='|ΔT| (°C)')
        ax2.set_title(f'|T_FVM − T_exact| ({tag}) — log')
        ax2.set_xlabel('x'); ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        fig2.tight_layout()
        save(fig2, f'steady_error_{tag}')

    print(f"  max error = {err.max():.4e}")


# ------------------------------------------------------------------ #
# 1b. Serial vs parallel difference plots                             #
# ------------------------------------------------------------------ #

def plot_serial_vs_parallel(serial_fname='steady_result_serial.txt',
                            par_fname='steady_result.txt', tag='omp'):
    for fname in (serial_fname, par_fname):
        if not os.path.exists(fname):
            print(f"  [skip] {fname} not found")
            return

    serial = np.loadtxt(serial_fname, skiprows=1)
    par    = np.loadtxt(par_fname,    skiprows=1)

    def sort_key(d):
        return np.lexsort((d[:, 1], d[:, 0]))

    serial = serial[sort_key(serial)]
    par    = par[sort_key(par)]

    if serial.shape[0] != par.shape[0]:
        print(f"  [skip] serial vs {tag}: mesh size mismatch "
              f"({serial.shape[0]} vs {par.shape[0]} elements) — re-run both at the same lc")
        return

    x, y = serial[:, 0], serial[:, 1]
    diff  = np.abs(serial[:, 2] - par[:, 2])

    print(f"  serial vs {tag}: max|ΔT| = {diff.max():.3e},  mean = {diff.mean():.3e}")

    xd, yd, di = inject_boundary_zeros(x, y, diff)
    vmax = diff.max() if diff.max() > 0 else 1.0
    title = f'|T_serial − T_{tag}|' + (' (identically zero)' if diff.max() == 0 else '')
    fig = field_plot(xd, yd, di, title, cmap='inferno', clabel='|ΔT| (°C)')
    save(fig, f'diff_serial_vs_{tag}')


# ------------------------------------------------------------------ #
# 2.  Unsteady snapshots                                              #
# ------------------------------------------------------------------ #

def plot_unsteady(fname='Usteady.txt', tag='omp', n_snapshots=5):
    if not os.path.exists(fname):
        print(f"  [skip] {fname} not found")
        return

    data = np.loadtxt(fname)
    time = data[:, 0]
    x    = data[:, 1]
    y    = data[:, 2]
    T    = data[:, 3]

    unique_times = np.unique(time)
    n_nodes      = len(x) // len(unique_times)

    indices = np.linspace(0, len(unique_times) - 1, n_snapshots, dtype=int)

    for idx in indices:
        tval  = unique_times[idx]
        start = idx * n_nodes
        end   = (idx + 1) * n_nodes

        x_t = x[start:end]
        y_t = y[start:end]
        T_t = T[start:end]

        xi, yi, Ti = inject_boundary(x_t, y_t, T_t)
        fig = field_plot(xi, yi, Ti, f'T  t={tval:.4f}  ({tag})', cmap='inferno')
        save(fig, f'unsteady_{tag}_t{idx:04d}')


# ------------------------------------------------------------------ #
# 3.  Strong scaling — speedup & efficiency                           #
# ------------------------------------------------------------------ #

def plot_strong_scaling(csv_omp='results/scaling_omp.csv',
                        csv_mpi='results/scaling_mpi.csv'):

    fig_sp, ax_sp = plt.subplots(figsize=(6, 4.5))
    fig_ef, ax_ef = plt.subplots(figsize=(6, 4.5))

    ax_sp.set_xlabel('Cores / Threads (p)')
    ax_sp.set_ylabel('Speedup  S(p)')
    ax_sp.set_title('Strong scaling — speedup')

    ax_ef.set_xlabel('Cores / Threads (p)')
    ax_ef.set_ylabel('Efficiency  S(p)/p')
    ax_ef.set_title('Strong scaling — efficiency')

    any_data = False

    for csv_path, paradigm, marker in [
            (csv_omp, 'omp', 'o'),
            (csv_mpi, 'mpi', 's')]:

        data = load_csv(csv_path)
        if data is None:
            print(f"  [skip] {csv_path} not found")
            continue

        for solver in ('steady', 'unsteady'):
            mask = (data['study'] == 'strong') & \
                   (data['solver'] == solver) & \
                   (data['paradigm'] == paradigm)
            if not np.any(mask):
                continue

            subset = data[mask]
            ps     = subset['p'].astype(float)
            ts     = subset['solver_time'].astype(float)

            order  = np.argsort(ps)
            ps, ts = ps[order], ts[order]

            T1 = ts[ps == ps.min()][0]
            speedup    = T1 / ts
            efficiency = speedup / ps

            label = f'{paradigm.upper()} {solver}'
            ax_sp.plot(ps, speedup,    marker=marker, label=label)
            ax_ef.plot(ps, efficiency, marker=marker, label=label)
            any_data = True

    if any_data:
        p_max = 8
        ax_sp.plot([1, p_max], [1, p_max], 'k--', lw=0.8, label='ideal')
        ax_ef.axhline(1.0, color='k', lw=0.8, ls='--', label='ideal')
        for ax in (ax_sp, ax_ef):
            ax.legend(); ax.grid(True, alpha=0.3)
        save(fig_sp, 'strong_speedup')
        save(fig_ef, 'strong_efficiency')
    else:
        plt.close(fig_sp); plt.close(fig_ef)
        print("  no strong-scaling data found in CSVs")


# ------------------------------------------------------------------ #
# 4.  Weak scaling — solver time vs process count                     #
# ------------------------------------------------------------------ #

def plot_weak_scaling(csv_omp='results/scaling_omp.csv',
                      csv_mpi='results/scaling_mpi.csv'):

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlabel('Cores / Threads (p)')
    ax.set_ylabel('Solver time (s)')
    ax.set_title('Weak scaling — solver time')

    any_data = False

    for csv_path, paradigm, marker in [
            (csv_omp, 'omp', 'o'),
            (csv_mpi, 'mpi', 's')]:

        data = load_csv(csv_path)
        if data is None:
            continue

        for solver in ('steady', 'unsteady'):
            mask = (data['study'] == 'weak') & \
                   (data['solver'] == solver) & \
                   (data['paradigm'] == paradigm)
            if not np.any(mask):
                continue

            subset = data[mask]
            ps     = subset['p'].astype(float)
            ts     = subset['solver_time'].astype(float)

            order  = np.argsort(ps)
            ps, ts = ps[order], ts[order]

            label = f'{paradigm.upper()} {solver}'
            ax.plot(ps, ts, marker=marker, label=label)
            any_data = True

    if any_data:
        ax.legend(); ax.grid(True, alpha=0.3)
        save(fig, 'weak_scaling')
    else:
        plt.close(fig)
        print("  no weak-scaling data found in CSVs")


# ------------------------------------------------------------------ #
# 5.  Setup vs solver time breakdown                                  #
# ------------------------------------------------------------------ #

def plot_time_breakdown(csv_omp='results/scaling_omp.csv',
                        csv_mpi='results/scaling_mpi.csv'):

    for csv_path, paradigm in [(csv_omp, 'omp'), (csv_mpi, 'mpi')]:
        data = load_csv(csv_path)
        if data is None:
            continue

        mask = data['study'] == 'strong'
        if not np.any(mask):
            continue

        subset = data[mask]
        for solver in ('steady', 'unsteady'):
            smask = subset['solver'] == solver
            if not np.any(smask):
                continue
            s = subset[smask]
            ps = s['p'].astype(float)
            setup  = s['setup_time'].astype(float)
            solve  = s['solver_time'].astype(float)

            order = np.argsort(ps)
            ps, setup, solve = ps[order], setup[order], solve[order]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(ps - 0.2, setup, 0.4, label='setup')
            ax.bar(ps + 0.2, solve, 0.4, label='solver')
            ax.set_xlabel('p')
            ax.set_ylabel('Time (s)')
            ax.set_title(f'Time breakdown — {paradigm.upper()} {solver}')
            ax.legend(); ax.grid(True, alpha=0.3, axis='y')
            save(fig, f'breakdown_{paradigm}_{solver}')


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

args = set(sys.argv[1:]) or {'--all'}

if '--all' in args or '--steady' in args:
    print("\n--- Steady plots ---")
    plot_steady('steady_result.txt',     tag='omp')
    plot_steady('steady_result_mpi.txt', tag='mpi')
    plot_serial_vs_parallel('steady_result_serial.txt', 'steady_result.txt',     tag='omp')
    plot_serial_vs_parallel('steady_result_serial.txt', 'steady_result_mpi.txt', tag='mpi')

if '--all' in args or '--unsteady' in args:
    print("\n--- Unsteady snapshots ---")
    plot_unsteady('Usteady.txt',     tag='omp')
    plot_unsteady('Usteady_mpi.txt', tag='mpi')

if '--all' in args or '--scaling' in args:
    print("\n--- Scaling plots ---")
    plot_strong_scaling()
    plot_weak_scaling()
    plot_time_breakdown()

print("\nDone. Figures saved in ./plots/")
