"""
PBC vs OBC chi-convergence benchmark for Fig 4 parameter regime.

Runs finite DMRG at half-filling for a grid of (U, V) values,
comparing OBC and PBC at increasing bond dimension chi.
Saves incrementally and updates plots after each (U, V, chi) point.
"""
import os
import sys
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import viridis
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aah_code.real_space_dmrg import get_gnd_fixed_filling

# ── Parameters ──────────────────────────────────────────────────────────
L = 48
T_HOP = -1.0
V_SEP = (1, 2)       # β = 1/2 (Fig 4 convention)
FILLING = 1.0         # half-filling
CHI_VALUES = [16, 32, 64, 72, 128]

U_VALUES = [0.0, 2.0]
V_VALUES = [0.0, 1.0, 5.0]

RESULTS_FILE = Path(__file__).parent / 'pbc_fig4_convergence_results.pkl'
PLOT_FILE = Path(__file__).parent / 'pbc_fig4_convergence.png'


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)


def make_plot(results):
    """Update the convergence plot from current results."""
    n_U = len(U_VALUES)
    n_V = len(V_VALUES)

    chi_colors = [viridis(v) for v in [0.2, 0.5, 0.85]]  # not used per-chi; we plot chi on x-axis

    fig, axes = plt.subplots(n_U, n_V, figsize=(6 * n_V, 5 * n_U), squeeze=False)
    fig.suptitle(
        f'PBC vs OBC convergence — L={L}, half-fill, $v_{{sep}}$={V_SEP}',
        fontsize=14, y=0.98,
    )

    for row, U in enumerate(U_VALUES):
        for col, V in enumerate(V_VALUES):
            ax = axes[row, col]
            key = (U, V)

            if key not in results:
                ax.set_title(f'U={U}, V={V}\n(no data yet)', fontsize=11)
                continue

            data = results[key]
            chi_done = sorted(data['obc'].keys())
            if not chi_done:
                continue

            chi_arr = np.array(chi_done)
            obc_arr = np.array([data['obc'][c] for c in chi_done])
            pbc_arr = np.array([data['pbc'][c] for c in chi_done])

            ax.plot(chi_arr, obc_arr, 'o--', color='steelblue', lw=1.5, markersize=6, label='OBC')
            ax.plot(chi_arr, pbc_arr, 's-', color='firebrick', lw=2, markersize=7, label='PBC')

            # Annotate latest values
            for i, c in enumerate(chi_done):
                ax.annotate(f'{obc_arr[i]:.4f}', (chi_arr[i], obc_arr[i]),
                            textcoords='offset points', xytext=(0, 8), fontsize=7,
                            color='steelblue', ha='center')
                ax.annotate(f'{pbc_arr[i]:.4f}', (chi_arr[i], pbc_arr[i]),
                            textcoords='offset points', xytext=(0, -14), fontsize=7,
                            color='firebrick', ha='center')

            ax.set_title(f'U={U}, V={V}', fontsize=11)
            ax.set_xlabel(r'$\chi$')
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel('Energy per site')
            ax.legend(fontsize=9, loc='upper right')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_FILE, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved to {PLOT_FILE}')


def run_single(L, chi, filling, U, t, V, V_sep, bc):
    """Run a single DMRG calc and return (E/site, time)."""
    t0 = time.perf_counter()
    E, _, fill = get_gnd_fixed_filling(L, chi, filling, U=U, t=t, V=V, V_sep=V_sep, bc=bc)
    dt = time.perf_counter() - t0
    return E / L, dt


def main():
    results = load_results()

    total = len(U_VALUES) * len(V_VALUES) * len(CHI_VALUES)
    done = 0

    for U in U_VALUES:
        for V in V_VALUES:
            key = (U, V)
            if key not in results:
                results[key] = {'obc': {}, 'pbc': {}, 'obc_time': {}, 'pbc_time': {}}

            for chi in CHI_VALUES:
                done += 1

                # Skip if already computed
                if chi in results[key]['obc'] and chi in results[key]['pbc']:
                    print(f'[{done}/{total}] U={U}, V={V}, chi={chi} — already done, skipping')
                    continue

                print(f'[{done}/{total}] U={U}, V={V}, chi={chi}', flush=True)

                # OBC
                if chi not in results[key]['obc']:
                    print(f'  OBC ...', end='', flush=True)
                    e_obc, dt_obc = run_single(L, chi, FILLING, U, T_HOP, V, V_SEP, 'open')
                    results[key]['obc'][chi] = e_obc
                    results[key]['obc_time'][chi] = dt_obc
                    print(f' E/site={e_obc:.8f}  ({dt_obc:.1f}s)')
                    save_results(results)

                # PBC
                if chi not in results[key]['pbc']:
                    print(f'  PBC ...', end='', flush=True)
                    e_pbc, dt_pbc = run_single(L, chi, FILLING, U, T_HOP, V, V_SEP, 'periodic')
                    results[key]['pbc'][chi] = e_pbc
                    results[key]['pbc_time'][chi] = dt_pbc
                    print(f' E/site={e_pbc:.8f}  ({dt_pbc:.1f}s)')
                    save_results(results)

                # Update plot after each chi
                make_plot(results)

    # Final summary
    print('\n' + '=' * 70)
    print('RESULTS SUMMARY')
    print('=' * 70)
    for U in U_VALUES:
        for V in V_VALUES:
            key = (U, V)
            data = results[key]
            print(f'\n  U={U}, V={V}:')
            print(f'  {"chi":>5s}  {"OBC E/site":>14s}  {"OBC time":>8s}  {"PBC E/site":>14s}  {"PBC time":>8s}')
            for chi in CHI_VALUES:
                if chi in data['obc'] and chi in data['pbc']:
                    print(f'  {chi:5d}  {data["obc"][chi]:14.8f}  {data["obc_time"][chi]:7.1f}s'
                          f'  {data["pbc"][chi]:14.8f}  {data["pbc_time"][chi]:7.1f}s')

    print(f'\nResults: {RESULTS_FILE}')
    print(f'Plot:    {PLOT_FILE}')


if __name__ == '__main__':
    main()
