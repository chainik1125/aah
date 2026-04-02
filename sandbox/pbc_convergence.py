"""
PBC vs OBC finite DMRG convergence test.

Sweeps bond dimension chi across multiple system sizes and compares against
Bethe ansatz (exact, thermodynamic limit) as the reference.

Run from the aah/ directory:
    python -m sandbox.pbc_convergence
"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from aah_code.real_space_dmrg import (
    get_gnd_fixed_filling,
    get_gnd_fixed_filling_pbc,
)
from aah_code.bethe_ansatz import lieb_wu_energy_general_filling

# ── Configuration ──────────────────────────────────────────────────────────

L_VALUES = [16, 32, 48]
CHI_VALUES = [16, 32, 64, 128]
T = 1.0

TEST_POINTS = [
    # (U, filling, label)
    (4.0, 1.0, "half-fill U=4 (gapped)"),
    (8.0, 1.0, "half-fill U=8 (strongly gapped)"),
    (0.0, 1.0, "half-fill U=0 (free fermion)"),
    (4.0, 0.5, "quarter-fill U=4 (metallic)"),
]

OUTPUT_DIR = Path(__file__).parent
RESULTS_PATH = OUTPUT_DIR / "pbc_convergence_results.pkl"


# ── Helpers ────────────────────────────────────────────────────────────────

def _timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


# ── Main ───────────────────────────────────────────────────────────────────

def run_convergence():
    all_results = {}

    for L in L_VALUES:
        print(f"\n{'#'*60}", flush=True)
        print(f"  L = {L}", flush=True)
        print(f"{'#'*60}", flush=True)

        for U, filling, label in TEST_POINTS:
            print(f"\n  --- {label}  (U={U}, n={filling}) ---", flush=True)

            e_bethe = lieb_wu_energy_general_filling(U, filling, T)
            print(f"  Bethe ansatz (exact): {e_bethe:.10f}  E/site", flush=True)

            obc_energies, pbc_energies = [], []
            obc_times, pbc_times = [], []

            for chi in CHI_VALUES:
                # OBC
                (E_obc, _, fill_obc), dt_obc = _timed(
                    get_gnd_fixed_filling, L, chi, filling, U, T,
                )
                e_obc = E_obc / L
                obc_energies.append(e_obc)
                obc_times.append(dt_obc)

                # PBC
                (E_pbc, _, fill_pbc), dt_pbc = _timed(
                    get_gnd_fixed_filling_pbc, L, chi, filling, U, T,
                )
                e_pbc = E_pbc / L
                pbc_energies.append(e_pbc)
                pbc_times.append(dt_pbc)

                obc_err = abs(e_obc - e_bethe) / abs(e_bethe) * 100
                pbc_err = abs(e_pbc - e_bethe) / abs(e_bethe) * 100
                print(f"  chi={chi:>3d}  OBC={e_obc:.10f} (err={obc_err:.2f}%, {dt_obc:5.1f}s)  "
                      f"PBC={e_pbc:.10f} (err={pbc_err:.2f}%, {dt_pbc:5.1f}s)", flush=True)

            all_results[(L, U, filling)] = {
                'label': label,
                'L': L,
                'e_bethe': e_bethe,
                'chi_values': CHI_VALUES,
                'obc_energies': obc_energies,
                'pbc_energies': pbc_energies,
                'obc_times': obc_times,
                'pbc_times': pbc_times,
            }

    # Save results
    with open(RESULTS_PATH, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {RESULTS_PATH}", flush=True)

    return all_results


def plot_results(all_results):
    """One page (figure) per L, 2 rows x 4 cols: top=relative error, bottom=absolute energy."""
    n_cols = len(TEST_POINTS)

    for L in L_VALUES:
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)
        fig.suptitle(f'PBC vs OBC  —  L = {L}', fontsize=14, y=0.98)

        col = 0
        for U, filling, label in TEST_POINTS:
            key = (L, U, filling)
            if key not in all_results:
                col += 1
                continue
            data = all_results[key]
            chi_arr = np.array(data['chi_values'])
            e_bethe = data['e_bethe']
            obc = np.array(data['obc_energies'])
            pbc = np.array(data['pbc_energies'])

            # ── Top row: relative error ──
            ax_err = axes[0, col]
            obc_err = np.abs(obc - e_bethe) / abs(e_bethe)
            pbc_err = np.abs(pbc - e_bethe) / abs(e_bethe)

            ax_err.plot(chi_arr, obc_err, 'o-', color='tab:blue', label='OBC')
            ax_err.plot(chi_arr, pbc_err, 's-', color='tab:red', label='PBC')
            ax_err.set_yscale('log')
            ax_err.set_ylabel(r'$|E - E_{\mathrm{Bethe}}| / |E_{\mathrm{Bethe}}|$')
            ax_err.set_title(data['label'], fontsize=10)
            ax_err.legend(fontsize=8)
            ax_err.grid(True, alpha=0.3)

            # ── Bottom row: absolute energy ──
            ax_abs = axes[1, col]
            ax_abs.axhline(e_bethe, color='black', ls='--', lw=1.5, label='Bethe ansatz')
            ax_abs.plot(chi_arr, obc, 'o-', color='tab:blue', label='OBC')
            ax_abs.plot(chi_arr, pbc, 's-', color='tab:red', label='PBC')
            ax_abs.set_xlabel(r'Bond dimension $\chi$')
            ax_abs.set_ylabel('E / site')
            ax_abs.legend(fontsize=8)
            ax_abs.grid(True, alpha=0.3)

            col += 1

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = OUTPUT_DIR / f"pbc_convergence_L{L}.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}", flush=True)

    plt.show()


if __name__ == '__main__':
    results = run_convergence()
    plot_results(results)
