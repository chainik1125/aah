"""
Quarter-filled constrained-c1 summary for all Fig. 2 schemes.

This fixes c0 to the exact U=infinity charge-sector value and tests whether
the observed winners still line up with the best c1 correction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aah_code.cluster_model.analyze_fig2_bethe_thermodynamics import DEFAULT_OUTPUT
from aah_code.cluster_model.plot_fig2_bethe_stage1_metrics import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_PLOT_DIR,
    format_sep_as_pi,
    load_or_compute_results,
)
from aah_code.cluster_model.plot_fig2_bethe_strong_coupling import fit_quarter_energy


def sep_key_to_ratio(key: str) -> tuple[int, int]:
    num, den = key.split("_", 1)
    return int(num), int(den)


def exact_u_infty_charge_energy(n_target: float, t: float) -> float:
    if not (0.0 <= n_target <= 1.0):
        raise ValueError(f"Expected filling in [0, 1] for the spinless charge-sector formula, got {n_target}.")
    return -(2.0 * abs(t) / np.pi) * np.sin(np.pi * n_target)


def fit_quarter_energy_constrained_c0(
    U_values: np.ndarray,
    energies: np.ndarray,
    *,
    c0_fixed: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([
        1.0 / U_values,
        1.0 / (U_values**2),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(X, energies - c0_fixed, rcond=None)
    return coeffs, c0_fixed + X @ coeffs


def collect_summary(results: dict, *, Nc: int, u_min: float) -> tuple[np.ndarray, dict, list[dict]]:
    U_values = np.asarray(results["U_values"], dtype=float)
    mask = U_values >= u_min
    fit_U = U_values[mask]
    t = float(results["parameters"]["t"])
    c0_exact = exact_u_infty_charge_energy(0.5, t)

    bethe_energy = np.asarray(results["bethe"]["quarter"]["energy"], dtype=float)[mask]
    bethe_unconstrained, _ = fit_quarter_energy(fit_U, bethe_energy)
    bethe_constrained, _ = fit_quarter_energy_constrained_c0(fit_U, bethe_energy, c0_fixed=c0_exact)

    rows: list[dict] = []
    for scheme_key in sorted(results["cluster"][str(Nc)].keys(), key=sep_key_to_ratio):
        payload = results["cluster"][str(Nc)][scheme_key]["quarter"]
        energy = np.asarray(payload["observables"]["energy"], dtype=float)[mask]
        unconstrained, _ = fit_quarter_energy(fit_U, energy)
        constrained, _ = fit_quarter_energy_constrained_c0(fit_U, energy, c0_fixed=c0_exact)
        rows.append(
            {
                "scheme_key": scheme_key,
                "label": format_sep_as_pi(sep_key_to_ratio(scheme_key)),
                "mean_energy_error": float(np.mean(np.asarray(payload["abs_error"]["energy"], dtype=float))),
                "dc0": abs(float(unconstrained[0] - bethe_unconstrained[0])),
                "dc1_unconstrained": abs(float(unconstrained[1] - bethe_unconstrained[1])),
                "dc1_constrained": abs(float(constrained[0] - bethe_constrained[0])),
            }
        )

    meta = {
        "c0_exact": c0_exact,
        "bethe_c0_float": float(bethe_unconstrained[0]),
        "bethe_c1_unconstrained": float(bethe_unconstrained[1]),
        "bethe_c1_constrained": float(bethe_constrained[0]),
    }
    return fit_U, meta, rows


def print_summary(*, Nc: int, meta: dict, rows: list[dict], u_min: float) -> None:
    print("\n" + "=" * 72)
    print(f"Quarter-filled constrained-c1 summary for Nc={Nc} using U >= {u_min:g}")
    print("=" * 72)
    print(
        f"  exact c0 = {meta['c0_exact']:.8f}, "
        f"Bethe floated c0 = {meta['bethe_c0_float']:.8f}, "
        f"Bethe constrained c1 = {meta['bethe_c1_constrained']:.8f}"
    )
    by_energy = sorted(rows, key=lambda row: row["mean_energy_error"])
    for row in by_energy:
        print(
            f"  {row['scheme_key']:>4s} ({row['label']:>4s})  "
            f"mean |Δe|={row['mean_energy_error']:.6e}  "
            f"|Δc0|={row['dc0']:.6e}  "
            f"|Δc1|={row['dc1_unconstrained']:.6e}  "
            f"|Δĉ1|={row['dc1_constrained']:.6e}"
        )


def make_figure(
    results: dict,
    *,
    cluster_sizes: list[int],
    u_min: float,
    output_dir: Path,
    show: bool,
) -> Path:
    fig, axes = plt.subplots(len(cluster_sizes), 2, figsize=(14.0, 4.8 * len(cluster_sizes)))
    if len(cluster_sizes) == 1:
        axes = np.asarray([axes])

    fig.suptitle(f"Quarter-filled constrained-c1 analysis for U >= {u_min:g}", fontsize=16, y=0.995)

    for row_idx, Nc in enumerate(cluster_sizes):
        _, meta, rows = collect_summary(results, Nc=Nc, u_min=u_min)
        print_summary(Nc=Nc, meta=meta, rows=rows, u_min=u_min)

        labels = [row["label"] for row in rows]
        x = np.arange(len(rows))
        width = 0.26
        best_scheme_key = min(rows, key=lambda row: row["mean_energy_error"])["scheme_key"]

        ax = axes[row_idx, 0]
        bars_c0 = ax.bar(x - width, [row["dc0"] for row in rows], width, label=r"$|\Delta c_0|$", color="#4c78a8")
        bars_c1 = ax.bar(x, [row["dc1_unconstrained"] for row in rows], width, label=r"$|\Delta c_1|$", color="#f58518")
        bars_c1c = ax.bar(x + width, [row["dc1_constrained"] for row in rows], width, label=r"$|\Delta \hat c_1|$", color="#54a24b")
        for idx, row in enumerate(rows):
            if row["scheme_key"] == best_scheme_key:
                for container in (bars_c0, bars_c1, bars_c1c):
                    container[idx].set_edgecolor("black")
                    container[idx].set_linewidth(1.8)
        ax.set_xticks(x, labels)
        ax.set_yscale("log")
        ax.set_title(f"$N_c={Nc}$: quarter-filled coefficient errors")
        ax.grid(True, axis="y", alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=10)

        ax = axes[row_idx, 1]
        mean_de = np.array([row["mean_energy_error"] for row in rows], dtype=float)
        constrained = np.array([row["dc1_constrained"] for row in rows], dtype=float)
        ax.plot(x, mean_de, marker="o", color="black", linewidth=2, label=r"mean quarter-filled $|\Delta e|$")
        scatter = ax.scatter(
            x,
            mean_de,
            c=constrained,
            cmap="viridis_r",
            s=110,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        best_idx = next(idx for idx, row in enumerate(rows) if row["scheme_key"] == best_scheme_key)
        ax.scatter(x[best_idx], mean_de[best_idx], marker="*", s=220, color="#d62728", zorder=4, label="Fig. 2 winner")
        for idx, row in enumerate(rows):
            ax.annotate(row["label"], (x[idx], mean_de[idx]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_yscale("log")
        ax.set_title(f"$N_c={Nc}$: energy error colored by " + r"$|\Delta \hat c_1|$")
        ax.grid(True, axis="y", alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=10)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$|\Delta \hat c_1|$")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig2_quarter_constrained_c1_summary.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"Saved {png_path}")
    if show:
        plt.show()
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    parser.add_argument("--analysis-pkl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-sizes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--u-min", type=float, default=5.0)
    parser.add_argument("--delta-u", type=float, default=5e-2)
    parser.add_argument("--N-k", type=int, default=256)
    parser.add_argument("--N-lam", type=int, default=256)
    parser.add_argument("--B", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--refresh-analysis", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    results = load_or_compute_results(
        analysis_pkl=args.analysis_pkl,
        fig2_cache=args.fig2_cache,
        cluster_sizes=args.cluster_sizes,
        refresh=args.refresh_analysis,
        delta_u=args.delta_u,
        N_k=args.N_k,
        N_lam=args.N_lam,
        B=args.B,
    )
    make_figure(
        results,
        cluster_sizes=args.cluster_sizes,
        u_min=args.u_min,
        output_dir=args.output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
