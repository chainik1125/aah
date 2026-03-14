"""
Summarize quarter-filled strong-coupling coefficient errors for all Fig. 2 schemes.

This plot is designed to answer whether the observed quarter-filled winners track
the Bethe charge-limit coefficient c0, the leading interaction correction c1,
or neither.
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


def collect_quarter_summary(results: dict, *, Nc: int, u_min: float) -> tuple[np.ndarray, list[dict]]:
    U_values = np.asarray(results["U_values"], dtype=float)
    mask = U_values >= u_min
    fit_U = U_values[mask]

    bethe_energy = np.asarray(results["bethe"]["quarter"]["energy"], dtype=float)[mask]
    bethe_coeffs, _ = fit_quarter_energy(fit_U, bethe_energy)

    summary_rows: list[dict] = []
    for scheme_key in sorted(results["cluster"][str(Nc)].keys(), key=sep_key_to_ratio):
        payload = results["cluster"][str(Nc)][scheme_key]["quarter"]
        energy = np.asarray(payload["observables"]["energy"], dtype=float)[mask]
        coeffs, _ = fit_quarter_energy(fit_U, energy)
        mean_energy_error = float(np.mean(np.asarray(payload["abs_error"]["energy"], dtype=float)))
        summary_rows.append(
            {
                "scheme_key": scheme_key,
                "ratio": sep_key_to_ratio(scheme_key),
                "label": format_sep_as_pi(sep_key_to_ratio(scheme_key)),
                "coeffs": coeffs,
                "dc0": abs(float(coeffs[0] - bethe_coeffs[0])),
                "dc1": abs(float(coeffs[1] - bethe_coeffs[1])),
                "dc2": abs(float(coeffs[2] - bethe_coeffs[2])),
                "mean_energy_error": mean_energy_error,
            }
        )

    return fit_U, summary_rows


def print_quarter_summary(*, Nc: int, summary_rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"Quarter-filled coefficient summary for Nc={Nc}")
    print("=" * 72)
    by_energy = sorted(summary_rows, key=lambda row: row["mean_energy_error"])
    print("Ranked by mean quarter-filled |Δe|:")
    for row in by_energy:
        print(
            f"  {row['scheme_key']:>4s} ({row['label']:>4s})  "
            f"mean |Δe|={row['mean_energy_error']:.6e}  "
            f"|Δc0|={row['dc0']:.6e}  |Δc1|={row['dc1']:.6e}"
        )


def make_quarter_summary_figure(
    results: dict,
    *,
    cluster_sizes: list[int],
    u_min: float,
    output_dir: Path,
    show: bool,
) -> Path:
    fig, axes = plt.subplots(len(cluster_sizes), 2, figsize=(14.0, 4.6 * len(cluster_sizes)))
    if len(cluster_sizes) == 1:
        axes = np.asarray([axes])

    fig.suptitle(f"Quarter-filled Bethe coefficient errors for U >= {u_min:g}", fontsize=16, y=0.995)

    for row_idx, Nc in enumerate(cluster_sizes):
        _, summary_rows = collect_quarter_summary(results, Nc=Nc, u_min=u_min)
        print_quarter_summary(Nc=Nc, summary_rows=summary_rows)
        labels = [row["label"] for row in summary_rows]
        x = np.arange(len(summary_rows))
        width = 0.38
        best_scheme_key = min(summary_rows, key=lambda row: row["mean_energy_error"])["scheme_key"]

        ax = axes[row_idx, 0]
        dc0 = [row["dc0"] for row in summary_rows]
        dc1 = [row["dc1"] for row in summary_rows]
        bars0 = ax.bar(x - width / 2, dc0, width, label=r"$|\Delta c_0|$", color="#4c78a8")
        bars1 = ax.bar(x + width / 2, dc1, width, label=r"$|\Delta c_1|$", color="#f58518")
        for idx, row in enumerate(summary_rows):
            if row["scheme_key"] == best_scheme_key:
                bars0[idx].set_edgecolor("black")
                bars0[idx].set_linewidth(1.8)
                bars1[idx].set_edgecolor("black")
                bars1[idx].set_linewidth(1.8)
        ax.set_xticks(x, labels)
        ax.set_yscale("log")
        ax.set_title(f"$N_c={Nc}$: coefficient errors")
        ax.grid(True, axis="y", alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=10)

        ax = axes[row_idx, 1]
        mean_de = np.array([row["mean_energy_error"] for row in summary_rows], dtype=float)
        ax.plot(x, mean_de, marker="o", color="black", linewidth=2, label=r"mean quarter-filled $|\Delta e|$")
        ax.scatter(
            x,
            mean_de,
            c=dc1,
            cmap="viridis_r",
            s=90,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        best_idx = next(idx for idx, row in enumerate(summary_rows) if row["scheme_key"] == best_scheme_key)
        ax.scatter(x[best_idx], mean_de[best_idx], marker="*", s=220, color="#d62728", zorder=4, label="Fig. 2 winner")
        for idx, row in enumerate(summary_rows):
            ax.annotate(
                row["label"],
                (x[idx], mean_de[idx]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
            )
        ax.set_xticks(x, labels)
        ax.set_yscale("log")
        ax.set_title(f"$N_c={Nc}$: mean quarter-filled energy error")
        ax.grid(True, axis="y", alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig2_quarter_coefficient_summary.png"
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
    make_quarter_summary_figure(
        results,
        cluster_sizes=args.cluster_sizes,
        u_min=args.u_min,
        output_dir=args.output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
