"""
Plot Bethe-ansatz metallic charge-sector parameters versus U.

Shown quantities:
  - K_rho
  - v_c
  - D_c
  - compressibility kappa
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aah_code.bethe_ansatz import lieb_wu_charge_sector_parameters
from aah_code.cluster_model.plot_fig2_bethe_stage1_metrics import DEFAULT_PLOT_DIR


def compute_charge_sector_curves(
    *,
    U_values: list[float],
    fillings: list[float],
    t: float = 1.0,
    delta_n: float = 1e-3,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> dict[float, dict[str, list[float]]]:
    curves: dict[float, dict[str, list[float]]] = {}
    for n in fillings:
        payload = {"K_rho": [], "v_c": [], "D_c": [], "compressibility": []}
        for U in U_values:
            vals = lieb_wu_charge_sector_parameters(
                float(U),
                float(n),
                t=t,
                delta_n=delta_n,
                N_k=N_k,
                N_lam=N_lam,
                B=B,
            )
            for key in payload:
                payload[key].append(float(vals[key]))
        curves[n] = payload
    return curves


def make_figure(
    *,
    U_values: list[float],
    curves: dict[float, dict[str, list[float]]],
    output_dir: Path,
    show: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    fig.suptitle("Bethe charge-sector parameters", fontsize=16, y=0.995)

    observables = [
        ("K_rho", r"$K_\rho$"),
        ("v_c", r"$v_c$"),
        ("D_c", r"$D_c$"),
        ("compressibility", r"$\kappa$"),
    ]
    colors = list(plt.cm.tab10.colors)

    U_arr = np.asarray(U_values, dtype=float)
    for ax, (observable, ylabel) in zip(axes.flatten(), observables):
        for idx, (n, payload) in enumerate(sorted(curves.items())):
            ax.plot(
                U_arr,
                np.asarray(payload[observable], dtype=float),
                color=colors[idx % len(colors)],
                marker="o",
                linewidth=2,
                label=fr"$n={n:g}$",
            )
        ax.set_title(observable)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if observable in {"D_c", "compressibility"}:
            ax.set_yscale("log")

    for ax in axes[1]:
        ax.set_xlabel("U")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(curves)), frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png_path = output_dir / "bethe_charge_sector_parameters.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"Saved {png_path}")
    if show:
        plt.show()
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--U-values", nargs="+", type=float, default=[0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--fillings", nargs="+", type=float, default=[0.5, 0.75])
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--delta-n", type=float, default=1e-3)
    parser.add_argument("--N-k", type=int, default=256)
    parser.add_argument("--N-lam", type=int, default=256)
    parser.add_argument("--B", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    curves = compute_charge_sector_curves(
        U_values=args.U_values,
        fillings=args.fillings,
        t=args.t,
        delta_n=args.delta_n,
        N_k=args.N_k,
        N_lam=args.N_lam,
        B=args.B,
    )
    make_figure(U_values=args.U_values, curves=curves, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
