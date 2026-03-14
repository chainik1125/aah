"""
Plot exact Bethe-ansatz half-filled spin and charge scales versus U.

Shown quantities:
  - charge gap Delta_c(U)
  - spin velocity v_s(U)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aah_code.bethe_ansatz import (
    lieb_wu_charge_gap_half_filling,
    lieb_wu_spin_velocity_half_filling,
)
from aah_code.cluster_model.plot_fig2_bethe_stage1_metrics import DEFAULT_PLOT_DIR


def compute_half_filled_scales(
    *,
    U_values: list[float],
    t: float = 1.0,
    delta_n: float = 1e-3,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> dict[str, list[float]]:
    payload = {"charge_gap": [], "spin_velocity": [], "spin_velocity_asymptotic": []}
    for U in U_values:
        U_float = float(U)
        payload["charge_gap"].append(
            float(
                lieb_wu_charge_gap_half_filling(
                    U_float,
                    t=t,
                    delta_n=delta_n,
                    N_k=N_k,
                    N_lam=N_lam,
                    B=B,
                )
            )
        )
        payload["spin_velocity"].append(float(lieb_wu_spin_velocity_half_filling(U_float, t=t)))
        if U_float <= 1e-12:
            payload["spin_velocity_asymptotic"].append(float("nan"))
        else:
            payload["spin_velocity_asymptotic"].append(float(2.0 * np.pi * abs(t) * abs(t) / U_float))
    return payload


def make_figure(
    *,
    U_values: list[float],
    payload: dict[str, list[float]],
    output_dir: Path,
    show: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    U_arr = np.asarray(U_values, dtype=float)
    charge_gap = np.asarray(payload["charge_gap"], dtype=float)
    spin_velocity = np.asarray(payload["spin_velocity"], dtype=float)
    spin_velocity_asymptotic = np.asarray(payload["spin_velocity_asymptotic"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True)
    fig.suptitle("Bethe half-filled spin and charge scales", fontsize=16, y=0.995)

    ax = axes[0]
    ax.plot(U_arr, charge_gap, color=plt.cm.tab10(0), marker="o", linewidth=2)
    ax.set_title(r"Charge gap $\Delta_c(U)$")
    ax.set_xlabel("U")
    ax.set_ylabel(r"$\Delta_c$")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(U_arr, spin_velocity, color=plt.cm.tab10(1), marker="o", linewidth=2, label=r"Bethe $v_s$")
    finite_mask = np.isfinite(spin_velocity_asymptotic)
    if np.any(finite_mask):
        ax.plot(
            U_arr[finite_mask],
            spin_velocity_asymptotic[finite_mask],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=r"$2\pi t^2/U$",
        )
    ax.set_title(r"Spin velocity $v_s(U)$")
    ax.set_xlabel("U")
    ax.set_ylabel(r"$v_s$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = output_dir / "bethe_half_filled_spin_charge_scales.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    print(f"Saved {png_path}")
    if show:
        plt.show()
    plt.close(fig)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--U-values", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0])
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--delta-n", type=float, default=1e-3)
    parser.add_argument("--N-k", type=int, default=256)
    parser.add_argument("--N-lam", type=int, default=256)
    parser.add_argument("--B", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    payload = compute_half_filled_scales(
        U_values=args.U_values,
        t=args.t,
        delta_n=args.delta_n,
        N_k=args.N_k,
        N_lam=args.N_lam,
        B=args.B,
    )
    make_figure(U_values=args.U_values, payload=payload, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
