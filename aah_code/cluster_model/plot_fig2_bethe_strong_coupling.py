"""
Strong-coupling coefficient analysis for Fig. 2 schemes versus Bethe Ansatz.

Quarter filling is fit to
    e(U) = c0 + c1/U + c2/U^2

Half filling is fit to
    e(U) = a1/U + a3/U^3

for the maximal and next-maximal schemes at selected cluster sizes.
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
    parse_scheme_overrides,
    resolve_scheme_selection,
    scheme_display_label,
)


def fit_quarter_energy(U_values: np.ndarray, energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([
        np.ones_like(U_values),
        1.0 / U_values,
        1.0 / (U_values**2),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(X, energies, rcond=None)
    return coeffs, X @ coeffs


def fit_half_energy(U_values: np.ndarray, energies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([
        1.0 / U_values,
        1.0 / (U_values**3),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(X, energies, rcond=None)
    return coeffs, X @ coeffs


def summarize_coefficients(results: dict, *, Nc: int, scheme_keys: list[str], u_min: float) -> None:
    U_values = np.asarray(results["U_values"], dtype=float)
    mask = U_values >= u_min
    fit_U = U_values[mask]

    ba_quarter = np.asarray(results["bethe"]["quarter"]["energy"], dtype=float)[mask]
    ba_half = np.asarray(results["bethe"]["half"]["energy"], dtype=float)[mask]
    q_ba, _ = fit_quarter_energy(fit_U, ba_quarter)
    h_ba, _ = fit_half_energy(fit_U, ba_half)

    print("\n" + "=" * 72)
    print(f"Strong-coupling fits for Nc={Nc} using U >= {u_min:g}")
    print("=" * 72)
    print(f"  Bethe quarter: c0={q_ba[0]: .8f}, c1={q_ba[1]: .8f}, c2={q_ba[2]: .8f}")
    print(f"  Bethe half   : a1={h_ba[0]: .8f}, a3={h_ba[1]: .8f}")

    for scheme_key in scheme_keys:
        ratio = sep_key_to_ratio_local(scheme_key)
        label = format_sep_as_pi(ratio)
        q_energy = np.asarray(
            results["cluster"][str(Nc)][scheme_key]["quarter"]["observables"]["energy"],
            dtype=float,
        )[mask]
        h_energy = np.asarray(
            results["cluster"][str(Nc)][scheme_key]["half"]["observables"]["energy"],
            dtype=float,
        )[mask]
        q_fit, _ = fit_quarter_energy(fit_U, q_energy)
        h_fit, _ = fit_half_energy(fit_U, h_energy)
        print(
            f"  {scheme_key:>4s} ({label:>4s}) quarter: "
            f"|Δc0|={abs(q_fit[0]-q_ba[0]):.6e}, |Δc1|={abs(q_fit[1]-q_ba[1]):.6e}"
        )
        print(
            f"  {scheme_key:>4s} ({label:>4s}) half   : "
            f"|Δa1|={abs(h_fit[0]-h_ba[0]):.6e}, |Δa3|={abs(h_fit[1]-h_ba[1]):.6e}"
        )


def sep_key_to_ratio_local(key: str) -> tuple[int, int]:
    num, den = key.split("_", 1)
    return int(num), int(den)


def make_strong_coupling_figure(
    results: dict,
    *,
    Nc: int,
    scheme_keys: list[str],
    u_min: float,
    output_dir: Path,
    show: bool,
) -> Path:
    U_values = np.asarray(results["U_values"], dtype=float)
    mask = U_values >= u_min
    fit_U = U_values[mask]
    inv_U = 1.0 / fit_U
    inv_U_sq = inv_U**2

    colors = list(plt.cm.tab10.colors)

    quarter_ba = np.asarray(results["bethe"]["quarter"]["energy"], dtype=float)[mask]
    half_ba = np.asarray(results["bethe"]["half"]["energy"], dtype=float)[mask]
    quarter_ba_coeffs, quarter_ba_fit = fit_quarter_energy(fit_U, quarter_ba)
    half_ba_coeffs, half_ba_fit = fit_half_energy(fit_U, half_ba)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    fig.suptitle(f"Strong-coupling fits vs Bethe, $N_c={Nc}$", fontsize=16, y=0.995)

    ax = axes[0, 0]
    ax.scatter(inv_U, quarter_ba, color="black", marker="s", s=35, label="Bethe")
    inv_dense = np.linspace(inv_U.min(), inv_U.max(), 200)
    U_dense = 1.0 / inv_dense
    ax.plot(inv_dense, quarter_ba_coeffs[0] + quarter_ba_coeffs[1] / U_dense + quarter_ba_coeffs[2] / (U_dense**2),
            color="black", linewidth=2, linestyle="--")
    quarter_error_bars = []
    for idx, scheme_key in enumerate(scheme_keys):
        ratio = sep_key_to_ratio_local(scheme_key)
        label = scheme_display_label(idx, len(scheme_keys), ratio)
        series = np.asarray(results["cluster"][str(Nc)][scheme_key]["quarter"]["observables"]["energy"], dtype=float)[mask]
        coeffs, _ = fit_quarter_energy(fit_U, series)
        ax.scatter(inv_U, series, color=colors[idx % len(colors)], s=35, label=label)
        ax.plot(
            inv_dense,
            coeffs[0] + coeffs[1] / U_dense + coeffs[2] / (U_dense**2),
            color=colors[idx % len(colors)],
            linewidth=2,
        )
        quarter_error_bars.append((scheme_key, coeffs))
    ax.set_title(r"Quarter filling: $e(U) = c_0 + c_1/U + c_2/U^2$")
    ax.set_xlabel(r"$1/U$")
    ax.set_ylabel(r"$e$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=10)

    ax = axes[0, 1]
    labels = [format_sep_as_pi(sep_key_to_ratio_local(key)) for key, _ in quarter_error_bars]
    x = np.arange(len(labels))
    width = 0.33
    c0_err = [abs(coeffs[0] - quarter_ba_coeffs[0]) for _, coeffs in quarter_error_bars]
    c1_err = [abs(coeffs[1] - quarter_ba_coeffs[1]) for _, coeffs in quarter_error_bars]
    ax.bar(x - width / 2, c0_err, width, label=r"$|\Delta c_0|$", color="#4c78a8")
    ax.bar(x + width / 2, c1_err, width, label=r"$|\Delta c_1|$", color="#f58518")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_title("Quarter-filling coefficient errors")
    ax.grid(True, axis="y", alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=10)

    ax = axes[1, 0]
    ax.scatter(inv_U_sq, fit_U * half_ba, color="black", marker="s", s=35, label="Bethe")
    inv_sq_dense = np.linspace(inv_U_sq.min(), inv_U_sq.max(), 200)
    ax.plot(inv_sq_dense, half_ba_coeffs[0] + half_ba_coeffs[1] * inv_sq_dense, color="black", linewidth=2, linestyle="--")
    half_error_bars = []
    for idx, scheme_key in enumerate(scheme_keys):
        ratio = sep_key_to_ratio_local(scheme_key)
        label = scheme_display_label(idx, len(scheme_keys), ratio)
        series = np.asarray(results["cluster"][str(Nc)][scheme_key]["half"]["observables"]["energy"], dtype=float)[mask]
        coeffs, _ = fit_half_energy(fit_U, series)
        ax.scatter(inv_U_sq, fit_U * series, color=colors[idx % len(colors)], s=35, label=label)
        ax.plot(inv_sq_dense, coeffs[0] + coeffs[1] * inv_sq_dense, color=colors[idx % len(colors)], linewidth=2)
        half_error_bars.append((scheme_key, coeffs))
    ax.set_title(r"Half filling: $U e(U) = a_1 + a_3/U^2$")
    ax.set_xlabel(r"$1/U^2$")
    ax.set_ylabel(r"$U e$")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    labels = [format_sep_as_pi(sep_key_to_ratio_local(key)) for key, _ in half_error_bars]
    x = np.arange(len(labels))
    a1_err = [abs(coeffs[0] - half_ba_coeffs[0]) for _, coeffs in half_error_bars]
    a3_err = [abs(coeffs[1] - half_ba_coeffs[1]) for _, coeffs in half_error_bars]
    ax.bar(x - width / 2, a1_err, width, label=r"$|\Delta a_1|$", color="#54a24b")
    ax.bar(x + width / 2, a3_err, width, label=r"$|\Delta a_3|$", color="#e45756")
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_title("Half-filling coefficient errors")
    ax.grid(True, axis="y", alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"fig2_bethe_strong_coupling_Nc{Nc}.png"
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
    parser.add_argument(
        "--schemes",
        nargs="*",
        default=None,
        help="Explicit scheme selections in the form Nc:key1,key2,... e.g. 2:1_2,1_4,1_8 4:1_4,1_8",
    )
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
    scheme_pairs = resolve_scheme_selection(
        results,
        args.cluster_sizes,
        overrides=parse_scheme_overrides(args.schemes),
    )
    for Nc in args.cluster_sizes:
        summarize_coefficients(results, Nc=Nc, scheme_keys=scheme_pairs[Nc], u_min=args.u_min)
        make_strong_coupling_figure(
            results,
            Nc=Nc,
            scheme_keys=scheme_pairs[Nc],
            u_min=args.u_min,
            output_dir=args.output_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()
