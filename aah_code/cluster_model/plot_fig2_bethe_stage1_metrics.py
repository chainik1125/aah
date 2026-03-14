"""
Plot stage-1 Bethe agreement metrics for the maximal and next-maximal schemes.

For each selected cluster size N_c, this script compares the smallest two
interaction-separation denominators present in the Fig. 2 cache:
    - maximal scheme      : smallest denominator (largest separation)
    - next-maximal scheme : second-smallest denominator

The panels show absolute errors against exact Bethe Ansatz thermodynamics for
energy, double occupancy, and kinetic energy at half and quarter filling.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aah_code.cluster_model.analyze_fig2_bethe_thermodynamics import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_OUTPUT,
    compute_fig2_bethe_thermodynamics,
    save_fig2_bethe_thermodynamics,
    sep_key_to_ratio,
)


DEFAULT_PLOT_DIR = Path("/tmp/aah-observables/large_files/plots")


def format_sep_as_pi(frac: tuple[int, int]) -> str:
    p, q = frac
    if p == 0:
        return "0"
    if q == 1:
        return f"{p}\u03c0"
    if p == 1:
        return "\u03c0" if q == 2 else f"\u03c0/{q // 2}" if q % 2 == 0 else f"2\u03c0/{q}"
    return f"{2 * p}\u03c0/{q}"


def select_default_scheme_pairs(results: dict, cluster_sizes: list[int]) -> dict[int, list[str]]:
    scheme_pairs: dict[int, list[str]] = {}
    for Nc in cluster_sizes:
        scheme_keys = sorted(results["cluster"][str(Nc)].keys(), key=sep_key_to_ratio)
        if len(scheme_keys) < 2:
            raise ValueError(f"Nc={Nc} needs at least two schemes to compare, found {scheme_keys}")
        scheme_pairs[Nc] = scheme_keys[:2]
    return scheme_pairs


def parse_scheme_overrides(specs: list[str] | None) -> dict[int, list[str]]:
    overrides: dict[int, list[str]] = {}
    if not specs:
        return overrides
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid scheme override '{spec}'. Expected format Nc:key1,key2,...")
        nc_text, keys_text = spec.split(":", 1)
        Nc = int(nc_text)
        keys = [key.strip() for key in keys_text.split(",") if key.strip()]
        if not keys:
            raise ValueError(f"Scheme override for Nc={Nc} must include at least one scheme key.")
        overrides[Nc] = keys
    return overrides


def resolve_scheme_selection(
    results: dict,
    cluster_sizes: list[int],
    overrides: dict[int, list[str]] | None = None,
) -> dict[int, list[str]]:
    available = {
        Nc: set(results["cluster"][str(Nc)].keys())
        for Nc in cluster_sizes
    }
    scheme_selection = select_default_scheme_pairs(results, cluster_sizes)
    if not overrides:
        return scheme_selection

    for Nc, keys in overrides.items():
        if Nc not in available:
            raise ValueError(f"Nc={Nc} not present in analysis results.")
        missing = [key for key in keys if key not in available[Nc]]
        if missing:
            raise ValueError(f"Nc={Nc} scheme(s) {missing} not found. Available: {sorted(available[Nc])}")
        scheme_selection[Nc] = keys
    return scheme_selection


def scheme_display_label(idx: int, total: int, ratio: tuple[int, int]) -> str:
    sep_label = format_sep_as_pi(ratio)
    if total == 1:
        return sep_label
    if idx == 0:
        return f"maximal: {sep_label}"
    if idx == 1:
        return f"winner: {sep_label}"
    return f"alt: {sep_label}"


def load_or_compute_results(
    *,
    analysis_pkl: Path,
    fig2_cache: Path,
    cluster_sizes: list[int],
    refresh: bool,
    delta_u: float,
    N_k: int,
    N_lam: int,
    B: float,
) -> dict:
    if analysis_pkl.exists() and not refresh:
        with analysis_pkl.open("rb") as fh:
            return pickle.load(fh)

    results = compute_fig2_bethe_thermodynamics(
        fig2_cache=fig2_cache,
        cluster_sizes=cluster_sizes,
        delta_u=delta_u,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    save_fig2_bethe_thermodynamics(results, analysis_pkl)
    return results


def make_stage1_figure(
    results: dict,
    *,
    Nc: int,
    scheme_keys: list[str],
    output_dir: Path,
    show: bool,
) -> Path:
    U_values = np.asarray(results["U_values"], dtype=float)
    fill_modes = [("half", "Half filling"), ("quarter", "Quarter filling")]
    observables = [
        ("energy", r"$|\Delta e|$"),
        ("double_occupancy", r"$|\Delta D|$"),
        ("kinetic", r"$|\Delta T|$"),
    ]
    colors = list(plt.cm.tab10.colors)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), sharex=True)
    fig.suptitle(f"Bethe Agreement, $N_c={Nc}$", fontsize=16, y=0.995)

    for row_idx, (fill_mode, fill_label) in enumerate(fill_modes):
        for col_idx, (observable, ylabel) in enumerate(observables):
            ax = axes[row_idx, col_idx]
            for scheme_idx, scheme_key in enumerate(scheme_keys):
                ratio = sep_key_to_ratio(scheme_key)
                errors = np.asarray(
                    results["cluster"][str(Nc)][scheme_key][fill_mode]["abs_error"][observable],
                    dtype=float,
                )
                safe_errors = np.clip(errors, 1e-12, None)
                ax.plot(
                    U_values,
                    safe_errors,
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    color=colors[scheme_idx % len(colors)],
                    label=scheme_display_label(scheme_idx, len(scheme_keys), ratio),
                )

            ax.set_yscale("log")
            ax.grid(True, alpha=0.25, which="both")
            ax.set_title(f"{fill_label}, {observable.replace('_', ' ')}")
            if row_idx == 1:
                ax.set_xlabel("U")
            ax.set_ylabel(ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"fig2_bethe_stage1_Nc{Nc}.png"
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
        make_stage1_figure(
            results,
            Nc=Nc,
            scheme_keys=scheme_pairs[Nc],
            output_dir=args.output_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()
