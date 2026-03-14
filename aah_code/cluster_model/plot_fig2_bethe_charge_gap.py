"""
Half-filled charge-gap comparison between Bethe Ansatz and selected cluster schemes.

The gap is estimated from the jump in the chemical potential around n=1:
    Delta_c = mu_+ - mu_-
with one-sided finite differences in the energy density.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aah_code.bethe_ansatz import lieb_wu_charge_gap_half_filling
from aah_code.cluster_model.plot_fig2_bethe_stage1_metrics import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_PLOT_DIR,
    format_sep_as_pi,
    parse_scheme_overrides,
    scheme_display_label,
)
from aah_code.cluster_model.thermodynamic_components import cluster_charge_gap_half_filling


def sep_key_to_ratio(key: str) -> tuple[int, int]:
    num, den = key.split("_", 1)
    return int(num), int(den)


def default_scheme_selection(fig2_data: dict, cluster_sizes: list[int]) -> dict[int, list[str]]:
    selection: dict[int, list[str]] = {}
    for Nc in cluster_sizes:
        scheme_keys = sorted(fig2_data["cluster_energies"][str(Nc)].keys(), key=sep_key_to_ratio)
        selection[Nc] = scheme_keys[:2]
    return selection


def resolve_scheme_selection(fig2_data: dict, cluster_sizes: list[int], overrides: dict[int, list[str]] | None) -> dict[int, list[str]]:
    selection = default_scheme_selection(fig2_data, cluster_sizes)
    if not overrides:
        return selection

    for Nc, keys in overrides.items():
        available = set(fig2_data["cluster_energies"][str(Nc)].keys())
        missing = [key for key in keys if key not in available]
        if missing:
            raise ValueError(f"Nc={Nc} scheme(s) {missing} not found. Available: {sorted(available)}")
        selection[Nc] = keys
    return selection


def compute_charge_gap_results(
    *,
    fig2_cache: Path,
    cluster_sizes: list[int],
    scheme_overrides: dict[int, list[str]] | None = None,
    delta_n: float = 1e-2,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> dict:
    with fig2_cache.open("rb") as fh:
        fig2_data = pickle.load(fh)

    U_values = np.asarray(fig2_data["U_values"], dtype=float)
    params = dict(fig2_data["parameters"])
    L = int(params["L"])
    t = float(params["t"])
    V = float(fig2_data.get("V", params.get("V", 0.0)))
    v_sep_ratio = tuple(params.get("v_sep_ratio", (1, 1)))
    solver_method = str(params.get("solver_method", "sparse_ED"))
    states_retained = params.get("states_retained", 6)

    bethe_gap = []
    for U in U_values:
        bethe_gap.append(
            float(
                lieb_wu_charge_gap_half_filling(
                    float(U),
                    t=t,
                    delta_n=delta_n,
                    N_k=N_k,
                    N_lam=N_lam,
                    B=B,
                )
            )
        )

    selected = resolve_scheme_selection(fig2_data, cluster_sizes, scheme_overrides)
    cluster = {}
    for Nc in cluster_sizes:
        cluster[str(Nc)] = {}
        for scheme_key in selected[Nc]:
            ratio = sep_key_to_ratio(scheme_key)
            payload = {
                "charge_gap": [],
                "abs_error": [],
            }
            for u_idx, U in enumerate(U_values):
                comps = cluster_charge_gap_half_filling(
                    L=L,
                    Nc=Nc,
                    int_sep_ratio=ratio,
                    U=float(U),
                    t=t,
                    V=V,
                    v_sep_ratio=v_sep_ratio,
                    solver_method=solver_method,
                    states_retained=states_retained,
                    delta_n=delta_n,
                )
                gap = float(comps["charge_gap"])
                payload["charge_gap"].append(gap)
                payload["abs_error"].append(abs(gap - bethe_gap[u_idx]))
            cluster[str(Nc)][scheme_key] = payload

    return {
        "U_values": U_values.tolist(),
        "bethe": {"charge_gap": bethe_gap},
        "cluster": cluster,
        "selected_pairs": selected,
        "parameters": {
            "fig2_cache": str(fig2_cache),
            "L": L,
            "t": t,
            "V": V,
            "v_sep_ratio": v_sep_ratio,
            "solver_method": solver_method,
            "states_retained": states_retained,
            "delta_n": delta_n,
            "N_k": N_k,
            "N_lam": N_lam,
            "B": B,
        },
    }


def make_charge_gap_figures(results: dict, *, output_dir: Path, show: bool) -> list[Path]:
    U_values = np.asarray(results["U_values"], dtype=float)
    bethe_gap = np.asarray(results["bethe"]["charge_gap"], dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for Nc_str, scheme_payload in results["cluster"].items():
        Nc = int(Nc_str)
        scheme_keys = results["selected_pairs"][Nc]
        colors = list(plt.cm.tab10.colors)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
        fig.suptitle(f"Half-filled charge gap vs Bethe, $N_c={Nc}$", fontsize=16, y=0.995)

        ax = axes[0]
        ax.plot(U_values, bethe_gap, color="black", linestyle="--", linewidth=2, marker="s", label="Bethe")
        for idx, scheme_key in enumerate(scheme_keys):
            ratio = sep_key_to_ratio(scheme_key)
            ax.plot(
                U_values,
                scheme_payload[scheme_key]["charge_gap"],
                color=colors[idx % len(colors)],
                linewidth=2,
                marker="o",
                label=scheme_display_label(idx, len(scheme_keys), ratio),
            )
        ax.set_title(r"Charge gap $\Delta_c(U)$")
        ax.set_xlabel("U")
        ax.set_ylabel(r"$\Delta_c$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=10)

        ax = axes[1]
        for idx, scheme_key in enumerate(scheme_keys):
            ratio = sep_key_to_ratio(scheme_key)
            ax.plot(
                U_values,
                np.clip(np.asarray(scheme_payload[scheme_key]["abs_error"], dtype=float), 1e-12, None),
                color=colors[idx % len(colors)],
                linewidth=2,
                marker="o",
                label=scheme_display_label(idx, len(scheme_keys), ratio),
            )
        ax.set_yscale("log")
        ax.set_title(r"Charge-gap error $|\Delta \Delta_c|$")
        ax.set_xlabel("U")
        ax.set_ylabel(r"$|\Delta \Delta_c|$")
        ax.grid(True, alpha=0.25, which="both")

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        png_path = output_dir / f"fig2_bethe_charge_gap_Nc{Nc}.png"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        print(f"Saved {png_path}")
        if show:
            plt.show()
        plt.close(fig)
        saved_paths.append(png_path)

        print("\n" + "=" * 72)
        print(f"Half-filled charge-gap agreement for Nc={Nc}")
        print("=" * 72)
        for scheme_key in scheme_keys:
            ratio = format_sep_as_pi(sep_key_to_ratio(scheme_key))
            gap_err = float(np.mean(np.asarray(scheme_payload[scheme_key]["abs_error"], dtype=float)))
            print(f"  {scheme_key:>4s} ({ratio:>4s}): mean |ΔΔ_c|={gap_err:.6e}")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    parser.add_argument("--cluster-sizes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--delta-n", type=float, default=1e-2)
    parser.add_argument("--N-k", type=int, default=256)
    parser.add_argument("--N-lam", type=int, default=256)
    parser.add_argument("--B", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--schemes",
        nargs="*",
        default=None,
        help="Explicit scheme selections in the form Nc:key1,key2,... e.g. 2:1_2,1_4,1_8 4:1_4,1_8",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    results = compute_charge_gap_results(
        fig2_cache=args.fig2_cache,
        cluster_sizes=args.cluster_sizes,
        scheme_overrides=parse_scheme_overrides(args.schemes),
        delta_n=args.delta_n,
        N_k=args.N_k,
        N_lam=args.N_lam,
        B=args.B,
    )
    make_charge_gap_figures(results, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
