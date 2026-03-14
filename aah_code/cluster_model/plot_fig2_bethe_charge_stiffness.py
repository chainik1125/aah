"""
Quarter-filled charge-stiffness comparison between Bethe Ansatz and cluster schemes.

This script uses the canonical cluster charge-sector helpers and compares the
charge stiffness D_c(U) against the Bethe-ansatz metallic charge sector.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aah_code.bethe_ansatz import lieb_wu_charge_sector_parameters
from aah_code.cluster_model.plot_fig2_bethe_stage1_metrics import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_PLOT_DIR,
    format_sep_as_pi,
    parse_scheme_overrides,
    scheme_display_label,
)
from aah_code.cluster_model.thermodynamic_components import cluster_charge_sector_parameters


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


def compute_charge_stiffness_results(
    *,
    fig2_cache: Path,
    cluster_sizes: list[int],
    scheme_overrides: dict[int, list[str]] | None = None,
    delta_n: float = 1e-2,
    delta_phi: float = 5e-2,
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

    bethe = {"charge_stiffness": []}
    for U in U_values:
        vals = lieb_wu_charge_sector_parameters(
            float(U),
            0.5,
            t=t,
            delta_n=1e-3,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )
        bethe["charge_stiffness"].append(float(vals["D_c"]))

    selected = resolve_scheme_selection(fig2_data, cluster_sizes, scheme_overrides)
    cluster: dict[str, dict[str, dict[str, list[float]]]] = {}
    for Nc in cluster_sizes:
        cluster[str(Nc)] = {}
        for scheme_key in selected[Nc]:
            ratio = sep_key_to_ratio(scheme_key)
            payload = {"charge_stiffness": [], "abs_error": []}
            for u_idx, U in enumerate(U_values):
                vals = cluster_charge_sector_parameters(
                    L=L,
                    Nc=Nc,
                    int_sep_ratio=ratio,
                    filling_target=0.5,
                    U=float(U),
                    t=t,
                    V=V,
                    v_sep_ratio=v_sep_ratio,
                    solver_method=solver_method,
                    states_retained=states_retained,
                    delta_n=delta_n,
                    delta_phi=delta_phi,
                )
                d_c = float(vals["charge_stiffness"])
                payload["charge_stiffness"].append(d_c)
                payload["abs_error"].append(abs(d_c - bethe["charge_stiffness"][u_idx]))
            cluster[str(Nc)][scheme_key] = payload

    return {
        "U_values": U_values.tolist(),
        "bethe": bethe,
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
            "delta_phi": delta_phi,
            "N_k": N_k,
            "N_lam": N_lam,
            "B": B,
        },
    }


def make_charge_stiffness_figures(results: dict, *, output_dir: Path, show: bool) -> list[Path]:
    U_values = np.asarray(results["U_values"], dtype=float)
    bethe_dc = np.asarray(results["bethe"]["charge_stiffness"], dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for Nc_str, scheme_payload in results["cluster"].items():
        Nc = int(Nc_str)
        scheme_keys = results["selected_pairs"][Nc]
        colors = list(plt.cm.tab10.colors)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
        fig.suptitle(f"Quarter-filled charge stiffness vs Bethe, $N_c={Nc}$", fontsize=16, y=0.995)

        ax = axes[0]
        ax.plot(U_values, bethe_dc, color="black", linestyle="--", linewidth=2, marker="s", label="Bethe")
        for idx, scheme_key in enumerate(scheme_keys):
            ratio = sep_key_to_ratio(scheme_key)
            ax.plot(
                U_values,
                scheme_payload[scheme_key]["charge_stiffness"],
                color=colors[idx % len(colors)],
                linewidth=2,
                marker="o",
                label=scheme_display_label(idx, len(scheme_keys), ratio),
            )
        ax.set_title(r"Charge stiffness $D_c(U)$")
        ax.set_xlabel("U")
        ax.set_ylabel(r"$D_c$")
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
        ax.set_title(r"Charge-stiffness error $|\Delta D_c|$")
        ax.set_xlabel("U")
        ax.set_ylabel(r"$|\Delta D_c|$")
        ax.grid(True, alpha=0.25, which="both")

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        png_path = output_dir / f"fig2_bethe_charge_stiffness_Nc{Nc}.png"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        print(f"Saved {png_path}")
        if show:
            plt.show()
        plt.close(fig)
        saved_paths.append(png_path)

        print("\n" + "=" * 72)
        print(f"Quarter-filled charge-stiffness agreement for Nc={Nc}")
        print("=" * 72)
        for scheme_key in scheme_keys:
            ratio = format_sep_as_pi(sep_key_to_ratio(scheme_key))
            d_err = float(np.mean(np.asarray(scheme_payload[scheme_key]["abs_error"], dtype=float)))
            print(f"  {scheme_key:>4s} ({ratio:>4s}): mean |ΔD_c|={d_err:.6e}")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    parser.add_argument("--cluster-sizes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--delta-n", type=float, default=1e-2)
    parser.add_argument("--delta-phi", type=float, default=5e-2)
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

    results = compute_charge_stiffness_results(
        fig2_cache=args.fig2_cache,
        cluster_sizes=args.cluster_sizes,
        scheme_overrides=parse_scheme_overrides(args.schemes),
        delta_n=args.delta_n,
        delta_phi=args.delta_phi,
        N_k=args.N_k,
        N_lam=args.N_lam,
        B=args.B,
    )
    make_charge_stiffness_figures(results, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
