"""
Quarter-filled density-response comparison between Bethe Ansatz and cluster schemes.

This script compares the maximal scheme against the actual quarter-filled
Fig. 2 winner for each selected cluster size. The observables are:
  - chemical potential mu(n, U)
  - inverse compressibility kappa^{-1} = n^2 d mu / d n
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aah_code.bethe_ansatz import lieb_wu_thermodynamic_components
from aah_code.cluster_model.plot_fig2_bethe_stage1_metrics import (
    DEFAULT_FIG2_CACHE,
    DEFAULT_PLOT_DIR,
    format_sep_as_pi,
    parse_scheme_overrides,
    scheme_display_label,
)
from aah_code.cluster_model.thermodynamic_components import cluster_density_response_components


def sep_key_to_ratio(key: str) -> tuple[int, int]:
    num, den = key.split("_", 1)
    return int(num), int(den)


def select_maximal_and_quarter_winner(fig2_data: dict, *, Nc: int) -> list[str]:
    scheme_keys = sorted(fig2_data["cluster_energies"][str(Nc)].keys(), key=sep_key_to_ratio)
    maximal_key = scheme_keys[0]

    finite_dmrg = np.asarray(fig2_data["finite_dmrg_energies"], dtype=float)[:, 1]
    winner_key = None
    winner_score = None
    for scheme_key in scheme_keys:
        series = np.asarray(fig2_data["cluster_energies"][str(Nc)][scheme_key]["quarter"], dtype=float)
        score = float(np.nanmean(np.abs(series - finite_dmrg)))
        if winner_score is None or score < winner_score:
            winner_score = score
            winner_key = scheme_key

    if winner_key is None:
        raise ValueError(f"Could not determine quarter-filled winner for Nc={Nc}")
    if winner_key == maximal_key:
        return [maximal_key]
    return [maximal_key, winner_key]


def resolve_scheme_selection(fig2_data: dict, cluster_sizes: list[int], overrides: dict[int, list[str]] | None) -> dict[int, list[str]]:
    selection: dict[int, list[str]] = {}
    for Nc in cluster_sizes:
        available = set(fig2_data["cluster_energies"][str(Nc)].keys())
        if overrides and Nc in overrides:
            keys = overrides[Nc]
            missing = [key for key in keys if key not in available]
            if missing:
                raise ValueError(f"Nc={Nc} scheme(s) {missing} not found. Available: {sorted(available)}")
            selection[Nc] = keys
        else:
            selection[Nc] = select_maximal_and_quarter_winner(fig2_data, Nc=Nc)
    return selection


def compute_density_response_results(
    *,
    fig2_cache: Path,
    cluster_sizes: list[int],
    scheme_overrides: dict[int, list[str]] | None = None,
    delta_n: float = 2e-2,
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

    bethe = {"chemical_potential": [], "inverse_compressibility": []}
    for U in U_values:
        comps = lieb_wu_thermodynamic_components(
            float(U),
            0.5,
            t=t,
            delta_n=delta_n,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )
        bethe["chemical_potential"].append(float(comps["chemical_potential"]))
        bethe["inverse_compressibility"].append(float(comps["inverse_compressibility"]))

    cluster = {}
    selected_pairs = resolve_scheme_selection(fig2_data, cluster_sizes, scheme_overrides)
    for Nc in cluster_sizes:
        keys = selected_pairs[Nc]
        cluster[str(Nc)] = {}
        for scheme_key in keys:
            ratio = sep_key_to_ratio(scheme_key)
            scheme_payload = {
                "chemical_potential": [],
                "inverse_compressibility": [],
                "abs_error": {
                    "chemical_potential": [],
                    "inverse_compressibility": [],
                },
            }
            for u_idx, U in enumerate(U_values):
                comps = cluster_density_response_components(
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
                )
                mu = float(comps["chemical_potential"])
                inv_kappa = float(comps["inverse_compressibility"])
                scheme_payload["chemical_potential"].append(mu)
                scheme_payload["inverse_compressibility"].append(inv_kappa)
                scheme_payload["abs_error"]["chemical_potential"].append(abs(mu - bethe["chemical_potential"][u_idx]))
                scheme_payload["abs_error"]["inverse_compressibility"].append(
                    abs(inv_kappa - bethe["inverse_compressibility"][u_idx])
                )
            cluster[str(Nc)][scheme_key] = scheme_payload

    return {
        "U_values": U_values.tolist(),
        "bethe": bethe,
        "cluster": cluster,
        "selected_pairs": selected_pairs,
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


def make_density_response_figures(results: dict, *, output_dir: Path, show: bool) -> list[Path]:
    U_values = np.asarray(results["U_values"], dtype=float)
    bethe_mu = np.asarray(results["bethe"]["chemical_potential"], dtype=float)
    bethe_inv_kappa = np.asarray(results["bethe"]["inverse_compressibility"], dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for Nc_str, scheme_payload in results["cluster"].items():
        Nc = int(Nc_str)
        scheme_keys = results["selected_pairs"][Nc]
        colors = list(plt.cm.tab10.colors)
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
        fig.suptitle(f"Quarter-filled density response vs Bethe, $N_c={Nc}$", fontsize=16, y=0.995)

        ax = axes[0, 0]
        ax.plot(U_values, bethe_mu, color="black", linestyle="--", linewidth=2, marker="s", label="Bethe")
        for idx, scheme_key in enumerate(scheme_keys):
            ratio = sep_key_to_ratio(scheme_key)
            ax.plot(
                U_values,
                scheme_payload[scheme_key]["chemical_potential"],
                color=colors[idx % len(colors)],
                linewidth=2,
                marker="o",
                label=scheme_display_label(idx, len(scheme_keys), ratio),
            )
        ax.set_title(r"Chemical potential $\mu(U)$ at $n=1/2$")
        ax.set_ylabel(r"$\mu$")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=10)

        ax = axes[0, 1]
        ax.plot(U_values, bethe_inv_kappa, color="black", linestyle="--", linewidth=2, marker="s", label="Bethe")
        for idx, scheme_key in enumerate(scheme_keys):
            ratio = sep_key_to_ratio(scheme_key)
            ax.plot(
                U_values,
                scheme_payload[scheme_key]["inverse_compressibility"],
                color=colors[idx % len(colors)],
                linewidth=2,
                marker="o",
                label=scheme_display_label(idx, len(scheme_keys), ratio),
            )
        ax.set_title(r"Inverse compressibility $\kappa^{-1}(U)$")
        ax.set_ylabel(r"$\kappa^{-1}$")
        ax.grid(True, alpha=0.25)

        ax = axes[1, 0]
        for idx, scheme_key in enumerate(scheme_keys):
            ratio = sep_key_to_ratio(scheme_key)
            ax.plot(
                U_values,
                np.clip(np.asarray(scheme_payload[scheme_key]["abs_error"]["chemical_potential"], dtype=float), 1e-12, None),
                color=colors[idx % len(colors)],
                linewidth=2,
                marker="o",
                label=scheme_display_label(idx, len(scheme_keys), ratio),
            )
        ax.set_yscale("log")
        ax.set_title(r"Chemical-potential error $|\Delta \mu|$")
        ax.set_xlabel("U")
        ax.set_ylabel(r"$|\Delta \mu|$")
        ax.grid(True, alpha=0.25, which="both")

        ax = axes[1, 1]
        for idx, scheme_key in enumerate(scheme_keys):
            ratio = sep_key_to_ratio(scheme_key)
            ax.plot(
                U_values,
                np.clip(
                    np.asarray(scheme_payload[scheme_key]["abs_error"]["inverse_compressibility"], dtype=float),
                    1e-12,
                    None,
                ),
                color=colors[idx % len(colors)],
                linewidth=2,
                marker="o",
                label=scheme_display_label(idx, len(scheme_keys), ratio),
            )
        ax.set_yscale("log")
        ax.set_title(r"Inverse-compressibility error $|\Delta \kappa^{-1}|$")
        ax.set_xlabel("U")
        ax.set_ylabel(r"$|\Delta \kappa^{-1}|$")
        ax.grid(True, alpha=0.25, which="both")

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        png_path = output_dir / f"fig2_bethe_density_response_Nc{Nc}.png"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        print(f"Saved {png_path}")
        if show:
            plt.show()
        plt.close(fig)
        saved_paths.append(png_path)

        print("\n" + "=" * 72)
        print(f"Quarter-filled density-response agreement for Nc={Nc}")
        print("=" * 72)
        for scheme_key in scheme_keys:
            ratio = format_sep_as_pi(sep_key_to_ratio(scheme_key))
            mu_err = float(np.mean(np.asarray(scheme_payload[scheme_key]["abs_error"]["chemical_potential"], dtype=float)))
            invk_err = float(
                np.mean(np.asarray(scheme_payload[scheme_key]["abs_error"]["inverse_compressibility"], dtype=float))
            )
            print(f"  {scheme_key:>4s} ({ratio:>4s}): mean |Δμ|={mu_err:.6e}, mean |Δκ^-1|={invk_err:.6e}")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    parser.add_argument("--cluster-sizes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--delta-n", type=float, default=2e-2)
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

    results = compute_density_response_results(
        fig2_cache=args.fig2_cache,
        cluster_sizes=args.cluster_sizes,
        scheme_overrides=parse_scheme_overrides(args.schemes),
        delta_n=args.delta_n,
        N_k=args.N_k,
        N_lam=args.N_lam,
        B=args.B,
    )
    make_density_response_figures(results, output_dir=args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
