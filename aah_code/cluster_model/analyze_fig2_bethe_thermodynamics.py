"""
Compare Fig. 2 cluster schemes against exact Lieb-Wu thermodynamics.

This script evaluates, for selected cluster sizes and separations:
  - exact Bethe Ansatz energy / double occupancy / kinetic energy
  - cluster energy / double occupancy / kinetic energy
at the half-filled and quarter-filled benchmark points used in Fig. 2.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from aah_code.bethe_ansatz import lieb_wu_thermodynamic_components
from aah_code.cluster_model.thermodynamic_components import cluster_energy_components


DEFAULT_FIG2_CACHE = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/"
    "new_code/aah/aah_code/cluster_model/large_files/plots/"
    "filling_int_cluster_comparison_L48_chi32_20260206_113516.pkl"
)
DEFAULT_OUTPUT = Path("/tmp/aah-observables/large_files/plots/fig2_bethe_thermodynamics.pkl")


def sep_key_to_ratio(key: str) -> tuple[int, int]:
    num, den = key.split("_", 1)
    return int(num), int(den)


def summarize_fill_mode(fill_mode: str, fill_results: dict, U_values: np.ndarray) -> None:
    print("\n" + "=" * 72)
    print(f"{fill_mode.upper()} FILLING")
    print("=" * 72)
    for Nc, scheme_payload in fill_results.items():
        print(f"\nNc = {Nc}")
        for observable in ("energy", "double_occupancy", "kinetic"):
            ranking = []
            winner_counts: dict[str, int] = {}
            for scheme_key, payload in scheme_payload.items():
                err = np.asarray(payload["abs_error"][observable], dtype=float)
                ranking.append((float(np.nanmean(err)), scheme_key))
                winner_counts[scheme_key] = 0

            for u_idx, _ in enumerate(U_values):
                row = []
                for scheme_key, payload in scheme_payload.items():
                    err = float(payload["abs_error"][observable][u_idx])
                    row.append((err, scheme_key))
                row.sort()
                if row:
                    winner_counts[row[0][1]] += 1

            ranking.sort()
            print(f"  {observable}:")
            for mean_err, scheme_key in ranking:
                print(
                    f"    {scheme_key:>4s}  mean |Δ| = {mean_err:.6e}  "
                    f"winner count = {winner_counts[scheme_key]}/{len(U_values)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig2-cache", type=Path, default=DEFAULT_FIG2_CACHE)
    parser.add_argument("--cluster-sizes", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--delta-u", type=float, default=5e-2)
    parser.add_argument("--N-k", type=int, default=256)
    parser.add_argument("--N-lam", type=int, default=256)
    parser.add_argument("--B", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.fig2_cache.open("rb") as fh:
        fig2_data = pickle.load(fh)

    U_values = np.asarray(fig2_data["U_values"], dtype=float)
    params = dict(fig2_data["parameters"])
    L = int(params["L"])
    t = float(params["t"])
    V = float(fig2_data.get("V", params.get("V", 0.0)))
    v_sep_ratio = tuple(params.get("v_sep_ratio", (1, 1)))
    solver_method = str(params.get("solver_method", "sparse_ED"))
    states_retained = params.get("states_retained", 6)

    filling_targets = {"half": 1.0, "quarter": 0.5}
    if "filling_modes" in fig2_data:
        fill_modes = [mode for mode in fig2_data["filling_modes"] if mode in filling_targets]
    else:
        fill_modes = ["half", "quarter"]

    results = {
        "parameters": {
            "fig2_cache": str(args.fig2_cache),
            "L": L,
            "t": t,
            "V": V,
            "v_sep_ratio": v_sep_ratio,
            "solver_method": solver_method,
            "states_retained": states_retained,
            "delta_u": args.delta_u,
            "N_k": args.N_k,
            "N_lam": args.N_lam,
            "B": args.B,
            "cluster_sizes": args.cluster_sizes,
        },
        "U_values": U_values.tolist(),
        "bethe": {},
        "cluster": {},
    }

    for fill_mode in fill_modes:
        n_target = filling_targets[fill_mode]
        ba_components = {
            "energy": [],
            "double_occupancy": [],
            "kinetic": [],
        }
        print("\n" + "-" * 72)
        print(f"Computing Bethe Ansatz thermodynamics for {fill_mode} filling (n={n_target})")
        print("-" * 72)
        for U in U_values:
            comps = lieb_wu_thermodynamic_components(
                float(U),
                n_target,
                t=t,
                delta_u=args.delta_u,
                N_k=args.N_k,
                N_lam=args.N_lam,
                B=args.B,
            )
            for observable in ba_components:
                ba_components[observable].append(float(comps[observable]))
        results["bethe"][fill_mode] = ba_components

    for Nc in args.cluster_sizes:
        Nc_key = str(Nc)
        if Nc_key not in fig2_data["cluster_energies"]:
            raise KeyError(f"Nc={Nc} not present in Fig. 2 cache")
        scheme_keys = sorted(fig2_data["cluster_energies"][Nc_key].keys(), key=sep_key_to_ratio)
        results["cluster"][Nc_key] = {}

        print("\n" + "=" * 72)
        print(f"Cluster thermodynamics for Nc={Nc}")
        print("=" * 72)

        for scheme_key in scheme_keys:
            int_sep_ratio = sep_key_to_ratio(scheme_key)
            scheme_result = {}
            print(f"\nScheme {scheme_key} ({int_sep_ratio})")
            for fill_mode in fill_modes:
                n_target = filling_targets[fill_mode]
                central_cache = np.asarray(fig2_data["cluster_energies"][Nc_key][scheme_key][fill_mode], dtype=float)
                central_fill_cache = np.asarray(fig2_data["cluster_fillings"][Nc_key][scheme_key][fill_mode], dtype=float)

                obs = {
                    "energy": [],
                    "double_occupancy": [],
                    "kinetic": [],
                    "filling": [],
                    "mu_eff": [],
                    "cache_energy": central_cache.tolist(),
                    "cache_filling": central_fill_cache.tolist(),
                    "cache_energy_mismatch": [],
                }
                abs_error = {
                    "energy": [],
                    "double_occupancy": [],
                    "kinetic": [],
                }
                ba_fill = results["bethe"][fill_mode]

                for u_idx, U in enumerate(U_values):
                    comps = cluster_energy_components(
                        L=L,
                        Nc=Nc,
                        int_sep_ratio=int_sep_ratio,
                        filling_target=n_target,
                        U=float(U),
                        t=t,
                        V=V,
                        v_sep_ratio=v_sep_ratio,
                        solver_method=solver_method,
                        states_retained=states_retained,
                        delta_u=args.delta_u,
                    )
                    for key in ("energy", "double_occupancy", "kinetic", "filling", "mu_eff"):
                        obs[key].append(float(comps[key]))
                    obs["cache_energy_mismatch"].append(float(comps["energy"] - central_cache[u_idx]))

                    abs_error["energy"].append(abs(float(comps["energy"]) - float(ba_fill["energy"][u_idx])))
                    abs_error["double_occupancy"].append(
                        abs(float(comps["double_occupancy"]) - float(ba_fill["double_occupancy"][u_idx]))
                    )
                    abs_error["kinetic"].append(abs(float(comps["kinetic"]) - float(ba_fill["kinetic"][u_idx])))

                scheme_result[fill_mode] = {
                    "observables": obs,
                    "abs_error": abs_error,
                }

                mean_energy_error = float(np.mean(abs_error["energy"]))
                mean_d_error = float(np.mean(abs_error["double_occupancy"]))
                mean_t_error = float(np.mean(abs_error["kinetic"]))
                mean_fill = float(np.mean(obs["filling"]))
                print(
                    f"  {fill_mode:>7s}: mean |Δe|={mean_energy_error:.6e}, "
                    f"mean |ΔD|={mean_d_error:.6e}, mean |ΔT|={mean_t_error:.6e}, "
                    f"mean n={mean_fill:.6f}"
                )

            results["cluster"][Nc_key][scheme_key] = scheme_result

    summary_payload: dict[str, dict[int, dict[str, dict]]] = {}
    for fill_mode in fill_modes:
        summary_payload[fill_mode] = {}
        fill_results = {}
        for Nc in args.cluster_sizes:
            Nc_key = str(Nc)
            fill_results[Nc] = {
                scheme_key: results["cluster"][Nc_key][scheme_key][fill_mode]
                for scheme_key in results["cluster"][Nc_key]
            }
        summarize_fill_mode(fill_mode, fill_results, U_values)
        summary_payload[fill_mode] = fill_results

    results["summary"] = summary_payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        pickle.dump(results, fh)
    print(f"\nSaved analysis payload to {args.output}")


if __name__ == "__main__":
    main()
