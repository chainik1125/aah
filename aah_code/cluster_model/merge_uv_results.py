"""
Merge per-(U,V) partial pickles into a full results payload compatible with compare_U_values_with_dmrg.
"""

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np


def _write_cluster_timing_svg(timings, plots_dir: Path, filename: str = "sep_uv_timings.svg") -> Path | None:
    if not timings:
        return None

    # We only want per-supercluster diagonalisation timings.
    points = []
    for rec in timings:
        if rec.get("method") != "cluster_ED_supercluster":
            continue
        try:
            sc_size = float(rec.get("super_cluster_size", np.nan))
            elapsed = float(rec.get("elapsed_sec", np.nan))
        except Exception:
            continue
        if not (np.isfinite(sc_size) and np.isfinite(elapsed)):
            continue
        points.append((sc_size, elapsed))

    if not points:
        return None

    # Aggregate with mean and min/max errorbars.
    agg = {}
    for sc_size, elapsed in points:
        agg.setdefault(sc_size, []).append(elapsed)

    xs = sorted(agg.keys())
    means = [float(np.mean(agg[x])) for x in xs]
    mins = [float(np.min(agg[x])) for x in xs]
    maxs = [float(np.max(agg[x])) for x in xs]
    err_down = [m - lo for m, lo in zip(means, mins)]
    err_up = [hi - m for hi, m in zip(maxs, means)]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: skipping timing SVG (matplotlib unavailable): {exc}")
        return None

    plots_dir.mkdir(parents=True, exist_ok=True)
    svg_path = plots_dir / filename

    plt.figure(figsize=(8, 5))
    plt.errorbar(xs, means, yerr=[err_down, err_up], fmt="-o", capsize=4)
    plt.xlabel("Supercluster size (sites)")
    plt.ylabel("Elapsed time per supercluster (s)")
    plt.title("Cluster ED runtime per supercluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(svg_path, format="svg")
    plt.close()

    print(f"Saved timing SVG to {svg_path}")
    return svg_path


def parse_args():
    p = argparse.ArgumentParser(description="Merge partial UV result pickles.")
    p.add_argument("--partials", required=True, help="Glob pattern for partial pickles (e.g., 'large_files/partials/*.pkl').")
    p.add_argument("--output", required=True, help="Path to write merged pickle.")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.partials))
    if not paths:
        raise SystemExit(f"No partials matched pattern {args.partials}")

    partials = []
    for p in paths:
        with open(p, "rb") as fh:
            partials.append(pickle.load(fh))

    # Collect parameter grids
    U_set = set()
    V_set = set()
    cluster_sizes = None
    int_sep_ratios = None
    base_params = {}
    timings = []

    for res in partials:
        U_list = res.get("U_values", [])
        V_list = res.get("V_values", [])
        if len(U_list) != 1 or len(V_list) != 1:
            raise ValueError("Partial payloads must have singleton U_values and V_values.")
        U_set.add(float(U_list[0]))
        V_set.add(float(V_list[0]))
        if cluster_sizes is None:
            cluster_sizes = res["cluster_sizes"]
        if int_sep_ratios is None:
            int_sep_ratios = res.get("int_sep_ratios")
        if not base_params:
            base_params = res.get("parameters", {})
        if "timings" in res:
            timings.extend(res["timings"])

    U_values = sorted(U_set)
    V_values = sorted(V_set)
    u_idx = {u: i for i, u in enumerate(U_values)}
    v_idx = {v: i for i, v in enumerate(V_values)}

    # Initialise accumulators
    cluster_results = {(Nc, V): np.full(len(U_values), np.nan) for Nc in cluster_sizes for V in V_values}
    cluster_fillings = {(Nc, V): np.full(len(U_values), np.nan) for Nc in cluster_sizes for V in V_values}
    idmrg_cache = np.full((len(U_values), len(V_values)), np.nan)
    idmrg_fill_cache = np.full((len(U_values), len(V_values)), np.nan)
    finite_dmrg_cache = np.full((len(U_values), len(V_values)), np.nan)
    finite_dmrg_fill_cache = np.full((len(U_values), len(V_values)), np.nan)

    # Fill accumulators
    for res in partials:
        U = float(res["U_values"][0])
        V = float(res["V_values"][0])
        ui = u_idx[U]
        vi = v_idx[V]

        for Nc, v_map in res.get("cluster_energies", {}).items():
            Nc_int = int(Nc)
            # keys may be strings or floats
            series = None
            for v_key, vals in v_map.items():
                if float(v_key) == V:
                    series = np.array(vals, dtype=float)
                    break
            if series is not None and series.size:
                cluster_results[(Nc_int, V)][ui] = series.flatten()[0]

        for Nc, v_map in res.get("cluster_fillings", {}).items():
            Nc_int = int(Nc)
            series = None
            for v_key, vals in v_map.items():
                if float(v_key) == V:
                    series = np.array(vals, dtype=float)
                    break
            if series is not None and series.size:
                cluster_fillings[(Nc_int, V)][ui] = series.flatten()[0]

        dmrg_E = np.array(res.get("idmrg_energies", []), dtype=float)
        if dmrg_E.size:
            idmrg_cache[ui, vi] = dmrg_E.flatten()[0]
        dmrg_fill = np.array(res.get("idmrg_fillings", []), dtype=float)
        if dmrg_fill.size:
            idmrg_fill_cache[ui, vi] = dmrg_fill.flatten()[0]

        f_dmrg_E = np.array(res.get("finite_dmrg_energies", []), dtype=float)
        if f_dmrg_E.size:
            finite_dmrg_cache[ui, vi] = f_dmrg_E.flatten()[0]
        f_dmrg_fill = np.array(res.get("finite_dmrg_fillings", []), dtype=float)
        if f_dmrg_fill.size:
            finite_dmrg_fill_cache[ui, vi] = f_dmrg_fill.flatten()[0]

    cluster_energy_serialized = {
        Nc: {V: cluster_results[(Nc, V)].tolist() for V in V_values}
        for Nc in cluster_sizes
    }
    cluster_fillings_serialized = {
        Nc: {V: cluster_fillings[(Nc, V)].tolist() for V in V_values}
        for Nc in cluster_sizes
    }

    merged = {
        "cluster_sizes": cluster_sizes,
        "U_values": U_values,
        "V_values": V_values,
        "cluster_energies": cluster_energy_serialized,
        "cluster_fillings": cluster_fillings_serialized,
        "idmrg_energies": idmrg_cache.tolist(),
        "finite_dmrg_energies": finite_dmrg_cache.tolist(),
        "idmrg_fillings": idmrg_fill_cache.tolist(),
        "finite_dmrg_fillings": finite_dmrg_fill_cache.tolist(),
        "int_sep_ratios": int_sep_ratios,
        "parameters": base_params,
        "failures": [],
    }
    if timings:
        merged["timings"] = timings

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(merged, fh)
    print(f"Saved merged results to {out_path}")

    # Also create a compact timing SVG next to the energy SVG output location.
    plots_dir = out_path.parent / "plots"
    _write_cluster_timing_svg(merged.get("timings", []), plots_dir=plots_dir)


if __name__ == "__main__":
    main()
