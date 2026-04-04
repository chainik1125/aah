"""
Benchmark script: compare cluster ED and finite DMRG performance.

Runs a fixed parameter set (Nc=4, L=48, chi=32) and reports per-task timings.
Designed to compare local Mac vs RunPod CPU (and later, parallel scaling).

Usage:
    python -m aah_code.cluster_model.benchmark
    python -m aah_code.cluster_model.benchmark --n_jobs=4
    python -m aah_code.cluster_model.benchmark --skip_dmrg
"""

import argparse
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations
from aah_code.real_space_dmrg import get_gnd


# Fixed benchmark parameters
L = 48
CHI = 32
V = 0.0
T = 1.0
V_SEP_RATIO = (1, 2)
NC = 4
INT_SEP = (1, 4)
U_VALUES = [0.0, 1.0, 5.0, 10.0]
FILLING_TARGETS = {"half": 0.5, "quarter": 0.25}


def run_single_cluster_ed(U: float, fill_mode: str) -> tuple[float, float]:
    """Run a single cluster ED computation. Returns (energy_per_site, elapsed_seconds)."""
    target_filling = FILLING_TARGETS[fill_mode]
    mu0 = U / 2.0

    physical_params = PhysicalParams(U=U, mu_0=mu0, V=V, t=T)
    run_config = ClusterModelConfig(
        L=L,
        int_cluster_size=NC,
        cluster_separation_ratio=INT_SEP,
        V_separation_ratio=V_SEP_RATIO,
        ham_lib="quspin",
        physical_params=physical_params,
        model_bc="periodic",
        int_cluster_bc="periodic",
        super_cluster_bc="periodic",
        solver_method="sparse_ED",
        states_retained=6,
    )

    t0 = time.perf_counter()
    system_expectations, _, _ = get_general_expectations(
        run_config, set_filling=target_filling, return_mu=True
    )
    elapsed = time.perf_counter() - t0

    energy, filling, _ = system_expectations
    energy_per_site = (energy + mu0 * filling) / L
    return energy_per_site, elapsed


def run_single_dmrg(U: float, fill_mode: str) -> tuple[float, float]:
    """Run a single finite DMRG computation. Returns (energy_per_site, elapsed_seconds)."""
    mu0 = U / 2.0

    t0 = time.perf_counter()
    energy, _, filling = get_gnd(
        L, CHI, U=U, t=T, mu=mu0, V=V, V_sep=V_SEP_RATIO, bc="periodic"
    )
    elapsed = time.perf_counter() - t0

    energy_per_site = (energy + mu0 * filling) / L
    return energy_per_site, elapsed


def run_benchmark(n_jobs: int = 1, skip_dmrg: bool = False) -> dict:
    """Run the full benchmark and return results dict."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "parameters": {
            "L": L, "chi": CHI, "V": V, "Nc": NC,
            "int_sep": INT_SEP, "U_values": U_VALUES,
            "n_jobs": n_jobs,
        },
        "cluster_ed": [],
        "finite_dmrg": [],
    }

    # --- Cluster ED benchmark ---
    tasks = [
        (U, fill_mode)
        for fill_mode in FILLING_TARGETS
        for U in U_VALUES
    ]
    total_ed = len(tasks)

    print(f"\n{'='*60}")
    print(f"Cluster ED benchmark: {total_ed} tasks (Nc={NC}, L={L})")
    print(f"n_jobs={n_jobs}")
    print(f"{'='*60}")

    ed_total_start = time.perf_counter()

    if n_jobs == 1:
        for i, (U, fill_mode) in enumerate(tasks):
            print(f"  [{i+1}/{total_ed}] U={U}, {fill_mode}...", end=" ", flush=True)
            energy, elapsed = run_single_cluster_ed(U, fill_mode)
            print(f"{elapsed:.2f}s  E/L={energy:.6f}")
            results["cluster_ed"].append({
                "U": U, "fill_mode": fill_mode,
                "energy_per_site": float(energy),
                "elapsed_s": elapsed,
            })
    else:
        from multiprocessing import Pool

        # Limit BLAS threads per worker to avoid oversubscription
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["OPENBLAS_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"

        def _ed_worker(args):
            U, fill_mode = args
            energy, elapsed = run_single_cluster_ed(U, fill_mode)
            return {"U": U, "fill_mode": fill_mode,
                    "energy_per_site": float(energy), "elapsed_s": elapsed}

        actual_jobs = min(n_jobs if n_jobs > 0 else os.cpu_count(), total_ed)
        print(f"  Using {actual_jobs} workers for {total_ed} tasks")
        with Pool(actual_jobs) as pool:
            results["cluster_ed"] = pool.map(_ed_worker, tasks)
        for r in results["cluster_ed"]:
            print(f"  U={r['U']}, {r['fill_mode']}: {r['elapsed_s']:.2f}s  E/L={r['energy_per_site']:.6f}")

    ed_total_elapsed = time.perf_counter() - ed_total_start
    results["cluster_ed_total_s"] = ed_total_elapsed
    print(f"\nCluster ED total: {ed_total_elapsed:.2f}s")

    # --- Finite DMRG benchmark ---
    if not skip_dmrg:
        dmrg_tasks = [
            (U, fill_mode)
            for fill_mode in FILLING_TARGETS
            for U in U_VALUES
        ]
        total_dmrg = len(dmrg_tasks)

        print(f"\n{'='*60}")
        print(f"Finite DMRG benchmark: {total_dmrg} tasks (L={L}, chi={CHI}, PBC)")
        print(f"n_jobs={n_jobs}")
        print(f"{'='*60}")

        dmrg_total_start = time.perf_counter()

        if n_jobs == 1:
            for i, (U, fill_mode) in enumerate(dmrg_tasks):
                print(f"  [{i+1}/{total_dmrg}] U={U}, {fill_mode}...", end=" ", flush=True)
                energy, elapsed = run_single_dmrg(U, fill_mode)
                print(f"{elapsed:.2f}s  E/L={energy:.6f}")
                results["finite_dmrg"].append({
                    "U": U, "fill_mode": fill_mode,
                    "energy_per_site": float(energy),
                    "elapsed_s": elapsed,
                })
        else:
            from multiprocessing import Pool

            def _dmrg_worker(args):
                U, fill_mode = args
                energy, elapsed = run_single_dmrg(U, fill_mode)
                return {"U": U, "fill_mode": fill_mode,
                        "energy_per_site": float(energy), "elapsed_s": elapsed}

            actual_jobs = min(n_jobs if n_jobs > 0 else os.cpu_count(), total_dmrg)
            print(f"  Using {actual_jobs} workers for {total_dmrg} tasks")
            with Pool(actual_jobs) as pool:
                results["finite_dmrg"] = pool.map(_dmrg_worker, dmrg_tasks)
            for r in results["finite_dmrg"]:
                print(f"  U={r['U']}, {r['fill_mode']}: {r['elapsed_s']:.2f}s  E/L={r['energy_per_site']:.6f}")

        dmrg_total_elapsed = time.perf_counter() - dmrg_total_start
        results["finite_dmrg_total_s"] = dmrg_total_elapsed
        print(f"\nFinite DMRG total: {dmrg_total_elapsed:.2f}s")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    ed_times = [r["elapsed_s"] for r in results["cluster_ed"]]
    print(f"Cluster ED: {len(ed_times)} tasks, "
          f"total={results['cluster_ed_total_s']:.1f}s, "
          f"mean={np.mean(ed_times):.2f}s, "
          f"min={np.min(ed_times):.2f}s, max={np.max(ed_times):.2f}s")
    if results["finite_dmrg"]:
        dmrg_times = [r["elapsed_s"] for r in results["finite_dmrg"]]
        print(f"Finite DMRG: {len(dmrg_times)} tasks, "
              f"total={results['finite_dmrg_total_s']:.1f}s, "
              f"mean={np.mean(dmrg_times):.2f}s, "
              f"min={np.min(dmrg_times):.2f}s, max={np.max(dmrg_times):.2f}s")

    # Save results
    out_dir = Path(__file__).parent / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    hostname = platform.node().split(".")[0] or "unknown"
    fname = f"benchmark_{hostname}_nj{n_jobs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark cluster ED and DMRG")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel workers (1=serial, -1=all cores)")
    parser.add_argument("--skip_dmrg", action="store_true",
                        help="Skip finite DMRG benchmark (cluster ED only)")
    args = parser.parse_args()

    run_benchmark(n_jobs=args.n_jobs, skip_dmrg=args.skip_dmrg)
