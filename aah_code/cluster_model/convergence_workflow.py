"""
Workflow helpers for studying convergence of the K-blocking scheme across
different Aubry-Andre approximants and interaction cluster separations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from aah_code.cluster_model.clustering import enumerate_m, step_from_ratio
from aah_code.cluster_model.hurwitz import Approximant, generate_optimal_approximants
from aah_code.cluster_model.plots import compare_int_seps_with_dmrg


# --------------------------------------------------------------------------- #
# Dataclasses describing sweep configuration and per-approximant tasks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SweepGrid:
    """
    Container describing the parameter sweep shared by all tasks.
    """

    x_axis: Dict[str, Sequence[float]]
    varying_parameter: Dict[str, Sequence[float]]
    fixed_parameter: Dict[str, float]
    solver_method: str = "sparse_ED"
    chi: int = 32
    states_retained: int = 6
    include_idmrg: bool = False
    include_finite_dmrg: bool = True


@dataclass(frozen=True)
class ApproximantTask:
    """
    Defines a single convergence experiment for a given approximant and
    cluster size.
    """

    approximant: Approximant
    Nc: int
    max_supercluster_size: int
    base_L: int
    L_override: Optional[int] = None

    @property
    def v_sep_ratio(self) -> Tuple[int, int]:
        return self.approximant.as_ratio()


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        raise ValueError("LCM requires non-zero integers.")
    return abs(a * b) // gcd(a, b)


def _lcm_many(values: Iterable[int]) -> int:
    filtered = [abs(v) for v in values if v]
    if not filtered:
        raise ValueError("Need at least one non-zero integer to compute LCM.")
    return reduce(_lcm, filtered)


def resolve_system_size(task: ApproximantTask) -> int:
    """
    Choose a system size compatible with the modulation ratio and cluster size.
    """
    num, den = task.v_sep_ratio
    candidates = [task.base_L, den, task.Nc]
    L = _lcm_many(candidates)
    if task.L_override:
        L = _lcm(L, task.L_override)
    return L


def enumerate_int_separations(
    task: ApproximantTask,
    L: int,
) -> Tuple[List[Tuple[int, int]], List[Dict]]:
    """
    Enumerate all admissible interaction separations for the given task.
    """
    v_step = step_from_ratio(L, task.v_sep_ratio)
    details = enumerate_m(
        L,
        task.Nc,
        [v_step],
        task.max_supercluster_size,
        return_details=True,
    )
    if not details:
        raise ValueError(
            f"No admissible int_sep found for L={L}, Nc={task.Nc}, "
            f"v_step={v_step}, max_supercluster_size={task.max_supercluster_size}."
        )

    int_sep_list = [(entry["m"], L) for entry in details]
    return int_sep_list, details


def _mean_abs_difference(series: np.ndarray, reference: np.ndarray) -> Optional[float]:
    if series.shape != reference.shape or series.size == 0:
        return None
    mask = ~np.isnan(series) & ~np.isnan(reference)
    if not np.any(mask):
        return None
    return float(np.mean(np.abs(series[mask] - reference[mask])))


def summarize_results(
    all_results: Dict,
    int_sep_list: List[Tuple[int, int]],
    *,
    include_idmrg: bool,
    include_finite_dmrg: bool,
) -> Dict:
    """
    Build lightweight summary statistics for quick inspection.
    """
    per_vary_value: List[Dict] = []
    best_overall: Optional[Dict] = None

    for vary_val, result in all_results.items():
        ref_series: Optional[np.ndarray] = None
        ref_label: Optional[str] = None

        if include_finite_dmrg and "energies_finite_dmrg" in result:
            finite = np.asarray(result["energies_finite_dmrg"], dtype=float)
            if finite.size and not np.all(np.isnan(finite)):
                ref_series = finite
                ref_label = "finite_dmrg"

        if ref_series is None and include_idmrg and "energies_idmrg" in result:
            idmrg = np.asarray(result["energies_idmrg"], dtype=float)
            if idmrg.size and not np.all(np.isnan(idmrg)):
                ref_series = idmrg
                ref_label = "idmrg"

        entry = {
            "vary_value": float(vary_val),
            "reference": ref_label,
            "int_sep_errors": {},
        }

        if ref_series is not None:
            for int_sep in int_sep_list:
                key = f"int_sep_{int_sep[0]}_{int_sep[1]}"
                series = np.asarray(result.get(f"energies_{key}", []), dtype=float)
                err = _mean_abs_difference(series, ref_series)
                entry["int_sep_errors"][key] = err

                if err is not None:
                    candidate = {
                        "vary_value": float(vary_val),
                        "int_sep_key": key,
                        "mean_abs_error": err,
                        "reference": ref_label,
                    }
                    if (
                        best_overall is None
                        or err < best_overall["mean_abs_error"]
                    ):
                        best_overall = candidate

        per_vary_value.append(entry)

    return {
        "per_vary_value": per_vary_value,
        "best_overall": best_overall,
    }


def execute_task(
    task: ApproximantTask,
    sweep: SweepGrid,
    output_root: Path,
) -> Dict:
    """
    Run the convergence comparison for a single approximant/cluster size pair.
    """
    L = resolve_system_size(task)
    int_sep_list, details = enumerate_int_separations(task, L)

    num, den = task.v_sep_ratio
    task_dir = (
        output_root
        / f"Nc_{task.Nc}"
        / f"beta_{num}_{den}"
        / f"L_{L}"
    )
    task_dir.mkdir(parents=True, exist_ok=True)

    figures, all_results = compare_int_seps_with_dmrg(
        v_sep_ratio=task.v_sep_ratio,
        int_sep_list=int_sep_list,
        x_axis=sweep.x_axis,
        varying_parameter=sweep.varying_parameter,
        fixed_parameter=sweep.fixed_parameter,
        L=L,
        Nc=task.Nc,
        solver_method=sweep.solver_method,
        chi=sweep.chi,
        states_retained=sweep.states_retained,
        include_idmrg=sweep.include_idmrg,
        include_finite_dmrg=sweep.include_finite_dmrg,
        output_dir=str(task_dir),
        show_plots=False,
        save_pickle=True,
    )

    summary = summarize_results(
        all_results,
        int_sep_list,
        include_idmrg=sweep.include_idmrg,
        include_finite_dmrg=sweep.include_finite_dmrg,
    )

    metadata = {
        "beta_ratio": [num, den],
        "Nc": task.Nc,
        "resolved_L": L,
        "base_L": task.base_L,
        "max_supercluster_size": task.max_supercluster_size,
        "int_sep_list": [list(sep) for sep in int_sep_list],
        "int_sep_details": details,
        "x_axis": {k: list(v) for k, v in sweep.x_axis.items()},
        "varying_parameter": {k: list(v) for k, v in sweep.varying_parameter.items()},
        "fixed_parameter": sweep.fixed_parameter,
        "solver_method": sweep.solver_method,
        "chi": sweep.chi,
        "states_retained": sweep.states_retained,
        "include_idmrg": sweep.include_idmrg,
        "include_finite_dmrg": sweep.include_finite_dmrg,
        "artifact_dir": str(task_dir),
    }

    metadata_path = task_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    summary_path = task_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "task_dir": str(task_dir),
        "metadata_path": str(metadata_path),
        "summary_path": str(summary_path),
        "figures_saved": len(figures),
    }


def run_convergence_study(
    beta: float,
    *,
    max_supercluster_size: int,
    cluster_sizes: Sequence[int],
    sweep: SweepGrid,
    base_L: int = 24,
    max_beta_denominator: Optional[int] = None,
    output_root: str = "large_files/convergence_study",
    timestamp: Optional[str] = None,
    skip_failed_tasks: bool = True,
) -> Dict:
    """
    High-level driver coordinating the convergence analysis.
    """
    if max_beta_denominator is None:
        max_beta_denominator = max_supercluster_size

    approximants = [
        Approximant(frac)
        for frac in generate_optimal_approximants(beta, max_beta_denominator)
        if frac.numerator not in (0,frac.denominator)  # skip trivial 1/1 approximant
    ]
    if not approximants:
        raise ValueError(
            "No Hurwitz approximants generated. "
            "Increase max_beta_denominator or check beta value."

        )
    study_ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    study_dir = Path(output_root) / study_ts
    study_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict] = []

    for Nc in cluster_sizes:
        for approximant in approximants:
            task = ApproximantTask(
                approximant=approximant,
                Nc=Nc,
                max_supercluster_size=max_supercluster_size,
                base_L=base_L,
            )
            try:
                result = execute_task(task, sweep, study_dir)
                manifest_entries.append(
                    {
                        "Nc": Nc,
                        "beta_ratio": list(approximant.as_ratio()),
                        "status": "completed",
                        "artifacts": result,
                    }
                )
            except ValueError as exc:
                if not skip_failed_tasks:
                    raise
                manifest_entries.append(
                    {
                        "Nc": Nc,
                        "beta_ratio": list(approximant.as_ratio()),
                        "status": "skipped",
                        "reason": str(exc),
                    }
                )

    manifest = {
        "beta": beta,
        "max_supercluster_size": max_supercluster_size,
        "cluster_sizes": list(cluster_sizes),
        "max_beta_denominator": max_beta_denominator,
        "base_L": base_L,
        "sweep": {
            "x_axis": {k: list(v) for k, v in sweep.x_axis.items()},
            "varying_parameter": {
                k: list(v) for k, v in sweep.varying_parameter.items()
            },
            "fixed_parameter": sweep.fixed_parameter,
            "solver_method": sweep.solver_method,
            "chi": sweep.chi,
            "states_retained": sweep.states_retained,
            "include_idmrg": sweep.include_idmrg,
            "include_finite_dmrg": sweep.include_finite_dmrg,
        },
        "timestamp": study_ts,
        "output_root": str(study_dir),
        "entries": manifest_entries,
    }

    manifest_path = study_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest
