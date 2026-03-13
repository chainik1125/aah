"""
Cluster-side thermodynamic decomposition helpers.

These helpers evaluate fixed-filling cluster energies and infer
double-occupancy and kinetic components from finite differences in U.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from io import StringIO
from typing import Any

import numpy as np

from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations


def _default_mu0_guess(U: float, filling_target: float) -> float:
    return U / 2.0 if abs(filling_target - 1.0) < 1e-8 else 0.0


def _finite_difference_scalar(
    func,
    x0: float,
    delta: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if delta <= 0:
        raise ValueError("delta must be positive")

    use_forward = lower is not None and (x0 - delta) < lower
    use_backward = upper is not None and (x0 + delta) > upper

    if use_forward and use_backward:
        raise ValueError("finite-difference stencil collapsed at both bounds")
    if use_forward:
        return (func(x0 + delta) - func(x0)) / delta
    if use_backward:
        return (func(x0) - func(x0 - delta)) / delta
    return (func(x0 + delta) - func(x0 - delta)) / (2.0 * delta)


@lru_cache(maxsize=None)
def _cluster_energy_bundle_cached(
    *,
    L: int,
    Nc: int,
    int_sep_ratio: tuple[int, int],
    v_sep_ratio: tuple[int, int],
    U: float,
    filling_target: float,
    t: float,
    V: float,
    solver_method: str,
    states_retained: Any,
    temperature: float,
) -> tuple[float, float, float]:
    mu0_guess = _default_mu0_guess(U, filling_target)
    physical_params = PhysicalParams(U=U, mu_0=mu0_guess, V=V, t=t)
    run_config = ClusterModelConfig(
        L=L,
        int_cluster_size=Nc,
        cluster_separation_ratio=int_sep_ratio,
        V_separation_ratio=v_sep_ratio,
        ham_lib="quspin",
        physical_params=physical_params,
        model_bc="periodic",
        int_cluster_bc="periodic",
        super_cluster_bc="periodic",
        solver_method=solver_method,
        states_retained=states_retained,
    )

    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        system_expectations, _, mu_eff = get_general_expectations(
            run_config,
            set_filling=filling_target,
            temperature=temperature,
            return_mu=True,
        )

    energy_gc, filling_total, _ = system_expectations
    energy_bare_per_site = float((energy_gc + mu0_guess * filling_total) / L)
    filling_per_site = float(filling_total / L)
    mu_eff_used = float(mu_eff) if mu_eff is not None else float("nan")
    return energy_bare_per_site, filling_per_site, mu_eff_used


def cluster_energy_components(
    *,
    L: int,
    Nc: int,
    int_sep_ratio: tuple[int, int],
    filling_target: float,
    U: float,
    t: float = -1.0,
    V: float = 0.0,
    v_sep_ratio: tuple[int, int] = (1, 1),
    solver_method: str = "sparse_ED",
    states_retained: Any = 6,
    temperature: float = 1e-2,
    delta_u: float = 5e-2,
) -> dict[str, float]:
    """
    Evaluate cluster energy, double occupancy, and kinetic energy per site.

    The energy convention matches the cached Fig. 2 data: bare Hubbard energy
    per site with the explicit `mu_0` term added back.
    """

    def energy_fn(U_value: float) -> float:
        energy, _, _ = _cluster_energy_bundle_cached(
            L=L,
            Nc=Nc,
            int_sep_ratio=int_sep_ratio,
            v_sep_ratio=v_sep_ratio,
            U=float(U_value),
            filling_target=float(filling_target),
            t=float(t),
            V=float(V),
            solver_method=solver_method,
            states_retained=states_retained,
            temperature=float(temperature),
        )
        return energy

    energy, filling, mu_eff = _cluster_energy_bundle_cached(
        L=L,
        Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio,
        U=float(U),
        filling_target=float(filling_target),
        t=float(t),
        V=float(V),
        solver_method=solver_method,
        states_retained=states_retained,
        temperature=float(temperature),
    )

    double_occ = _finite_difference_scalar(energy_fn, U, delta_u, lower=0.0)
    return {
        "energy": energy,
        "double_occupancy": double_occ,
        "kinetic": energy - U * double_occ,
        "filling": filling,
        "mu_eff": mu_eff,
    }
