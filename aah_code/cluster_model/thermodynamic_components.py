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
from aah_code.cluster_model.run_scripts_me import get_general_expectations, get_general_spectra


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


def _canonical_ground_state_energy(
    total_energy_spectrum: np.ndarray,
    total_number_spectrum: np.ndarray,
    target_total_number: int,
) -> float:
    """
    Minimize the independent-supercluster energy at fixed total particle number.
    """
    total_energy_spectrum = np.asarray(total_energy_spectrum, dtype=float)
    total_number_spectrum = np.asarray(total_number_spectrum)
    number_spectrum_site_sum = np.rint(total_number_spectrum.sum(axis=-1)).astype(int)

    num_superclusters, states_retained = total_energy_spectrum.shape
    if number_spectrum_site_sum.shape != (num_superclusters, states_retained):
        raise ValueError(
            "number spectrum must reduce to shape matching the energy spectrum; "
            f"got {number_spectrum_site_sum.shape} and {total_energy_spectrum.shape}"
        )

    inf = np.inf
    dp = np.full(target_total_number + 1, inf, dtype=float)
    dp[0] = 0.0

    for sc_idx in range(num_superclusters):
        next_dp = np.full_like(dp, inf)
        for n_prev in range(target_total_number + 1):
            if not np.isfinite(dp[n_prev]):
                continue
            base_energy = dp[n_prev]
            for state_idx in range(states_retained):
                n_state = int(number_spectrum_site_sum[sc_idx, state_idx])
                n_new = n_prev + n_state
                if n_new > target_total_number:
                    continue
                e_new = base_energy + float(total_energy_spectrum[sc_idx, state_idx])
                if e_new < next_dp[n_new]:
                    next_dp[n_new] = e_new
        dp = next_dp

    e_target = float(dp[target_total_number])
    if not np.isfinite(e_target):
        raise ValueError(
            f"Target total number N={target_total_number} is not reachable with the retained spectra."
        )
    return e_target


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
    twist_phi: float,
    solver_method: str,
    states_retained: Any,
    temperature: float,
) -> tuple[float, float, float]:
    mu0_guess = _default_mu0_guess(U, filling_target)
    physical_params = PhysicalParams(U=U, mu_0=mu0_guess, V=V, t=t, twist_phi=twist_phi)
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


@lru_cache(maxsize=None)
def _cluster_canonical_energy_bundle_cached(
    *,
    L: int,
    Nc: int,
    int_sep_ratio: tuple[int, int],
    v_sep_ratio: tuple[int, int],
    U: float,
    filling_target: float,
    t: float,
    V: float,
    twist_phi: float,
    solver_method: str,
    states_retained: Any,
) -> tuple[float, float]:
    target_total_number = int(round(float(filling_target) * L))
    target_total_number = max(0, min(target_total_number, 2 * L))

    physical_params = PhysicalParams(U=U, mu_0=0.0, V=V, t=t, twist_phi=twist_phi)
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
        _, energy_spectrum, number_spectrum, _ = get_general_spectra(run_config)

    total_energy = _canonical_ground_state_energy(energy_spectrum, number_spectrum, target_total_number)
    actual_filling = target_total_number / float(L)
    return total_energy / float(L), actual_filling


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
    twist_phi: float = 0.0,
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
            twist_phi=float(twist_phi),
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
        twist_phi=float(twist_phi),
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


def cluster_density_response_components(
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
    delta_n: float = 1e-2,
    twist_phi: float = 0.0,
) -> dict[str, float]:
    """
    Evaluate chemical potential and compressibility-like density response from
    the fixed-filling cluster thermodynamics.

    `mu_eff` is taken as the thermodynamic chemical potential of the fixed-filling
    cluster approximation. The inverse compressibility is estimated as
        n^2 * d mu / d n
    via finite differences in the target filling.
    """

    def mu_fn(n_value: float) -> float:
        _, _, mu_eff = _cluster_energy_bundle_cached(
            L=L,
            Nc=Nc,
            int_sep_ratio=int_sep_ratio,
            v_sep_ratio=v_sep_ratio,
            U=float(U),
            filling_target=float(n_value),
            t=float(t),
            V=float(V),
            twist_phi=float(twist_phi),
            solver_method=solver_method,
            states_retained=states_retained,
            temperature=float(temperature),
        )
        return float(mu_eff)

    energy, filling, mu_eff = _cluster_energy_bundle_cached(
        L=L,
        Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio,
        U=float(U),
        filling_target=float(filling_target),
        t=float(t),
        V=float(V),
        twist_phi=float(twist_phi),
        solver_method=solver_method,
        states_retained=states_retained,
        temperature=float(temperature),
    )

    dmu_dn = _finite_difference_scalar(mu_fn, filling_target, delta_n, lower=1e-6, upper=2.0 - 1e-6)
    inverse_compressibility = (filling_target**2) * dmu_dn
    compressibility = np.inf if abs(inverse_compressibility) < 1e-14 else 1.0 / inverse_compressibility

    return {
        "energy": energy,
        "filling": filling,
        "chemical_potential": mu_eff,
        "mu_eff": mu_eff,
        "inverse_compressibility": inverse_compressibility,
        "compressibility": compressibility,
    }


def cluster_charge_gap_half_filling(
    *,
    L: int,
    Nc: int,
    int_sep_ratio: tuple[int, int],
    U: float,
    t: float = -1.0,
    V: float = 0.0,
    v_sep_ratio: tuple[int, int] = (1, 1),
    solver_method: str = "sparse_ED",
    states_retained: Any = 6,
    temperature: float = 1e-2,
    delta_n: float = 1e-2,
    twist_phi: float = 0.0,
) -> dict[str, float]:
    """
    Half-filled charge gap from one-sided energy derivatives around n=1.

    The estimate mirrors the Bethe-side definition:
        Delta_c = mu_+ - mu_-
    with
        mu_- = [e(1) - e(1-dn)] / dn
        mu_+ = [e(1+dn) - e(1)] / dn
    """
    if delta_n <= 0:
        raise ValueError("delta_n must be positive")
    if 1.0 - delta_n <= 0.0 or 1.0 + delta_n >= 2.0:
        raise ValueError(f"delta_n={delta_n} is too large for half-filling charge-gap evaluation.")

    common_kwargs = dict(
        L=L,
        Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        v_sep_ratio=v_sep_ratio,
        U=float(U),
        t=float(t),
        V=float(V),
        solver_method=solver_method,
        states_retained=states_retained,
        temperature=float(temperature),
    )

    e_minus, fill_minus, mu_minus_target = _cluster_energy_bundle_cached(
        **common_kwargs,
        filling_target=1.0 - delta_n,
        twist_phi=float(twist_phi),
    )
    e_half, fill_half, mu_half_target = _cluster_energy_bundle_cached(
        **common_kwargs,
        filling_target=1.0,
        twist_phi=float(twist_phi),
    )
    e_plus, fill_plus, mu_plus_target = _cluster_energy_bundle_cached(
        **common_kwargs,
        filling_target=1.0 + delta_n,
        twist_phi=float(twist_phi),
    )

    mu_minus = (e_half - e_minus) / delta_n
    mu_plus = (e_plus - e_half) / delta_n
    charge_gap = mu_plus - mu_minus
    return {
        "charge_gap": charge_gap,
        "mu_minus": mu_minus,
        "mu_plus": mu_plus,
        "energy_minus": e_minus,
        "energy_half": e_half,
        "energy_plus": e_plus,
        "filling_minus": fill_minus,
        "filling_half": fill_half,
        "filling_plus": fill_plus,
        "mu_eff_minus": mu_minus_target,
        "mu_eff_half": mu_half_target,
        "mu_eff_plus": mu_plus_target,
    }


def cluster_charge_stiffness(
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
    delta_phi: float = 5e-2,
) -> dict[str, float]:
    """
    Charge stiffness from the flux curvature of the fixed-filling cluster energy.

    With a total twist Phi threaded through the ring, we estimate
        D_c = (L / 2) d^2 E(Phi) / d Phi^2 |_{Phi=0}
    Using the per-site energy e(Phi) = E(Phi) / L, this becomes
        D_c = (L^2 / 2) d^2 e(Phi) / d Phi^2 |_{Phi=0}.
    """
    if delta_phi <= 0:
        raise ValueError("delta_phi must be positive")

    common_kwargs = dict(
        L=L,
        Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        filling_target=filling_target,
        U=float(U),
        t=float(t),
        V=float(V),
        v_sep_ratio=v_sep_ratio,
        solver_method=solver_method,
        states_retained=states_retained,
    )

    e_minus, filling_minus = _cluster_canonical_energy_bundle_cached(**common_kwargs, twist_phi=-delta_phi)
    e_zero, filling_zero = _cluster_canonical_energy_bundle_cached(**common_kwargs, twist_phi=0.0)
    e_plus, filling_plus = _cluster_canonical_energy_bundle_cached(**common_kwargs, twist_phi=delta_phi)
    second_derivative = (e_plus - 2.0 * e_zero + e_minus) / (delta_phi**2)
    d_c = -0.5 * float(second_derivative)
    return {
        "charge_stiffness": d_c,
        "energy_minus": e_minus,
        "energy_zero": e_zero,
        "energy_plus": e_plus,
        "filling_minus": filling_minus,
        "filling_zero": filling_zero,
        "filling_plus": filling_plus,
        "second_derivative": second_derivative,
    }


def cluster_canonical_density_response_components(
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
) -> dict[str, float]:
    """
    Canonical finite-difference density response from E(N) at fixed total particle number.
    """
    target_total_number = int(round(float(filling_target) * L))
    n_minus = (target_total_number - 1) / float(L)
    n_zero = target_total_number / float(L)
    n_plus = (target_total_number + 1) / float(L)

    if target_total_number <= 0 or target_total_number >= 2 * L:
        raise ValueError("Canonical density response requires 0 < N < 2L.")

    common_kwargs = dict(
        L=L,
        Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        U=float(U),
        t=float(t),
        V=float(V),
        v_sep_ratio=v_sep_ratio,
        twist_phi=0.0,
        solver_method=solver_method,
        states_retained=states_retained,
    )
    e_minus, _ = _cluster_canonical_energy_bundle_cached(**common_kwargs, filling_target=n_minus)
    e_zero, _ = _cluster_canonical_energy_bundle_cached(**common_kwargs, filling_target=n_zero)
    e_plus, _ = _cluster_canonical_energy_bundle_cached(**common_kwargs, filling_target=n_plus)

    total_minus = e_minus * L
    total_zero = e_zero * L
    total_plus = e_plus * L
    chemical_potential = 0.5 * (total_plus - total_minus)
    second_total = total_plus - 2.0 * total_zero + total_minus
    inverse_compressibility = (n_zero**2) * L * second_total
    compressibility = np.inf if abs(inverse_compressibility) < 1e-14 else 1.0 / inverse_compressibility
    return {
        "energy_minus": e_minus,
        "energy_zero": e_zero,
        "energy_plus": e_plus,
        "chemical_potential": chemical_potential,
        "inverse_compressibility": inverse_compressibility,
        "compressibility": compressibility,
        "filling_minus": n_minus,
        "filling_zero": n_zero,
        "filling_plus": n_plus,
    }


def cluster_charge_sector_parameters(
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
    delta_n: float = 1e-2,
    delta_phi: float = 5e-2,
) -> dict[str, float]:
    """
    Cluster analog of the metallic charge-sector parameters.

    Uses the fixed-filling cluster compressibility and the flux curvature of the
    cluster energy. The Luttinger-liquid relations are:
        kappa = 2 K_rho / (pi n^2 v_c)
        D_c   = v_c K_rho / pi
    """
    density_response = cluster_canonical_density_response_components(
        L=L,
        Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        filling_target=filling_target,
        U=U,
        t=t,
        V=V,
        v_sep_ratio=v_sep_ratio,
        solver_method=solver_method,
        states_retained=states_retained,
    )
    stiffness = cluster_charge_stiffness(
        L=L,
        Nc=Nc,
        int_sep_ratio=int_sep_ratio,
        filling_target=filling_target,
        U=U,
        t=t,
        V=V,
        v_sep_ratio=v_sep_ratio,
        solver_method=solver_method,
        states_retained=states_retained,
        temperature=temperature,
        delta_phi=delta_phi,
    )
    kappa = float(density_response["compressibility"])
    d_c = float(stiffness["charge_stiffness"])
    if not np.isfinite(kappa) or not np.isfinite(d_c) or kappa <= 0.0 or d_c <= 0.0:
        return {
            "chemical_potential": float(density_response["chemical_potential"]),
            "compressibility": kappa,
            "inverse_compressibility": float(density_response["inverse_compressibility"]),
            "charge_stiffness": d_c,
            "K_rho": float("nan"),
            "v_c": float("nan"),
        }

    k_rho = (np.pi * filling_target / np.sqrt(2.0)) * np.sqrt(d_c * kappa)
    v_c = 2.0 * k_rho / (np.pi * (filling_target**2) * kappa)
    return {
        "chemical_potential": float(density_response["chemical_potential"]),
        "compressibility": kappa,
        "inverse_compressibility": float(density_response["inverse_compressibility"]),
        "charge_stiffness": d_c,
        "K_rho": float(k_rho),
        "v_c": float(v_c),
    }
