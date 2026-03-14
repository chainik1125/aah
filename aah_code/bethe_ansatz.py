"""
Lieb-Wu Bethe Ansatz benchmarks for the 1D Hubbard model.

This module provides exact thermodynamic-limit ground-state quantities for the
repulsive 1D Hubbard model at zero temperature and zero magnetization.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import i0, i1, j0, j1
except ImportError as exc:  # pragma: no cover - runtime dependency
    _SCIPY_IMPORT_ERROR = exc
    quad = None
    brentq = None
    i0 = None
    i1 = None
    j0 = None
    j1 = None
else:
    _SCIPY_IMPORT_ERROR = None


def _require_scipy() -> None:
    if _SCIPY_IMPORT_ERROR is not None:  # pragma: no cover - runtime dependency
        raise ImportError(
            "scipy is required for aah_code.bethe_ansatz; run this module with the project venv."
        ) from _SCIPY_IMPORT_ERROR


def _free_electron_energy(n_target: float, t: float) -> float:
    """Non-interacting energy per site at filling n (0 < n <= 1)."""
    q = np.pi * n_target / 2.0
    return -4.0 * abs(t) * np.sin(q) / np.pi


def _kernel(x: np.ndarray, n: int, c: float) -> np.ndarray:
    """Lorentzian kernel a_n(x) = (1/pi) * (n c) / ((n c)^2 + x^2)."""
    nc = n * c
    return nc / (np.pi * (nc**2 + x**2))


def _finite_difference_scalar(
    func: Callable[[float], float],
    x0: float,
    delta: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """Stable first derivative with automatic one-sided fallback near boundaries."""
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


def _lieb_wu_linear_system(
    Q: float,
    U: float,
    t: float,
    *,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> dict[str, np.ndarray | float | int]:
    """
    Build the discretized zero-magnetization Lieb-Wu linear system at fixed Q.

    Returns the matrix, quadrature weights, and grids used for both the root-density
    and dressed-charge equations.
    """
    _require_scipy()

    t_abs = abs(t)
    if t_abs == 0:
        raise ValueError("t must be nonzero for the Lieb-Wu solver.")

    c = U / (4.0 * t_abs) if U > 0 else 1e-15

    k_grid = np.linspace(-Q, Q, N_k)
    dk = k_grid[1] - k_grid[0] if N_k > 1 else 2.0 * Q
    w_k = np.full(N_k, dk)
    w_k[0] = w_k[-1] = dk / 2.0

    lam_grid = np.linspace(-B, B, N_lam)
    dlam = lam_grid[1] - lam_grid[0] if N_lam > 1 else 2.0 * B
    w_lam = np.full(N_lam, dlam)
    w_lam[0] = w_lam[-1] = dlam / 2.0

    sin_k = np.sin(k_grid)
    cos_k = np.cos(k_grid)

    diff_kl = sin_k[:, None] - lam_grid[None, :]
    A1_kl = cos_k[:, None] * _kernel(diff_kl, 1, c) * w_lam[None, :]

    diff_lk = lam_grid[:, None] - sin_k[None, :]
    A1_lk = _kernel(diff_lk, 1, c) * w_k[None, :]

    diff_ll = lam_grid[:, None] - lam_grid[None, :]
    A2_ll = _kernel(diff_ll, 2, c) * w_lam[None, :]

    N = N_k + N_lam
    M = np.zeros((N, N), dtype=float)
    M[:N_k, :N_k] = np.eye(N_k)
    M[:N_k, N_k:] = -A1_kl
    M[N_k:, :N_k] = -A1_lk
    M[N_k:, N_k:] = np.eye(N_lam) + A2_ll
    return {
        "matrix": M,
        "k_grid": k_grid,
        "lam_grid": lam_grid,
        "w_k": w_k,
        "w_lam": w_lam,
        "cos_k": cos_k,
        "N_k": N_k,
        "N_lam": N_lam,
        "t_abs": t_abs,
    }


def lieb_wu_energy_half_filling(U: float, t: float = 1.0) -> float:
    """
    Exact ground-state energy per site at half filling.

    The returned energy is for the bare Hubbard Hamiltonian and does not
    include any chemical-potential term.
    """
    _require_scipy()

    t_abs = abs(t)
    if t_abs == 0:
        return 0.0
    if U < 0:
        raise ValueError("This implementation requires U >= 0.")
    if abs(U) < 1e-12:
        return -4.0 * t_abs / np.pi

    def integrand(omega: float) -> float:
        exp_arg = U * omega / (2.0 * t_abs)
        if exp_arg > 500:
            return 0.0
        return j0(omega) * j1(omega) / (omega * (1.0 + np.exp(exp_arg)))

    result, _ = quad(integrand, 0.0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-12)
    return -4.0 * t_abs * result


def _solve_lieb_wu_at_Q(
    Q: float,
    U: float,
    t: float,
    *,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> tuple[float, float]:
    """
    Solve the zero-magnetization Lieb-Wu integral equations at fixed Q.

    Returns
    -------
    filling_per_site, energy_per_site
    """
    _require_scipy()

    if abs(t) == 0:
        return 0.0, 0.0

    system = _lieb_wu_linear_system(Q, U, t, N_k=N_k, N_lam=N_lam, B=B)
    N_k = int(system["N_k"])
    M = np.asarray(system["matrix"], dtype=float)
    rhs = np.zeros(M.shape[0], dtype=float)
    rhs[:N_k] = 1.0 / (2.0 * np.pi)

    x = np.linalg.solve(M, rhs)
    rho = x[:N_k]

    w_k = np.asarray(system["w_k"], dtype=float)
    cos_k = np.asarray(system["cos_k"], dtype=float)
    t_abs = float(system["t_abs"])
    filling = float(np.dot(rho, w_k))
    energy = float(-2.0 * t_abs * np.dot(cos_k * rho, w_k))
    return filling, energy


def _solve_lieb_wu_dressed_charge_at_Q(
    Q: float,
    U: float,
    t: float,
    *,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the zero-magnetization dressed-charge equations at fixed Q.

    Returns
    -------
    k_grid, xi_charge(k), xi_spin(lambda)
    """
    system = _lieb_wu_linear_system(Q, U, t, N_k=N_k, N_lam=N_lam, B=B)
    N_k = int(system["N_k"])
    M = np.asarray(system["matrix"], dtype=float)
    rhs = np.zeros(M.shape[0], dtype=float)
    rhs[:N_k] = 1.0
    x = np.linalg.solve(M, rhs)
    xi_charge = x[:N_k]
    xi_spin = x[N_k:]
    return (
        np.asarray(system["k_grid"], dtype=float),
        np.asarray(xi_charge, dtype=float),
        np.asarray(xi_spin, dtype=float),
    )


def lieb_wu_energy_general_filling(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """
    Exact ground-state energy per site at arbitrary filling.

    Parameters
    ----------
    U
        Hubbard interaction, with U >= 0.
    n_target
        Filling per site, with 0 < n <= 2.
    t
        Hopping amplitude.
    """
    _require_scipy()

    if U < 0:
        raise ValueError("This implementation requires U >= 0.")
    if n_target <= 0.0 or n_target > 2.0:
        raise ValueError(f"n_target must be in (0, 2], got {n_target}.")

    if n_target > 1.0 + 1e-12:
        e_hole = lieb_wu_energy_general_filling(
            U,
            2.0 - n_target,
            t,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )
        return e_hole + U * (n_target - 1.0)

    if abs(n_target - 1.0) < 1e-8:
        return lieb_wu_energy_half_filling(U, t)

    if abs(U) < 1e-12:
        return _free_electron_energy(n_target, t)

    def residual(Q: float) -> float:
        filling, _ = _solve_lieb_wu_at_Q(Q, U, t, N_k=N_k, N_lam=N_lam, B=B)
        return filling - n_target

    Q_opt = brentq(residual, 1e-6, np.pi - 1e-6, xtol=1e-10, rtol=1e-10)
    _, energy = _solve_lieb_wu_at_Q(Q_opt, U, t, N_k=N_k, N_lam=N_lam, B=B)
    return energy


def lieb_wu_charge_luttinger_parameter(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """
    Charge Luttinger parameter K_rho for the metallic zero-magnetization phase.

    Uses the dressed charge evaluated at the charge Fermi boundary:
        K_rho = Z(Q)^2 / 2
    with the standard spinful-Hubbard convention.
    """
    _require_scipy()

    if U < 0:
        raise ValueError("This implementation requires U >= 0.")
    if n_target <= 0.0 or n_target >= 2.0:
        raise ValueError(f"n_target must be in (0, 2), got {n_target}.")
    if abs(n_target - 1.0) < 1e-8 and U > 1e-12:
        raise ValueError("K_rho is not defined in the gapped half-filled charge sector.")
    if abs(U) < 1e-12:
        return 1.0

    if n_target > 1.0:
        n_target = 2.0 - n_target

    def residual(Q: float) -> float:
        filling, _ = _solve_lieb_wu_at_Q(Q, U, t, N_k=N_k, N_lam=N_lam, B=B)
        return filling - n_target

    Q_opt = brentq(residual, 1e-6, np.pi - 1e-6, xtol=1e-10, rtol=1e-10)
    _, xi_charge, _ = _solve_lieb_wu_dressed_charge_at_Q(Q_opt, U, t, N_k=N_k, N_lam=N_lam, B=B)
    z_q = float(xi_charge[-1])
    return 0.5 * z_q * z_q


def lieb_wu_double_occupancy(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_u: float = 5e-2,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """Double occupancy from Hellmann-Feynman: D = d e / dU."""

    def energy_fn(U_value: float) -> float:
        return lieb_wu_energy_general_filling(
            U_value,
            n_target,
            t,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )

    return _finite_difference_scalar(energy_fn, U, delta_u, lower=0.0)


def lieb_wu_kinetic_energy(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_u: float = 5e-2,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """Kinetic energy per site from T = e - U D."""
    energy = lieb_wu_energy_general_filling(
        U,
        n_target,
        t,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    double_occ = lieb_wu_double_occupancy(
        U,
        n_target,
        t,
        delta_u=delta_u,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    return energy - U * double_occ


def lieb_wu_chemical_potential(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_n: float = 1e-3,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """Chemical potential mu = d e / d n at fixed U."""

    def energy_fn(n_value: float) -> float:
        return lieb_wu_energy_general_filling(
            U,
            n_value,
            t,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )

    return _finite_difference_scalar(energy_fn, n_target, delta_n, lower=1e-6, upper=2.0 - 1e-6)


def lieb_wu_inverse_compressibility(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_n: float = 1e-3,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """Inverse compressibility n^2 * d mu / d n at fixed U."""

    def mu_fn(n_value: float) -> float:
        return lieb_wu_chemical_potential(
            U,
            n_value,
            t,
            delta_n=delta_n,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )

    dmu_dn = _finite_difference_scalar(mu_fn, n_target, delta_n, lower=1e-6, upper=2.0 - 1e-6)
    return (n_target**2) * dmu_dn


def lieb_wu_compressibility(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_n: float = 1e-3,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """Compressibility kappa = 1 / (n^2 d mu / d n) at fixed U."""
    inv_kappa = lieb_wu_inverse_compressibility(
        U,
        n_target,
        t,
        delta_n=delta_n,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    if abs(inv_kappa) < 1e-14:
        return np.inf
    return 1.0 / inv_kappa


def lieb_wu_charge_gap_half_filling(
    U: float,
    t: float = 1.0,
    *,
    delta_n: float = 1e-3,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """
    Half-filled charge gap from the jump in the chemical potential.

    We estimate
        Delta_c = mu_+ - mu_-
    with one-sided derivatives of the thermodynamic-limit energy density:
        mu_- = d e / d n |_{1^-}
        mu_+ = d e / d n |_{1^+}
    """
    if delta_n <= 0:
        raise ValueError("delta_n must be positive")
    if U < 0:
        raise ValueError("This implementation requires U >= 0.")
    if abs(U) < 1e-12:
        return 0.0

    n_minus = 1.0 - delta_n
    n_plus = 1.0 + delta_n
    if n_minus <= 0.0 or n_plus >= 2.0:
        raise ValueError(f"delta_n={delta_n} is too large for half-filling charge-gap evaluation.")

    e_half = lieb_wu_energy_general_filling(
        U,
        1.0,
        t,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    e_minus = lieb_wu_energy_general_filling(
        U,
        n_minus,
        t,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    e_plus = lieb_wu_energy_general_filling(
        U,
        n_plus,
        t,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    mu_minus = (e_half - e_minus) / delta_n
    mu_plus = (e_plus - e_half) / delta_n
    return mu_plus - mu_minus


def lieb_wu_spin_velocity_half_filling(U: float, t: float = 1.0) -> float:
    """
    Half-filled spin velocity from Takahashi's low-temperature specific heat.

    Using the exact low-T result for the half-filled Hubbard model together with
    the c=1 spin-sector relation C/T = pi / (3 v_s), one obtains

        v_s(U) = 2 t * I_1(2 pi t / U) / I_0(2 pi t / U)

    for U > 0. The U -> 0^+ limit is 2 t.
    """
    _require_scipy()

    t_abs = abs(t)
    if t_abs == 0:
        return 0.0
    if U < 0:
        raise ValueError("This implementation requires U >= 0.")
    if abs(U) < 1e-12:
        return 2.0 * t_abs

    x = 2.0 * np.pi * t_abs / U
    return 2.0 * t_abs * (i1(x) / i0(x))


def lieb_wu_thermodynamic_components(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_u: float = 5e-2,
    delta_n: float | None = None,
    include_charge_gap: bool = False,
    include_charge_sector: bool = False,
    include_spin_velocity: bool = False,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> dict[str, float]:
    """
    Bundle exact thermodynamic quantities for direct comparison to cluster data.
    """
    energy = lieb_wu_energy_general_filling(
        U,
        n_target,
        t,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    double_occ = lieb_wu_double_occupancy(
        U,
        n_target,
        t,
        delta_u=delta_u,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    result = {
        "energy": energy,
        "double_occupancy": double_occ,
        "kinetic": energy - U * double_occ,
    }
    if delta_n is not None:
        result["chemical_potential"] = lieb_wu_chemical_potential(
            U,
            n_target,
            t,
            delta_n=delta_n,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )
        result["inverse_compressibility"] = lieb_wu_inverse_compressibility(
            U,
            n_target,
            t,
            delta_n=delta_n,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )
        result["compressibility"] = (
            np.inf
            if abs(result["inverse_compressibility"]) < 1e-14
            else 1.0 / result["inverse_compressibility"]
        )
    if include_charge_gap:
        if abs(n_target - 1.0) >= 1e-8:
            raise ValueError("include_charge_gap=True is only defined at half filling (n_target=1).")
        gap_delta_n = 1e-3 if delta_n is None else delta_n
        result["charge_gap"] = lieb_wu_charge_gap_half_filling(
            U,
            t,
            delta_n=gap_delta_n,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )
    if include_charge_sector:
        if delta_n is None:
            raise ValueError("include_charge_sector=True requires delta_n to be provided.")
        result.update(
            lieb_wu_charge_sector_parameters(
                U,
                n_target,
                t,
                delta_n=delta_n,
                N_k=N_k,
                N_lam=N_lam,
                B=B,
            )
        )
    if include_spin_velocity:
        if abs(n_target - 1.0) >= 1e-8:
            raise ValueError(
                "include_spin_velocity=True is currently only defined at half filling (n_target=1)."
            )
        result["spin_velocity"] = lieb_wu_spin_velocity_half_filling(U, t=t)
    return result


def lieb_wu_charge_sector_parameters(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_n: float = 1e-3,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> dict[str, float]:
    """
    Metallic charge-sector parameters from Bethe Ansatz.

    Returns
    -------
    K_rho
        Charge Luttinger parameter from the dressed charge.
    compressibility
        kappa as defined in this module.
    inverse_compressibility
        kappa^{-1} = n^2 d mu / d n.
    v_c
        Charge velocity inferred from K_rho and kappa via the standard spinful
        Luttinger-liquid relation kappa = 2 K_rho / (pi n^2 v_c).
    D_c
        Charge stiffness inferred from v_c and K_rho via
        D_c = v_c K_rho / pi.
    """
    if abs(n_target - 1.0) < 1e-8 and U > 1e-12:
        raise ValueError("Charge-sector LL parameters are not defined at half filling for U>0.")

    if abs(U) < 1e-12:
        k_rho = 1.0
    else:
        k_rho = lieb_wu_charge_luttinger_parameter(
            U,
            n_target,
            t,
            N_k=N_k,
            N_lam=N_lam,
            B=B,
        )
    inverse_kappa = lieb_wu_inverse_compressibility(
        U,
        n_target,
        t,
        delta_n=delta_n,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )
    if abs(inverse_kappa) < 1e-14:
        raise ValueError("Compressibility is singular; cannot infer charge-sector LL parameters.")
    kappa = 1.0 / inverse_kappa
    v_c = 2.0 * k_rho / (np.pi * (n_target**2) * kappa)
    d_c = v_c * k_rho / np.pi
    return {
        "K_rho": float(k_rho),
        "compressibility": float(kappa),
        "inverse_compressibility": float(inverse_kappa),
        "v_c": float(v_c),
        "D_c": float(d_c),
    }


def lieb_wu_charge_stiffness(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_n: float = 1e-3,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """Charge stiffness D_c in the metallic phase."""
    return lieb_wu_charge_sector_parameters(
        U,
        n_target,
        t,
        delta_n=delta_n,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )["D_c"]


def lieb_wu_charge_velocity(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_n: float = 1e-3,
    N_k: int = 256,
    N_lam: int = 256,
    B: float = 20.0,
) -> float:
    """Charge velocity v_c in the metallic phase."""
    return lieb_wu_charge_sector_parameters(
        U,
        n_target,
        t,
        delta_n=delta_n,
        N_k=N_k,
        N_lam=N_lam,
        B=B,
    )["v_c"]
