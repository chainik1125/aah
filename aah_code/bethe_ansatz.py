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
    from scipy.special import j0, j1
except ImportError as exc:  # pragma: no cover - runtime dependency
    _SCIPY_IMPORT_ERROR = exc
    quad = None
    brentq = None
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

    t_abs = abs(t)
    if t_abs == 0:
        return 0.0, 0.0

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

    rhs = np.zeros(N, dtype=float)
    rhs[:N_k] = 1.0 / (2.0 * np.pi)

    x = np.linalg.solve(M, rhs)
    rho = x[:N_k]

    filling = float(np.dot(rho, w_k))
    energy = float(-2.0 * t_abs * np.dot(cos_k * rho, w_k))
    return filling, energy


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


def lieb_wu_thermodynamic_components(
    U: float,
    n_target: float,
    t: float = 1.0,
    *,
    delta_u: float = 5e-2,
    delta_n: float | None = None,
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
    return result
