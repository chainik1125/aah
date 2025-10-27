"""
Utilities for generating optimal rational approximants to irrational numbers
using their Hurwitz (continued fraction) expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import floor
from typing import Iterable, Iterator, List, Sequence


def continued_fraction_terms(
    value: float,
    *,
    max_terms: int = 32,
    tol: float = 1e-12,
) -> List[int]:
    """
    Generate continued fraction coefficients a_0, a_1, ..., a_n for ``value``.

    Args:
        value: Positive real number to approximate. Typically irrational.
        max_terms: Hard cap on number of coefficients to produce.
        tol: Threshold on the fractional part used to detect termination.

    Returns:
        List of continued fraction coefficients.
    """
    if value <= 0:
        raise ValueError("continued_fraction_terms expects a positive value.")

    coeffs: List[int] = []
    remainder = float(value)

    for _ in range(max_terms):
        a = floor(remainder)
        coeffs.append(int(a))
        frac_part = remainder - a
        if frac_part < tol:
            break
        remainder = 1.0 / frac_part

    return coeffs


def convergents_from_cf(coeffs: Sequence[int]) -> Iterator[Fraction]:
    """
    Yield convergents for a continued fraction described by ``coeffs``.
    """
    if not coeffs:
        raise ValueError("Need at least one continued fraction coefficient.")

    # Standard recurrence: h_{-2}=0, h_{-1}=1; k_{-2}=1, k_{-1}=0
    h_m2, h_m1 = 0, 1
    k_m2, k_m1 = 1, 0

    for a in coeffs:
        h = a * h_m1 + h_m2
        k = a * k_m1 + k_m2
        yield Fraction(h, k)
        h_m2, h_m1 = h_m1, h
        k_m2, k_m1 = k_m1, k


def generate_optimal_approximants(
    value: float,
    max_denominator: int,
    *,
    max_terms: int = 32,
    tol: float = 1e-12,
) -> List[Fraction]:
    """
    Return convergents with denominator <= ``max_denominator``.

    Args:
        value: Target irrational number (e.g., Aubry-Andre modulation ratio).
        max_denominator: Only keep convergents whose denominator is <= this cap.
        max_terms: Maximum number of continued fraction coefficients to use.
        tol: Tolerance for terminating the continued fraction expansion.

    Returns:
        List of Fractions in increasing denominator order.
    """
    if max_denominator < 1:
        raise ValueError("max_denominator must be >= 1.")

    coeffs = continued_fraction_terms(value, max_terms=max_terms, tol=tol)

    approximants: List[Fraction] = []
    seen: set[Fraction] = set()

    for frac in convergents_from_cf(coeffs):
        reduced = Fraction(frac.numerator, frac.denominator)  # ensure canonical
        if reduced.denominator > max_denominator:
            break
        if reduced not in seen:
            approximants.append(reduced)
            seen.add(reduced)

    return approximants


@dataclass(frozen=True)
class Approximant:
    """
    Convenience wrapper describing a Hurwitz convergent.
    """

    fraction: Fraction

    def as_ratio(self) -> tuple[int, int]:
        return (self.fraction.numerator, self.fraction.denominator)

    def __str__(self) -> str:
        num, den = self.as_ratio()
        return f"{num}/{den}"

