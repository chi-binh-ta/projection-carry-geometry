"""Support-sumset geometry for digit multiplication."""

from __future__ import annotations

from typing import Any


def _validate_support(name: str, S: set[int]) -> None:
    """Validate a set of nonnegative integer support indices."""
    if not isinstance(S, set):
        raise ValueError(f"{name} must be a set of nonnegative integers.")
    for value in S:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must contain only integers.")
        if value < 0:
            raise ValueError(f"{name} must contain only nonnegative integers.")


def digit_support_from_digits(digits: list[int]) -> set[int]:
    """Return the digit support ``{i : digits[i] != 0}``."""
    if not isinstance(digits, list):
        raise ValueError("digits must be a list of integers.")
    return {index for index, digit in enumerate(digits) if digit != 0}


def support_sumset(Sx: set[int], Sy: set[int]) -> set[int]:
    """Return the additive support sumset ``{i + j for i in Sx for j in Sy}``."""
    _validate_support("Sx", Sx)
    _validate_support("Sy", Sy)
    return {i + j for i in Sx for j in Sy}


def representation_counts(Sx: set[int], Sy: set[int]) -> dict[int, int]:
    """Return ``rho_k``, the number of pairs ``(i, j)`` with ``i + j = k``."""
    _validate_support("Sx", Sx)
    _validate_support("Sy", Sy)
    counts: dict[int, int] = {}
    for i in Sx:
        for j in Sy:
            k = i + j
            counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def support_gap_profile(S: set[int]) -> list[int]:
    """Return sorted consecutive gaps in a support set."""
    _validate_support("S", S)
    values = sorted(S)
    return [right - left for left, right in zip(values, values[1:])]


def support_diameter(S: set[int]) -> int:
    """Return ``max(S) - min(S)``, or ``0`` for empty or singleton support."""
    _validate_support("S", S)
    if len(S) <= 1:
        return 0
    return max(S) - min(S)


def support_density(S: set[int]) -> float:
    """Return ``|S| / (diameter + 1)``, with convention ``0`` for empty support."""
    _validate_support("S", S)
    if not S:
        return 0.0
    return len(S) / (support_diameter(S) + 1)


def additive_energy(Sx: set[int], Sy: set[int]) -> int:
    """Return ``sum_k rho_k^2`` for representation counts of ``Sx + Sy``."""
    counts = representation_counts(Sx, Sy)
    return sum(count * count for count in counts.values())


def sumset_summary(Sx: set[int], Sy: set[int]) -> dict[str, Any]:
    """Return summary statistics for support-sumset geometry."""
    counts = representation_counts(Sx, Sy)
    Ssum = set(counts)
    return {
        "size_Sx": len(Sx),
        "size_Sy": len(Sy),
        "size_sumset": len(Ssum),
        "max_representation_count": max(counts.values(), default=0),
        "additive_energy": additive_energy(Sx, Sy),
        "support_density_x": support_density(Sx),
        "support_density_y": support_density(Sy),
        "diameter_x": support_diameter(Sx),
        "diameter_y": support_diameter(Sy),
    }
