"""Outer-product interaction matrices for digit multiplication."""

from __future__ import annotations

import numpy as np


def _validate_digit_vector(name: str, digits: list[int]) -> None:
    """Validate a nonempty vector of nonnegative integer digits."""
    if not isinstance(digits, list):
        raise ValueError(f"{name} must be a list of integers.")
    if not digits:
        raise ValueError(f"{name} must not be empty.")
    for index, digit in enumerate(digits):
        if not isinstance(digit, int) or isinstance(digit, bool):
            raise ValueError(f"{name}[{index}] must be an integer.")
        if digit < 0:
            raise ValueError(f"{name}[{index}] must be nonnegative.")


def interaction_matrix(a: list[int], b: list[int]) -> np.ndarray:
    """Return the matrix ``M`` with entries ``M[i, j] = a[i] * b[j]``."""
    _validate_digit_vector("a", a)
    _validate_digit_vector("b", b)
    return np.outer(np.array(a, dtype=int), np.array(b, dtype=int))


def interaction_entries(
    a: list[int], b: list[int], include_zeros: bool = False
) -> list[tuple[int, int, int]]:
    """Return interaction entries ``(i, j, a_i * b_j)``.

    Parameters
    ----------
    a, b:
        Low-to-high digit vectors.
    include_zeros:
        If ``False`` (default), entries whose product is zero are omitted.
    """
    _validate_digit_vector("a", a)
    _validate_digit_vector("b", b)
    if not isinstance(include_zeros, bool):
        raise ValueError("include_zeros must be a boolean.")

    entries: list[tuple[int, int, int]] = []
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            value = ai * bj
            if include_zeros or value != 0:
                entries.append((i, j, value))
    return entries
