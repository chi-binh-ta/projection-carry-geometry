"""Anti-diagonal projection for interaction matrices."""

from __future__ import annotations

import numpy as np


def _validate_matrix(M: np.ndarray) -> None:
    """Validate a two-dimensional nonempty NumPy array."""
    if not isinstance(M, np.ndarray):
        raise ValueError("M must be a numpy.ndarray.")
    if M.ndim != 2:
        raise ValueError("M must be two-dimensional.")
    if M.shape[0] == 0 or M.shape[1] == 0:
        raise ValueError("M must have positive shape in both dimensions.")


def diagonal_projection(M: np.ndarray) -> list[int]:
    """Compute raw coefficients ``c_k = sum_{i+j=k} M[i, j]``."""
    _validate_matrix(M)
    n_rows, n_cols = M.shape
    diagonal_sums = [0 for _ in range(n_rows + n_cols - 1)]
    for i in range(n_rows):
        for j in range(n_cols):
            diagonal_sums[i + j] += int(M[i, j])
    return diagonal_sums


def diagonal_density(M: np.ndarray, nonzero_only: bool = True) -> list[int]:
    """Count entries on each anti-diagonal.

    If ``nonzero_only`` is ``True``, only nonzero entries are counted.
    Otherwise every geometric cell on the anti-diagonal is counted.
    """
    _validate_matrix(M)
    if not isinstance(nonzero_only, bool):
        raise ValueError("nonzero_only must be a boolean.")

    n_rows, n_cols = M.shape
    density = [0 for _ in range(n_rows + n_cols - 1)]
    for i in range(n_rows):
        for j in range(n_cols):
            if not nonzero_only or M[i, j] != 0:
                density[i + j] += 1
    return density


def support_sumset(Sx: set[int], Sy: set[int]) -> set[int]:
    """Return the additive sumset ``{i + j : i in Sx, j in Sy}``."""
    if not isinstance(Sx, set) or not isinstance(Sy, set):
        raise ValueError("Sx and Sy must be sets of nonnegative integers.")
    for name, support in (("Sx", Sx), ("Sy", Sy)):
        for value in support:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must contain only nonnegative integers.")
    return {i + j for i in Sx for j in Sy}
