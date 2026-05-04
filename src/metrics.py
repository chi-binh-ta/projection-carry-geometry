"""Carry and support metrics for Projection--Carry Geometry."""

from __future__ import annotations

import math
from typing import Any

from carry import multiply_projection_carry
from diagonal_projection import diagonal_density
from digits import support_digits


def _validate_int_list(name: str, values: list[int]) -> None:
    """Validate a list of integers."""
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list of integers.")
    for index, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name}[{index}] must be an integer.")


def carry_mass(carries: list[int]) -> int:
    """Return the total positive carry mass, ignoring ``carries[0]``."""
    _validate_int_list("carries", carries)
    return sum(carry for carry in carries[1:] if carry > 0)


def carry_depth(carries: list[int]) -> int:
    """Return the largest carry index with nonzero carry, or ``-1``."""
    _validate_int_list("carries", carries)
    for index in range(len(carries) - 1, 0, -1):
        if carries[index] != 0:
            return index
    return -1


def carry_entropy(carries: list[int]) -> float:
    """Return Shannon entropy of the positive carry distribution."""
    _validate_int_list("carries", carries)
    positive = [carry for carry in carries[1:] if carry > 0]
    total = sum(positive)
    if total == 0:
        return 0.0
    return -sum((carry / total) * math.log2(carry / total) for carry in positive)


def raw_support(c: list[int]) -> set[int]:
    """Return support of raw diagonal coefficients."""
    _validate_int_list("c", c)
    return {index for index, coefficient in enumerate(c) if coefficient != 0}


def final_support(d: list[int]) -> set[int]:
    """Return support of final normalized digits."""
    _validate_int_list("d", d)
    return {index for index, digit in enumerate(d) if digit != 0}


def max_raw_coefficient(c: list[int]) -> int:
    """Return the maximum raw diagonal coefficient, or ``0`` for empty input."""
    _validate_int_list("c", c)
    return max(c, default=0)


def carry_count(carries: list[int]) -> int:
    """Return the number of nonzero carry positions, ignoring ``carries[0]``."""
    _validate_int_list("carries", carries)
    return sum(1 for carry in carries[1:] if carry != 0)


def projection_summary(x: int, y: int, B: int) -> dict[str, Any]:
    """Run multiplication and return summary metrics."""
    pipeline = multiply_projection_carry(x, y, B)
    x_digits = pipeline["x_digits"]
    y_digits = pipeline["y_digits"]
    raw = pipeline["raw_diagonal_sums"]
    final = pipeline["final_digits"]
    carries = pipeline["carries"]
    matrix = pipeline["matrix"]

    Sx = support_digits(x_digits)
    Sy = support_digits(y_digits)
    raw_supp = raw_support(raw)
    final_supp = final_support(final)

    return {
        "x": x,
        "y": y,
        "B": B,
        "result": pipeline["result"],
        "x_num_digits": len(x_digits),
        "y_num_digits": len(y_digits),
        "x_support_size": len(Sx),
        "y_support_size": len(Sy),
        "raw_support_size": len(raw_supp),
        "final_support_size": len(final_supp),
        "raw_support": raw_supp,
        "final_support": final_supp,
        "max_raw_coefficient": max_raw_coefficient(raw),
        "carry_mass": carry_mass(carries),
        "carry_depth": carry_depth(carries),
        "carry_entropy": carry_entropy(carries),
        "carry_count": carry_count(carries),
        "diagonal_nonzero_density": diagonal_density(matrix, nonzero_only=True),
        "diagonal_geometric_density": diagonal_density(matrix, nonzero_only=False),
    }
