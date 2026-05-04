"""Carry complexity metrics linking diagonal mass to carry flow."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from carry import carry_normalize
from carry_cascade import max_carry_cascade_length
from carry_flow import carry_flow_table, carry_value
from metrics import carry_count, carry_depth, carry_entropy, carry_mass


def _validate_base(B: int) -> None:
    """Validate a positional base."""
    if not isinstance(B, int) or isinstance(B, bool):
        raise ValueError("B must be an integer base.")
    if B < 2:
        raise ValueError("B must be at least 2.")


def _validate_coefficients(c: list[int]) -> None:
    """Validate a raw coefficient list."""
    if not isinstance(c, list):
        raise ValueError("c must be a list of nonnegative integers.")
    for index, coefficient in enumerate(c):
        if not isinstance(coefficient, int) or isinstance(coefficient, bool):
            raise ValueError(f"c[{index}] must be an integer.")
        if coefficient < 0:
            raise ValueError(f"c[{index}] must be nonnegative.")


def carry_activation_table(c: list[int], B: int) -> pd.DataFrame:
    """Return carry-flow rows with activation and excess-mass columns."""
    _validate_base(B)
    _validate_coefficients(c)
    table = carry_flow_table(c, B).copy()
    table["carry_active"] = table["outgoing_carry"] > 0
    table["excess"] = (table["total_s"] - (B - 1)).clip(lower=0)
    return table


def carry_amplification_ratio(c: list[int], B: int) -> float:
    """Return total outgoing carry divided by total raw diagonal mass."""
    _validate_base(B)
    _validate_coefficients(c)
    table = carry_flow_table(c, B)
    outgoing_total = int(table["outgoing_carry"].sum())
    raw_total = max(sum(c), 1)
    return outgoing_total / raw_total


def diagonal_mass_entropy(c: list[int]) -> float:
    """Return Shannon entropy of positive raw coefficient mass."""
    _validate_coefficients(c)
    positive = [coefficient for coefficient in c if coefficient > 0]
    total = sum(positive)
    if total == 0:
        return 0.0
    return -sum((coefficient / total) * math.log2(coefficient / total) for coefficient in positive)


def diagonal_mass_concentration(c: list[int]) -> float:
    """Return the largest raw coefficient divided by total raw mass."""
    _validate_coefficients(c)
    return max(c, default=0) / max(sum(c), 1)


def value_weighted_carry(c: list[int], B: int) -> int:
    """Return carry flow weighted by positional value."""
    _validate_base(B)
    _validate_coefficients(c)
    table = carry_flow_table(c, B)
    return sum(
        int(row["outgoing_carry"]) * (B ** (int(row["k"]) + 1))
        for row in table.to_dict("records")
    )


def normalized_value_weighted_carry(c: list[int], B: int) -> float:
    """Return value-weighted carry divided by the raw represented value."""
    _validate_base(B)
    _validate_coefficients(c)
    raw_value = max(carry_value(c, B), 1)
    return value_weighted_carry(c, B) / raw_value


def carry_complexity_summary(c: list[int], B: int) -> dict[str, Any]:
    """Summarize diagonal mass concentration and carry-flow complexity."""
    _validate_base(B)
    _validate_coefficients(c)
    _, carries = carry_normalize(c, B)

    return {
        "raw_mass_total": sum(c),
        "max_raw_mass": max(c, default=0),
        "diagonal_mass_entropy": diagonal_mass_entropy(c),
        "diagonal_mass_concentration": diagonal_mass_concentration(c),
        "carry_mass": carry_mass(carries),
        "carry_count": carry_count(carries),
        "carry_depth": carry_depth(carries),
        "carry_entropy": carry_entropy(carries),
        "carry_amplification_ratio": carry_amplification_ratio(c, B),
        "max_carry_cascade_length": max_carry_cascade_length(c, B),
        "value_weighted_carry": value_weighted_carry(c, B),
        "normalized_value_weighted_carry": normalized_value_weighted_carry(c, B),
    }
