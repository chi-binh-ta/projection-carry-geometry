"""Carry flow tables and value-preservation helpers.

This module treats raw coefficients ``c_k`` as mass on the degree axis.  Carry
normalization keeps a digit at degree ``k`` and transports outgoing carry mass
from ``k`` to ``k + 1``.
"""

from __future__ import annotations

import pandas as pd


def _validate_base(B: int) -> None:
    """Validate a positional base."""
    if not isinstance(B, int) or isinstance(B, bool):
        raise ValueError("B must be an integer base.")
    if B < 2:
        raise ValueError("B must be at least 2.")


def _validate_coefficients(c: list[int]) -> None:
    """Validate nonnegative integer raw coefficients."""
    if not isinstance(c, list):
        raise ValueError("c must be a list of nonnegative integers.")
    for index, coefficient in enumerate(c):
        if not isinstance(coefficient, int) or isinstance(coefficient, bool):
            raise ValueError(f"c[{index}] must be an integer.")
        if coefficient < 0:
            raise ValueError(f"c[{index}] must be nonnegative.")


def carry_flow_table(c: list[int], B: int) -> pd.DataFrame:
    """Return a table describing carry flow across degree positions.

    Columns are ``k``, ``raw_c``, ``incoming_carry``, ``total_s``, ``digit``,
    and ``outgoing_carry``.
    """
    _validate_base(B)
    _validate_coefficients(c)

    rows: list[dict[str, int]] = []
    carry = 0
    k = 0
    while k < len(c) or carry > 0:
        raw_c = c[k] if k < len(c) else 0
        incoming = carry
        total_s = raw_c + incoming
        digit = total_s % B
        outgoing = total_s // B
        rows.append(
            {
                "k": k,
                "raw_c": raw_c,
                "incoming_carry": incoming,
                "total_s": total_s,
                "digit": digit,
                "outgoing_carry": outgoing,
            }
        )
        carry = outgoing
        k += 1

    return pd.DataFrame(
        rows,
        columns=[
            "k",
            "raw_c",
            "incoming_carry",
            "total_s",
            "digit",
            "outgoing_carry",
        ],
    )


def carry_flow_edges(c: list[int], B: int) -> list[dict[str, int]]:
    """Return positive carry-flow edges from degree ``k`` to ``k + 1``."""
    table = carry_flow_table(c, B)
    edges: list[dict[str, int]] = []
    for row in table.to_dict("records"):
        mass = int(row["outgoing_carry"])
        if mass > 0:
            k = int(row["k"])
            edges.append({"source": k, "target": k + 1, "mass": mass})
    return edges


def carry_value(c: list[int], B: int) -> int:
    """Evaluate a coefficient or digit sequence as ``sum c_k B**k``."""
    _validate_base(B)
    _validate_coefficients(c)
    return sum(coefficient * (B**k) for k, coefficient in enumerate(c))


def carry_value_preservation(c: list[int], d: list[int], B: int) -> bool:
    """Return whether raw coefficients ``c`` and normalized digits ``d`` agree."""
    return carry_value(c, B) == carry_value(d, B)
