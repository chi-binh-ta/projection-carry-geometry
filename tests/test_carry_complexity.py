"""Carry complexity tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from carry_complexity import (
    carry_activation_table,
    carry_amplification_ratio,
    carry_complexity_summary,
    diagonal_mass_concentration,
    diagonal_mass_entropy,
)


def test_main_example_carry_complexity() -> None:
    """The 123 times 321 raw coefficients activate exactly one carry."""
    c = [3, 8, 14, 8, 3]
    summary = carry_complexity_summary(c, 10)

    assert summary["carry_mass"] == 1
    assert summary["carry_count"] == 1
    assert summary["diagonal_mass_concentration"] == 14 / 36
    assert summary["carry_amplification_ratio"] == 1 / 36
    assert summary["max_carry_cascade_length"] == 1
    assert summary["value_weighted_carry"] == 1000


def test_no_carry_complexity() -> None:
    """Small raw coefficients in base 10 should not activate carry."""
    c = [1, 2, 3]
    summary = carry_complexity_summary(c, 10)

    assert summary["carry_mass"] == 0
    assert summary["carry_count"] == 0
    assert summary["carry_amplification_ratio"] == 0


def test_zero_mass_entropy_and_concentration() -> None:
    """All-zero raw mass should have zero entropy and concentration."""
    assert diagonal_mass_entropy([0, 0, 0]) == 0.0
    assert diagonal_mass_concentration([0, 0, 0]) == 0.0


def test_carry_activation_table_columns() -> None:
    """Activation table should mark active outgoing carries and excess mass."""
    table = carry_activation_table([3, 8, 14, 8, 3], 10)

    assert "carry_active" in table.columns
    assert "excess" in table.columns
    assert table["carry_active"].tolist() == [False, False, True, False, False]
    assert table["excess"].tolist() == [0, 0, 5, 0, 0]


def test_carry_amplification_zero_mass() -> None:
    """Zero raw mass should use denominator one and return zero."""
    assert carry_amplification_ratio([0, 0], 10) == 0.0
