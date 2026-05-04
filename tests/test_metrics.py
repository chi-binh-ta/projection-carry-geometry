"""Metric tests for Projection--Carry Geometry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics import (
    carry_count,
    carry_depth,
    carry_entropy,
    carry_mass,
    final_support,
    max_raw_coefficient,
    projection_summary,
    raw_support,
)


def test_basic_carry_metrics() -> None:
    """Carry metrics should be computed from positive carry positions."""
    carries = [0, 0, 1, 0, 0, 0]
    assert carry_mass(carries) == 1
    assert carry_depth(carries) == 2
    assert carry_count(carries) == 1
    assert carry_entropy(carries) == 0.0


def test_support_metrics() -> None:
    """Support helpers should report nonzero coefficient indices."""
    assert raw_support([0, 5, 0, 2]) == {1, 3}
    assert final_support([3, 0, 4, 0]) == {0, 2}
    assert max_raw_coefficient([1, 0, 25, 4]) == 25


def test_projection_summary_contains_carry_metrics() -> None:
    """The summary should expose the main carry complexity fields."""
    summary = projection_summary(123, 321, 10)
    assert summary["result"] == 39483
    assert summary["carry_mass"] >= 1
    assert summary["carry_depth"] >= 0
    assert "carry_entropy" in summary
    assert summary["raw_support_size"] == 5
