"""Carry flow tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from carry import carry_normalize
from carry_flow import (
    carry_flow_edges,
    carry_flow_table,
    carry_value,
    carry_value_preservation,
)


def test_carry_flow_table_main_example() -> None:
    """The flow table should expose each carry-normalization step."""
    table = carry_flow_table([3, 8, 14, 8, 3], 10)
    assert list(table.columns) == [
        "k",
        "raw_c",
        "incoming_carry",
        "total_s",
        "digit",
        "outgoing_carry",
    ]
    assert table["raw_c"].tolist() == [3, 8, 14, 8, 3]
    assert table["digit"].tolist() == [3, 8, 4, 9, 3]
    assert table["outgoing_carry"].tolist() == [0, 0, 1, 0, 0]


def test_carry_flow_edges_include_only_positive_mass() -> None:
    """Only positive outgoing carries should become flow edges."""
    assert carry_flow_edges([3, 8, 14, 8, 3], 10) == [
        {"source": 2, "target": 3, "mass": 1}
    ]


def test_carry_value_and_preservation() -> None:
    """The carry-flow value helper should agree with normalization."""
    c = [15, 15, 15]
    d, _ = carry_normalize(c, 10)
    assert carry_value(c, 10) == 1665
    assert carry_value_preservation(c, d, 10)


def test_carry_flow_extends_when_final_carry_remains() -> None:
    """The table should include extra rows needed to exhaust final carry."""
    table = carry_flow_table([81], 10)
    assert table["raw_c"].tolist() == [81, 0]
    assert table["digit"].tolist() == [1, 8]
    assert table["outgoing_carry"].tolist() == [8, 0]
