"""Tests for carry cascade measurements."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from carry_cascade import (
    carry_active_sequence,
    carry_cascade_segments,
    max_carry_cascade_length,
)


def test_no_carry_cascade() -> None:
    """No active carries means cascade length zero."""
    assert max_carry_cascade_length([1, 2, 3], 10) == 0
    assert carry_cascade_segments([1, 2, 3], 10) == []


def test_main_example_cascade() -> None:
    """The 123 times 321 raw coefficients have one active carry edge."""
    assert carry_active_sequence([3, 8, 14, 8, 3], 10) == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert carry_cascade_segments([3, 8, 14, 8, 3], 10) == [
        {"start_k": 2, "end_k": 2, "length": 1}
    ]


def test_multi_step_cascade() -> None:
    """Large raw coefficients can trigger consecutive active carry edges."""
    assert carry_active_sequence([99, 99, 99], 10) == [True, True, True, True, False]
    assert max_carry_cascade_length([99, 99, 99], 10) == 4
    assert carry_cascade_segments([99, 99, 99], 10) == [
        {"start_k": 0, "end_k": 3, "length": 4}
    ]
