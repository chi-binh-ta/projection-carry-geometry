"""Digit utility tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from digits import digits_base, value_from_digits


def test_digits_roundtrip_across_bases() -> None:
    """Converting to low-to-high digits and back should preserve value."""
    bases = [2, 3, 5, 10, 16, 100]
    values = [0, 1, 2, 9, 10, 123, 9999, 123456]

    for B in bases:
        for n in values:
            assert value_from_digits(digits_base(n, B), B) == n


def test_digits_are_low_to_high() -> None:
    """The project convention is low-to-high digit order."""
    assert digits_base(123, 10) == [3, 2, 1]
