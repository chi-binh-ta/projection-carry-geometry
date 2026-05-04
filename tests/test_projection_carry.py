"""Projection--carry multiplication correctness tests."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from carry import carry_normalize, multiply_projection_carry
from carry_flow import carry_value


def test_main_example() -> None:
    """The canonical 123 times 321 example should match the specification."""
    out = multiply_projection_carry(123, 321, 10)
    assert out["raw_diagonal_sums"] == [3, 8, 14, 8, 3]
    assert out["final_digits"] == [3, 8, 4, 9, 3]
    assert out["result"] == 39483


def test_99_times_99() -> None:
    """A decimal example with cascading carry."""
    assert multiply_projection_carry(99, 99, 10)["result"] == 9801


def test_random_correctness() -> None:
    """Projection--carry multiplication should agree with Python integers."""
    rng = random.Random(20260504)
    for _ in range(100):
        B = rng.choice([2, 3, 5, 10, 16])
        x = rng.randrange(0, 100_000)
        y = rng.randrange(0, 100_000)
        assert multiply_projection_carry(x, y, B)["result"] == x * y


def test_carry_value_preservation() -> None:
    """Carry normalization should preserve the represented integer value."""
    coefficient_lists = [
        [3, 8, 14, 8, 3],
        [81],
        [0],
        [15, 15, 15],
        [1, 0, 25, 4],
    ]
    for c in coefficient_lists:
        for B in [2, 10, 16]:
            d, _ = carry_normalize(c, B)
            assert carry_value(c, B) == carry_value(d, B)
