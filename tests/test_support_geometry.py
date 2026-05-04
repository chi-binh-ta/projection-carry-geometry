"""Tests for support-sumset geometry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from support_geometry import (
    additive_energy,
    representation_counts,
    support_density,
    support_diameter,
    support_gap_profile,
    support_sumset,
    sumset_summary,
)


def test_dense_three_point_sumset() -> None:
    """Three consecutive support points have triangular representation counts."""
    Sx = {0, 1, 2}
    Sy = {0, 1, 2}

    assert support_sumset(Sx, Sy) == {0, 1, 2, 3, 4}
    assert representation_counts(Sx, Sy) == {0: 1, 1: 2, 2: 3, 3: 2, 4: 1}
    assert additive_energy(Sx, Sy) == 19


def test_sparse_two_point_sumset() -> None:
    """Separated supports can still collide at the middle sum."""
    counts = representation_counts({0, 10}, {0, 10})

    assert support_sumset({0, 10}, {0, 10}) == {0, 10, 20}
    assert counts[10] == 2


def test_support_gap_profile() -> None:
    """Gap profile records consecutive sorted support gaps."""
    assert support_gap_profile({0, 3, 10}) == [3, 7]


def test_support_density_and_summary() -> None:
    """Support summaries should include density and diameter statistics."""
    summary = sumset_summary({0, 1, 2}, {0, 1, 2})

    assert support_diameter({0, 1, 2}) == 2
    assert support_density({0, 1, 2}) == 1.0
    assert support_density(set()) == 0.0
    assert summary["size_sumset"] == 5
    assert summary["max_representation_count"] == 3
