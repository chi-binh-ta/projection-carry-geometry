"""Tests for local carry activation laws."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from carry import carry_normalize
from carry_complexity import carry_activation_table
from carry_flow import carry_value_preservation
from local_carry_laws import (
    build_local_carry_dataset,
    check_local_activation_law,
    local_carry_dataset_for_pair,
)


def test_local_activation_law_main_coefficients() -> None:
    """Carry is active exactly when total_s reaches the base threshold."""
    table = carry_activation_table([3, 8, 14, 8, 3], 10)
    row = table.loc[table["k"] == 2].iloc[0]

    assert row["total_s"] == 14
    assert row["outgoing_carry"] == 1
    assert bool(row["carry_active"])
    assert bool((table["carry_active"] == (table["total_s"] >= 10)).all())


def test_local_dataset_for_main_pair() -> None:
    """The pair-level dataset should satisfy the local activation law."""
    df = local_carry_dataset_for_pair(123, 321, 10)
    row = df.loc[df["k"] == 2].iloc[0]

    assert row["raw_c"] == 14
    assert row["total_s"] == 14
    assert row["outgoing_carry"] == 1
    assert bool(row["carry_active"])
    assert check_local_activation_law(df)


def test_value_preservation_still_holds() -> None:
    """Carry normalization should still preserve value in Sprint 4."""
    c = [3, 8, 14, 8, 3]
    d, _ = carry_normalize(c, 10)
    assert carry_value_preservation(c, d, 10)


def test_random_local_activation_law() -> None:
    """Random local datasets should satisfy carry_active iff total_s >= B."""
    for B in [2, 10, 16]:
        df = build_local_carry_dataset(
            n_trials=10,
            num_digits=8,
            B=B,
            density=0.7,
            rng_seed=20260507 + B,
        )
        assert check_local_activation_law(df)
