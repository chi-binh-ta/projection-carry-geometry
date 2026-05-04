"""Tests for paper table generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from paper_tables import make_correlation_tables
from scaling_experiments import run_scaling_grid


def test_make_correlation_tables(tmp_path: Path) -> None:
    """Paper table generation should write all expected CSV files."""
    scaling_csv = tmp_path / "scaling_grid.csv"
    table_dir = tmp_path / "tables"
    run_scaling_grid(
        output_csv=str(scaling_csv),
        bases=[2],
        num_digits_list=[4],
        densities=[0.5],
        n_trials=3,
        rng_seed=20260510,
    )

    make_correlation_tables(str(scaling_csv), str(table_dir))

    assert (table_dir / "correlation_table.csv").exists()
    assert (table_dir / "normalized_correlation_table.csv").exists()
    assert (table_dir / "grouped_scaling_summary.csv").exists()
