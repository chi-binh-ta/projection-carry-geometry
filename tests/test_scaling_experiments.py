"""Tests for Sprint 6 scaling experiments and paper figures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from paper_figures import make_normalized_scaling_figures, make_scaling_figures
from scaling_experiments import (
    CARRY_OUTCOMES,
    GEOMETRY_PREDICTORS,
    NORMALIZED_CARRY_OUTCOMES,
    NORMALIZED_GEOMETRY_PREDICTORS,
    correlation_table,
    normalized_correlation_table,
    run_scaling_grid,
    summarize_scaling_results,
)


def test_run_scaling_grid_tiny_config(tmp_path: Path) -> None:
    """A tiny scaling grid should return one row per random trial."""
    csv_path = tmp_path / "scaling_grid.csv"
    frame = run_scaling_grid(
        output_csv=str(csv_path),
        bases=[2],
        num_digits_list=[4],
        densities=[0.5],
        n_trials=3,
        rng_seed=123,
    )

    assert len(frame) == 3
    assert csv_path.exists()
    assert {"additive_energy", "carry_mass", "max_carry_cascade_length"}.issubset(frame.columns)


def test_summarize_scaling_results_nonempty(tmp_path: Path) -> None:
    """Scaling summaries should aggregate by base, digit count, and density."""
    frame = run_scaling_grid(
        output_csv=str(tmp_path / "scaling_grid.csv"),
        bases=[2],
        num_digits_list=[4],
        densities=[0.5],
        n_trials=3,
        rng_seed=124,
    )
    summary = summarize_scaling_results(frame)

    assert not summary.empty
    assert {"B", "num_digits", "density", "carry_mass_mean"}.issubset(summary.columns)


def test_correlation_table_shape(tmp_path: Path) -> None:
    """Correlation table should map geometry predictors to carry outcomes."""
    frame = run_scaling_grid(
        output_csv=str(tmp_path / "scaling_grid.csv"),
        bases=[2],
        num_digits_list=[4],
        densities=[0.5],
        n_trials=3,
        rng_seed=125,
    )
    correlations = correlation_table(frame)

    assert list(correlations.index) == GEOMETRY_PREDICTORS
    assert list(correlations.columns) == CARRY_OUTCOMES


def test_normalized_correlation_table_shape(tmp_path: Path) -> None:
    """Normalized correlation table should use normalized predictors/outcomes."""
    frame = run_scaling_grid(
        output_csv=str(tmp_path / "scaling_grid.csv"),
        bases=[2],
        num_digits_list=[4],
        densities=[0.5],
        n_trials=3,
        rng_seed=127,
    )
    correlations = normalized_correlation_table(frame)

    assert list(correlations.index) == NORMALIZED_GEOMETRY_PREDICTORS
    assert list(correlations.columns) == NORMALIZED_CARRY_OUTCOMES


def test_make_scaling_figures_creates_png(tmp_path: Path) -> None:
    """Paper figure generation should create PNG files from a tiny CSV."""
    csv_path = tmp_path / "scaling_grid.csv"
    figure_dir = tmp_path / "figures"
    run_scaling_grid(
        output_csv=str(csv_path),
        bases=[2],
        num_digits_list=[4],
        densities=[0.5],
        n_trials=3,
        rng_seed=126,
    )

    make_scaling_figures(str(csv_path), str(figure_dir))

    pngs = list(figure_dir.glob("*.png"))
    assert pngs
    assert (figure_dir / "additive_energy_vs_carry_mass.png").exists()


def test_make_normalized_scaling_figures_creates_png(tmp_path: Path) -> None:
    """Normalized figure generation should create normalized PNG files."""
    csv_path = tmp_path / "scaling_grid.csv"
    figure_dir = tmp_path / "figures"
    run_scaling_grid(
        output_csv=str(csv_path),
        bases=[2],
        num_digits_list=[4],
        densities=[0.5],
        n_trials=3,
        rng_seed=128,
    )

    make_normalized_scaling_figures(str(csv_path), str(figure_dir))

    assert (figure_dir / "normalized_additive_energy_vs_carry_mass_per_digit.png").exists()
