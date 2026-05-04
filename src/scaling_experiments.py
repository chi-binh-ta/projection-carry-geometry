"""Scaling experiments for paper-ready Projection--Carry Geometry results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from carry import multiply_projection_carry
from carry_complexity import carry_complexity_summary
from digits import value_from_digits
from metrics import final_support, raw_support
from support_geometry import digit_support_from_digits, sumset_summary

GEOMETRY_PREDICTORS = [
    "size_sumset",
    "max_representation_count",
    "additive_energy",
    "support_density_x",
    "support_density_y",
    "diagonal_mass_concentration",
]

CARRY_OUTCOMES = [
    "carry_mass",
    "carry_count",
    "max_carry_cascade_length",
    "carry_amplification_ratio",
    "normalized_value_weighted_carry",
]

NORMALIZED_GEOMETRY_PREDICTORS = [
    "normalized_additive_energy",
    "normalized_sumset_size",
    "support_density_x",
    "support_density_y",
    "diagonal_mass_concentration",
]

NORMALIZED_CARRY_OUTCOMES = [
    "carry_mass_per_digit",
    "carry_count_per_digit",
    "cascade_fraction",
    "carry_amplification_ratio",
    "normalized_value_weighted_carry",
]

SCALING_METRICS = [
    "additive_energy",
    "max_representation_count",
    "size_sumset",
    "carry_mass",
    "carry_count",
    "max_carry_cascade_length",
    "carry_amplification_ratio",
    "normalized_value_weighted_carry",
]


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    """Return Pearson correlation, using 0.0 for constant or undersized input."""
    if len(left) < 2 or len(right) < 2:
        return 0.0
    if float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return 0.0
    value = left.corr(right)
    if pd.isna(value):
        return 0.0
    return float(value)


def _validate_base(B: int) -> None:
    """Validate a positional base."""
    if not isinstance(B, int) or isinstance(B, bool):
        raise ValueError("B must be an integer base.")
    if B < 2:
        raise ValueError("B must be at least 2.")


def _validate_positive_int(name: str, value: int) -> None:
    """Validate a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_density(density: float) -> None:
    """Validate support density."""
    if not isinstance(density, (int, float)) or not 0.0 <= float(density) <= 1.0:
        raise ValueError("density must be a number in [0, 1].")


def _random_digit_number(num_digits: int, B: int, density: float, rng: Any) -> int:
    """Generate a random integer with controlled digit support density."""
    _validate_positive_int("num_digits", num_digits)
    _validate_base(B)
    _validate_density(density)

    digits: list[int] = []
    for _ in range(num_digits - 1):
        if float(rng.random()) <= density:
            digits.append(int(rng.integers(1, B)))
        else:
            digits.append(0)
    digits.append(int(rng.integers(1, B)))
    return value_from_digits(digits, B)


def run_scaling_grid(
    output_csv: str,
    bases: list[int] = [2, 10, 16],
    num_digits_list: list[int] = [8, 16, 32],
    densities: list[float] = [0.2, 0.5, 0.7, 1.0],
    n_trials: int = 50,
    rng_seed: int = 20260509,
) -> pd.DataFrame:
    """Run a grid of random support-geometry and carry-complexity experiments."""
    if not isinstance(output_csv, str) or not output_csv:
        raise ValueError("output_csv must be a nonempty string.")
    if not isinstance(bases, list) or not bases:
        raise ValueError("bases must be a nonempty list.")
    if not isinstance(num_digits_list, list) or not num_digits_list:
        raise ValueError("num_digits_list must be a nonempty list.")
    if not isinstance(densities, list) or not densities:
        raise ValueError("densities must be a nonempty list.")
    _validate_positive_int("n_trials", n_trials)
    if not isinstance(rng_seed, int) or isinstance(rng_seed, bool):
        raise ValueError("rng_seed must be an integer.")

    for B in bases:
        _validate_base(B)
    for num_digits in num_digits_list:
        _validate_positive_int("num_digits", num_digits)
    for density in densities:
        _validate_density(density)

    rng = np.random.default_rng(rng_seed)
    rows: list[dict[str, Any]] = []
    for B in bases:
        for num_digits in num_digits_list:
            for density in densities:
                for trial in range(n_trials):
                    x = _random_digit_number(num_digits, B, float(density), rng)
                    y = _random_digit_number(num_digits, B, float(density), rng)
                    pipeline = multiply_projection_carry(x, y, B)
                    raw = pipeline["raw_diagonal_sums"]
                    final_digits = pipeline["final_digits"]
                    x_digits = pipeline["x_digits"]
                    y_digits = pipeline["y_digits"]
                    Sx = digit_support_from_digits(x_digits)
                    Sy = digit_support_from_digits(y_digits)

                    row: dict[str, Any] = {
                        "B": B,
                        "num_digits": num_digits,
                        "density": float(density),
                        "trial": trial,
                        "x": x,
                        "y": y,
                        "x_support_size": len(Sx),
                        "y_support_size": len(Sy),
                        "raw_support_size": len(raw_support(raw)),
                        "final_support_size": len(final_support(final_digits)),
                    }
                    row.update(sumset_summary(Sx, Sy))
                    row.update(carry_complexity_summary(raw, B))
                    rows.append(row)

    frame = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def summarize_scaling_results(df: pd.DataFrame) -> pd.DataFrame:
    """Group scaling results by configuration and compute mean/std metrics."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas.DataFrame.")
    required = {"B", "num_digits", "density", *SCALING_METRICS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    summary = df.groupby(["B", "num_digits", "density"])[SCALING_METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return predictor/outcome correlation table for scaling results."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas.DataFrame.")
    required = set(GEOMETRY_PREDICTORS + CARRY_OUTCOMES)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    rows: dict[str, dict[str, float]] = {}
    for predictor in GEOMETRY_PREDICTORS:
        rows[predictor] = {}
        for outcome in CARRY_OUTCOMES:
            rows[predictor][outcome] = _safe_corr(df[predictor], df[outcome])
    return pd.DataFrame.from_dict(rows, orient="index", columns=CARRY_OUTCOMES)


def _add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with Sprint 7 normalized metrics added."""
    required = {
        "additive_energy",
        "x_support_size",
        "y_support_size",
        "size_sumset",
        "raw_support_size",
        "carry_mass",
        "carry_count",
        "max_carry_cascade_length",
        "num_digits",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    normalized = df.copy()
    support_pair_count = normalized["x_support_size"] * normalized["y_support_size"]
    normalized["normalized_additive_energy"] = normalized["additive_energy"] / (
        support_pair_count.pow(2).clip(lower=1)
    )
    normalized["normalized_sumset_size"] = normalized["size_sumset"] / normalized[
        "raw_support_size"
    ].clip(lower=1)
    normalized["carry_mass_per_digit"] = normalized["carry_mass"] / normalized[
        "num_digits"
    ].clip(lower=1)
    normalized["carry_count_per_digit"] = normalized["carry_count"] / normalized[
        "num_digits"
    ].clip(lower=1)
    normalized["cascade_fraction"] = normalized["max_carry_cascade_length"] / (
        (2 * normalized["num_digits"] - 1).clip(lower=1)
    )
    return normalized


def normalized_correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return correlations between normalized geometry and carry metrics."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas.DataFrame.")
    normalized = _add_normalized_columns(df)
    required = set(NORMALIZED_GEOMETRY_PREDICTORS + NORMALIZED_CARRY_OUTCOMES)
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    rows: dict[str, dict[str, float]] = {}
    for predictor in NORMALIZED_GEOMETRY_PREDICTORS:
        rows[predictor] = {}
        for outcome in NORMALIZED_CARRY_OUTCOMES:
            rows[predictor][outcome] = _safe_corr(normalized[predictor], normalized[outcome])
    return pd.DataFrame.from_dict(rows, orient="index", columns=NORMALIZED_CARRY_OUTCOMES)
