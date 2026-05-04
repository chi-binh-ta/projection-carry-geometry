"""Local carry activation datasets and laws."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from carry import multiply_projection_carry
from carry_flow import carry_flow_table
from diagonal_projection import diagonal_density
from digits import value_from_digits


def _validate_base(B: int) -> None:
    """Validate a positional base."""
    if not isinstance(B, int) or isinstance(B, bool):
        raise ValueError("B must be an integer base.")
    if B < 2:
        raise ValueError("B must be at least 2.")


def _validate_nonnegative_int(name: str, value: int) -> None:
    """Validate a nonnegative integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _validate_density(density: float) -> None:
    """Validate support density."""
    if not isinstance(density, (int, float)) or not 0.0 <= float(density) <= 1.0:
        raise ValueError("density must be a number in [0, 1].")


def _random_digit_number(num_digits: int, B: int, density: float, rng: Any) -> int:
    """Generate a random number with low-to-high digits and nonzero top digit."""
    _validate_nonnegative_int("num_digits", num_digits)
    if num_digits < 1:
        raise ValueError("num_digits must be positive.")
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


def local_carry_dataset_for_pair(x: int, y: int, B: int) -> pd.DataFrame:
    """Return one local carry-law row per degree for one multiplication pair."""
    _validate_nonnegative_int("x", x)
    _validate_nonnegative_int("y", y)
    _validate_base(B)

    pipeline = multiply_projection_carry(x, y, B)
    raw = pipeline["raw_diagonal_sums"]
    flow = carry_flow_table(raw, B)
    nonzero_density = diagonal_density(pipeline["matrix"], nonzero_only=True)

    rows: list[dict[str, int | bool]] = []
    for row in flow.to_dict("records"):
        k = int(row["k"])
        total_s = int(row["total_s"])
        outgoing = int(row["outgoing_carry"])
        carry_active = outgoing > 0
        rows.append(
            {
                "x": x,
                "y": y,
                "B": B,
                "k": k,
                "raw_c": int(row["raw_c"]),
                "diagonal_density_nonzero": (
                    int(nonzero_density[k]) if k < len(nonzero_density) else 0
                ),
                "incoming_carry": int(row["incoming_carry"]),
                "total_s": total_s,
                "digit": int(row["digit"]),
                "outgoing_carry": outgoing,
                "carry_active": carry_active,
                "activation_threshold": B,
                "excess": max(total_s - (B - 1), 0),
                "local_activation_correct": carry_active == (total_s >= B),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "x",
            "y",
            "B",
            "k",
            "raw_c",
            "diagonal_density_nonzero",
            "incoming_carry",
            "total_s",
            "digit",
            "outgoing_carry",
            "carry_active",
            "activation_threshold",
            "excess",
            "local_activation_correct",
        ],
    )


def build_local_carry_dataset(
    n_trials: int,
    num_digits: int,
    B: int,
    density: float,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Build a local carry-law dataset from random multiplication pairs."""
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if not isinstance(num_digits, int) or isinstance(num_digits, bool) or num_digits < 1:
        raise ValueError("num_digits must be a positive integer.")
    _validate_base(B)
    _validate_density(density)
    if not isinstance(rng_seed, int) or isinstance(rng_seed, bool):
        raise ValueError("rng_seed must be an integer.")

    rng = np.random.default_rng(rng_seed)
    frames: list[pd.DataFrame] = []
    for trial in range(n_trials):
        x = _random_digit_number(num_digits, B, density, rng)
        y = _random_digit_number(num_digits, B, density, rng)
        frame = local_carry_dataset_for_pair(x, y, B)
        frame.insert(0, "trial", trial)
        frame.insert(1, "num_digits", num_digits)
        frame.insert(2, "density", float(density))
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def check_local_activation_law(df: pd.DataFrame) -> bool:
    """Return whether all rows satisfy ``carry_active == (total_s >= B)``."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas.DataFrame.")
    if "local_activation_correct" in df.columns:
        return bool(df["local_activation_correct"].all())
    required = {"carry_active", "total_s", "B"}
    if not required.issubset(set(df.columns)):
        raise ValueError("df must contain local_activation_correct or carry_active, total_s, and B.")
    return bool((df["carry_active"] == (df["total_s"] >= df["B"])).all())
