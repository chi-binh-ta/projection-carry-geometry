"""Paper table generation for Projection--Carry Geometry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scaling_experiments import (
    correlation_table,
    normalized_correlation_table,
    summarize_scaling_results,
)


def make_correlation_tables(input_csv: str, output_dir: str = "paper/tables") -> None:
    """Generate paper-ready correlation and grouped summary CSV tables."""
    if not isinstance(input_csv, str) or not input_csv:
        raise ValueError("input_csv must be a nonempty string.")
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)

    correlation_table(df).to_csv(output_path / "correlation_table.csv")
    normalized_correlation_table(df).to_csv(output_path / "normalized_correlation_table.csv")
    summarize_scaling_results(df).to_csv(output_path / "grouped_scaling_summary.csv", index=False)
