"""Paper figure generation for Projection--Carry Geometry scaling results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from scaling_experiments import _add_normalized_columns


def _save_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    save_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Save a grouped scatter plot."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if "density" in df.columns:
        for density, group in df.groupby("density"):
            ax.scatter(group[x_column], group[y_column], alpha=0.65, label=f"density={density}")
        ax.legend(fontsize=8)
    else:
        ax.scatter(df[x_column], df[y_column], alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _save_grouped_line(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_column: str,
    save_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Save a grouped mean line plot."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if group_column in df.columns:
        grouped = df.groupby([group_column, x_column])[y_column].mean().reset_index()
        for group_value, group in grouped.groupby(group_column):
            ax.plot(group[x_column], group[y_column], marker="o", label=f"{group_column}={group_value}")
        ax.legend(fontsize=8)
    else:
        grouped = df.groupby(x_column)[y_column].mean().reset_index()
        ax.plot(grouped[x_column], grouped[y_column], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def make_scaling_figures(input_csv: str, output_dir: str = "paper/figures") -> None:
    """Generate paper-ready figures from a scaling grid CSV."""
    if not isinstance(input_csv, str) or not input_csv:
        raise ValueError("input_csv must be a nonempty string.")
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    df = pd.read_csv(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _save_scatter(
        df,
        "additive_energy",
        "carry_mass",
        output_path / "additive_energy_vs_carry_mass.png",
        "Additive Energy vs Carry Mass",
        "additive_energy",
        "carry_mass",
    )
    _save_scatter(
        df,
        "max_representation_count",
        "carry_mass",
        output_path / "max_representation_vs_carry_mass.png",
        "Max Representation Count vs Carry Mass",
        "max_representation_count",
        "carry_mass",
    )
    _save_scatter(
        df,
        "size_sumset",
        "carry_count",
        output_path / "size_sumset_vs_carry_count.png",
        "Sumset Size vs Carry Count",
        "size_sumset",
        "carry_count",
    )
    _save_grouped_line(
        df,
        "density",
        "max_carry_cascade_length",
        "B",
        output_path / "density_vs_carry_cascade.png",
        "Density vs Carry Cascade Length",
        "density",
        "mean max_carry_cascade_length",
    )
    _save_grouped_line(
        df,
        "B",
        "carry_amplification_ratio",
        "density",
        output_path / "base_vs_carry_amplification.png",
        "Base vs Carry Amplification",
        "base B",
        "mean carry_amplification_ratio",
    )


def make_normalized_scaling_figures(input_csv: str, output_dir: str = "paper/figures") -> None:
    """Generate normalized paper-ready scaling figures."""
    if not isinstance(input_csv, str) or not input_csv:
        raise ValueError("input_csv must be a nonempty string.")
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    df = _add_normalized_columns(pd.read_csv(input_path))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _save_scatter(
        df,
        "normalized_additive_energy",
        "carry_mass_per_digit",
        output_path / "normalized_additive_energy_vs_carry_mass_per_digit.png",
        "Normalized Additive Energy vs Carry Mass per Digit",
        "normalized_additive_energy",
        "carry_mass_per_digit",
    )
    _save_grouped_line(
        df,
        "density",
        "cascade_fraction",
        "B",
        output_path / "density_vs_cascade_fraction.png",
        "Density vs Cascade Fraction",
        "density",
        "mean cascade_fraction",
    )
    _save_grouped_line(
        df,
        "B",
        "carry_mass_per_digit",
        "density",
        output_path / "base_vs_carry_mass_per_digit.png",
        "Base vs Carry Mass per Digit",
        "base B",
        "mean carry_mass_per_digit",
    )
    _save_grouped_line(
        df,
        "num_digits",
        "carry_count_per_digit",
        "B",
        output_path / "num_digits_vs_carry_count_per_digit.png",
        "Digit Length vs Carry Count per Digit",
        "num_digits",
        "mean carry_count_per_digit",
    )
