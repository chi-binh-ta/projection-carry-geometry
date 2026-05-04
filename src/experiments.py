"""Demos and experiments for Projection--Carry Geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from carry import multiply_projection_carry
from carry_complexity import carry_complexity_summary
from carry_flow import carry_flow_table
from diagonal_projection import diagonal_density, support_sumset
from digits import support_digits, value_from_digits
from local_carry_laws import build_local_carry_dataset, check_local_activation_law
from metrics import projection_summary, raw_support
from paper_figures import make_normalized_scaling_figures, make_scaling_figures
from paper_tables import make_correlation_tables
from scaling_experiments import run_scaling_grid
from support_geometry import (
    digit_support_from_digits,
    representation_counts,
    sumset_summary,
)
from visualize import (
    plot_carry_flow,
    plot_carry_profile,
    plot_diagonal_sums,
    plot_interaction_matrix,
    plot_support_sumsets,
)


def _figure_path(output_dir: str, subdir: str, filename: str) -> str:
    """Return a figure path and ensure its parent directory exists."""
    path = Path(output_dir) / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _results_path(filename: str) -> str:
    """Return a results path and ensure the results directory exists."""
    path = Path("results") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def demo_123_times_321(output_dir: str = "figures") -> dict[str, Any]:
    """Run and visualize the canonical ``123 * 321`` base-10 demo."""
    pipeline = multiply_projection_carry(123, 321, 10)
    flow_table = carry_flow_table(pipeline["raw_diagonal_sums"], 10)
    Sx = support_digits(pipeline["x_digits"])
    Sy = support_digits(pipeline["y_digits"])
    Ssum = support_sumset(Sx, Sy)

    plot_interaction_matrix(
        pipeline["matrix"],
        _figure_path(output_dir, "matrices", "123_times_321_matrix.png"),
        "123 x 321 interaction matrix",
    )
    plot_diagonal_sums(
        pipeline["raw_diagonal_sums"],
        _figure_path(output_dir, "diagonal_sums", "123_times_321_diagonal_sums.png"),
        "123 x 321 raw diagonal sums",
    )
    plot_carry_profile(
        pipeline["carries"],
        _figure_path(output_dir, "carry_profiles", "123_times_321_carry_profile.png"),
        "123 x 321 carry profile",
    )
    plot_carry_flow(
        flow_table,
        _figure_path(output_dir, "carry_profiles", "123_times_321_carry_flow.png"),
        "123 x 321 carry as mass flow",
    )
    plot_support_sumsets(
        Sx,
        Sy,
        Ssum,
        _figure_path(output_dir, "support_sumsets", "123_times_321_support_sumset.png"),
        "123 x 321 support sumset",
    )
    flow_table.to_csv(_results_path("123_times_321_carry_flow.csv"), index=False)

    assert pipeline["raw_diagonal_sums"] == [3, 8, 14, 8, 3]
    assert pipeline["final_digits"] == [3, 8, 4, 9, 3]
    assert pipeline["result"] == 39483
    pipeline["carry_flow_table"] = flow_table
    return pipeline


def demo_binary(output_dir: str = "figures") -> dict[str, Any]:
    """Run and visualize a binary multiplication demo."""
    x = int("10101", 2)
    y = int("11001", 2)
    B = 2
    pipeline = multiply_projection_carry(x, y, B)
    Sx = support_digits(pipeline["x_digits"])
    Sy = support_digits(pipeline["y_digits"])
    Ssum = support_sumset(Sx, Sy)

    plot_interaction_matrix(
        pipeline["matrix"],
        _figure_path(output_dir, "matrices", "binary_10101_times_11001_matrix.png"),
        "Binary 10101 x 11001 interaction matrix",
    )
    plot_diagonal_sums(
        pipeline["raw_diagonal_sums"],
        _figure_path(output_dir, "diagonal_sums", "binary_10101_times_11001_diagonal_sums.png"),
        "Binary 10101 x 11001 raw diagonal sums",
    )
    plot_carry_profile(
        pipeline["carries"],
        _figure_path(output_dir, "carry_profiles", "binary_10101_times_11001_carry_profile.png"),
        "Binary 10101 x 11001 carry profile",
    )
    plot_support_sumsets(
        Sx,
        Sy,
        Ssum,
        _figure_path(output_dir, "support_sumsets", "binary_10101_times_11001_support_sumset.png"),
        "Binary 10101 x 11001 support sumset",
    )
    return pipeline


def _rng_random(rng: Any) -> float:
    """Return a random float from an object resembling NumPy or stdlib RNG."""
    return float(rng.random())


def _rng_digit(rng: Any, B: int) -> int:
    """Return a random nonzero base-``B`` digit."""
    if hasattr(rng, "integers"):
        return int(rng.integers(1, B))
    if hasattr(rng, "randrange"):
        return int(rng.randrange(1, B))
    raise ValueError("rng must provide integers() or randrange().")


def random_digit_number(num_digits: int, B: int, density: float = 1.0, rng: Any = None) -> int:
    """Generate a random nonnegative integer with controlled digit support density.

    The highest digit is forced to be nonzero so the returned number has exactly
    ``num_digits`` base-``B`` digits.
    """
    if not isinstance(num_digits, int) or isinstance(num_digits, bool) or num_digits < 1:
        raise ValueError("num_digits must be a positive integer.")
    if not isinstance(B, int) or isinstance(B, bool) or B < 2:
        raise ValueError("B must be an integer at least 2.")
    if not isinstance(density, (int, float)) or not 0.0 <= float(density) <= 1.0:
        raise ValueError("density must be a number in [0, 1].")

    rng = rng if rng is not None else np.random.default_rng()
    digits: list[int] = []
    for _ in range(num_digits - 1):
        if _rng_random(rng) <= density:
            digits.append(_rng_digit(rng, B))
        else:
            digits.append(0)
    digits.append(_rng_digit(rng, B))
    return value_from_digits(digits, B)


def experiment_sparse_support(
    n_trials: int,
    num_digits: int,
    B: int,
    densities: list[float],
    output_csv: str,
) -> pd.DataFrame:
    """Run sparse-support random multiplication trials and save metrics to CSV."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if not densities:
        raise ValueError("densities must not be empty.")

    rng = np.random.default_rng(20260504)
    rows: list[dict[str, Any]] = []
    for density in densities:
        for trial in range(n_trials):
            x = random_digit_number(num_digits, B, density, rng)
            y = random_digit_number(num_digits, B, density, rng)
            summary = projection_summary(x, y, B)
            rows.append(
                {
                    "trial": trial,
                    "density": float(density),
                    "B": B,
                    "num_digits": num_digits,
                    "x": x,
                    "y": y,
                    "carry_mass": summary["carry_mass"],
                    "carry_depth": summary["carry_depth"],
                    "carry_entropy": summary["carry_entropy"],
                    "carry_count": summary["carry_count"],
                    "x_support_size": summary["x_support_size"],
                    "y_support_size": summary["y_support_size"],
                    "raw_support_size": summary["raw_support_size"],
                    "final_support_size": summary["final_support_size"],
                    "max_raw_coefficient": summary["max_raw_coefficient"],
                }
            )

    frame = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def experiment_base_dependence(
    n_trials: int,
    num_digits: int,
    bases: list[int],
    output_csv: str,
) -> pd.DataFrame:
    """Compare carry metrics across bases for dense random digit numbers."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if not bases:
        raise ValueError("bases must not be empty.")

    rng = np.random.default_rng(20260505)
    rows: list[dict[str, Any]] = []
    for B in bases:
        for trial in range(n_trials):
            x = random_digit_number(num_digits, B, density=1.0, rng=rng)
            y = random_digit_number(num_digits, B, density=1.0, rng=rng)
            summary = projection_summary(x, y, B)
            rows.append(
                {
                    "trial": trial,
                    "B": B,
                    "num_digits": num_digits,
                    "x": x,
                    "y": y,
                    "carry_mass": summary["carry_mass"],
                    "carry_depth": summary["carry_depth"],
                    "carry_entropy": summary["carry_entropy"],
                    "carry_count": summary["carry_count"],
                    "raw_support_size": summary["raw_support_size"],
                    "final_support_size": summary["final_support_size"],
                    "max_raw_coefficient": summary["max_raw_coefficient"],
                }
            )

    frame = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def experiment_carry_complexity(
    n_trials: int,
    num_digits: int,
    B: int,
    density: float,
    output_csv: str,
) -> pd.DataFrame:
    """Run random trials connecting diagonal mass concentration to carry flow."""
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if not isinstance(num_digits, int) or isinstance(num_digits, bool) or num_digits < 1:
        raise ValueError("num_digits must be a positive integer.")
    if not isinstance(B, int) or isinstance(B, bool) or B < 2:
        raise ValueError("B must be an integer at least 2.")
    if not isinstance(density, (int, float)) or not 0.0 <= float(density) <= 1.0:
        raise ValueError("density must be a number in [0, 1].")

    rng = np.random.default_rng(20260506)
    rows: list[dict[str, Any]] = []
    for trial in range(n_trials):
        x = random_digit_number(num_digits, B, density, rng)
        y = random_digit_number(num_digits, B, density, rng)
        pipeline = multiply_projection_carry(x, y, B)
        raw = pipeline["raw_diagonal_sums"]
        matrix = pipeline["matrix"]
        nonzero_density = diagonal_density(matrix, nonzero_only=True)
        geometric_density = diagonal_density(matrix, nonzero_only=False)
        complexity = carry_complexity_summary(raw, B)

        row: dict[str, Any] = {
            "trial": trial,
            "x": x,
            "y": y,
            "B": B,
            "density": float(density),
            "num_digits": num_digits,
            "x_support_size": len(support_digits(pipeline["x_digits"])),
            "y_support_size": len(support_digits(pipeline["y_digits"])),
            "raw_support_size": len(raw_support(raw)),
            "diagonal_nonzero_density": repr(nonzero_density),
            "diagonal_geometric_density": repr(geometric_density),
            "diagonal_nonzero_density_max": max(nonzero_density, default=0),
            "diagonal_geometric_density_max": max(geometric_density, default=0),
            "diagonal_nonzero_density_total": sum(nonzero_density),
        }
        row.update(complexity)
        rows.append(row)

    frame = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def experiment_local_carry_laws(
    n_trials: int,
    num_digits: int,
    B: int,
    densities: list[float],
    output_csv: str,
) -> pd.DataFrame:
    """Generate local carry-law rows across support densities."""
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if not isinstance(num_digits, int) or isinstance(num_digits, bool) or num_digits < 1:
        raise ValueError("num_digits must be a positive integer.")
    if not isinstance(B, int) or isinstance(B, bool) or B < 2:
        raise ValueError("B must be an integer at least 2.")
    if not isinstance(densities, list) or not densities:
        raise ValueError("densities must be a nonempty list.")

    frames: list[pd.DataFrame] = []
    for index, density in enumerate(densities):
        if not isinstance(density, (int, float)) or not 0.0 <= float(density) <= 1.0:
            raise ValueError("each density must be a number in [0, 1].")
        frames.append(
            build_local_carry_dataset(
                n_trials=n_trials,
                num_digits=num_digits,
                B=B,
                density=float(density),
                rng_seed=20260507 + index,
            )
        )

    frame = pd.concat(frames, ignore_index=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def experiment_support_geometry_carry(
    n_trials: int,
    num_digits: int,
    B: int,
    densities: list[float],
    output_csv: str,
) -> pd.DataFrame:
    """Connect support-sumset geometry to carry complexity metrics."""
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer.")
    if not isinstance(num_digits, int) or isinstance(num_digits, bool) or num_digits < 1:
        raise ValueError("num_digits must be a positive integer.")
    if not isinstance(B, int) or isinstance(B, bool) or B < 2:
        raise ValueError("B must be an integer at least 2.")
    if not isinstance(densities, list) or not densities:
        raise ValueError("densities must be a nonempty list.")

    rng = np.random.default_rng(20260508)
    rows: list[dict[str, Any]] = []
    for density in densities:
        if not isinstance(density, (int, float)) or not 0.0 <= float(density) <= 1.0:
            raise ValueError("each density must be a number in [0, 1].")
        for trial in range(n_trials):
            x = random_digit_number(num_digits, B, float(density), rng)
            y = random_digit_number(num_digits, B, float(density), rng)
            pipeline = multiply_projection_carry(x, y, B)
            x_digits = pipeline["x_digits"]
            y_digits = pipeline["y_digits"]
            Sx = digit_support_from_digits(x_digits)
            Sy = digit_support_from_digits(y_digits)
            raw = pipeline["raw_diagonal_sums"]
            counts = representation_counts(Sx, Sy)

            row: dict[str, Any] = {
                "trial": trial,
                "x": x,
                "y": y,
                "B": B,
                "density": float(density),
                "num_digits": num_digits,
                "raw_support_size": len(raw_support(raw)),
                "representation_counts": repr(counts),
            }
            row.update(sumset_summary(Sx, Sy))
            row.update(carry_complexity_summary(raw, B))
            rows.append(row)

    frame = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def experiment_scaling_grid(output_csv: str = "results/scaling_grid.csv") -> pd.DataFrame:
    """Run the Sprint 6 small scaling grid for paper-ready experiments."""
    return run_scaling_grid(
        output_csv=output_csv,
        bases=[2, 10, 16],
        num_digits_list=[8, 16],
        densities=[0.2, 0.5, 0.7, 1.0],
        n_trials=20,
        rng_seed=20260509,
    )


def _run_required_assertions() -> None:
    """Run the correctness checks requested for the project."""
    demo = multiply_projection_carry(123, 321, 10)
    assert demo["result"] == 39483
    assert demo["raw_diagonal_sums"] == [3, 8, 14, 8, 3]
    assert demo["final_digits"] == [3, 8, 4, 9, 3]

    assert multiply_projection_carry(99, 99, 10)["result"] == 9801

    rng = np.random.default_rng(12345)
    for _ in range(100):
        B = int(rng.choice([2, 3, 5, 10, 16, 100]))
        x = int(rng.integers(0, 100_000))
        y = int(rng.integers(0, 100_000))
        assert multiply_projection_carry(x, y, B)["result"] == x * y


def main() -> None:
    """Run demos, small experiments, and required correctness checks."""
    Path("results").mkdir(parents=True, exist_ok=True)
    _run_required_assertions()
    demo = demo_123_times_321("figures")
    binary = demo_binary("figures")
    sparse_frame = experiment_sparse_support(
        n_trials=20,
        num_digits=8,
        B=10,
        densities=[0.2, 0.5, 1.0],
        output_csv="figures/sparse_support_metrics.csv",
    )
    base_frame = experiment_base_dependence(
        n_trials=20,
        num_digits=8,
        bases=[2, 10, 16, 100],
        output_csv="figures/base_dependence_metrics.csv",
    )
    carry_complexity_csv = _results_path("carry_complexity_metrics.csv")
    complexity_frame = experiment_carry_complexity(
        n_trials=40,
        num_digits=10,
        B=10,
        density=0.7,
        output_csv=carry_complexity_csv,
    )
    local_carry_csv = _results_path("local_carry_laws.csv")
    local_frame = experiment_local_carry_laws(
        n_trials=20,
        num_digits=10,
        B=10,
        densities=[0.2, 0.5, 0.7, 1.0],
        output_csv=local_carry_csv,
    )
    local_law_passed = check_local_activation_law(local_frame)
    support_geometry_csv = _results_path("support_geometry_carry.csv")
    support_frame = experiment_support_geometry_carry(
        n_trials=20,
        num_digits=10,
        B=10,
        densities=[0.2, 0.5, 0.7, 1.0],
        output_csv=support_geometry_csv,
    )
    scaling_grid_csv = _results_path("scaling_grid.csv")
    scaling_frame = experiment_scaling_grid(scaling_grid_csv)
    paper_figure_dir = Path("paper") / "figures"
    make_scaling_figures(scaling_grid_csv, str(paper_figure_dir))
    make_normalized_scaling_figures(scaling_grid_csv, str(paper_figure_dir))
    paper_table_dir = Path("paper") / "tables"
    make_correlation_tables(scaling_grid_csv, str(paper_table_dir))
    normalized_figure_paths = sorted(
        str(path)
        for path in paper_figure_dir.glob("*.png")
        if path.name
        in {
            "normalized_additive_energy_vs_carry_mass_per_digit.png",
            "density_vs_cascade_fraction.png",
            "base_vs_carry_mass_per_digit.png",
            "num_digits_vs_carry_count_per_digit.png",
        }
    )
    paper_figure_paths = sorted(str(path) for path in paper_figure_dir.glob("*.png"))
    paper_table_paths = sorted(str(path) for path in paper_table_dir.glob("*.csv"))

    print("Projection--Carry Geometry demos completed.")
    print(f"123 * 321 raw diagonal sums: {demo['raw_diagonal_sums']}")
    print(f"123 * 321 final digits: {demo['final_digits']}")
    print(f"123 * 321 result: {demo['result']}")
    print(
        "Binary demo: "
        f"{binary['x']} * {binary['y']} = {binary['result']} "
        f"in base {binary['B']}"
    )
    print(f"Sparse-support metrics saved with {len(sparse_frame)} rows.")
    print(f"Base-dependence metrics saved with {len(base_frame)} rows.")
    print(f"Carry-complexity metrics saved with {len(complexity_frame)} rows.")
    print(f"Carry-complexity CSV: {carry_complexity_csv}")
    print(f"Local carry-law rows saved with {len(local_frame)} rows.")
    print(f"Local carry-law CSV: {local_carry_csv}")
    print(f"Local activation law passed: {local_law_passed}")
    print(f"Support-geometry carry rows saved with {len(support_frame)} rows.")
    print(f"Support-geometry carry CSV: {support_geometry_csv}")
    print(f"Scaling grid rows saved with {len(scaling_frame)} rows.")
    print(f"Scaling grid CSV: {scaling_grid_csv}")
    print("Paper figure paths:")
    for path in paper_figure_paths:
        print(f"  {path}")
    print("Normalized paper figure paths:")
    for path in normalized_figure_paths:
        print(f"  {path}")
    print("Paper table paths:")
    for path in paper_table_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
