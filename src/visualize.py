"""Visualization helpers for Projection--Carry Geometry."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(save_path: str) -> Path:
    """Create the parent directory for a figure path and return it as ``Path``."""
    if not isinstance(save_path, str) or not save_path:
        raise ValueError("save_path must be a nonempty string.")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_interaction_matrix(M: np.ndarray, save_path: str, title: str = "") -> None:
    """Plot and save a heatmap of an interaction matrix."""
    if not isinstance(M, np.ndarray) or M.ndim != 2:
        raise ValueError("M must be a two-dimensional numpy.ndarray.")
    path = _ensure_parent(save_path)

    fig_width = max(5.0, 0.9 * M.shape[1] + 2.0)
    fig_height = max(4.0, 0.75 * M.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(M, cmap="viridis", origin="upper")
    ax.set_xlabel("j: digit index of y")
    ax.set_ylabel("i: digit index of x")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_title(title or "Outer-product interaction matrix")

    max_value = float(np.max(M)) if M.size else 0.0
    threshold = max_value / 2.0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            value = int(M[i, j])
            text_color = "white" if value > threshold else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=text_color, fontsize=10)
            ax.text(
                j + 0.34,
                i - 0.34,
                f"k={i + j}",
                ha="right",
                va="top",
                color=text_color,
                fontsize=7,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_diagonal_sums(c: list[int], save_path: str, title: str = "") -> None:
    """Plot and save raw anti-diagonal sums ``c_k``."""
    if not isinstance(c, list):
        raise ValueError("c must be a list of integers.")
    path = _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(c) + 2.0), 3.8))
    ax.bar(range(len(c)), c, color="#4C78A8")
    ax.set_xlabel("k")
    ax.set_ylabel("raw sum c_k")
    ax.set_title(title or "Raw anti-diagonal sums")
    ax.set_xticks(range(len(c)))
    for index, value in enumerate(c):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_carry_profile(carries: list[int], save_path: str, title: str = "") -> None:
    """Plot and save the carry sequence."""
    if not isinstance(carries, list):
        raise ValueError("carries must be a list of integers.")
    path = _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(carries) + 2.0), 3.8))
    ax.bar(range(len(carries)), carries, color="#F58518")
    ax.set_xlabel("carry index")
    ax.set_ylabel("carry value")
    ax.set_title(title or "Carry profile")
    ax.set_xticks(range(len(carries)))
    for index, value in enumerate(carries):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_carry_flow(flow_table, save_path: str, title: str = "") -> None:
    """Plot raw coefficients and outgoing carry as a degree-axis flow profile."""
    required = {"k", "raw_c", "outgoing_carry"}
    if not hasattr(flow_table, "columns") or not required.issubset(set(flow_table.columns)):
        raise ValueError("flow_table must contain k, raw_c, and outgoing_carry columns.")
    path = _ensure_parent(save_path)

    k_values = list(flow_table["k"])
    raw_values = list(flow_table["raw_c"])
    carry_values = list(flow_table["outgoing_carry"])

    fig, ax = plt.subplots(figsize=(max(6.0, 0.65 * len(k_values) + 2.0), 4.0))
    ax.bar(k_values, raw_values, width=0.72, color="#4C78A8", alpha=0.78, label="raw_c")
    ax.plot(
        k_values,
        carry_values,
        color="#E45756",
        marker="o",
        linewidth=2.0,
        label="outgoing_carry",
    )
    for k, raw, carry in zip(k_values, raw_values, carry_values):
        ax.text(k, raw, str(raw), ha="center", va="bottom", fontsize=9)
        if carry > 0:
            ax.text(k, carry, str(carry), ha="center", va="bottom", fontsize=9, color="#B33A3A")

    ax.set_xlabel("degree k")
    ax.set_ylabel("mass")
    ax.set_title(title or "Carry as nonlinear mass flow")
    ax.set_xticks(k_values)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_support_sumsets(
    Sx: set[int], Sy: set[int], Ssum: set[int], save_path: str, title: str = ""
) -> None:
    """Plot supports of x, y, and the additive sumset ``Sx + Sy``."""
    for name, support in (("Sx", Sx), ("Sy", Sy), ("Ssum", Ssum)):
        if not isinstance(support, set):
            raise ValueError(f"{name} must be a set of integers.")
    path = _ensure_parent(save_path)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    rows = [(Ssum, 0, "Sx + Sy", "#54A24B"), (Sy, 1, "Sy", "#E45756"), (Sx, 2, "Sx", "#4C78A8")]
    for support, y_value, label, color in rows:
        xs = sorted(support)
        ys = [y_value] * len(xs)
        if xs:
            ax.scatter(xs, ys, s=80, label=label, color=color)
        ax.hlines(y_value, min(xs, default=0), max(xs, default=0), color=color, alpha=0.2)

    all_values = sorted(Sx | Sy | Ssum)
    if all_values:
        ax.set_xlim(min(all_values) - 1, max(all_values) + 1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Sx + Sy", "Sy", "Sx"])
    ax.set_xlabel("digit index / degree")
    ax.set_title(title or "Digit supports and support sumset")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
