"""Carry cascade measurements."""

from __future__ import annotations

from carry_flow import carry_flow_table


def carry_active_sequence(c: list[int], B: int) -> list[bool]:
    """Return whether outgoing carry is active at each carry-flow row."""
    table = carry_flow_table(c, B)
    return [bool(value > 0) for value in table["outgoing_carry"].tolist()]


def carry_cascade_segments(c: list[int], B: int) -> list[dict[str, int]]:
    """Return consecutive active-carry segments."""
    active = carry_active_sequence(c, B)
    segments: list[dict[str, int]] = []
    start: int | None = None

    for k, is_active in enumerate(active):
        if is_active and start is None:
            start = k
        elif not is_active and start is not None:
            end = k - 1
            segments.append({"start_k": start, "end_k": end, "length": end - start + 1})
            start = None

    if start is not None:
        end = len(active) - 1
        segments.append({"start_k": start, "end_k": end, "length": end - start + 1})

    return segments


def max_carry_cascade_length(c: list[int], B: int) -> int:
    """Return the maximum number of consecutive active carry edges."""
    segments = carry_cascade_segments(c, B)
    return max((segment["length"] for segment in segments), default=0)
