"""Base-B carry normalization for projected multiplication coefficients."""

from __future__ import annotations

from typing import Any

from diagonal_projection import diagonal_projection
from digits import digits_base, value_from_digits
from interaction_matrix import interaction_matrix


def _validate_base(B: int) -> None:
    """Validate that ``B`` is a positional base."""
    if not isinstance(B, int) or isinstance(B, bool):
        raise ValueError("B must be an integer base.")
    if B < 2:
        raise ValueError("B must be at least 2.")


def _validate_coefficients(c: list[int]) -> None:
    """Validate raw nonnegative coefficients before carry."""
    if not isinstance(c, list):
        raise ValueError("c must be a list of nonnegative integers.")
    for index, coefficient in enumerate(c):
        if not isinstance(coefficient, int) or isinstance(coefficient, bool):
            raise ValueError(f"c[{index}] must be an integer.")
        if coefficient < 0:
            raise ValueError(f"c[{index}] must be nonnegative.")


def carry_normalize(c: list[int], B: int) -> tuple[list[int], list[int]]:
    """Normalize raw coefficients into base-``B`` digits.

    Returns
    -------
    d:
        Final low-to-high digits with redundant trailing zeros removed,
        except that zero is represented as ``[0]``.
    carries:
        Carry sequence with ``carries[0] = 0``.  For digit position ``k``,
        ``carries[k]`` is the incoming carry and ``carries[k + 1]`` is the
        outgoing carry.
    """
    _validate_base(B)
    _validate_coefficients(c)

    if not c:
        return [0], [0]

    digits: list[int] = []
    carries = [0]
    carry = 0

    for coefficient in c:
        s = coefficient + carry
        digits.append(s % B)
        carry = s // B
        carries.append(carry)

    while carry:
        s = carry
        digits.append(s % B)
        carry = s // B
        carries.append(carry)

    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()

    target_carry_length = len(digits) + 1
    if len(carries) > target_carry_length and all(v == 0 for v in carries[target_carry_length:]):
        carries = carries[:target_carry_length]

    return digits, carries


def carry_steps(c: list[int], B: int) -> list[dict[str, int]]:
    """Return a row-by-row description of carry normalization."""
    _validate_base(B)
    _validate_coefficients(c)

    steps: list[dict[str, int]] = []
    carry = 0
    k = 0
    for coefficient in c:
        incoming = carry
        s = coefficient + incoming
        digit = s % B
        carry = s // B
        steps.append(
            {
                "k": k,
                "raw_c": coefficient,
                "incoming_carry": incoming,
                "s": s,
                "digit": digit,
                "outgoing_carry": carry,
            }
        )
        k += 1

    while carry:
        incoming = carry
        s = incoming
        digit = s % B
        carry = s // B
        steps.append(
            {
                "k": k,
                "raw_c": 0,
                "incoming_carry": incoming,
                "s": s,
                "digit": digit,
                "outgoing_carry": carry,
            }
        )
        k += 1

    return steps


def multiply_projection_carry(x: int, y: int, B: int = 10) -> dict[str, Any]:
    """Run the full Projection--Carry multiplication pipeline."""
    _validate_base(B)
    for name, value in (("x", x), ("y", y)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be a nonnegative integer.")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative.")

    x_digits = digits_base(x, B)
    y_digits = digits_base(y, B)
    matrix = interaction_matrix(x_digits, y_digits)
    raw_diagonal_sums = diagonal_projection(matrix)
    final_digits, carries = carry_normalize(raw_diagonal_sums, B)
    result = value_from_digits(final_digits, B)

    if result != x * y:
        raise AssertionError(
            f"Projection-carry result {result} does not match {x} * {y} = {x * y}."
        )

    return {
        "x": x,
        "y": y,
        "B": B,
        "x_digits": x_digits,
        "y_digits": y_digits,
        "matrix": matrix,
        "raw_diagonal_sums": raw_diagonal_sums,
        "final_digits": final_digits,
        "carries": carries,
        "result": result,
    }
