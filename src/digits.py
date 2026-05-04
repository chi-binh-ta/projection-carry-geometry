"""Digit representation utilities for positional bases.

All digit lists in this project are low-to-high:
the integer 123 in base 10 is represented as ``[3, 2, 1]``.
"""

from __future__ import annotations


def _validate_base(B: int) -> None:
    """Validate that ``B`` is a positional base."""
    if not isinstance(B, int) or isinstance(B, bool):
        raise ValueError("B must be an integer base.")
    if B < 2:
        raise ValueError("B must be at least 2.")


def digits_base(n: int, B: int) -> list[int]:
    """Return the base-``B`` digits of ``n`` in low-to-high order.

    Examples
    --------
    >>> digits_base(123, 10)
    [3, 2, 1]
    >>> digits_base(0, 10)
    [0]
    """
    _validate_base(B)
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValueError("n must be a nonnegative integer.")
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        return [0]

    digits: list[int] = []
    while n:
        digits.append(n % B)
        n //= B
    return digits


def validate_digits(digits: list[int], B: int) -> None:
    """Raise ``ValueError`` if any digit is outside ``[0, B - 1]``."""
    _validate_base(B)
    if not isinstance(digits, list):
        raise ValueError("digits must be a list of integers.")
    if not digits:
        raise ValueError("digits must not be empty.")

    for index, digit in enumerate(digits):
        if not isinstance(digit, int) or isinstance(digit, bool):
            raise ValueError(f"digits[{index}] must be an integer.")
        if digit < 0 or digit >= B:
            raise ValueError(f"digits[{index}]={digit} is outside [0, {B - 1}].")


def value_from_digits(digits: list[int], B: int) -> int:
    """Reconstruct an integer from low-to-high base-``B`` digits.

    Examples
    --------
    >>> value_from_digits([3, 2, 1], 10)
    123
    """
    validate_digits(digits, B)

    value = 0
    place = 1
    for digit in digits:
        value += digit * place
        place *= B
    return value


def support_digits(digits: list[int]) -> set[int]:
    """Return the digit support ``{i : digits[i] != 0}``."""
    if not isinstance(digits, list):
        raise ValueError("digits must be a list of integers.")
    return {index for index, digit in enumerate(digits) if digit != 0}
