"""Parsing and formatting of human-entered timestamps."""

from __future__ import annotations


def parse_timecode(value: str | float | int | None) -> float:
    """Parse ``SS``, ``MM:SS`` or ``HH:MM:SS`` (fractions allowed) into seconds.

    Blank or ``None`` returns 0.0, which callers treat as "not set".

    Raises:
        ValueError: the text is not a valid timestamp.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("Time cannot be negative.")
        return float(value)

    text = value.strip()
    if not text:
        return 0.0

    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid time {value!r}. Use HH:MM:SS, MM:SS or seconds.")

    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid time {value!r}. Use HH:MM:SS, MM:SS or seconds.") from exc

    if any(n < 0 for n in numbers):
        raise ValueError("Time cannot be negative.")
    if len(numbers) > 1 and any(n >= 60 for n in numbers[1:]):
        raise ValueError(f"Invalid time {value!r}: minutes and seconds must be below 60.")

    seconds = 0.0
    for n in numbers:
        seconds = seconds * 60 + n
    return seconds
