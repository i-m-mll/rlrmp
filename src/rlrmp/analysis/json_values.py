"""JSON-safe scalar conversion for analysis payloads."""

from __future__ import annotations

import math


def json_float(value: float) -> float | str:
    """Convert a float to JSON-safe finite or named non-finite form."""

    resolved = float(value)
    if math.isinf(resolved):
        return "inf" if resolved > 0 else "-inf"
    if math.isnan(resolved):
        return "nan"
    return resolved


def json_float_or_none(value: float | None) -> float | str | None:
    """Convert an optional float with :func:`json_float`."""

    if value is None:
        return None
    return json_float(value)
