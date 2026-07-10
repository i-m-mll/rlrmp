"""Color conversion helpers shared by analysis figure builders."""

from __future__ import annotations


def hex_to_rgba(color: str, alpha: float) -> str:
    """Return a Plotly-compatible RGBA string for a six-digit hex color."""

    value = color.removeprefix("#")
    if len(value) != 6:
        raise ValueError(f"expected six-digit hex color, got {color!r}")
    try:
        red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"invalid hex color {color!r}") from exc
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return f"rgba({red},{green},{blue},{alpha})"
