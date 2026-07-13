"""Shared mapping coercion for payload readers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def as_mapping(value: Any) -> Mapping[str, Any]:
    """Return mapping values unchanged and coerce other values to empty."""

    return value if isinstance(value, Mapping) else {}
