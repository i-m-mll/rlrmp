"""Shared finite-array summary helpers for analysis pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

CountKey = Literal["count", "n"]


def summary_stats(
    values: Any,
    *,
    count_key: CountKey = "count",
    quantiles: Iterable[float] = (0.5, 0.95),
) -> dict[str, float | int]:
    """Return flat mean/std/min/max summaries with optional quantiles."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            count_key: 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    flat = array.reshape(-1)
    summary: dict[str, float | int] = {
        count_key: int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }
    for probability in quantiles:
        percentile = int(round(float(probability) * 100.0))
        summary[f"p{percentile:02d}"] = float(np.quantile(flat, probability))
    return summary
