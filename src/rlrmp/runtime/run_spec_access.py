"""Fail-closed accessors for consumed run-spec fields."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class RunSpecAccessError(ValueError):
    """Raised when consumed run-spec metadata is absent."""

    def __init__(self, field: str, source: str, message: str) -> None:
        self.field = field
        self.source = source
        super().__init__(f"Run spec {source} is missing required {field}: {message}")


def require_run_seed(
    run_spec: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> int:
    """Return the run seed, raising instead of substituting a default."""

    if run_spec.get("seed") is not None:
        return int(run_spec["seed"])
    source_label = _run_spec_source(run_spec, source)
    raise RunSpecAccessError(
        "seed",
        source_label,
        "seed is consumed to reconstruct deterministic model initialization.",
    )


def require_run_dt(
    run_spec: Mapping[str, Any],
    hps: Any,
    *,
    source: str | Path | None = None,
) -> float:
    """Return the run timestep from game_card.dt or hps.dt, raising if absent."""

    game_card = run_spec.get("game_card", {})
    if isinstance(game_card, Mapping) and game_card.get("dt") is not None:
        return float(game_card["dt"])

    hps_dt = getattr(hps, "dt", None)
    if hps_dt is not None:
        return float(hps_dt)

    source_label = _run_spec_source(run_spec, source)
    raise RunSpecAccessError(
        "game_card.dt",
        source_label,
        "dt is plant physics metadata; no literal default is substituted.",
    )


def _run_spec_source(run_spec: Mapping[str, Any], source: str | Path | None) -> str:
    if source is not None:
        return str(source)
    for key in ("run_spec_path", "spec_path", "source_path", "path"):
        value = run_spec.get(key)
        if value:
            return str(value)
    run = run_spec.get("run")
    issue = run_spec.get("issue")
    if issue and run:
        return f"results/{issue}/runs/{run}.json"
    return "<unknown>"
