"""Run-spec validation helpers for tracked RLRMP training recipes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


NOMINAL_GRU_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "game_card",
        "task_timing",
        "model_summary",
        "training_summary",
        "provenance",
        "feedbax_graph",
    }
)
NOMINAL_GRU_REQUIRED_PROVENANCE_KEYS = frozenset(
    {
        "git",
        "dependencies",
        "modal",
        "gpu",
    }
)
FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS = frozenset(
    {
        "graph_spec_path",
        "manifest_path",
    }
)


class RunSpecValidationError(ValueError):
    """Raised when a tracked run spec is missing required metadata."""


def validate_nominal_gru_run_spec(run_spec: dict[str, Any], *, spec_dir: Path) -> None:
    """Validate the C&S-fidelity GRU run metadata contract.

    Args:
        run_spec: Decoded ``run.json`` payload.
        spec_dir: Directory containing the ``run.json`` file and graph sidecars.

    Raises:
        RunSpecValidationError: If the run spec is missing top-level metadata,
            provenance groups, graph pointers, or adjacent graph sidecar files.
    """

    missing_top_level = _missing_keys(run_spec, NOMINAL_GRU_REQUIRED_TOP_LEVEL_KEYS)
    if missing_top_level:
        raise RunSpecValidationError(
            "nominal GRU run spec is missing required top-level metadata keys: "
            + ", ".join(missing_top_level)
        )

    model_summary = _mapping(run_spec, "model_summary")
    controller_kind = model_summary.get("controller_kind")
    if controller_kind != "gru":
        raise RunSpecValidationError(
            f"nominal GRU run spec must declare model_summary.controller_kind='gru'; "
            f"found {controller_kind!r}"
        )

    training_summary = _mapping(run_spec, "training_summary")
    training_mode = training_summary.get("training_mode")
    if training_mode != "nominal":
        raise RunSpecValidationError(
            f"nominal GRU run spec must declare training_summary.training_mode='nominal'; "
            f"found {training_mode!r}"
        )

    missing_provenance = _missing_keys(
        _mapping(run_spec, "provenance"),
        NOMINAL_GRU_REQUIRED_PROVENANCE_KEYS,
    )
    if missing_provenance:
        raise RunSpecValidationError(
            "nominal GRU run spec is missing required provenance groups: "
            + ", ".join(missing_provenance)
        )

    graph_metadata = _mapping(run_spec, "feedbax_graph")
    missing_graph_pointers = _missing_keys(
        graph_metadata,
        FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS,
    )
    if missing_graph_pointers:
        raise RunSpecValidationError(
            "nominal GRU run spec is missing Feedbax graph pointer keys: "
            + ", ".join(missing_graph_pointers)
        )

    for key in sorted(FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS):
        sidecar = spec_dir / str(graph_metadata[key])
        if not sidecar.is_file():
            raise RunSpecValidationError(
                f"nominal GRU run spec points to missing Feedbax graph sidecar: {sidecar}"
            )


def validate_nominal_gru_run_spec_file(run_spec_path: Path | str) -> None:
    """Load and validate a C&S-fidelity GRU ``run.json`` file."""

    path = Path(run_spec_path)
    validate_nominal_gru_run_spec(
        json.loads(path.read_text(encoding="utf-8")),
        spec_dir=path.parent,
    )


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise RunSpecValidationError(f"nominal GRU run spec key {key!r} must be an object")
    return value


def _missing_keys(mapping: dict[str, Any], required_keys: frozenset[str]) -> list[str]:
    return sorted(key for key in required_keys if key not in mapping)


__all__ = [
    "FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS",
    "NOMINAL_GRU_REQUIRED_PROVENANCE_KEYS",
    "NOMINAL_GRU_REQUIRED_TOP_LEVEL_KEYS",
    "RunSpecValidationError",
    "validate_nominal_gru_run_spec",
    "validate_nominal_gru_run_spec_file",
]
