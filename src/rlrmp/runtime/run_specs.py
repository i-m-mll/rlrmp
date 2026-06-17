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
        "loss_summary",
        "provenance",
        "feedbax_graph",
    }
)
NOMINAL_GRU_LOSS_OBJECTIVES = frozenset(
    {
        "partial_feedbax_terms",
        "partial_net_output_force_filter",
        "full_analytical_qrf",
    }
)
NOMINAL_GRU_TRAINING_MODES = frozenset(
    {
        "nominal",
        "fixed_target_perturbation_randomized",
        "fixed_target_perturbation_generalized",
        "broad_full_state_epsilon_l2",
        "broad_full_state_epsilon_pgd_l2",
        "target_relative_multitarget_static",
        "target_relative_multitarget_static_h0",
        "delayed_reach_target_visible_go_cue",
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
CS_LSS_PLANT_BACKEND = "cs_lss"
CS_LSS_REQUIRED_MECHANICS_TYPE = "LinearStateSpace"
CS_LSS_FEEDBACK_COMPONENT_TYPES = frozenset(
    {
        "StateFeedbackSelector",
        "RLRMPCsLssDelayedPositionVelocityFeedback",
        "RLRMPCsLssTargetRelativeDelayedFeedback",
        "RLRMPCsLssTargetRelativeDelayedProprioceptiveFeedback",
    }
)
LEGACY_POINT_MASS_GRAPH_TYPES = frozenset(
    {
        "FirstOrderFilter",
        "PointMass",
        "RLRMPFeedbackChannels",
        "RLRMPPointMass",
    }
)


class RunSpecValidationError(ValueError):
    """Raised when a tracked run spec is missing required metadata."""


def validate_nominal_gru_run_spec(
    run_spec: dict[str, Any],
    *,
    spec_dir: Path,
    require_graph_sidecars: bool = True,
) -> None:
    """Validate the C&S-fidelity GRU run metadata contract.

    Args:
        run_spec: Decoded ``run.json`` payload.
        spec_dir: Directory containing the ``run.json`` file and graph sidecars.
        require_graph_sidecars: When false, validate pointer metadata without
            requiring adjacent sidecar files. This is used only for replaying a
            historical flat run spec into a fresh output/spec directory.

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
    training_modes = str(training_mode).split("+") if training_mode is not None else []
    if not training_modes or any(mode not in NOMINAL_GRU_TRAINING_MODES for mode in training_modes):
        raise RunSpecValidationError(
            "nominal GRU run spec must declare training_summary.training_mode as one of "
            f"{sorted(NOMINAL_GRU_TRAINING_MODES)} or a '+'-joined composite; "
            f"found {training_mode!r}"
        )

    loss_objective = run_spec.get("loss_objective")
    if loss_objective not in NOMINAL_GRU_LOSS_OBJECTIVES:
        raise RunSpecValidationError(
            "nominal GRU run spec must declare loss_objective as one of "
            f"{sorted(NOMINAL_GRU_LOSS_OBJECTIVES)}; found {loss_objective!r}"
        )
    loss_summary = _mapping(run_spec, "loss_summary")
    loss_profile = loss_summary.get("objective_profile")
    if loss_profile != loss_objective:
        raise RunSpecValidationError(
            "nominal GRU run spec loss_summary.objective_profile must match "
            f"loss_objective; found {loss_profile!r} vs {loss_objective!r}"
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

    graph_spec_sidecar: Path | None = None
    for key in sorted(FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS):
        pointer = graph_metadata[key]
        if pointer is None and key == "graph_spec_path":
            status = graph_metadata.get("graph_export_status")
            if status == "unavailable":
                continue
            raise RunSpecValidationError(
                "nominal GRU run spec has no Feedbax graph sidecar but does not "
                "declare feedbax_graph.graph_export_status='unavailable'"
            )
        sidecar = spec_dir / str(pointer)
        if not require_graph_sidecars:
            continue
        if not sidecar.is_file():
            raise RunSpecValidationError(
                f"nominal GRU run spec points to missing Feedbax graph sidecar: {sidecar}"
            )
        if key == "graph_spec_path":
            graph_spec_sidecar = sidecar

    if graph_spec_sidecar is not None and _is_cs_lss_run_spec(run_spec):
        _validate_cs_lss_graph_spec_sidecar(graph_spec_sidecar)


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


def _is_cs_lss_run_spec(run_spec: dict[str, Any]) -> bool:
    for path in (
        ("model_summary", "plant_backend"),
        ("training_summary", "plant_backend"),
        ("fidelity_status", "plant_backend"),
        ("hps", "model", "plant_backend"),
    ):
        value = _nested_get(run_spec, path)
        if value == CS_LSS_PLANT_BACKEND:
            return True
    return _nested_get(run_spec, ("model_summary", "exact_cs_linear_state_space")) is True


def _nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _validate_cs_lss_graph_spec_sidecar(graph_spec_path: Path) -> None:
    try:
        payload = json.loads(graph_spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunSpecValidationError(
            f"CS-LSS Feedbax graph sidecar is not valid JSON: {graph_spec_path}"
        ) from exc

    nodes = payload.get("nodes")
    if not isinstance(nodes, dict):
        raise RunSpecValidationError(
            f"CS-LSS Feedbax graph sidecar must contain an object 'nodes': {graph_spec_path}"
        )

    node_types = {
        str(node_id): node.get("type") for node_id, node in nodes.items() if isinstance(node, dict)
    }
    mechanics_type = node_types.get("mechanics")
    feedback_type = node_types.get("feedback")
    legacy_types = sorted(
        node_type
        for node_type in set(node_types.values())
        if node_type in LEGACY_POINT_MASS_GRAPH_TYPES
    )
    if mechanics_type != CS_LSS_REQUIRED_MECHANICS_TYPE:
        legacy_note = (
            f"; stale legacy graph types present: {', '.join(legacy_types)}" if legacy_types else ""
        )
        raise RunSpecValidationError(
            "CS-LSS Feedbax graph sidecar must declare mechanics node type "
            f"{CS_LSS_REQUIRED_MECHANICS_TYPE!r}; found {mechanics_type!r}{legacy_note}"
        )
    if feedback_type not in CS_LSS_FEEDBACK_COMPONENT_TYPES:
        raise RunSpecValidationError(
            "CS-LSS Feedbax graph sidecar must declare a delayed position/velocity "
            f"feedback selector; found feedback node type {feedback_type!r}"
        )


def _missing_keys(mapping: dict[str, Any], required_keys: frozenset[str]) -> list[str]:
    return sorted(key for key in required_keys if key not in mapping)


__all__ = [
    "CS_LSS_FEEDBACK_COMPONENT_TYPES",
    "CS_LSS_PLANT_BACKEND",
    "CS_LSS_REQUIRED_MECHANICS_TYPE",
    "FEEDBAX_GRAPH_REQUIRED_POINTER_KEYS",
    "LEGACY_POINT_MASS_GRAPH_TYPES",
    "NOMINAL_GRU_LOSS_OBJECTIVES",
    "NOMINAL_GRU_REQUIRED_PROVENANCE_KEYS",
    "NOMINAL_GRU_REQUIRED_TOP_LEVEL_KEYS",
    "NOMINAL_GRU_TRAINING_MODES",
    "RunSpecValidationError",
    "validate_nominal_gru_run_spec",
    "validate_nominal_gru_run_spec_file",
]
