"""In-memory aggregation helpers for structured bridge analysis results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rlrmp.analysis.bridge_results import (
    BridgeAnalysisResult,
    BridgeCertificateComponent,
)

BRIDGE_SUMMARY_FORMAT = "rlrmp.bridge_summary.v1"
BRIDGE_ROW_BASE_COLUMNS = (
    "issue_id",
    "run_id",
    "status",
    "objective",
    "architecture",
    "controller_label",
    "optimizer_label",
    "training_distribution",
    "evaluation_lane",
    "reference_controller",
    "seed",
    "gamma_factor",
)
BRIDGE_COMPONENT_STATUSES = {"available", "not_applicable", "missing"}


class BridgeResultValidationError(ValueError):
    """Raised when a bridge result is missing required aggregation fields."""


def validate_bridge_result(
    result: BridgeAnalysisResult,
    *,
    required_artifact_labels: Iterable[str] = (),
    required_certificate_labels: Iterable[str] = (),
    allow_missing_certificate_components: bool = False,
) -> None:
    """Validate aggregation-relevant artifact and certificate labels.

    Args:
        result: Structured bridge analysis result to validate.
        required_artifact_labels: Artifact dictionary keys that must be present
            with nonempty string values.
        required_certificate_labels: Certificate component names that must be
            present.
        allow_missing_certificate_components: If false, required certificate
            components with status ``"missing"`` are rejected. Components marked
            ``"not_applicable"`` are accepted because they are explicit
            unsupported/component-not-needed declarations.

    Raises:
        BridgeResultValidationError: If aggregation cannot safely compare the
            result against the requested labels.
    """

    errors: list[str] = []
    for label in required_artifact_labels:
        artifact = result.artifacts.get(label)
        if not isinstance(artifact, str) or not artifact.strip():
            errors.append(f"missing required artifact label {label!r}")

    components = _certificate_components_by_name(result.certificate_components, errors)
    for label in required_certificate_labels:
        component = components.get(label)
        if component is None:
            errors.append(f"missing required certificate component {label!r}")
            continue
        if component.status == "missing" and not allow_missing_certificate_components:
            errors.append(f"required certificate component {label!r} has status 'missing'")

    if errors:
        raise BridgeResultValidationError(
            f"bridge result {result.spec.run_id!r} is invalid: " + "; ".join(errors)
        )


def bridge_result_row(
    result: BridgeAnalysisResult,
) -> dict[str, Any]:
    """Flatten a structured bridge result into one compact comparable row."""

    row: dict[str, Any] = {
        "issue_id": result.spec.issue_id,
        "run_id": result.spec.run_id,
        "status": result.status,
        "objective": result.spec.objective,
        "architecture": result.spec.architecture,
        "controller_label": result.spec.controller_label,
        "optimizer_label": result.spec.optimizer_label,
        "training_distribution": result.spec.training_distribution,
        "evaluation_lane": result.spec.evaluation_lane,
        "reference_controller": result.spec.reference_controller,
        "seed": result.spec.seed,
        "gamma_factor": result.spec.gamma_factor,
    }
    _flatten_mapping(row, "parameter", result.spec.parameters)
    _flatten_mapping(row, "metric", result.metrics)
    _flatten_mapping(row, "artifact", result.artifacts)
    for component in sorted(result.certificate_components, key=lambda value: value.name):
        prefix = f"certificate.{component.name}"
        _set_row_value(row, f"{prefix}.status", component.status)
        if component.reason:
            _set_row_value(row, f"{prefix}.reason", component.reason)
        _flatten_mapping(row, f"{prefix}.summary", component.summary)
    return row


def summarize_bridge_results(
    results: Iterable[BridgeAnalysisResult],
    *,
    required_artifact_labels: Iterable[str] = (),
    required_certificate_labels: Iterable[str] = (),
    allow_missing_certificate_components: bool = False,
) -> dict[str, Any]:
    """Validate and flatten structured bridge results into a summary payload."""

    required_artifacts = tuple(required_artifact_labels)
    required_certificates = tuple(required_certificate_labels)
    rows: list[dict[str, Any]] = []
    for result in results:
        validate_bridge_result(
            result,
            required_artifact_labels=required_artifacts,
            required_certificate_labels=required_certificates,
            allow_missing_certificate_components=allow_missing_certificate_components,
        )
        rows.append(bridge_result_row(result))
    return {
        "format": BRIDGE_SUMMARY_FORMAT,
        "required_artifact_labels": list(required_artifacts),
        "required_certificate_labels": list(required_certificates),
        "allow_missing_certificate_components": allow_missing_certificate_components,
        "rows": rows,
    }


def _certificate_components_by_name(
    components: Iterable[BridgeCertificateComponent],
    errors: list[str],
) -> dict[str, BridgeCertificateComponent]:
    by_name: dict[str, BridgeCertificateComponent] = {}
    for component in components:
        if component.name in by_name:
            errors.append(f"duplicate certificate component {component.name!r}")
        else:
            by_name[component.name] = component
        if component.status not in BRIDGE_COMPONENT_STATUSES:
            errors.append(
                f"certificate component {component.name!r} has unsupported status "
                f"{component.status!r}"
            )
    return by_name


def _flatten_mapping(row: dict[str, Any], prefix: str, values: Mapping[str, Any]) -> None:
    for key in sorted(values):
        value = values[key]
        row_key = f"{prefix}.{key}"
        if isinstance(value, Mapping):
            _flatten_mapping(row, row_key, value)
        else:
            _set_row_value(row, row_key, value)


def _set_row_value(row: dict[str, Any], key: str, value: Any) -> None:
    if key in row:
        raise BridgeResultValidationError(f"duplicate bridge row label {key!r}")
    row[key] = value


__all__ = [
    "BRIDGE_ROW_BASE_COLUMNS",
    "BRIDGE_SUMMARY_FORMAT",
    "BridgeResultValidationError",
    "bridge_result_row",
    "summarize_bridge_results",
    "validate_bridge_result",
]
