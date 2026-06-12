"""Aggregation helpers for analytical bridge run manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rlrmp.analysis.pipelines.bridge_contracts import (
    BridgeCertificateComponent,
    BridgeRunManifest,
    read_bridge_manifest,
)

BRIDGE_SUMMARY_FORMAT = "rlrmp.bridge_summary.v1"
BRIDGE_ROW_BASE_COLUMNS = (
    "source_path",
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
BRIDGE_MARKDOWN_BASE_COLUMNS = (
    "run_id",
    "status",
    "objective",
    "architecture",
    "controller_label",
    "optimizer_label",
    "training_distribution",
    "evaluation_lane",
)
BRIDGE_COMPONENT_STATUSES = {"available", "not_applicable", "missing"}


class BridgeManifestValidationError(ValueError):
    """Raised when a bridge manifest is missing required aggregation fields."""


def read_bridge_manifests(paths: Iterable[Path]) -> tuple[BridgeRunManifest, ...]:
    """Read one or more bridge manifest JSON files."""

    return tuple(read_bridge_manifest(Path(path)) for path in paths)


def validate_bridge_manifest(
    manifest: BridgeRunManifest,
    *,
    required_artifact_labels: Iterable[str] = (),
    required_certificate_labels: Iterable[str] = (),
    allow_missing_certificate_components: bool = False,
) -> None:
    """Validate aggregation-relevant artifact and certificate labels.

    Args:
        manifest: Manifest to validate.
        required_artifact_labels: Artifact dictionary keys that must be present
            with nonempty string values.
        required_certificate_labels: Certificate component names that must be
            present.
        allow_missing_certificate_components: If false, required certificate
            components with status ``"missing"`` are rejected. Components marked
            ``"not_applicable"`` are accepted because they are explicit
            unsupported/component-not-needed declarations.

    Raises:
        BridgeManifestValidationError: If aggregation cannot safely compare the
            manifest against the requested labels.
    """

    errors: list[str] = []
    for label in required_artifact_labels:
        artifact = manifest.artifacts.get(label)
        if not isinstance(artifact, str) or not artifact.strip():
            errors.append(f"missing required artifact label {label!r}")

    components = _certificate_components_by_name(manifest.certificate_components, errors)
    for label in required_certificate_labels:
        component = components.get(label)
        if component is None:
            errors.append(f"missing required certificate component {label!r}")
            continue
        if component.status == "missing" and not allow_missing_certificate_components:
            errors.append(f"required certificate component {label!r} has status 'missing'")

    if errors:
        raise BridgeManifestValidationError(
            f"bridge manifest {manifest.spec.run_id!r} is invalid: " + "; ".join(errors)
        )


def bridge_manifest_row(
    manifest: BridgeRunManifest,
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Flatten a bridge manifest into one compact comparable row."""

    row: dict[str, Any] = {
        "source_path": "" if source_path is None else str(source_path),
        "issue_id": manifest.spec.issue_id,
        "run_id": manifest.spec.run_id,
        "status": manifest.status,
        "objective": manifest.spec.objective,
        "architecture": manifest.spec.architecture,
        "controller_label": manifest.spec.controller_label,
        "optimizer_label": manifest.spec.optimizer_label,
        "training_distribution": manifest.spec.training_distribution,
        "evaluation_lane": manifest.spec.evaluation_lane,
        "reference_controller": manifest.spec.reference_controller,
        "seed": manifest.spec.seed,
        "gamma_factor": manifest.spec.gamma_factor,
    }
    _flatten_mapping(row, "parameter", manifest.spec.parameters)
    _flatten_mapping(row, "metric", manifest.metrics)
    _flatten_mapping(row, "artifact", manifest.artifacts)
    for component in sorted(manifest.certificate_components, key=lambda value: value.name):
        prefix = f"certificate.{component.name}"
        _set_row_value(row, f"{prefix}.status", component.status)
        if component.reason:
            _set_row_value(row, f"{prefix}.reason", component.reason)
        _flatten_mapping(row, f"{prefix}.summary", component.summary)
    return row


def summarize_bridge_manifests(
    manifest_paths: Iterable[Path],
    *,
    required_artifact_labels: Iterable[str] = (),
    required_certificate_labels: Iterable[str] = (),
    allow_missing_certificate_components: bool = False,
) -> dict[str, Any]:
    """Read, validate, and flatten bridge manifests into a summary object."""

    paths = tuple(Path(path) for path in manifest_paths)
    required_artifacts = tuple(required_artifact_labels)
    required_certificates = tuple(required_certificate_labels)
    rows: list[dict[str, Any]] = []
    for path in paths:
        manifest = read_bridge_manifest(path)
        validate_bridge_manifest(
            manifest,
            required_artifact_labels=required_artifacts,
            required_certificate_labels=required_certificates,
            allow_missing_certificate_components=allow_missing_certificate_components,
        )
        rows.append(bridge_manifest_row(manifest, source_path=path))
    return {
        "format": BRIDGE_SUMMARY_FORMAT,
        "required_artifact_labels": list(required_artifacts),
        "required_certificate_labels": list(required_certificates),
        "allow_missing_certificate_components": allow_missing_certificate_components,
        "rows": rows,
    }


def write_bridge_summary(summary: Mapping[str, Any], path: Path) -> None:
    """Write a bridge summary JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_bridge_summary(path: Path) -> dict[str, Any]:
    """Read a bridge summary JSON document written by :func:`write_bridge_summary`."""

    return json.loads(path.read_text(encoding="utf-8"))


def render_bridge_summary_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
) -> str:
    """Render flattened bridge rows as a compact Markdown table."""

    if not rows:
        return "No bridge manifests.\n"

    table_columns = tuple(columns) if columns is not None else bridge_markdown_columns(rows)
    header = "| " + " | ".join(_escape_markdown_cell(column) for column in table_columns) + " |"
    separator = "| " + " | ".join("---" for _ in table_columns) + " |"
    body = [
        "| "
        + " | ".join(
            _escape_markdown_cell(_format_markdown_value(row.get(column)))
            for column in table_columns
        )
        + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body)) + "\n"


def write_bridge_summary_markdown(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    columns: Sequence[str] | None = None,
) -> None:
    """Write a Markdown table for flattened bridge summary rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_bridge_summary_markdown(rows, columns=columns), encoding="utf-8")


def bridge_markdown_columns(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return stable default Markdown columns for flattened bridge rows."""

    row_keys = set().union(*(row.keys() for row in rows))
    metric_keys = sorted(key for key in row_keys if key.startswith("metric."))
    certificate_status_keys = sorted(
        key for key in row_keys if key.startswith("certificate.") and key.endswith(".status")
    )
    return (
        tuple(column for column in BRIDGE_MARKDOWN_BASE_COLUMNS if column in row_keys)
        + tuple(metric_keys)
        + tuple(certificate_status_keys)
    )


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
        raise BridgeManifestValidationError(f"duplicate bridge row label {key!r}")
    row[key] = value


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


__all__ = [
    "BRIDGE_MARKDOWN_BASE_COLUMNS",
    "BRIDGE_ROW_BASE_COLUMNS",
    "BRIDGE_SUMMARY_FORMAT",
    "BridgeManifestValidationError",
    "bridge_manifest_row",
    "bridge_markdown_columns",
    "read_bridge_manifests",
    "read_bridge_summary",
    "render_bridge_summary_markdown",
    "summarize_bridge_manifests",
    "validate_bridge_manifest",
    "write_bridge_summary",
    "write_bridge_summary_markdown",
]
