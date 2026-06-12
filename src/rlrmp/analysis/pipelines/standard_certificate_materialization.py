"""Reusable bridge standard-certificate materialization helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlrmp.analysis.pipelines.bridge_certificates import build_standard_certificate_components
from rlrmp.analysis.pipelines.bridge_contracts import (
    BridgeArchitecture,
    BridgeCertificateComponent,
    BridgeCertificateMode,
    BridgeRunManifest,
    BridgeRunSpec,
)


@dataclass(frozen=True)
class StandardCertificateRowRequest:
    """In-memory request for one standard-certificate row.

    ``component_kwargs`` is deliberately not part of the serialized manifest.
    It contains arrays and metadata passed through to
    :func:`build_standard_certificate_components`, which remains the umbrella
    standard-certificate entry point for all modes.
    """

    spec: BridgeRunSpec
    architecture: BridgeArchitecture
    status: str
    certificate_mode: BridgeCertificateMode | None = None
    component_kwargs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def build_standard_certificate_manifest(
    request: StandardCertificateRowRequest,
) -> BridgeRunManifest:
    """Build one serialized manifest row from a standard-certificate request."""

    components = build_standard_certificate_components(
        architecture=request.architecture,
        certificate_mode=request.certificate_mode,
        **request.component_kwargs,
    )
    return BridgeRunManifest(
        spec=request.spec,
        status=request.status,
        metrics=request.metrics,
        artifacts=request.artifacts,
        certificate_components=components,
    )


def component_by_name(
    row: BridgeRunManifest | dict[str, Any],
) -> dict[str, BridgeCertificateComponent | dict[str, Any]]:
    """Return certificate components keyed by component name."""

    components: Any
    if isinstance(row, BridgeRunManifest):
        components = row.certificate_components
    else:
        components = row.get("certificate_components", ())
    return {
        component.name
        if isinstance(component, BridgeCertificateComponent)
        else component["name"]: component
        for component in components
    }


def component_status_counts(rows: list[BridgeRunManifest] | list[dict[str, Any]]) -> dict[str, int]:
    """Return stable ``component:status`` counts for materialized rows."""

    counts: Counter[str] = Counter()
    for row in rows:
        components = (
            row.certificate_components
            if isinstance(row, BridgeRunManifest)
            else row.get("certificate_components", ())
        )
        for component in components:
            if isinstance(component, BridgeCertificateComponent):
                name = component.name
                status = component.status
            else:
                name = component["name"]
                status = component["status"]
            counts[f"{name}:{status}"] += 1
    return dict(sorted(counts.items()))


def materialization_summary(rows: list[BridgeRunManifest]) -> dict[str, Any]:
    """Return a compact JSON summary for a standard-certificate result."""

    status_counts = Counter(row.status for row in rows)
    return {
        "n_rows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "component_status_counts": component_status_counts(rows),
    }


def repo_relative(path: Path, *, repo_root: Path) -> str:
    """Return a repo-relative path when possible."""

    try:
        return str(path.absolute().relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "StandardCertificateRowRequest",
    "build_standard_certificate_manifest",
    "component_by_name",
    "component_status_counts",
    "materialization_summary",
    "repo_relative",
]
