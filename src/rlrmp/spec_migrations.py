"""RLRMP schema identities layered onto Feedbax structured-spec policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from feedbax.migrations import (
    SpecFamilyMigrationPolicy,
    SpecMigrationResult,
    SpecSchemaFamily,
    SpecSchemaRegistry,
    UnknownSpecFamily,
    UnsupportedSpecVersion,
    default_spec_registry,
    migrate_structured_spec_payload,
)


DIAGNOSTIC_REGENERATION_SPEC_KIND = "RLRMPDiagnosticRegenerationSpec"
DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_ID = "rlrmp.diagnostic_regeneration_spec"
DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_VERSION = "rlrmp.diagnostic_regeneration_spec.v1"

GRU_EVALUATION_DIAGNOSTICS_KIND = "RLRMPGRUEvaluationDiagnosticsManifest"
GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID = "rlrmp.gru_evaluation_diagnostics"
GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION = "rlrmp.gru_evaluation_diagnostics.v1"

CS_GRU_STANDARD_CERTIFICATES_KIND = "RLRMPCSGrUStandardCertificateManifest"
CS_GRU_STANDARD_CERTIFICATES_SCHEMA_ID = "rlrmp.cs_gru_standard_certificates"
CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION = "rlrmp.cs_gru_standard_certificates.v1"

RUN_SPEC_KIND = "RLRMPRunSpec"
RUN_SPEC_SCHEMA_ID = "rlrmp.run_spec"
RUN_SPEC_SCHEMA_VERSION = "rlrmp.run_spec.v1"

LEGACY_TRAINING_CONFIG_KIND = "RLRMPLegacyTrainingConfig"
LEGACY_TRAINING_CONFIG_SCHEMA_ID = "rlrmp.legacy_training_config"
LEGACY_TRAINING_CONFIG_SCHEMA_VERSION = "rlrmp.legacy_training_config.archive.v1"


class ArchiveOnlySpecError(ValueError):
    """Raised when a historical JSON artifact is intentionally archive-only."""


def ensure_rlrmp_spec_families(
    registry: SpecSchemaRegistry | None = None,
) -> SpecSchemaRegistry:
    """Register RLRMP-owned durable spec families in a Feedbax registry."""

    active_registry = registry or default_spec_registry
    for family in _rlrmp_spec_families():
        try:
            active_registry.resolve(family.kind)
        except UnknownSpecFamily:
            active_registry.register_family(family)
            if family.policy is not None:
                for old_version in family.policy.rejected_old_versions:
                    active_registry.reject_version(
                        family.kind,
                        old_version,
                        reason=(
                            f"{family.kind} has no deterministic migration from "
                            f"{old_version!r}; {family.policy.owner_module} owns this "
                            "schema and must regenerate the current version or add an "
                            "explicit migration."
                        ),
                    )
    return active_registry


def stamp_current_schema(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``payload`` stamped with the registered current schema identity."""

    registry = ensure_rlrmp_spec_families()
    family = registry.resolve(kind)
    stamped = dict(payload)
    stamped.setdefault("schema_id", family.identity)
    stamped.setdefault("schema_version", family.current_version)
    return stamped


def accept_rlrmp_spec_payload(
    kind: str,
    payload: Mapping[str, Any],
    *,
    source_version: str | None = None,
    registry: SpecSchemaRegistry | None = None,
    path: str = "spec",
) -> SpecMigrationResult:
    """Accept, migrate, or explicitly reject an RLRMP-owned spec payload."""

    active_registry = ensure_rlrmp_spec_families(registry)
    family = active_registry.resolve(kind)
    schema_id = payload.get("schema_id")
    if schema_id is not None and schema_id != family.identity:
        raise UnsupportedSpecVersion(
            "Unsupported RLRMP structured spec schema identity: "
            f"path={path!r}, family={kind!r}, schema_id={schema_id!r}, "
            f"expected={family.identity!r}"
        )
    return migrate_structured_spec_payload(
        kind,
        payload,
        source_version=source_version,
        registry=active_registry,
        path=path,
    )


def load_rlrmp_spec_payload(
    kind: str,
    path: Path | str,
    *,
    registry: SpecSchemaRegistry | None = None,
) -> SpecMigrationResult:
    """Load a JSON spec payload and run it through the registered RLRMP policy."""

    payload_path = Path(path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if kind == LEGACY_TRAINING_CONFIG_KIND:
        raise ArchiveOnlySpecError(
            "Archive-only RLRMP legacy training config cannot be migrated through "
            "Feedbax structured-spec policy: "
            f"path={payload_path}, family={kind!r}, "
            f"schema_id={LEGACY_TRAINING_CONFIG_SCHEMA_ID!r}, "
            "reason=2ef67ca-era config.json files predate durable run-spec "
            "provenance and are retained only as historical training archives."
        )
    if not isinstance(payload, Mapping):
        raise TypeError(f"RLRMP spec payload must be a JSON object: {payload_path}")
    return accept_rlrmp_spec_payload(
        kind,
        payload,
        registry=registry,
        path=str(payload_path),
    )


def _rlrmp_spec_families() -> tuple[SpecSchemaFamily, ...]:
    return (
        _family(
            DIAGNOSTIC_REGENERATION_SPEC_KIND,
            DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_ID,
            DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.diagnostic_provenance",),
            consumed_by=("rlrmp analysis re-run tooling",),
            description="RLRMP-local recipe for regenerating diagnostic artifacts.",
            rejected_old_versions=("rlrmp.diagnostic_regeneration_spec.v0",),
        ),
        _family(
            GRU_EVALUATION_DIAGNOSTICS_KIND,
            GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID,
            GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.gru_evaluation_diagnostics",),
            consumed_by=("Feedbax AnalysisRunManifest artifacts", "rlrmp post-hoc reports"),
            description="RLRMP GRU rollout-diagnostics manifest payload.",
            rejected_old_versions=("rlrmp.gru_evaluation_diagnostics.v0",),
        ),
        _family(
            CS_GRU_STANDARD_CERTIFICATES_KIND,
            CS_GRU_STANDARD_CERTIFICATES_SCHEMA_ID,
            CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.cs_gru_standard_materialization",),
            consumed_by=("Feedbax AnalysisRunManifest artifacts", "standard certificate reports"),
            description="RLRMP C&S GRU standard-certificate manifest payload.",
            rejected_old_versions=("rlrmp.cs_gru_standard_certificates.v0",),
        ),
        _family(
            RUN_SPEC_KIND,
            RUN_SPEC_SCHEMA_ID,
            RUN_SPEC_SCHEMA_VERSION,
            emitted_by=("rlrmp post-run tracked specs",),
            consumed_by=("rlrmp.run_specs", "Feedbax TrainingRunManifest.training_spec"),
            description="Tracked RLRMP run recipe under results/<issue>/runs.",
            rejected_old_versions=("rlrmp.run_spec.v0",),
        ),
    )


def _family(
    kind: str,
    schema_id: str,
    current_version: str,
    *,
    emitted_by: tuple[str, ...],
    consumed_by: tuple[str, ...],
    description: str,
    rejected_old_versions: tuple[str, ...],
) -> SpecSchemaFamily:
    return SpecSchemaFamily(
        kind=kind,
        schema_id=schema_id,
        current_version=current_version,
        description=description,
        policy=SpecFamilyMigrationPolicy(
            owner_module="rlrmp.spec_migrations",
            emitted_by=emitted_by,
            consumed_by=consumed_by,
            stance="reject",
            rejected_old_versions=rejected_old_versions,
            required_tests=("tests/test_rlrmp_spec_migrations.py",),
            notes="RLRMP-owned payloads are current-version accepted; old durable versions must "
            "either be regenerated or gain explicit migrations.",
        ),
    )


__all__ = [
    "ArchiveOnlySpecError",
    "CS_GRU_STANDARD_CERTIFICATES_KIND",
    "CS_GRU_STANDARD_CERTIFICATES_SCHEMA_ID",
    "CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION",
    "DIAGNOSTIC_REGENERATION_SPEC_KIND",
    "DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_ID",
    "DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_VERSION",
    "GRU_EVALUATION_DIAGNOSTICS_KIND",
    "GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID",
    "GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION",
    "LEGACY_TRAINING_CONFIG_KIND",
    "RUN_SPEC_KIND",
    "RUN_SPEC_SCHEMA_ID",
    "RUN_SPEC_SCHEMA_VERSION",
    "accept_rlrmp_spec_payload",
    "ensure_rlrmp_spec_families",
    "load_rlrmp_spec_payload",
    "stamp_current_schema",
]
