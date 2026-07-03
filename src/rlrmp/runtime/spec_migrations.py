"""RLRMP schema identities layered onto Feedbax structured-spec policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from feedbax.contracts.migrations import (
    SchemaMigration,
    SpecFamilyMigrationPolicy,
    SpecMigrationResult,
    SpecSchemaFamily,
    SpecSchemaRegistry,
    UnknownSpecFamily,
    UnsupportedSpecVersion,
    default_spec_registry,
    migrate_structured_spec_payload,
)


GRU_EVALUATION_DIAGNOSTICS_KIND = "RLRMPGRUEvaluationDiagnosticsManifest"
GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID = "rlrmp.gru_evaluation_diagnostics"
GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION = "rlrmp.gru_evaluation_diagnostics.v1"

CS_GRU_STANDARD_CERTIFICATES_KIND = "RLRMPCSGrUStandardCertificateManifest"
CS_GRU_STANDARD_CERTIFICATES_SCHEMA_ID = "rlrmp.cs_gru_standard_certificates"
CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION = "rlrmp.cs_gru_standard_certificates.v1"

OBJECTIVE_COMPARATOR_SIDECAR_KIND = "RLRMPObjectiveComparatorSidecar"
OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_ID = "rlrmp.objective_comparator_sidecar"
OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_VERSION = "rlrmp.objective_comparator_sidecar.v6"

GRU_PERTURBATION_BANK_KIND = "RLRMPGRUPerturbationBank"
GRU_PERTURBATION_BANK_SCHEMA_ID = "rlrmp.gru_perturbation_bank"
GRU_PERTURBATION_BANK_SCHEMA_VERSION = "rlrmp.gru_perturbation_bank.v3"

GRU_PERTURBATION_RESPONSE_NORM_PLOTS_KIND = "RLRMPGRUPerturbationResponseNormPlots"
GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_ID = "rlrmp.gru_perturbation_response_norm_plots"
GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_VERSION = (
    "rlrmp.gru_perturbation_response_norm_plots.v1"
)

PERTURBATION_OPEN_LOOP_CALIBRATION_KIND = "RLRMPPerturbationOpenLoopCalibration"
PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_ID = "rlrmp.perturbation_open_loop_calibration"
PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_VERSION = "rlrmp.perturbation_open_loop_calibration.v2"

HINF_PHENOTYPE_SIDECAR_KIND = "RLRMPHinfPhenotypeSidecar"
HINF_PHENOTYPE_SIDECAR_SCHEMA_ID = "rlrmp.hinf_phenotype_sidecar"
HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION = "rlrmp.hinf_phenotype_sidecar.v1"

GRU_WORST_CASE_EPSILON_AUDIT_KIND = "RLRMPGRUWorstCaseEpsilonAudit"
GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_ID = "rlrmp.gru_worst_case_epsilon_audit"
GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_VERSION = "rlrmp.gru_worst_case_epsilon_audit.v1"

GRU_BROAD_EPSILON_ATTRIBUTION_KIND = "RLRMPGRUBroadEpsilonAttribution"
GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_ID = "rlrmp.gru_broad_epsilon_attribution"
GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_VERSION = "rlrmp.gru_broad_epsilon_attribution.v1"

GRU_MAP_ERROR_DECOMPOSITION_KIND = "RLRMPGRUMapErrorDecomposition"
GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_ID = "rlrmp.gru_map_error_decomposition"
GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_VERSION = "rlrmp.gru_map_error_decomposition.v1"

GRU_FEEDBACK_ABLATION_KIND = "RLRMPGRUFeedbackAblation"
GRU_FEEDBACK_ABLATION_SCHEMA_ID = "rlrmp.gru_feedback_ablation"
GRU_FEEDBACK_ABLATION_SCHEMA_VERSION = "rlrmp.gru_feedback_ablation.v1"

DELAYED_DIAGNOSTIC_BUNDLE_KIND = "RLRMPDelayedDiagnosticBundle"
DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_ID = "rlrmp.delayed_diagnostic_bundle"
DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION = "rlrmp.delayed_diagnostic_bundle.v1"

FEEDBACK_QUALITY_LENS_KIND = "RLRMPFeedbackQualityLens"
FEEDBACK_QUALITY_LENS_SCHEMA_ID = "rlrmp.feedback_quality_lens"
FEEDBACK_QUALITY_LENS_SCHEMA_VERSION = "rlrmp.feedback_quality_lens.v1"

RUN_SPEC_KIND = "RLRMPRunSpec"
RUN_SPEC_SCHEMA_ID = "rlrmp.run_spec"
RUN_SPEC_SCHEMA_VERSION = "rlrmp.run_spec.v2"
RUN_SPEC_SCHEMA_VERSION_V1 = "rlrmp.run_spec.v1"
RUN_SPEC_SCHEMA_VERSION_LEGACY_CS_GRU = "rlrmp.cs_stochastic_gru.v1"

FINITE_ADVERSARY_POLICY_METADATA_KIND = "RLRMPFiniteAdversaryPolicyMetadata"
FINITE_ADVERSARY_POLICY_METADATA_SCHEMA_ID = "rlrmp.finite_adversary_policy_metadata"
FINITE_ADVERSARY_POLICY_METADATA_SCHEMA_VERSION = "rlrmp.finite_adversary_policy_metadata.v1"

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
            if family.kind == RUN_SPEC_KIND:
                for source_version, migration_id in (
                    (RUN_SPEC_SCHEMA_VERSION_V1, "rlrmp-run-spec-v1-to-v2"),
                    (
                        RUN_SPEC_SCHEMA_VERSION_LEGACY_CS_GRU,
                        "rlrmp-cs-stochastic-gru-v1-to-run-spec-v2",
                    ),
                ):
                    active_registry.register_migration(
                        RUN_SPEC_KIND,
                        SchemaMigration(
                            source_version=source_version,
                            target_version=RUN_SPEC_SCHEMA_VERSION,
                            migration_id=migration_id,
                            migrate=_migrate_run_spec_v1_to_v2,
                            description=(
                                "Carry active tracked RLRMP run-spec payloads into the "
                                "v2 extension family used beside Feedbax TrainingRunSpec."
                            ),
                        ),
                    )
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
            OBJECTIVE_COMPARATOR_SIDECAR_KIND,
            OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_ID,
            OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.objective_comparator",),
            consumed_by=(
                "rlrmp.analysis.pipelines.hinf_phenotype_sidecar",
                "rlrmp post-run diagnostic summaries",
            ),
            description="RLRMP full-QRF objective-comparator sidecar.",
            rejected_old_versions=tuple(
                f"rlrmp.objective_comparator_sidecar.v{version}" for version in range(0, 6)
            ),
        ),
        _family(
            GRU_PERTURBATION_BANK_KIND,
            GRU_PERTURBATION_BANK_SCHEMA_ID,
            GRU_PERTURBATION_BANK_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.gru_perturbation_bank",),
            consumed_by=(
                "rlrmp.analysis.pipelines.gru_perturbation_calibration",
                "rlrmp.analysis.pipelines.gru_feedback_ablation",
                "rlrmp perturbation-response diagnostics",
            ),
            description="Controller-independent C&S GRU perturbation bank and response manifest.",
            rejected_old_versions=(
                "rlrmp.gru_perturbation_bank.v0",
                "rlrmp.gru_perturbation_bank.v1",
                "rlrmp.gru_perturbation_bank.v2",
                "rlrmp.gru_perturbation_response.v0",
                "rlrmp.gru_perturbation_response.v1",
                "rlrmp.gru_perturbation_response.v2",
            ),
        ),
        _family(
            GRU_PERTURBATION_RESPONSE_NORM_PLOTS_KIND,
            GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_ID,
            GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.gru_perturbation_response_norm_plots",),
            consumed_by=("rlrmp perturbation-response figure notes",),
            description="RLRMP perturbation-response norm-plot manifest.",
            rejected_old_versions=("rlrmp.gru_perturbation_response_norm_plots.v0",),
        ),
        _family(
            PERTURBATION_OPEN_LOOP_CALIBRATION_KIND,
            PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_ID,
            PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.gru_perturbation_calibration",),
            consumed_by=(
                "rlrmp.analysis.pipelines.gru_perturbation_bank",
                "rlrmp perturbation calibration notes",
            ),
            description="Open-loop perturbation calibration manifest for C&S GRU banks.",
            rejected_old_versions=(
                "rlrmp.perturbation_open_loop_calibration.v0",
                "rlrmp.perturbation_open_loop_calibration.v1",
            ),
        ),
        _family(
            HINF_PHENOTYPE_SIDECAR_KIND,
            HINF_PHENOTYPE_SIDECAR_SCHEMA_ID,
            HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.hinf_phenotype_sidecar",),
            consumed_by=("RLRMP robustness phenotype reports",),
            description="Interpretive H-infinity phenotype sidecar aggregation.",
            rejected_old_versions=("rlrmp.hinf_phenotype_sidecar.v0",),
        ),
        _family(
            GRU_WORST_CASE_EPSILON_AUDIT_KIND,
            GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_ID,
            GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.gru_worst_case_epsilon_audit",),
            consumed_by=("RLRMP broad-epsilon robustness diagnostics",),
            description="Worst-case full-state epsilon audit manifest for frozen GRU rollouts.",
            rejected_old_versions=("rlrmp.gru_worst_case_epsilon_audit.v0",),
        ),
        _family(
            GRU_BROAD_EPSILON_ATTRIBUTION_KIND,
            GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_ID,
            GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.gru_broad_epsilon_attribution",),
            consumed_by=("RLRMP broad-epsilon attribution reports",),
            description="Paired active-vs-zero broad-epsilon attribution manifest.",
            rejected_old_versions=("rlrmp.gru_broad_epsilon_attribution.v0",),
        ),
        _family(
            GRU_MAP_ERROR_DECOMPOSITION_KIND,
            GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_ID,
            GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.gru_map_error_decomposition",),
            consumed_by=(
                "rlrmp.analysis.pipelines.hinf_phenotype_sidecar",
                "rlrmp post-run diagnostic summaries",
            ),
            description="GRU observation-history-to-action map-error decomposition sidecar.",
            rejected_old_versions=("rlrmp.gru_map_error_decomposition.v0",),
        ),
        _family(
            GRU_FEEDBACK_ABLATION_KIND,
            GRU_FEEDBACK_ABLATION_SCHEMA_ID,
            GRU_FEEDBACK_ABLATION_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.gru_feedback_ablation",),
            consumed_by=(
                "rlrmp.analysis.pipelines.hinf_phenotype_sidecar",
                "rlrmp post-run diagnostic summaries",
            ),
            description="Validation-selected C&S GRU feedback-ablation diagnostics.",
            rejected_old_versions=("rlrmp.gru_feedback_ablation.v0",),
            notes=(
                "RLRMP owns this scientific sidecar schema only; conversion to "
                "Feedbax bundle outputs remains issue af77a06."
            ),
        ),
        _family(
            DELAYED_DIAGNOSTIC_BUNDLE_KIND,
            DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_ID,
            DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.pipelines.delayed_diagnostic_bundle",),
            consumed_by=(
                "rlrmp delayed-reach diagnostic reports",
                "Feedbax AnalysisRunManifest artifacts",
            ),
            description=(
                "RLRMP delayed-reach direction-split and peak/support-decay "
                "diagnostic bundle."
            ),
            rejected_old_versions=(
                "rlrmp.delayed_no_pgd_direction_split_velocity.v1",
                "rlrmp.delayed_peak_decay_diagnostics.v1",
            ),
            notes=(
                "Historical delayed direction-split and peak-decay payloads were "
                "one-off script outputs. Regenerate through the current bundle "
                "instead of migrating them structurally."
            ),
        ),
        _family(
            FEEDBACK_QUALITY_LENS_KIND,
            FEEDBACK_QUALITY_LENS_SCHEMA_ID,
            FEEDBACK_QUALITY_LENS_SCHEMA_VERSION,
            emitted_by=("rlrmp.analysis.declarative_materialization",),
            consumed_by=("Feedbax AnalysisRunManifest artifacts", "rlrmp feedback-quality reports"),
            description=(
                "Feedback-control quality lens inventory over RLRMP-owned diagnostic sidecars."
            ),
            rejected_old_versions=("rlrmp.feedback_quality_lens.v0",),
            notes=(
                "This schema is only the RLRMP-owned lens/index payload. "
                "Generic output status, lineage, and artifact custody remain "
                "Feedbax-owned bundle/manifest semantics."
            ),
        ),
        _family(
            RUN_SPEC_KIND,
            RUN_SPEC_SCHEMA_ID,
            RUN_SPEC_SCHEMA_VERSION,
            emitted_by=("rlrmp post-run tracked specs",),
            consumed_by=("rlrmp.runtime.run_specs", "Feedbax TrainingRunManifest.training_spec"),
            description="Tracked RLRMP run recipe under results/<issue>/runs.",
            rejected_old_versions=("rlrmp.run_spec.v0",),
            supported_old_versions=(
                RUN_SPEC_SCHEMA_VERSION_V1,
                RUN_SPEC_SCHEMA_VERSION_LEGACY_CS_GRU,
            ),
            stance="migrate",
            notes=(
                "Active v1 tracked run specs are migrated by carrying their scientific "
                "payload into the v2 RLRMPRunSpec extension beside a composed Feedbax "
                "TrainingRunSpec. Archive-only pre-run-spec config.json files remain "
                "outside this family."
            ),
        ),
        _family(
            FINITE_ADVERSARY_POLICY_METADATA_KIND,
            FINITE_ADVERSARY_POLICY_METADATA_SCHEMA_ID,
            FINITE_ADVERSARY_POLICY_METADATA_SCHEMA_VERSION,
            emitted_by=("rlrmp.train.closed_loop_finite_adversary",),
            consumed_by=(
                "rlrmp TrainingRunSpec method payloads",
                "rlrmp finite-policy run specs and audits",
            ),
            description="Governed finite closed-loop adversary policy metadata payload.",
            rejected_old_versions=("rlrmp.finite_adversary_policy_metadata.v0",),
            notes=(
                "FiniteAdversaryPolicyMetadata is carried through the governed "
                "RLRMP payload envelope; standalone ad-hoc metadata serializers are retired."
            ),
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
    supported_old_versions: tuple[str, ...] = (),
    stance: str = "reject",
    notes: str | None = None,
) -> SpecSchemaFamily:
    policy_notes = (
        notes
        or "RLRMP-owned payloads are current-version accepted; old durable versions must "
        "either be regenerated or gain explicit migrations."
    )
    return SpecSchemaFamily(
        kind=kind,
        schema_id=schema_id,
        current_version=current_version,
        description=description,
        policy=SpecFamilyMigrationPolicy(
            owner_module="rlrmp.runtime.spec_migrations",
            emitted_by=emitted_by,
            consumed_by=consumed_by,
            stance=stance,
            supported_old_versions=supported_old_versions,
            rejected_old_versions=rejected_old_versions,
            required_tests=("tests/test_rlrmp_spec_migrations.py",),
            notes=policy_notes,
        ),
    )


def _migrate_run_spec_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(payload)
    migrated.setdefault("schema_id", RUN_SPEC_SCHEMA_ID)
    migrated["schema_version"] = RUN_SPEC_SCHEMA_VERSION
    migrated.setdefault("migration_policy", "migrated_active_v1_to_v2")
    return migrated


__all__ = [
    "ArchiveOnlySpecError",
    "CS_GRU_STANDARD_CERTIFICATES_KIND",
    "CS_GRU_STANDARD_CERTIFICATES_SCHEMA_ID",
    "CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION",
    "DELAYED_DIAGNOSTIC_BUNDLE_KIND",
    "DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_ID",
    "DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION",
    "FEEDBACK_QUALITY_LENS_KIND",
    "FEEDBACK_QUALITY_LENS_SCHEMA_ID",
    "FEEDBACK_QUALITY_LENS_SCHEMA_VERSION",
    "FINITE_ADVERSARY_POLICY_METADATA_KIND",
    "FINITE_ADVERSARY_POLICY_METADATA_SCHEMA_ID",
    "FINITE_ADVERSARY_POLICY_METADATA_SCHEMA_VERSION",
    "GRU_EVALUATION_DIAGNOSTICS_KIND",
    "GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID",
    "GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION",
    "GRU_BROAD_EPSILON_ATTRIBUTION_KIND",
    "GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_ID",
    "GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_VERSION",
    "GRU_FEEDBACK_ABLATION_KIND",
    "GRU_FEEDBACK_ABLATION_SCHEMA_ID",
    "GRU_FEEDBACK_ABLATION_SCHEMA_VERSION",
    "GRU_MAP_ERROR_DECOMPOSITION_KIND",
    "GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_ID",
    "GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_VERSION",
    "GRU_PERTURBATION_BANK_KIND",
    "GRU_PERTURBATION_BANK_SCHEMA_ID",
    "GRU_PERTURBATION_BANK_SCHEMA_VERSION",
    "GRU_PERTURBATION_RESPONSE_NORM_PLOTS_KIND",
    "GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_ID",
    "GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_VERSION",
    "GRU_WORST_CASE_EPSILON_AUDIT_KIND",
    "GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_ID",
    "GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_VERSION",
    "HINF_PHENOTYPE_SIDECAR_KIND",
    "HINF_PHENOTYPE_SIDECAR_SCHEMA_ID",
    "HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION",
    "LEGACY_TRAINING_CONFIG_KIND",
    "OBJECTIVE_COMPARATOR_SIDECAR_KIND",
    "OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_ID",
    "OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_VERSION",
    "PERTURBATION_OPEN_LOOP_CALIBRATION_KIND",
    "PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_ID",
    "PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_VERSION",
    "RUN_SPEC_KIND",
    "RUN_SPEC_SCHEMA_ID",
    "RUN_SPEC_SCHEMA_VERSION",
    "RUN_SPEC_SCHEMA_VERSION_LEGACY_CS_GRU",
    "RUN_SPEC_SCHEMA_VERSION_V1",
    "accept_rlrmp_spec_payload",
    "ensure_rlrmp_spec_families",
    "load_rlrmp_spec_payload",
    "stamp_current_schema",
]
