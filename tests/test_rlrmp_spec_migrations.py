from __future__ import annotations

import json
from pathlib import Path

import pytest
from feedbax.manifest import TrainingRunManifest, load_manifest
from feedbax.migrations import SpecSchemaRegistry, UnsupportedSpecVersion

from rlrmp.spec_migrations import (
    ArchiveOnlySpecError,
    CS_GRU_STANDARD_CERTIFICATES_KIND,
    CS_GRU_STANDARD_CERTIFICATES_SCHEMA_ID,
    CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION,
    DIAGNOSTIC_REGENERATION_SPEC_KIND,
    DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_ID,
    DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_VERSION,
    GRU_EVALUATION_DIAGNOSTICS_KIND,
    GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID,
    GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION,
    LEGACY_TRAINING_CONFIG_KIND,
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    accept_rlrmp_spec_payload,
    ensure_rlrmp_spec_families,
    load_rlrmp_spec_payload,
    stamp_current_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rlrmp_spec_policy_registers_current_families_and_rejects_v0() -> None:
    registry = ensure_rlrmp_spec_families(SpecSchemaRegistry())

    expected = {
        DIAGNOSTIC_REGENERATION_SPEC_KIND: (
            DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_ID,
            DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_VERSION,
            "rlrmp.diagnostic_regeneration_spec.v0",
        ),
        GRU_EVALUATION_DIAGNOSTICS_KIND: (
            GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID,
            GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION,
            "rlrmp.gru_evaluation_diagnostics.v0",
        ),
        CS_GRU_STANDARD_CERTIFICATES_KIND: (
            CS_GRU_STANDARD_CERTIFICATES_SCHEMA_ID,
            CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION,
            "rlrmp.cs_gru_standard_certificates.v0",
        ),
        RUN_SPEC_KIND: (
            RUN_SPEC_SCHEMA_ID,
            RUN_SPEC_SCHEMA_VERSION,
            "rlrmp.run_spec.v0",
        ),
    }

    for kind, (schema_id, current_version, old_version) in expected.items():
        family = registry.resolve(kind)
        assert family.identity == schema_id
        assert family.current_version == current_version
        assert family.policy is not None
        assert family.policy.owner_module == "rlrmp.spec_migrations"

        result = registry.migrate(kind, {"schema_version": current_version})
        assert result.target_version == current_version
        assert not result.migrated

        with pytest.raises(UnsupportedSpecVersion) as excinfo:
            registry.migrate(kind, {"schema_version": old_version})
        message = str(excinfo.value)
        assert f"family={kind!r}" in message
        assert f"schema_id={schema_id!r}" in message
        assert "migration_intentionally_absent=yes" in message


def test_stamp_current_schema_adds_identity_and_version() -> None:
    payload = stamp_current_schema(
        GRU_EVALUATION_DIAGNOSTICS_KIND,
        {"issue": "unit", "runs": {}},
    )

    assert payload["schema_id"] == GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID
    assert payload["schema_version"] == GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION
    assert payload["issue"] == "unit"


def test_rlrmp_payload_acceptance_rejects_wrong_schema_identity() -> None:
    with pytest.raises(UnsupportedSpecVersion, match="schema identity"):
        accept_rlrmp_spec_payload(
            GRU_EVALUATION_DIAGNOSTICS_KIND,
            {
                "schema_id": "rlrmp.other",
                "schema_version": GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION,
            },
        )


def test_representative_historical_artifacts_load_or_reject_by_policy() -> None:
    run_spec = load_rlrmp_spec_payload(
        RUN_SPEC_KIND,
        REPO_ROOT / "results" / "9455785" / "runs" / "artifact_normalization_fixture__sample.json",
    )
    assert run_spec.schema_id == RUN_SPEC_SCHEMA_ID
    assert run_spec.target_version == RUN_SPEC_SCHEMA_VERSION

    regeneration_spec = load_rlrmp_spec_payload(
        DIAGNOSTIC_REGENERATION_SPEC_KIND,
        REPO_ROOT
        / "results"
        / "0203d1f"
        / "notes"
        / "gru_postrun_materialization_validation_selected_regeneration_spec.json",
    )
    assert regeneration_spec.target_version == DIAGNOSTIC_REGENERATION_SPEC_SCHEMA_VERSION
    assert regeneration_spec.payload["diagnostic_name"] == "gru_postrun_materialization_bundle"

    evaluation_manifest = load_rlrmp_spec_payload(
        GRU_EVALUATION_DIAGNOSTICS_KIND,
        REPO_ROOT
        / "results"
        / "0203d1f"
        / "notes"
        / "gru_evaluation_diagnostics_validation_selected.json",
    )
    assert evaluation_manifest.target_version == GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION
    assert evaluation_manifest.payload["scope"] == "post_hoc_evaluation_non_certificate_diagnostics"

    manifest = load_manifest(
        REPO_ROOT
        / "results"
        / "9455785"
        / "manifests"
        / "training_runs"
        / "feedbax_training_run_rlrmp_artifact_normalization_fixture.json"
    )
    assert isinstance(manifest, TrainingRunManifest)

    with pytest.raises(ArchiveOnlySpecError) as excinfo:
        load_rlrmp_spec_payload(
            LEGACY_TRAINING_CONFIG_KIND,
            REPO_ROOT / "results" / "2ef67ca" / "models" / "baseline_standard" / "config.json",
        )
    message = str(excinfo.value)
    assert "Archive-only RLRMP legacy training config" in message
    assert "2ef67ca-era config.json files" in message


def test_checked_in_current_gru_diagnostics_can_be_schema_stamped() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "0203d1f"
        / "notes"
        / "gru_evaluation_diagnostics_validation_selected.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    stamped = stamp_current_schema(GRU_EVALUATION_DIAGNOSTICS_KIND, payload)

    result = accept_rlrmp_spec_payload(GRU_EVALUATION_DIAGNOSTICS_KIND, stamped)

    assert result.payload["schema_id"] == GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID
    assert result.payload["schema_version"] == GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION
