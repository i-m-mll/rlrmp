from __future__ import annotations

import json
from pathlib import Path

import pytest
from feedbax.contracts.manifest import TrainingRunManifest, load_manifest
from feedbax.contracts.migrations import SpecSchemaRegistry, UnknownSpecFamily, UnsupportedSpecVersion
from feedbax.contracts.training import TRAINING_RUN_SPEC_SCHEMA_VERSION

from rlrmp.runtime.spec_migrations import (
    ArchiveOnlySpecError,
    BRIDGE_CERTIFICATE_REPORT_PARAMS_KIND,
    BRIDGE_CERTIFICATE_REPORT_PARAMS_SCHEMA_ID,
    BRIDGE_CERTIFICATE_REPORT_PARAMS_SCHEMA_VERSION,
    CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
    CENTER_OUT_ENSEMBLE_EVAL_PARAMS_SCHEMA_ID,
    CENTER_OUT_ENSEMBLE_EVAL_PARAMS_SCHEMA_VERSION,
    CS_GRU_STANDARD_CERTIFICATES_KIND,
    CS_GRU_STANDARD_CERTIFICATES_SCHEMA_ID,
    CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION,
    DELAYED_DIAGNOSTIC_BUNDLE_KIND,
    DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_ID,
    DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
    DELAYED_REACH_BANK_EVAL_PARAMS_KIND,
    DELAYED_REACH_BANK_EVAL_PARAMS_SCHEMA_ID,
    DELAYED_REACH_BANK_EVAL_PARAMS_SCHEMA_VERSION,
    DELAYED_REACH_EVAL_BANK_KIND,
    DELAYED_REACH_EVAL_BANK_LEGACY_VERSION,
    DELAYED_REACH_EVAL_BANK_SCHEMA_ID,
    DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION,
    FEEDBACK_ABLATION_EVAL_PARAMS_KIND,
    FEEDBACK_ABLATION_EVAL_PARAMS_SCHEMA_ID,
    FEEDBACK_ABLATION_EVAL_PARAMS_SCHEMA_VERSION,
    FEEDBACK_QUALITY_LENS_KIND,
    FEEDBACK_QUALITY_LENS_REPORT_PARAMS_KIND,
    FEEDBACK_QUALITY_LENS_REPORT_PARAMS_SCHEMA_ID,
    FEEDBACK_QUALITY_LENS_REPORT_PARAMS_SCHEMA_VERSION,
    FEEDBACK_QUALITY_LENS_SCHEMA_ID,
    FEEDBACK_QUALITY_LENS_SCHEMA_VERSION,
    FeedbaxTrainingRunSpecMigrationError,
    FIXED_BANK_GRU_CHECKPOINT_RESCORE_KIND,
    FIXED_BANK_GRU_CHECKPOINT_RESCORE_LEGACY_VERSION,
    FIXED_BANK_GRU_CHECKPOINT_RESCORE_SCHEMA_ID,
    FIXED_BANK_GRU_CHECKPOINT_RESCORE_SCHEMA_VERSION,
    GRU_BROAD_EPSILON_ATTRIBUTION_KIND,
    GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_ID,
    GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_VERSION,
    GRU_EVALUATION_DIAGNOSTICS_KIND,
    GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID,
    GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION,
    GRU_FEEDBACK_ABLATION_KIND,
    GRU_FEEDBACK_ABLATION_SCHEMA_ID,
    GRU_FEEDBACK_ABLATION_SCHEMA_VERSION,
    GRU_MAP_ERROR_DECOMPOSITION_KIND,
    GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_ID,
    GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_VERSION,
    GRU_PERTURBATION_BANK_KIND,
    GRU_PERTURBATION_BANK_SCHEMA_ID,
    GRU_PERTURBATION_BANK_SCHEMA_VERSION,
    GRU_POSTRUN_REPORT_PARAMS_KIND,
    GRU_POSTRUN_REPORT_PARAMS_SCHEMA_ID,
    GRU_POSTRUN_REPORT_PARAMS_SCHEMA_VERSION,
    GRU_PERTURBATION_RESPONSE_NORM_PLOTS_KIND,
    GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_ID,
    GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_VERSION,
    GRU_WORST_CASE_EPSILON_AUDIT_KIND,
    GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_ID,
    GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_VERSION,
    HINF_PHENOTYPE_SIDECAR_KIND,
    HINF_PHENOTYPE_SIDECAR_SCHEMA_ID,
    HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION,
    LEGACY_TRAINING_CONFIG_KIND,
    LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF,
    OBJECTIVE_COMPARATOR_SIDECAR_KIND,
    OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_ID,
    OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_VERSION,
    PERTURBATION_CLASS_RESPONSE_KIND,
    PERTURBATION_CLASS_RESPONSE_SCHEMA_ID,
    PERTURBATION_CLASS_RESPONSE_SCHEMA_VERSION,
    PERTURBATION_OPEN_LOOP_CALIBRATION_KIND,
    PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_ID,
    PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_VERSION,
    PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
    PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_SCHEMA_ID,
    PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_SCHEMA_VERSION,
    ROBUSTNESS_PHENOTYPE_REPORT_PARAMS_KIND,
    ROBUSTNESS_PHENOTYPE_REPORT_PARAMS_SCHEMA_ID,
    ROBUSTNESS_PHENOTYPE_REPORT_PARAMS_SCHEMA_VERSION,
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    RUN_SPEC_SCHEMA_VERSION_LEGACY_CS_GRU,
    RUN_SPEC_SCHEMA_VERSION_V1,
    SEMANTIC_METHOD_EXTENSION_METADATA_KEYS,
    STANDARD_MATRIX_EVAL_PARAMS_KIND,
    STANDARD_MATRIX_EVAL_PARAMS_SCHEMA_ID,
    STANDARD_MATRIX_EVAL_PARAMS_SCHEMA_VERSION,
    STANDARD_MATRIX_EVAL_PARAMS_SCHEMA_VERSION_V1,
    VALIDATION_SELECTED_GRU_CHECKPOINTS_KIND,
    VALIDATION_SELECTED_GRU_CHECKPOINTS_LEGACY_VERSION,
    VALIDATION_SELECTED_GRU_CHECKPOINTS_SCHEMA_ID,
    VALIDATION_SELECTED_GRU_CHECKPOINTS_SCHEMA_VERSION,
    WORST_CASE_EPSILON_EVAL_PARAMS_KIND,
    WORST_CASE_EPSILON_EVAL_PARAMS_SCHEMA_ID,
    WORST_CASE_EPSILON_EVAL_PARAMS_SCHEMA_VERSION,
    accept_rlrmp_spec_payload,
    ensure_rlrmp_spec_families,
    load_rlrmp_spec_payload,
    stamp_current_schema,
)
from rlrmp.runtime.training_run_specs import (
    CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    feedbax_training_run_spec_from_payload,
)
from rlrmp.train.executor.slots import CS_SUPERVISED_METHOD_REF


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rlrmp_spec_policy_registers_current_families_and_rejects_v0() -> None:
    registry = ensure_rlrmp_spec_families(SpecSchemaRegistry())

    expected = {
        GRU_EVALUATION_DIAGNOSTICS_KIND: (
            GRU_EVALUATION_DIAGNOSTICS_SCHEMA_ID,
            GRU_EVALUATION_DIAGNOSTICS_SCHEMA_VERSION,
            ("rlrmp.gru_evaluation_diagnostics.v0",),
        ),
        CS_GRU_STANDARD_CERTIFICATES_KIND: (
            CS_GRU_STANDARD_CERTIFICATES_SCHEMA_ID,
            CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION,
            ("rlrmp.cs_gru_standard_certificates.v0",),
        ),
        OBJECTIVE_COMPARATOR_SIDECAR_KIND: (
            OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_ID,
            OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_VERSION,
            tuple(f"rlrmp.objective_comparator_sidecar.v{version}" for version in range(0, 6)),
        ),
        GRU_PERTURBATION_BANK_KIND: (
            GRU_PERTURBATION_BANK_SCHEMA_ID,
            GRU_PERTURBATION_BANK_SCHEMA_VERSION,
            (
                "rlrmp.gru_perturbation_bank.v0",
                "rlrmp.gru_perturbation_bank.v2",
                "rlrmp.gru_perturbation_response.v2",
            ),
        ),
        PERTURBATION_CLASS_RESPONSE_KIND: (
            PERTURBATION_CLASS_RESPONSE_SCHEMA_ID,
            PERTURBATION_CLASS_RESPONSE_SCHEMA_VERSION,
            ("rlrmp.perturbation_class_response.v0",),
        ),
        GRU_PERTURBATION_RESPONSE_NORM_PLOTS_KIND: (
            GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_ID,
            GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_VERSION,
            ("rlrmp.gru_perturbation_response_norm_plots.v0",),
        ),
        PERTURBATION_OPEN_LOOP_CALIBRATION_KIND: (
            PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_ID,
            PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_VERSION,
            (
                "rlrmp.perturbation_open_loop_calibration.v0",
                "rlrmp.perturbation_open_loop_calibration.v1",
            ),
        ),
        HINF_PHENOTYPE_SIDECAR_KIND: (
            HINF_PHENOTYPE_SIDECAR_SCHEMA_ID,
            HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION,
            ("rlrmp.hinf_phenotype_sidecar.v0",),
        ),
        GRU_WORST_CASE_EPSILON_AUDIT_KIND: (
            GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_ID,
            GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_VERSION,
            ("rlrmp.gru_worst_case_epsilon_audit.v0",),
        ),
        GRU_BROAD_EPSILON_ATTRIBUTION_KIND: (
            GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_ID,
            GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_VERSION,
            ("rlrmp.gru_broad_epsilon_attribution.v0",),
        ),
        GRU_MAP_ERROR_DECOMPOSITION_KIND: (
            GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_ID,
            GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_VERSION,
            ("rlrmp.gru_map_error_decomposition.v0",),
        ),
        GRU_FEEDBACK_ABLATION_KIND: (
            GRU_FEEDBACK_ABLATION_SCHEMA_ID,
            GRU_FEEDBACK_ABLATION_SCHEMA_VERSION,
            ("rlrmp.gru_feedback_ablation.v0",),
        ),
        DELAYED_DIAGNOSTIC_BUNDLE_KIND: (
            DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_ID,
            DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
            (
                "rlrmp.delayed_no_pgd_direction_split_velocity.v1",
                "rlrmp.delayed_peak_decay_diagnostics.v1",
            ),
        ),
        FEEDBACK_QUALITY_LENS_KIND: (
            FEEDBACK_QUALITY_LENS_SCHEMA_ID,
            FEEDBACK_QUALITY_LENS_SCHEMA_VERSION,
            ("rlrmp.feedback_quality_lens.v0",),
        ),
        GRU_POSTRUN_REPORT_PARAMS_KIND: (
            GRU_POSTRUN_REPORT_PARAMS_SCHEMA_ID,
            GRU_POSTRUN_REPORT_PARAMS_SCHEMA_VERSION,
            ("rlrmp.report.gru_postrun_summary.params.v0",),
        ),
        BRIDGE_CERTIFICATE_REPORT_PARAMS_KIND: (
            BRIDGE_CERTIFICATE_REPORT_PARAMS_SCHEMA_ID,
            BRIDGE_CERTIFICATE_REPORT_PARAMS_SCHEMA_VERSION,
            ("rlrmp.report.bridge_certificate_notes.params.v0",),
        ),
        FEEDBACK_QUALITY_LENS_REPORT_PARAMS_KIND: (
            FEEDBACK_QUALITY_LENS_REPORT_PARAMS_SCHEMA_ID,
            FEEDBACK_QUALITY_LENS_REPORT_PARAMS_SCHEMA_VERSION,
            ("rlrmp.report.feedback_quality_lens_summary.params.v0",),
        ),
        ROBUSTNESS_PHENOTYPE_REPORT_PARAMS_KIND: (
            ROBUSTNESS_PHENOTYPE_REPORT_PARAMS_SCHEMA_ID,
            ROBUSTNESS_PHENOTYPE_REPORT_PARAMS_SCHEMA_VERSION,
            ("rlrmp.report.robustness_phenotype_markdown.params.v0",),
        ),
        VALIDATION_SELECTED_GRU_CHECKPOINTS_KIND: (
            VALIDATION_SELECTED_GRU_CHECKPOINTS_SCHEMA_ID,
            VALIDATION_SELECTED_GRU_CHECKPOINTS_SCHEMA_VERSION,
            (VALIDATION_SELECTED_GRU_CHECKPOINTS_LEGACY_VERSION,),
        ),
        FIXED_BANK_GRU_CHECKPOINT_RESCORE_KIND: (
            FIXED_BANK_GRU_CHECKPOINT_RESCORE_SCHEMA_ID,
            FIXED_BANK_GRU_CHECKPOINT_RESCORE_SCHEMA_VERSION,
            (FIXED_BANK_GRU_CHECKPOINT_RESCORE_LEGACY_VERSION,),
        ),
        DELAYED_REACH_EVAL_BANK_KIND: (
            DELAYED_REACH_EVAL_BANK_SCHEMA_ID,
            DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION,
            (DELAYED_REACH_EVAL_BANK_LEGACY_VERSION,),
        ),
        CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND: (
            CENTER_OUT_ENSEMBLE_EVAL_PARAMS_SCHEMA_ID,
            CENTER_OUT_ENSEMBLE_EVAL_PARAMS_SCHEMA_VERSION,
            ("rlrmp.eval.center_out_ensemble.params.v0",),
        ),
        PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND: (
            PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_SCHEMA_ID,
            PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_SCHEMA_VERSION,
            (
                "rlrmp.eval.perturbation_response_bank.params.v0",
                "rlrmp.eval.perturbation_response_bank.params.v1",
            ),
        ),
        FEEDBACK_ABLATION_EVAL_PARAMS_KIND: (
            FEEDBACK_ABLATION_EVAL_PARAMS_SCHEMA_ID,
            FEEDBACK_ABLATION_EVAL_PARAMS_SCHEMA_VERSION,
            ("rlrmp.eval.feedback_ablation.params.v0",),
        ),
        WORST_CASE_EPSILON_EVAL_PARAMS_KIND: (
            WORST_CASE_EPSILON_EVAL_PARAMS_SCHEMA_ID,
            WORST_CASE_EPSILON_EVAL_PARAMS_SCHEMA_VERSION,
            ("rlrmp.eval.worst_case_epsilon.params.v0",),
        ),
        DELAYED_REACH_BANK_EVAL_PARAMS_KIND: (
            DELAYED_REACH_BANK_EVAL_PARAMS_SCHEMA_ID,
            DELAYED_REACH_BANK_EVAL_PARAMS_SCHEMA_VERSION,
            ("rlrmp.eval.delayed_reach_bank.params.v0",),
        ),
        STANDARD_MATRIX_EVAL_PARAMS_KIND: (
            STANDARD_MATRIX_EVAL_PARAMS_SCHEMA_ID,
            STANDARD_MATRIX_EVAL_PARAMS_SCHEMA_VERSION,
            (STANDARD_MATRIX_EVAL_PARAMS_SCHEMA_VERSION_V1,),
        ),
        RUN_SPEC_KIND: (
            RUN_SPEC_SCHEMA_ID,
            RUN_SPEC_SCHEMA_VERSION,
            ("rlrmp.run_spec.v0",),
        ),
    }

    for kind, (schema_id, current_version, old_versions) in expected.items():
        family = registry.resolve(kind)
        assert family.identity == schema_id
        assert family.current_version == current_version
        assert family.policy is not None
        assert family.policy.owner_module == "rlrmp.runtime.spec_migrations"
        if kind == RUN_SPEC_KIND:
            assert family.policy.stance == "migrate"
            assert set(family.policy.supported_old_versions) == {
                RUN_SPEC_SCHEMA_VERSION_V1,
                RUN_SPEC_SCHEMA_VERSION_LEGACY_CS_GRU,
            }

        result = registry.migrate(kind, {"schema_version": current_version})
        assert result.target_version == current_version
        assert not result.migrated

        for old_version in old_versions:
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

    migrated_cs_gru = load_rlrmp_spec_payload(
        RUN_SPEC_KIND,
        REPO_ROOT
        / "results"
        / "30f2313"
        / "runs"
        / "cs_stochastic_gru__hidden_penalty"
        / "run.json",
    )
    assert migrated_cs_gru.source_version == RUN_SPEC_SCHEMA_VERSION_LEGACY_CS_GRU
    assert migrated_cs_gru.target_version == RUN_SPEC_SCHEMA_VERSION
    assert migrated_cs_gru.migrated
    assert migrated_cs_gru.payload["migration_policy"] == "migrated_active_v1_to_v2"

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


def test_historical_feedbax_training_run_spec_migrates_standard_supervised() -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "results"
            / "1ab1fef"
            / "runs"
            / "epsilon_scaled_short_3500to1000.json"
        ).read_text(encoding="utf-8")
    )
    original = payload[FEEDBAX_TRAINING_RUN_SPEC_KEY]
    assert (
        f"{original['method_ref']['package']}/"
        f"{original['method_ref']['name']}/"
        f"{original['method_ref']['version']}"
        == LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF
    )

    training_spec = feedbax_training_run_spec_from_payload(payload)

    assert training_spec.schema_version == TRAINING_RUN_SPEC_SCHEMA_VERSION
    assert training_spec.graph.inline["schema_version"] == "feedbax.spec.graph.v4"
    assert {
        subgraph["schema_version"]
        for subgraph in training_spec.graph.inline.get("subgraphs", {}).values()
    } == {"feedbax.spec.graph.v4"}
    assert training_spec.on_nan == "raise"
    assert training_spec.method_ref.key == CS_SUPERVISED_METHOD_REF
    assert training_spec.method_payload.schema_version == (
        CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION
    )
    assert (
        training_spec.method_payload.payload["training_mode"]
        == payload["training_summary"]["training_mode"]
    )
    assert training_spec.method_payload.payload["pre_step"]["kind"] == (
        "broad_epsilon_pgd_pre_step"
    )
    method_metadata = training_spec.method_extensions.metadata
    assert not (set(method_metadata) & SEMANTIC_METHOD_EXTENSION_METADATA_KEYS)
    assert method_metadata["runner"] == "rlrmp.train.cs_nominal_gru"
    assert training_spec.worker_execution.metadata["migrated_from_method_ref"] == (
        LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF
    )
    assert training_spec.metadata["feedbax_training_run_spec_migration"] == {
        "source_method_ref": LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF,
        "target_method_ref": CS_SUPERVISED_METHOD_REF,
        "semantic_metadata_keys_removed": sorted(SEMANTIC_METHOD_EXTENSION_METADATA_KEYS),
    }


def test_stripped_legacy_feedbax_training_run_spec_fails_closed() -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "results"
            / "1ab1fef"
            / "runs"
            / "epsilon_scaled_short_3500to1000.json"
        ).read_text(encoding="utf-8")
    )
    feedbax_spec = dict(payload[FEEDBAX_TRAINING_RUN_SPEC_KEY])
    method_extensions = dict(feedbax_spec["method_extensions"])
    metadata = {
        key: value
        for key, value in dict(method_extensions["metadata"]).items()
        if key not in SEMANTIC_METHOD_EXTENSION_METADATA_KEYS
    }
    method_extensions["metadata"] = metadata
    feedbax_spec["method_extensions"] = method_extensions
    payload[FEEDBAX_TRAINING_RUN_SPEC_KEY] = feedbax_spec

    with pytest.raises(FeedbaxTrainingRunSpecMigrationError) as excinfo:
        feedbax_training_run_spec_from_payload(payload)

    message = str(excinfo.value)
    assert LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF in message
    assert "method_extensions.metadata" in message
    assert "dfa0cd5" in message


def test_unsupported_semantic_feedbax_training_run_spec_fails_explicitly() -> None:
    payload = json.loads(
        (
            REPO_ROOT
            / "results"
            / "1ab1fef"
            / "runs"
            / "epsilon_scaled_short_3500to1000.json"
        ).read_text(encoding="utf-8")
    )
    feedbax_spec = dict(payload[FEEDBAX_TRAINING_RUN_SPEC_KEY])
    feedbax_spec["method_ref"] = {
        "package": "rlrmp",
        "name": "unsupported_historical_method",
        "version": "v1",
    }
    payload[FEEDBAX_TRAINING_RUN_SPEC_KEY] = feedbax_spec

    with pytest.raises(FeedbaxTrainingRunSpecMigrationError) as excinfo:
        feedbax_training_run_spec_from_payload(payload)

    message = str(excinfo.value)
    assert "unsupported_historical_method" in message
    assert "method_extensions.metadata" in message
    assert "rlrmp_training_mode" in message


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


def test_representative_analysis_sidecar_payloads_are_accepted() -> None:
    sidecars = {
        OBJECTIVE_COMPARATOR_SIDECAR_KIND: (
            OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_ID,
            OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_VERSION,
            {
                "schema_version": OBJECTIVE_COMPARATOR_SIDECAR_SCHEMA_VERSION,
                "scope": "unit",
                "rows": [{"run_id": "run_a"}],
                "standard_split_bank_comparator": {"status": "available"},
            },
        ),
        GRU_PERTURBATION_BANK_KIND: (
            GRU_PERTURBATION_BANK_SCHEMA_ID,
            GRU_PERTURBATION_BANK_SCHEMA_VERSION,
            {
                "schema_version": GRU_PERTURBATION_BANK_SCHEMA_VERSION,
                "bank_id": "unit-bank",
                "scope": "controller_independent_perturbation_response",
                "runs": {"run_a": {"status": "available"}},
            },
        ),
        PERTURBATION_OPEN_LOOP_CALIBRATION_KIND: (
            PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_ID,
            PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_VERSION,
            {
                "schema_version": PERTURBATION_OPEN_LOOP_CALIBRATION_SCHEMA_VERSION,
                "bank_schema_version": GRU_PERTURBATION_BANK_SCHEMA_VERSION,
                "calibration_points": [],
            },
        ),
        HINF_PHENOTYPE_SIDECAR_KIND: (
            HINF_PHENOTYPE_SIDECAR_SCHEMA_ID,
            HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION,
            {
                "schema_version": HINF_PHENOTYPE_SIDECAR_SCHEMA_VERSION,
                "scope": "validation_selected_gru_robustness_phenotype",
                "components": {},
                "rows": [],
            },
        ),
        GRU_BROAD_EPSILON_ATTRIBUTION_KIND: (
            GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_ID,
            GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_VERSION,
            {
                "schema_version": GRU_BROAD_EPSILON_ATTRIBUTION_SCHEMA_VERSION,
                "active_vs_zero_semantics": {"paired_condition": "zero_broad_epsilon"},
                "rows": [],
            },
        ),
        GRU_WORST_CASE_EPSILON_AUDIT_KIND: (
            GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_ID,
            GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_VERSION,
            {
                "schema_version": GRU_WORST_CASE_EPSILON_AUDIT_SCHEMA_VERSION,
                "scope": "worst_case_full_state_epsilon",
                "rows": [],
            },
        ),
        GRU_MAP_ERROR_DECOMPOSITION_KIND: (
            GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_ID,
            GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_VERSION,
            {
                "schema_version": GRU_MAP_ERROR_DECOMPOSITION_SCHEMA_VERSION,
                "rows": [{"run_id": "run_a", "decomposition": {}}],
            },
        ),
        GRU_FEEDBACK_ABLATION_KIND: (
            GRU_FEEDBACK_ABLATION_SCHEMA_ID,
            GRU_FEEDBACK_ABLATION_SCHEMA_VERSION,
            {
                "schema_version": GRU_FEEDBACK_ABLATION_SCHEMA_VERSION,
                "scope": "fixed_target_random_perturb_validation_selected",
                "rows": [],
                "feedback_checkpoint_selection_audit_status": "external_custody",
            },
        ),
        GRU_PERTURBATION_RESPONSE_NORM_PLOTS_KIND: (
            GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_ID,
            GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_VERSION,
            {
                "schema_version": GRU_PERTURBATION_RESPONSE_NORM_PLOTS_SCHEMA_VERSION,
                "source_manifest": "results/unit/manifest.json",
                "figures": [],
            },
        ),
        DELAYED_DIAGNOSTIC_BUNDLE_KIND: (
            DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_ID,
            DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
            {
                "schema_version": DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
                "issue": "21f4638",
                "scope": "unit",
                "direction_split": {"status": "not_applicable"},
                "peak_decay": {"status": "available", "signals": {}},
            },
        ),
        FEEDBACK_QUALITY_LENS_KIND: (
            FEEDBACK_QUALITY_LENS_SCHEMA_ID,
            FEEDBACK_QUALITY_LENS_SCHEMA_VERSION,
            {
                "schema_version": FEEDBACK_QUALITY_LENS_SCHEMA_VERSION,
                "scope": "feedback_control_quality_diagnostics",
                "outputs": {
                    "perturbation_response": {"status": "materialized"},
                    "feedback_ablation": {"status": "not_applicable"},
                },
            },
        ),
    }

    for kind, (schema_id, schema_version, payload) in sidecars.items():
        result = accept_rlrmp_spec_payload(kind, payload)

        assert result.schema_id == schema_id
        assert result.target_version == schema_version
        assert result.payload["schema_version"] == schema_version


def test_generic_feedbax_custody_families_are_not_registered_in_rlrmp() -> None:
    registry = ensure_rlrmp_spec_families(SpecSchemaRegistry())
    generic_or_sibling_owned_kinds = (
        "RLRMPDiagnosticRegenerationSpec",
        "RLRMPGRUPostrunMaterialization",
        "RLRMPDiagnosticOutputStatus",
        "RLRMPArtifactGroup",
    )

    for kind in generic_or_sibling_owned_kinds:
        with pytest.raises(UnknownSpecFamily):
            registry.resolve(kind)
