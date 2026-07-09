"""Tests for Feedbax declarative rlrmp materialization recipes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rlrmp
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.bundles import (
    execute_staged_analysis_bundle,
    load_analysis_bundle,
)
from feedbax.analysis.evaluation import execute_evaluation_run_spec
from feedbax.analysis.materialization import ContextMaterializer
from feedbax.analysis.specs import (
    AnalysisRecipeExecutionError,
    execute_analysis_run_spec,
    unregister_analysis_recipe,
)
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunSpec,
    ParentRef,
    TrainingRunManifest,
    load_manifest,
    write_manifest,
)
from feedbax.plugins.registry import ExperimentRegistry
from pydantic import ValidationError

from rlrmp.analysis import declarative_materialization as dm
from rlrmp.eval.recipes import (
    CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
    FEEDBACK_ABLATION_EVALUATION_TYPE,
    PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
)
from rlrmp.analysis.pipelines.gru_perturbation_bank import (
    PERTURBATION_BANK_PARAMS_TYPE,
    PerturbationBankParams,
)
from rlrmp.runtime.params_models import params_model_for
from rlrmp.runtime.spec_migrations import (
    CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
    FEEDBACK_ABLATION_EVAL_PARAMS_KIND,
    PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
    stamp_current_schema,
)


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
FEEDBACK_QUALITY_ALLOWED_PARITY_DIFFS = (
    "declarative_analysis",
    "bundle_contract.analysis_manifest_id",
)


def _artifact_roles(manifest) -> set[str]:
    return {artifact.role for artifact in manifest.artifacts}


def _artifact_payload(manifest, role: str) -> dict:
    artifact = next(artifact for artifact in manifest.artifacts if artifact.role == role)
    assert artifact.uri is not None
    return json.loads(Path(artifact.uri).read_text(encoding="utf-8"))


def _feedback_quality_parity_payload(payload: dict) -> dict:
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    normalized.pop("declarative_analysis", None)
    normalized.get("bundle_contract", {}).pop("analysis_manifest_id", None)
    return normalized


def _unregister_declarative_recipes() -> None:
    unregister_analysis_recipe(dm.GRU_STANDARD_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.FEEDBACK_ABLATION_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.PERTURBATION_CLASS_RESPONSE_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.PERTURBATION_BANK_AGGREGATE_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.POLICY_DIAGNOSTICS_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.RECURRENT_JACOBIAN_ANALYSIS_TYPE)
    for analysis_type in dm.FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES.values():
        unregister_analysis_recipe(analysis_type)
    unregister_analysis_recipe(dm.FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.ROBUSTNESS_PHENOTYPE_ANALYSIS_TYPE)
    unregister_analysis_recipe(dm.OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE)


def _training_ref(run_id: str = "training-run-a") -> ParentRef:
    return ParentRef(kind="TrainingRunManifest", id=run_id, role="training_run")


def _perturbation_row(
    perturbation_id: str,
    *,
    family: str,
    axis: str = "x",
) -> dict[str, object]:
    return {
        "perturbation_id": perturbation_id,
        "channel": "command_input",
        "family": family,
        "amplitude": 1.0,
        "units": "N",
        "axis": axis,
        "basis": "command_cartesian_force_xy",
        "sign": 1,
        "timing": {"start_time_index": 1, "duration_steps": 2},
        "adapter": "feedbax.additive_channel_adapter.command_input",
        "description": "Unit test command pulse.",
        "calibration_mode": "unit_test",
        "calibration_role": "reach_relative_calibrated_open_loop",
    }


def _perturbation_bank() -> dict[str, object]:
    return {
        "bank_id": "unit_test_bank",
        "perturbations": [
            _perturbation_row("row-a", family="command_input_pulse"),
            _perturbation_row(
                "row-b",
                family="target_aligned_lateral_command_load_pulse",
                axis="y",
            ),
        ],
    }


def _run_payload() -> dict[str, object]:
    return {
        "label": "Run A",
        "run_spec_path": "results/unit/runs/training-run-a.json",
        "artifact_dir": "_artifacts/unit/runs/training-run-a",
        "n_replicates": 1,
        "n_rollout_trials_per_replicate": 1,
        "n_time_steps": 2,
        "dt_s": 0.01,
        "status_counts": {"evaluated": 2},
        "perturbations": [
            {
                "perturbation_id": "row-a",
                "channel": "command_input",
                "family": "command_input_pulse",
                "axis": "x",
                "status": "evaluated",
                "metrics": {
                    "delta_action_norm_mean": 1.0,
                    "delta_position_norm_mean": 0.5,
                },
                "perturbation": _perturbation_row("row-a", family="command_input_pulse"),
            },
            {
                "perturbation_id": "row-b",
                "channel": "command_input",
                "family": "target_aligned_lateral_command_load_pulse",
                "axis": "y",
                "status": "evaluated",
                "metrics": {
                    "delta_action_norm_mean": 2.0,
                    "delta_position_norm_mean": 1.5,
                },
                "perturbation": _perturbation_row(
                    "row-b",
                    family="target_aligned_lateral_command_load_pulse",
                    axis="y",
                ),
            },
        ],
        "bulk_files": {
            "row-a": "_artifacts/unit/runs/training-run-a/row-a.npz",
            "row-b": "_artifacts/unit/runs/training-run-a/row-b.npz",
        },
    }


def _policy_diagnostics_payload() -> dict[str, object]:
    return {
        "schema_id": "rlrmp.eval.policy_diagnostics.fixture",
        "rows": [
            {
                "row_id": "policy-row-a",
                "blocks": {
                    "feedback": [1.0, -1.0],
                    "sisu": [0.25],
                },
                "roles": {
                    "feedback": "controller_visible_feedback",
                    "sisu": "sisu",
                },
                "absent_blocks": [
                    {
                        "name": "context",
                        "role": "context",
                        "reason": "no_hold_context",
                    }
                ],
                "action": [0.5, -0.25],
                "linear_map": [
                    [2.0, 0.0, 1.0],
                    [0.0, 3.0, -1.0],
                ],
                "sisu_values": [0.0, 1.0],
                "signed_pairs": [
                    {
                        "pair_id": "feedback-x",
                        "positive_response": [3.0, -1.0],
                        "negative_response": [-2.0, 1.0],
                        "baseline_response": [0.0, 0.0],
                    }
                ],
            }
        ],
    }


def _recurrent_jacobians_payload() -> dict[str, object]:
    return {
        "schema_id": "rlrmp.eval.recurrent_jacobians.fixture",
        "rows": [
            {
                "row_id": "recurrent-row-a",
                "h_pre": [0.1, -0.2],
                "feedback": [1.0, 2.0],
                "sisu": [0.5],
                "context": [1.0],
                "h_post": [0.25, -0.75],
                "u": [1.5, -0.5],
                "blocks": {
                    "A": [[0.8, 0.1], [0.0, 0.7]],
                    "B_y": [[1.0, 0.0], [0.5, -0.5]],
                    "B_s": [[0.2], [0.4]],
                    "B_c": [[0.1], [-0.2]],
                    "W": [[1.0, 2.0], [-1.0, 0.5]],
                },
            }
        ],
    }


def test_declarative_recipes_use_feedbax_context_materializers() -> None:
    standard = dm.gru_standard_certificate_recipe(
        dm.gru_standard_certificate_spec(),
        Path("."),
        (),
    )
    evaluation = dm.gru_evaluation_diagnostics_recipe(
        dm.gru_evaluation_diagnostics_spec(
            experiment="unitexp",
            run_ids=["unit_run"],
        ),
        Path("."),
        (),
    )
    rollout_recovery = dm.output_feedback_rollout_recovery_recipe(
        dm.output_feedback_rollout_recovery_spec(),
        Path("."),
        (),
    )

    assert isinstance(standard.analyses["gru_standard_certificate"], AbstractAnalysis)
    assert not isinstance(standard.analyses["gru_standard_certificate"], ContextMaterializer)
    assert isinstance(evaluation.analyses["gru_evaluation_diagnostics"], AbstractAnalysis)
    assert not isinstance(evaluation.analyses["gru_evaluation_diagnostics"], ContextMaterializer)
    assert isinstance(
        rollout_recovery.analyses["output_feedback_rollout_recovery"],
        AbstractAnalysis,
    )
    assert not isinstance(
        rollout_recovery.analyses["output_feedback_rollout_recovery"],
        ContextMaterializer,
    )
    perturbation_leaf = dm.perturbation_class_response_recipe(
        dm.perturbation_class_response_spec(
            family="command_input_pulse",
            evaluation_manifest_id="eval-manifest",
        ),
        Path("."),
        [
            type(
                "Resolved",
                (),
                {
                    "ref": ParentRef(
                        kind="EvaluationRunManifest",
                        id="eval-manifest",
                        role="evaluation_run",
                    ),
                    "states": {
                        "evaluation_type": PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
                        "evaluation_manifest_id": "eval-manifest",
                        "perturbation_battery": {"perturbations": []},
                        "response_tensors": {"runs": {}},
                        "class_index_map": {
                            "command_input_pulse": {
                                "perturbation_ids": [],
                                "row_indices": [],
                            }
                        },
                    },
                    "manifest": None,
                    "path": None,
                },
            )()
        ],
    )
    assert isinstance(
        perturbation_leaf.analyses["command_input_pulse"],
        AbstractAnalysis,
    )


def test_diagnostic_bank_recipes_register_params_models_and_eval_dependencies() -> None:
    rlrmp.register_experiment_package(ExperimentRegistry())

    assert params_model_for(dm.FEEDBACK_ABLATION_ANALYSIS_TYPE) is (
        dm.FeedbackAblationAnalysisParams
    )
    assert params_model_for(dm.FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["feedback_ablation"]) is (
        dm.FeedbackAblationAnalysisParams
    )
    assert params_model_for(dm.OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE) is (
        dm.OutputFeedbackRolloutRecoveryParams
    )
    assert params_model_for(dm.POLICY_DIAGNOSTICS_ANALYSIS_TYPE) is (
        dm.PolicyDiagnosticsAnalysisParams
    )
    assert params_model_for(dm.RECURRENT_JACOBIAN_ANALYSIS_TYPE) is (
        dm.RecurrentJacobianAnalysisParams
    )
    assert params_model_for(dm.PERTURBATION_CLASS_RESPONSE_ANALYSIS_TYPE) is (
        dm.PerturbationClassResponseAnalysisParams
    )
    assert params_model_for(dm.PERTURBATION_BANK_AGGREGATE_ANALYSIS_TYPE) is (
        dm.PerturbationBankAggregateAnalysisParams
    )
    assert params_model_for(PERTURBATION_BANK_PARAMS_TYPE) is PerturbationBankParams
    with pytest.raises(ValidationError):
        dm.PolicyDiagnosticsAnalysisParams.model_validate({"unknown": True})
    with pytest.raises(ValidationError):
        dm.RecurrentJacobianAnalysisParams.model_validate({"unknown": True})
    with pytest.raises(ValidationError):
        dm.PerturbationClassResponseAnalysisParams.model_validate({"unknown": True})
    with pytest.raises(ValidationError):
        dm.PerturbationBankAggregateAnalysisParams.model_validate({"unknown": True})
    assert dm.EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE[
        dm.FEEDBACK_ABLATION_ANALYSIS_TYPE
    ] == (FEEDBACK_ABLATION_EVALUATION_TYPE,)
    assert dm.EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE[
        dm.POLICY_DIAGNOSTICS_ANALYSIS_TYPE
    ] == ("evaluation_run",)
    assert dm.EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE[
        dm.FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["feedback_ablation"]
    ] == (FEEDBACK_ABLATION_EVALUATION_TYPE, "params.materialize_feedback_ablation")
    assert dm.EVAL_DEPENDENCIES_BY_ANALYSIS_TYPE[
        dm.RECURRENT_JACOBIAN_ANALYSIS_TYPE
    ] == ("evaluation_run",)


def test_feedback_ablation_recipe_consumes_cached_eval_states_and_records_custody(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    eval_manifest, eval_path = execute_evaluation_run_spec(
        EvaluationRunSpec(
            evaluation_type=FEEDBACK_ABLATION_EVALUATION_TYPE,
            inputs=[_training_ref()],
            params=stamp_current_schema(
                FEEDBACK_ABLATION_EVAL_PARAMS_KIND,
                {
                    "ablation_masks": {"normal": None},
                    "rollout_pairs": [
                        {
                            "bin": "nominal",
                            "mode": "normal",
                            "status": "evaluated",
                            "metrics": {
                                "baseline_action_norm": {"mean": 1.0},
                                "baseline_endpoint_error_m": {"mean": 0.01},
                                "baseline_terminal_speed_m_s": {"mean": 0.01},
                                "baseline_full_qrf_cost": {"total": {"mean": 1.0}},
                            },
                        },
                        {
                            "bin": "initial_state",
                            "mode": "lagged_observation_history",
                            "status": "evaluated",
                            "metrics": {
                                "baseline_action_norm": {"mean": 1.0},
                                "delta_action_norm": {"mean": 0.5},
                            },
                        },
                    ],
                },
            ),
        ),
        root=tmp_path,
        force=True,
    )

    analysis_manifest, _path = execute_analysis_run_spec(
        AnalysisRunSpec(
            analysis_type=dm.FEEDBACK_ABLATION_ANALYSIS_TYPE,
            inputs=[
                ParentRef(
                    kind="EvaluationRunManifest",
                    id=eval_manifest.id,
                    role="evaluation_run",
                    uri=str(eval_path),
                )
            ],
            params={"experiment": "unit", "scope": "cached_eval_fixture"},
        ),
        root=tmp_path,
        issues=["d0189db"],
    )

    payload = _artifact_payload(analysis_manifest, "rlrmp-gru-feedback-ablation-manifest")
    report = _artifact_payload(analysis_manifest, "rlrmp-gru-feedback-ablation-report-render")
    assert analysis_manifest.inputs[0].kind == "EvaluationRunManifest"
    assert analysis_manifest.provenance.parents[0].id == eval_manifest.id
    assert payload["analysis_type"] == dm.FEEDBACK_ABLATION_ANALYSIS_TYPE
    assert payload["scope"] == "cached_eval_fixture"
    run = next(iter(payload["runs"].values()))
    assert run["status_counts"] == {"evaluated": 2}
    assert run["normalized_feedback_use"]["status"] == "available"
    assert "GRU Feedback Ablation Diagnostic" in report["markdown"]

    component_manifest, _component_path = execute_analysis_run_spec(
        AnalysisRunSpec(
            analysis_type=dm.FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["feedback_ablation"],
            inputs=[
                ParentRef(
                    kind="EvaluationRunManifest",
                    id=eval_manifest.id,
                    role="evaluation_run",
                    uri=str(eval_path),
                )
            ],
            params={"experiment": "unit", "scope": "cached_eval_component_fixture"},
        ),
        root=tmp_path,
        issues=["d0189db"],
    )

    component_payload = _artifact_payload(
        component_manifest,
        "rlrmp-feedback-quality-feedback-ablation-status",
    )
    assert component_payload["analysis_type"] == (
        dm.FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["feedback_ablation"]
    )
    assert "rlrmp-feedback-quality-feedback-ablation-note" in _artifact_roles(
        component_manifest
    )


def test_policy_diagnostics_recipe_consumes_cached_eval_states_and_records_custody(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    eval_manifest, eval_path = execute_evaluation_run_spec(
        EvaluationRunSpec(
            evaluation_type=CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
            inputs=[_training_ref()],
            params=stamp_current_schema(
                CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
                {
                    "task": "center_out",
                    "policy_diagnostics": _policy_diagnostics_payload(),
                },
            ),
        ),
        root=tmp_path,
        force=True,
    )

    analysis_manifest, _path = execute_analysis_run_spec(
        dm.policy_diagnostics_spec(
            evaluation_manifest_id=eval_manifest.id,
            evaluation_manifest_uri=eval_path,
            include_finite_difference=True,
        ),
        root=tmp_path,
        issues=["a3a3716"],
    )

    payload = _artifact_payload(analysis_manifest, "rlrmp-policy-diagnostics-bank")
    row = payload["rows"][0]
    assert analysis_manifest.inputs[0].kind == "EvaluationRunManifest"
    assert analysis_manifest.provenance.parents[0].id == eval_manifest.id
    assert payload["adapter"]["kernel_module"] == "rlrmp.eval.policy_diagnostics"
    assert payload["evaluation_manifest_dependency"]["manifest_id"] == eval_manifest.id
    assert row["row_id"] == "policy-row-a"
    assert row["finite_difference"]["passed"] is True
    assert row["sisu_modulation"]["status"] == "available"
    assert row["signed_pairs"][0]["pair_id"] == "feedback-x"
    assert row["jacobian"]["blocks"]["feedback"]["shape"] == [2, 2]
    assert row["block_summaries"]["feedback"]["singular_values"]["status"] == "available"


def test_recurrent_jacobian_recipe_consumes_cached_eval_states_and_records_custody(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    eval_manifest, eval_path = execute_evaluation_run_spec(
        EvaluationRunSpec(
            evaluation_type=CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
            inputs=[_training_ref()],
            params=stamp_current_schema(
                CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
                {
                    "task": "center_out",
                    "recurrent_jacobians": _recurrent_jacobians_payload(),
                },
            ),
        ),
        root=tmp_path,
        force=True,
    )

    analysis_manifest, _path = execute_analysis_run_spec(
        dm.recurrent_jacobian_spec(
            evaluation_manifest_id=eval_manifest.id,
            evaluation_manifest_uri=eval_path,
            include_finite_difference=True,
        ),
        root=tmp_path,
        issues=["061a879"],
    )

    payload = _artifact_payload(analysis_manifest, "rlrmp-recurrent-jacobian-bank")
    row = payload["rows"][0]
    assert analysis_manifest.inputs[0].kind == "EvaluationRunManifest"
    assert analysis_manifest.provenance.parents[0].id == eval_manifest.id
    assert payload["adapter"]["kernel_module"] == "rlrmp.eval.recurrent_jacobians"
    assert payload["evaluation_manifest_dependency"]["manifest_id"] == eval_manifest.id
    assert row["row_id"] == "recurrent-row-a"
    assert row["metadata"]["domains"]["context"]["status"] == "available"
    assert row["summaries"]["matrix_summaries"]["K_y"]["status"] == "available"
    assert row["summaries"]["matrix_summaries"]["B_c"]["shape"] == [2, 1]
    assert row["finite_difference"]["A"]["status"] == "available"
    assert row["finite_difference"]["A"]["max_abs_error"] < 1e-3


def test_output_feedback_bridge_bundle_resource_loads() -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)

    bundle = load_analysis_bundle("rlrmp/output_feedback_bridge", registry=registry)

    assert bundle.name == "output_feedback_bridge"
    assert bundle.metadata["bundle_family"] == "rlrmp/output_feedback_bridge"
    assert bundle.templates[0].analysis_type == dm.OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE
    assert bundle.templates[0].requested_outputs == ["output_feedback_rollout_recovery"]


def test_feedback_quality_lens_bundle_resource_loads() -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)

    bundle = load_analysis_bundle("rlrmp/feedback_quality_lens", registry=registry)

    assert bundle.name == "feedback_quality_lens"
    assert bundle.metadata["bundle_family"] == "rlrmp/feedback_quality_lens"
    stages = {stage.name: stage for stage in bundle.stages}
    assert stages["evaluation_diagnostics"].analysis_type == (
        dm.FEEDBACK_QUALITY_COMPONENT_ANALYSIS_TYPES["evaluation_diagnostics"]
    )
    assert stages["response_norm_plots"].depends_on_roles[0].stage == "perturbation_response"
    assert stages["response_norm_plots"].depends_on_roles[0].role == (
        "rlrmp-feedback-quality-perturbation-response-manifest"
    )
    assert stages["response_norm_plots"].depends_on_roles[0].required is False
    assert stages["feedback_quality_lens"].analysis_type == dm.FEEDBACK_QUALITY_LENS_ANALYSIS_TYPE
    assert set(stages["feedback_quality_lens"].depends_on) == set(
        dm.FEEDBACK_QUALITY_COMPONENT_NAMES
    )


def test_feedback_quality_component_gating_expr_census_table() -> None:
    registrations = dm._feedback_quality_component_registrations()
    component_statuses = [
        ("unavailable", True),
        ("materialized", False),
    ]

    for name in dm.FEEDBACK_QUALITY_COMPONENT_NAMES:
        registration = registrations[name]
        decision_rows = [
            (
                "default_include",
                {},
                "unavailable",
                True,
                False,
                True,
                True,
            ),
            (
                "explicit_include",
                {f"include_{name}": True},
                "unavailable",
                True,
                False,
                True,
                True,
            ),
            (
                "disabled",
                {f"include_{name}": False},
                "unavailable",
                False,
                False,
                False,
                False,
            ),
            (
                "not_applicable",
                {"not_applicable_components": [name]},
                "unavailable",
                True,
                True,
                False,
                False,
            ),
        ]
        for status, should_materialize in component_statuses:
            decision_rows.append(
                (
                    f"default_include_{status}",
                    {},
                    status,
                    True,
                    False,
                    True,
                    should_materialize,
                )
            )

        for (
            _case,
            params,
            status,
            included,
            not_applicable,
            eligible,
            should_materialize,
        ) in decision_rows:
            decision = dm._feedback_quality_gating_decision(
                registration,
                params=params,
                component_status={"status": status},
            )
            assert decision.included is included
            assert decision.not_applicable is not_applicable
            assert decision.eligible is eligible
            assert decision.should_materialize is should_materialize

    example = registrations["evaluation_diagnostics"].gating_expr.model_dump(
        mode="json",
        exclude_none=True,
    )
    assert example == {
        "kind": "all",
        "exprs": [
            {
                "kind": "any",
                "exprs": [
                    {
                        "kind": "not",
                        "expr": {
                            "kind": "compare",
                            "item": "params",
                            "path": "include_evaluation_diagnostics",
                            "op": "exists",
                        },
                    },
                    {
                        "kind": "compare",
                        "item": "params",
                        "path": "include_evaluation_diagnostics",
                        "op": "eq",
                        "value": True,
                    },
                ],
            },
            {
                "kind": "not",
                "expr": {
                    "kind": "all",
                    "exprs": [
                        {
                            "kind": "compare",
                            "item": "params",
                            "path": "not_applicable_components",
                            "op": "exists",
                        },
                        {
                            "kind": "compare",
                            "item": "params",
                            "path": "not_applicable_components",
                            "op": "contains",
                            "value": "evaluation_diagnostics",
                        },
                    ],
                },
            },
        ],
    }


def test_gru_postrun_bundle_declares_perturbation_leaf_aggregate_stages() -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)

    bundle = load_analysis_bundle("rlrmp/gru_postrun", registry=registry)

    stages = {stage.name: stage for stage in bundle.stages}
    assert stages["perturbation_bank_eval"].evaluation_type == (
        PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE
    )
    assert stages["perturbation_class_command_input_pulse"].analysis_type == (
        dm.PERTURBATION_CLASS_RESPONSE_ANALYSIS_TYPE
    )
    assert stages["perturbation_class_command_input_pulse"].depends_on == [
        "perturbation_bank_eval"
    ]
    assert stages["perturbation_bank_aggregate"].analysis_type == (
        dm.PERTURBATION_BANK_AGGREGATE_ANALYSIS_TYPE
    )
    assert "perturbation_class_command_input_pulse" in (
        stages["perturbation_bank_aggregate"].depends_on
    )


def test_perturbation_class_leaves_aggregate_to_legacy_bank_payload(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    params = stamp_current_schema(
        PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
        {
            "checkpoint_bank_ref": {"kind": "CheckpointSelectionManifest", "id": "bank-a"},
            "perturbation_battery": _perturbation_bank(),
            "response_tensors": {"runs": {"training-run-a": _run_payload()}},
            "consumed_data_identities": [
                {
                    "role": "perturbation_open_loop_calibration",
                    "hash": "sha256:unit-calibration",
                }
            ],
            "legacy_payload_mode": True,
        },
    )
    eval_manifest, eval_path = execute_evaluation_run_spec(
        EvaluationRunSpec(
            evaluation_type=PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
            training_run_ids=["training-run-a"],
            inputs=[_training_ref()],
            params=params,
        ),
        root=tmp_path,
        force=True,
    )
    leaf_manifests = []
    for family in (
        "command_input_pulse",
        "target_aligned_lateral_command_load_pulse",
    ):
        manifest, path = execute_analysis_run_spec(
            dm.perturbation_class_response_spec(
                family=family,
                evaluation_manifest_id=eval_manifest.id,
                evaluation_manifest_uri=eval_path,
                expected_calibration_identity={"hash": "sha256:unit-calibration"},
            ),
            root=tmp_path,
        )
        leaf_payload = _artifact_payload(manifest, "rlrmp-perturbation-class-response")
        assert leaf_payload["family"] == family
        assert leaf_payload["evaluation_manifest"]["id"] == eval_manifest.id
        leaf_manifests.append(
            ParentRef(
                kind="AnalysisRunManifest",
                id=manifest.id,
                role="analysis_run",
                uri=str(path),
            )
        )

    aggregate_manifest, _aggregate_path = execute_analysis_run_spec(
        dm.perturbation_bank_aggregate_spec(
            leaf_manifest_refs=leaf_manifests,
            issue="unit",
            source_experiment="unit-exp",
            bank_mode="raw",
        ),
        root=tmp_path,
    )
    aggregate = _artifact_payload(
        aggregate_manifest,
        "rlrmp-gru-perturbation-response-manifest",
    )

    legacy_run = _run_payload()
    assert aggregate["schema_version"] == "rlrmp.gru_perturbation_bank.v3"
    assert [row["perturbation_id"] for row in aggregate["bank"]["perturbations"]] == [
        "row-a",
        "row-b",
    ]
    assert [
        row["perturbation_id"]
        for row in aggregate["runs"]["training-run-a"]["perturbations"]
    ] == ["row-a", "row-b"]
    assert aggregate["runs"]["training-run-a"]["status_counts"] == {
        "evaluated": 2,
    }
    assert aggregate["runs"]["training-run-a"]["bulk_files"] == legacy_run["bulk_files"]
    assert aggregate["runs"]["training-run-a"]["n_time_steps"] == legacy_run["n_time_steps"]


def test_perturbation_class_leaf_absent_family_fails_closed(tmp_path: Path) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    params = stamp_current_schema(
        PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
        {
            "checkpoint_bank_ref": {"kind": "CheckpointSelectionManifest", "id": "bank-a"},
            "perturbation_battery": _perturbation_bank(),
            "response_tensors": {"runs": {"training-run-a": _run_payload()}},
            "legacy_payload_mode": True,
        },
    )
    eval_manifest, eval_path = execute_evaluation_run_spec(
        EvaluationRunSpec(
            evaluation_type=PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
            training_run_ids=["training-run-a"],
            inputs=[_training_ref()],
            params=params,
        ),
        root=tmp_path,
        force=True,
    )

    with pytest.raises(AnalysisRecipeExecutionError) as excinfo:
        execute_analysis_run_spec(
            dm.perturbation_class_response_spec(
                family="missing_family",
                evaluation_manifest_id=eval_manifest.id,
                evaluation_manifest_uri=eval_path,
            ),
            root=tmp_path,
        )
    assert "missing_family" in str(excinfo.value.__cause__)
    assert "contains families" in str(excinfo.value.__cause__)


def test_legacy_perturbation_materializer_routes_to_leaf_aggregate_without_raw_outputs(
    tmp_path: Path,
) -> None:
    from rlrmp.analysis.pipelines import gru_perturbation_bank

    output_path = tmp_path / "legacy_manifest.json"
    note_path = tmp_path / "legacy_note.md"
    bulk_dir = tmp_path / "legacy_bulk"

    manifest = gru_perturbation_bank.materialize_gru_perturbation_response(
        source_experiment="unit-exp",
        result_experiment="e32c8bb",
        run_ids=("training-run-a",),
        evaluate=False,
        write_bulk_arrays=True,
        output_path=output_path,
        note_path=note_path,
        bulk_dir=bulk_dir,
        repo_root=tmp_path,
    )

    assert manifest["schema_version"] == "rlrmp.gru_perturbation_bank.v3"
    assert manifest["issue"] == "e32c8bb"
    assert manifest["source_experiment"] == "unit-exp"
    assert manifest["bank_summary"]["n_perturbations"] == len(
        manifest["bank"]["perturbations"]
    )
    assert manifest["compatibility_adapter"]["route"] == (
        "feedbax_evaluation_manifest_to_perturbation_class_leaf_aggregate"
    )
    assert manifest["compatibility_adapter"]["write_bulk_arrays_requested"] is True
    assert manifest["compatibility_adapter"]["write_bulk_arrays_effective"] is False
    assert not output_path.exists()
    assert not note_path.exists()
    assert not bulk_dir.exists()


def test_feedback_quality_lens_bundle_executes_fixture_and_groups_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    bundle = load_analysis_bundle("rlrmp/feedback_quality_lens", registry=registry)
    repo_root = tmp_path / "repo"
    feedbax_root = tmp_path / "feedbax_runs"
    run_id = "rlrmp-test-training-run:feedback-quality"
    run_manifest = TrainingRunManifest(
        id=run_id,
        job_id="feedback-quality-fixture",
        status="completed",
        metadata={
            "feedback_quality_candidate": True,
            "rlrmp_experiment": "5f70333",
        },
    )
    write_manifest(run_manifest, root=feedbax_root, index=False)
    monkeypatch.setattr(dm, "REPO_ROOT", repo_root)

    plan = dm.plan_gru_postrun_materialization(
        experiment="5f70333",
        run_ids=(run_id,),
        repo_root=repo_root,
    )
    for path, payload in (
        (
            plan.evaluation_manifest_path,
            {"schema_version": "rlrmp.gru_evaluation_diagnostics.v1", "runs": {}},
        ),
        (
            plan.perturbation_response_json_path,
            {"schema_version": "rlrmp.gru_perturbation_bank.v3", "runs": {}},
        ),
        (
            plan.feedback_ablation_json_path,
            {"schema_version": "rlrmp.gru_feedback_ablation.v1", "rows": []},
        ),
        (
            plan.objective_comparator_json_path,
            {"schema_version": "rlrmp.objective_comparator_sidecar.v6", "rows": []},
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    for path in (
        plan.perturbation_response_note_path,
        plan.feedback_ablation_note_path,
        plan.objective_comparator_note_path,
    ):
        path.write_text("# fixture\n", encoding="utf-8")

    plan.evaluation_bulk_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(plan.evaluation_bulk_dir / "unit_rollout.npz", x=np.ones((1,)))
    plan.perturbation_response_bulk_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        plan.perturbation_response_bulk_dir / "unit_perturbation.npz", x=np.ones((1,))
    )
    norm_manifest = (
        repo_root
        / "results"
        / "5f70333"
        / "notes"
        / "gru_perturbation_response_norm_plots_validation_selected_manifest.json"
    )
    norm_manifest.write_text(
        json.dumps({"schema_version": "rlrmp.gru_perturbation_response_norm_plots.v1"}) + "\n",
        encoding="utf-8",
    )
    norm_note = norm_manifest.with_name(
        "gru_perturbation_response_norm_plots_validation_selected.md"
    )
    norm_note.write_text("# norm plots\n", encoding="utf-8")
    norm_fig_dir = (
        repo_root
        / "_artifacts"
        / "5f70333"
        / "figures"
        / ("perturbation_response_norms_validation_selected")
    )
    norm_fig_dir.mkdir(parents=True, exist_ok=True)
    (norm_fig_dir / "figure.html").write_text("<html></html>\n", encoding="utf-8")

    calibration_dir = (
        repo_root
        / "_artifacts"
        / "5f70333"
        / "perturbation_open_loop_calibration"
    )
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "perturbation_open_loop_calibration.json").write_text(
        json.dumps({"schema_version": "rlrmp.perturbation_open_loop_calibration.v1"}) + "\n",
        encoding="utf-8",
    )
    (repo_root / "results" / "5f70333" / "notes").mkdir(parents=True, exist_ok=True)
    (
        repo_root / "results" / "5f70333" / "notes" / "perturbation_open_loop_calibration.md"
    ).write_text("# calibration\n", encoding="utf-8")
    (calibration_dir / "calibration_table.csv").write_text(
        "level,value\nsmall,1.0\n",
        encoding="utf-8",
    )

    execution = execute_staged_analysis_bundle(
        bundle,
        root=feedbax_root,
        run_ids=[run_id],
        issues=["af77a06"],
        fig_dump_formats=("json",),
    )

    stages = {stage.name: stage for stage in execution.stages}
    assert stages["perturbation_response"].status == "materialized"
    assert stages["response_norm_plots"].status == "materialized"
    assert stages["feedback_quality_lens"].status == "materialized"
    manifest_ref = stages["feedback_quality_lens"].manifest_refs[0]
    manifest = load_manifest(manifest_ref.uri)
    manifest_path = Path(manifest_ref.uri)
    assert manifest_path.exists()
    assert manifest.status == "completed"
    assert manifest.provenance.issues == ["af77a06"]
    stage_manifests = [
        load_manifest(stage.manifest_refs[0].uri)
        for stage in stages.values()
        if stage.manifest_refs
    ]
    roles = set().union(*(_artifact_roles(stage_manifest) for stage_manifest in stage_manifests))
    assert "rlrmp-feedback-quality-lens" in _artifact_roles(manifest)
    assert "rlrmp-feedback-quality-perturbation-response-bulk" in roles
    assert "rlrmp-feedback-quality-response-norm-figure" in roles
    assert "rlrmp-feedback-quality-perturbation-calibration-manifest" in roles

    payload_ref = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.role == "rlrmp-feedback-quality-lens"
    )
    payload = json.loads(Path(payload_ref.uri).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "rlrmp.feedback_quality_lens.v1"
    assert payload["bundle_contract"]["artifact_custody"] == "feedbax.AnalysisRunManifest"
    assert payload["outputs"]["perturbation_response"]["status"] == "materialized"
    assert payload["outputs"]["response_norm_plots"]["status"] == "materialized"
    assert payload["outputs"]["perturbation_calibration"]["status"] == "materialized"
    expected_payload = json.loads(
        (FIXTURES_DIR / "feedback_quality_lens_legacy_payload.json").read_text(
            encoding="utf-8"
        )
    )
    assert FEEDBACK_QUALITY_ALLOWED_PARITY_DIFFS == (
        "declarative_analysis",
        "bundle_contract.analysis_manifest_id",
    )
    assert _feedback_quality_parity_payload(payload) == expected_payload
    assert (
        "feedback_quality_perturbation_response_bulk"
        in payload["outputs"]["perturbation_response"]["artifact_group_ids"]
    )
    bulk_artifact = next(
        artifact
        for stage_manifest in stage_manifests
        for artifact in stage_manifest.artifacts
        if artifact.role == "rlrmp-feedback-quality-perturbation-response-bulk"
    )
    assert bulk_artifact.metadata["artifact_group"]["id"] == (
        "feedback_quality_perturbation_response_bulk"
    )
    assert load_manifest(manifest_path).id == manifest.id


def test_feedback_quality_lens_records_run_condition_skips(
    tmp_path: Path,
) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    bundle = load_analysis_bundle("rlrmp/feedback_quality_lens", registry=registry)
    stages = []
    for stage in bundle.stages:
        params = dict(stage.params)
        for name in dm.FEEDBACK_QUALITY_COMPONENT_NAMES:
            params[f"materialize_{name}"] = False
        stages.append(stage.model_copy(update={"params": params}))
    bundle = bundle.model_copy(update={"stages": stages})
    run_id = "rlrmp-test-training-run:feedback-quality-skip"
    write_manifest(
        TrainingRunManifest(
            id=run_id,
            job_id="feedback-quality-skip-fixture",
            status="completed",
            metadata={
                "feedback_quality_candidate": True,
                "rlrmp_experiment": "5f70333",
            },
        ),
        root=tmp_path,
        index=False,
    )

    execution = execute_staged_analysis_bundle(
        bundle,
        root=tmp_path,
        run_ids=[run_id],
        issues=["af77a06"],
    )

    records = {stage.name: stage for stage in execution.stages}
    assert records["evaluation_diagnostics"].status == "skipped"
    assert records["perturbation_calibration"].status == "skipped"
    assert records["response_norm_plots"].status == "skipped"
    assert records["feedback_quality_lens"].status == "skipped"
    assert records["evaluation_diagnostics"].reason is not None
    assert "run_condition evaluated false" in records["evaluation_diagnostics"].reason
    assert "materialize_evaluation_diagnostics" in records["evaluation_diagnostics"].reason


def test_gru_standard_recipe_records_opaque_certificate_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_note = tmp_path / "legacy" / "gru_standard_certificates.md"
    legacy_manifest = tmp_path / "legacy" / "gru_standard_certificates_manifest.json"

    def fake_materialize(**kwargs):
        return {
            "format": "rlrmp.cs_gru_standard_certificates.v1",
            "issue": kwargs["materializer_issue_id"],
            "source_issue": kwargs["experiment"],
            "rows": [
                {
                    "spec": {
                        "run_id": f"{kwargs['run_ids'][0]}__nominal_clean",
                        "architecture": "gru",
                    },
                    "certificate_components": [],
                }
            ],
            "summary": {"n_rows": 1},
            "failure_decomposition": {"rows": []},
        }

    def fake_write(result, *, note_path, manifest_path, regeneration_spec_path=None, repo_root):
        del regeneration_spec_path, repo_root
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("standard certificate note\n", encoding="utf-8")
        payload = {**result, "regeneration_spec": "results/unit/regeneration.json"}
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(dm, "materialize_gru_standard_result", fake_materialize)
    monkeypatch.setattr(dm, "write_gru_standard_result", fake_write)
    dm.register_certificate_analysis_recipes(replace=True)
    try:
        spec = AnalysisRunSpec(
            analysis_type=dm.GRU_STANDARD_ANALYSIS_TYPE,
            params={
                "run_ids": ["unit_run"],
                "experiment": "unitexp",
                "materializer_issue_id": "103db99",
                "load_models": False,
                "note_output": str(legacy_note),
                "manifest_output": str(legacy_manifest),
            },
        )

        manifest, path = execute_analysis_run_spec(spec, root=tmp_path, issues=["103db99"])

        assert path.exists()
        assert manifest.status == "completed"
        assert manifest.analysis_spec.inline["analysis_type"] == dm.GRU_STANDARD_ANALYSIS_TYPE
        assert manifest.summary_metrics["analysis_count"] == 1
        assert "rlrmp-bridge-standard-certificate" in _artifact_roles(manifest)
        assert "rlrmp-bridge-standard-certificate-manifest" in _artifact_roles(manifest)
        assert "rlrmp-bridge-standard-certificate-note" in _artifact_roles(manifest)
        payload_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-bridge-standard-certificate"
        )
        payload = json.loads(Path(payload_artifact.uri).read_text(encoding="utf-8"))
        assert payload["format"] == "rlrmp.cs_gru_standard_certificates.v1"
        assert payload["declarative_analysis"]["schema_owner"] == "rlrmp"
        assert legacy_manifest.exists()
        assert load_manifest(path).id == manifest.id
    finally:
        _unregister_declarative_recipes()


def test_gru_standard_recipe_consumes_evaluation_manifest_parent_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    eval_spec = EvaluationRunSpec(
        evaluation_type=CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
        training_run_ids=["unit_run"],
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id="unit_run",
                role="training_run",
                metadata={"rlrmp_experiment": "unitexp"},
            )
        ],
        params=stamp_current_schema(
            CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
            {
                "task": "center_out",
                "gru_standard_certificate": {
                    "mode": "precomputed",
                    "runs": {
                        "unit_run": {
                            "candidate_actions": [[[0.0, 0.0]]],
                            "evaluation_metadata": {"status": "fixture"},
                        }
                    },
                },
            },
        ),
    )
    eval_manifest, eval_manifest_path = execute_evaluation_run_spec(
        eval_spec,
        root=tmp_path,
        force=True,
    )

    def fake_materialize_from_states(evaluation_states, **kwargs):
        return {
            "format": "rlrmp.cs_gru_standard_certificates.v1",
            "issue": kwargs["materializer_issue_id"],
            "source_issue": kwargs["experiment"],
            "rows": [],
            "summary": {
                "n_rows": 0,
                "source_eval_manifest": evaluation_states["evaluation_manifest_id"],
            },
            "failure_decomposition": {"rows": []},
        }

    monkeypatch.setattr(
        dm,
        "materialize_gru_standard_result_from_evaluation_states",
        fake_materialize_from_states,
    )
    dm.register_certificate_analysis_recipes(replace=True)
    try:
        spec = dm.gru_standard_certificate_spec(
            run_ids=["unit_run"],
            experiment="unitexp",
            materializer_issue_id="103db99",
            evaluation_manifest_id=eval_manifest.id,
            evaluation_manifest_uri=eval_manifest_path,
            repo_root=tmp_path / "repo",
        )

        analysis_manifest, _path = execute_analysis_run_spec(spec, root=tmp_path)

        payload_ref = next(
            artifact
            for artifact in analysis_manifest.artifacts
            if artifact.role == "rlrmp-bridge-standard-certificate"
        )
        payload = json.loads(Path(payload_ref.uri).read_text(encoding="utf-8"))
        assert payload["summary"]["source_eval_manifest"] == eval_manifest.id
        assert payload["evaluation_manifest_dependency"]["manifest_id"] == eval_manifest.id
        assert analysis_manifest.inputs[0].kind == "EvaluationRunManifest"
        assert analysis_manifest.provenance.parents[0].id == eval_manifest.id
    finally:
        _unregister_declarative_recipes()


def test_gru_evaluation_recipe_groups_bulk_npz_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    output_path = tmp_path / "diagnostics.json"
    bulk_dir = tmp_path / "bulk"

    def fake_materialize(**kwargs):
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["bulk_dir"].mkdir(parents=True, exist_ok=True)
        bulk_path = kwargs["bulk_dir"] / "unit_run.npz"
        np.savez_compressed(bulk_path, command=np.array([1.0]), position=np.array([2.0]))
        manifest = {
            "schema_version": "rlrmp.gru_evaluation_diagnostics.v1",
            "issue": kwargs["experiment"],
            "scope": "post_hoc_evaluation_non_certificate_diagnostics",
            "runs": {
                "unit_run": {
                    "bulk_arrays": {
                        "path": str(bulk_path),
                        "format": "np.savez_compressed",
                        "arrays": ["command", "position"],
                    }
                }
            },
        }
        kwargs["output_path"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest

    monkeypatch.setattr(dm, "materialize_gru_evaluation_diagnostics", fake_materialize)
    dm.register_certificate_analysis_recipes(replace=True)
    try:
        spec = AnalysisRunSpec(
            analysis_type=dm.GRU_EVALUATION_DIAGNOSTICS_ANALYSIS_TYPE,
            params={
                "experiment": "unitexp",
                "run_ids": ["unit_run"],
                "repo_root": str(repo_root),
                "output_path": str(output_path),
                "bulk_dir": str(bulk_dir),
            },
        )

        manifest, path = execute_analysis_run_spec(spec, root=tmp_path, issues=["103db99"])

        assert path.exists()
        assert manifest.status == "completed"
        assert "rlrmp-gru-evaluation-diagnostics" in _artifact_roles(manifest)
        assert "rlrmp-gru-evaluation-diagnostics-manifest" in _artifact_roles(manifest)
        assert "rlrmp-gru-evaluation-diagnostics-bulk" in _artifact_roles(manifest)
        bulk_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-gru-evaluation-diagnostics-bulk"
        )
        assert bulk_artifact.logical_name == "bulk/unit_run.npz"
        assert bulk_artifact.metadata["artifact_group"]["id"] == ("gru_evaluation_diagnostics_bulk")
        assert bulk_artifact.metadata["artifact_group"]["member_role"] == "rollout_arrays"
        payload_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-gru-evaluation-diagnostics"
        )
        payload = json.loads(Path(payload_artifact.uri).read_text(encoding="utf-8"))
        assert payload["schema_version"] == "rlrmp.gru_evaluation_diagnostics.v1"
        assert payload["declarative_analysis"]["artifact_owner"] == ("feedbax.AnalysisRunManifest")
        assert load_manifest(path).id == manifest.id
    finally:
        _unregister_declarative_recipes()


def test_gru_evaluation_recipe_consumes_evaluation_manifest_parent_ref(tmp_path: Path) -> None:
    registry = ExperimentRegistry()
    rlrmp.register_experiment_package(registry)
    legacy_manifest = {
        "schema_version": "rlrmp.gru_evaluation_diagnostics.v1",
        "issue": "unitexp",
        "scope": "post_hoc_evaluation_non_certificate_diagnostics",
        "runs": {
            "unit_run": {
                "label": "Unit Run",
                "checkpoint_policy": "validation_selected_per_replicate",
            }
        },
    }
    eval_spec = EvaluationRunSpec(
        evaluation_type=CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id="unit_run",
                role="training_run",
                metadata={"rlrmp_experiment": "unitexp"},
            )
        ],
        params=stamp_current_schema(
            CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
            {
                "task": "center_out",
                "legacy_diagnostics_manifest": legacy_manifest,
            },
        ),
    )
    eval_manifest, eval_manifest_path = execute_evaluation_run_spec(
        eval_spec,
        root=tmp_path,
        force=True,
    )
    output_path = tmp_path / "diagnostics.json"
    dm.register_certificate_analysis_recipes(replace=True)
    try:
        spec = dm.gru_evaluation_diagnostics_spec(
            experiment="unitexp",
            run_ids=["unit_run"],
            output_path=output_path,
            bulk_dir=tmp_path / "bulk",
            evaluation_manifest_id=eval_manifest.id,
            evaluation_manifest_uri=eval_manifest_path,
            repo_root=tmp_path / "repo",
        )

        analysis_manifest, _path = execute_analysis_run_spec(spec, root=tmp_path)

        payload_ref = next(
            artifact
            for artifact in analysis_manifest.artifacts
            if artifact.role == "rlrmp-gru-evaluation-diagnostics"
        )
        payload = json.loads(Path(payload_ref.uri).read_text(encoding="utf-8"))
        written = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["runs"] == legacy_manifest["runs"]
        assert written["evaluation_manifest_dependency"]["manifest_id"] == eval_manifest.id
        assert analysis_manifest.inputs[0].kind == "EvaluationRunManifest"
        assert analysis_manifest.provenance.parents[0].id == eval_manifest.id
    finally:
        _unregister_declarative_recipes()


def test_output_feedback_rollout_recovery_recipe_records_manifest_and_bulk_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    note_output = repo_root / "results" / "7a459bb" / "notes" / "rollout.md"
    manifest_output = repo_root / "results" / "7a459bb" / "notes" / "rollout_manifest.json"
    artifact_output = (
        repo_root / "_artifacts" / "7a459bb" / "output_feedback_rollout_recovery" / "rollout.npz"
    )

    def fake_materialize(**kwargs):
        payload = {
            "issue": kwargs["issue_id"],
            "scope": "output-feedback bridge rollout recovery",
            "fits": [{"label": "unit__scratch"}],
            "artifact_npz_keys": ["gain", "rollout"],
            "rerun_metadata": {
                "discretization": kwargs["discretization"],
                "lane": kwargs["lane"],
            },
        }
        from rlrmp.analysis.pipelines.output_feedback_rollout_recovery import (
            RolloutRecoveryMaterialization,
        )

        return RolloutRecoveryMaterialization(
            summary=payload,
            markdown="# rollout recovery\n",
            arrays={"gain": np.ones((1,)), "rollout": np.zeros((1,))},
        )

    monkeypatch.setattr(
        dm,
        "materialize_output_feedback_rollout_recovery",
        fake_materialize,
    )
    dm.register_certificate_analysis_recipes(replace=True)
    try:
        spec = dm.output_feedback_rollout_recovery_spec(
            issue_id="7a459bb",
            discretization="jaxley",
            lane="analysis",
            note_output=note_output,
            manifest_output=manifest_output,
            artifact_output=artifact_output,
            repo_root=repo_root,
        )

        manifest, path = execute_analysis_run_spec(spec, root=tmp_path, issues=["c4416c5"])

        assert path.exists()
        assert manifest.status == "completed"
        assert manifest.analysis_spec.inline["analysis_type"] == (
            dm.OUTPUT_FEEDBACK_ROLLOUT_RECOVERY_ANALYSIS_TYPE
        )
        assert manifest.provenance.issues == ["c4416c5"]
        assert "rlrmp-output-feedback-rollout-recovery" in _artifact_roles(manifest)
        assert "rlrmp-output-feedback-rollout-recovery-note" in _artifact_roles(manifest)
        assert "rlrmp-output-feedback-rollout-recovery-bulk" in _artifact_roles(manifest)
        bulk_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-output-feedback-rollout-recovery-bulk"
        )
        assert bulk_artifact.logical_name == "bulk/output_feedback_rollout_recovery.npz"
        assert bulk_artifact.metadata["artifact_group"]["id"] == (
            "output_feedback_rollout_recovery_bulk"
        )
        assert bulk_artifact.metadata["artifact_group"]["member_role"] == (
            "rollout_recovery_arrays"
        )
        assert bulk_artifact.metadata["array_keys"] == ["gain", "rollout"]
        payload_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "rlrmp-output-feedback-rollout-recovery"
        )
        payload = json.loads(Path(payload_artifact.uri).read_text(encoding="utf-8"))
        assert payload["issue"] == "7a459bb"
        assert payload["declarative_analysis"]["schema_owner"] == "rlrmp"
        assert payload["markdown_artifact"]["role"] == (
            "rlrmp-output-feedback-rollout-recovery-note"
        )
        assert payload["bulk_artifact"]["role"] == "rlrmp-output-feedback-rollout-recovery-bulk"
        assert load_manifest(path).id == manifest.id
    finally:
        _unregister_declarative_recipes()
