from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import rlrmp
from feedbax.analysis.evaluation import EvaluationRecipeExecutionError, execute_evaluation_run_spec
from feedbax.contracts.manifest import EvaluationRunSpec, ParentRef, evaluation_run_manifest_id
from feedbax.plugins.registry import ExperimentRegistry
from pydantic import BaseModel, ValidationError
from rlrmp.analysis.matrix import STANDARD_MATRIX_EVALUATION_TYPE
from rlrmp.eval.recipes import (
    CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
    CenterOutEnsembleEvalParams,
    DELAYED_REACH_BANK_EVALUATION_TYPE,
    DelayedReachBankEvalParams,
    FEEDBACK_ABLATION_EVALUATION_TYPE,
    FeedbackAblationEvalParams,
    PerturbationResponseBankEvalParams,
    PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
    WorstCaseEpsilonEvalParams,
    WORST_CASE_EPSILON_EVALUATION_TYPE,
)
from rlrmp.runtime.params_models import params_model_for, registered_params_models
from rlrmp.runtime.spec_migrations import (
    CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
    DELAYED_REACH_BANK_EVAL_PARAMS_KIND,
    FEEDBACK_ABLATION_EVAL_PARAMS_KIND,
    PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
    STANDARD_MATRIX_EVAL_PARAMS_KIND,
    WORST_CASE_EPSILON_EVAL_PARAMS_KIND,
    stamp_current_schema,
)


def _register() -> None:
    rlrmp.register_experiment_package(ExperimentRegistry())


def _training_ref(run_id: str = "training-run-a") -> ParentRef:
    return ParentRef(kind="TrainingRunManifest", id=run_id, role="training_run")


def _spec(evaluation_type: str, params: dict[str, Any]) -> EvaluationRunSpec:
    return EvaluationRunSpec(
        evaluation_type=evaluation_type,
        training_run_ids=["training-run-a"],
        inputs=[_training_ref()],
        params=params,
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _identity_changed_params(evaluation_type: str, params: dict[str, Any]) -> dict[str, Any]:
    changed = dict(params)
    if evaluation_type == CENTER_OUT_ENSEMBLE_EVALUATION_TYPE:
        changed["seed"] = 999
    elif evaluation_type == PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE:
        changed["bank_status"] = {"training-run-a": {"changed": 1}}
    elif evaluation_type == FEEDBACK_ABLATION_EVALUATION_TYPE:
        changed["rollout_pairs"] = [{"pair": "changed"}]
    elif evaluation_type == WORST_CASE_EPSILON_EVALUATION_TYPE:
        changed["audit_inputs"] = {"changed": True}
    elif evaluation_type == DELAYED_REACH_BANK_EVALUATION_TYPE:
        changed["bank_tensors"] = {"changed": True}
    else:
        raise AssertionError(f"missing identity-change fixture for {evaluation_type}")
    return changed


def _perturbation_row(
    perturbation_id: str,
    *,
    family: str = "command_input_pulse",
) -> dict[str, Any]:
    return {
        "perturbation_id": perturbation_id,
        "channel": "command_input",
        "family": family,
        "amplitude": 1.0,
        "units": "N",
        "axis": "x",
        "basis": "command_cartesian_force_xy",
        "sign": 1,
        "timing": {"start_time_index": 1, "duration_steps": 2},
        "adapter": "feedbax.additive_channel_adapter.command_input",
        "description": "Unit test command pulse.",
        "calibration_mode": "unit_test",
        "calibration_role": "reach_relative_calibrated_open_loop",
        "level_name": "small",
        "reach_label": "test_reach",
    }


def _bank() -> dict[str, Any]:
    return {
        "bank_id": "unit_test_bank",
        "perturbations": [
            _perturbation_row("row-a", family="command_input_pulse"),
            _perturbation_row(
                "row-b",
                family="target_aligned_lateral_command_load_pulse",
            ),
        ],
    }


@pytest.mark.parametrize(
    ("model_class", "expected_defaults"),
    [
        (
            CenterOutEnsembleEvalParams,
            {
                "schema_id": None,
                "schema_version": None,
                "consumed_data_identities": [],
                "task": None,
                "checkpoint_selector": None,
                "replicate_selector": None,
                "n_directions": None,
                "n_trials_per_direction": None,
                "n_trials": None,
                "directions": None,
                "task_conditions": None,
                "perturbation": None,
                "pert_axis": None,
                "sisu_values": [],
                "seed": None,
                "trajectories": [],
                "kinematics_summary": {},
                "legacy_diagnostics_manifest": None,
                "legacy_bulk_arrays": {},
                "gru_standard_certificate": None,
            },
        ),
        (
            PerturbationResponseBankEvalParams,
            {
                "schema_id": None,
                "schema_version": None,
                "consumed_data_identities": [],
                "checkpoint_bank_ref": None,
                "checkpoint_bank": None,
                "perturbation_battery": None,
                "bank": None,
                "alignment_mode": "reach_locked",
                "response_tensors": None,
                "class_index_map": None,
                "bank_status": {},
                "bundle_contract": {},
                "states_custody": None,
                "legacy_payload_mode": False,
                "source_experiment": None,
                "experiment": None,
                "run_ids": None,
                "labels": None,
                "class_set": None,
                "families": None,
                "family_set": None,
                "perturbation_families": None,
                "perturbation_ids": None,
                "consume_open_loop_calibration": False,
                "bank_mode": None,
                "mode": "raw",
                "calibration_level": None,
                "calibration_reach": None,
                "feedback_scale_manifest": None,
                "feedback_scale_manifest_path": None,
                "repo_root": None,
                "bulk_dir": None,
                "write_bulk_arrays": False,
                "n_rollout_trials": 8,
                "extlqg_physical_dim": 8,
                "preferred_checkpoint_manifest_path": None,
                "checkpoint_selection_mode": "sparse_history",
            },
        ),
        (
            FeedbackAblationEvalParams,
            {
                "schema_id": None,
                "schema_version": None,
                "consumed_data_identities": [],
                "ablation_masks": None,
                "ablation_mask_set": None,
                "base_task": {},
                "rollout_pairs": [],
            },
        ),
        (
            WorstCaseEpsilonEvalParams,
            {
                "schema_id": None,
                "schema_version": None,
                "consumed_data_identities": [],
                "epsilon_budget_data_product_identity": None,
                "epsilon_budget_identity": None,
                "optimizer": {},
                "audit_inputs": {},
                "worst_case_rollouts": [],
            },
        ),
        (
            DelayedReachBankEvalParams,
            {
                "schema_id": None,
                "schema_version": None,
                "consumed_data_identities": [],
                "bank_spec": {},
                "bank_tensors": {},
                "selection_inputs": {},
            },
        ),
    ],
)
def test_eval_params_model_defaults_match_recipe_literals(
    model_class: type[BaseModel],
    expected_defaults: dict[str, Any],
) -> None:
    model = model_class.model_validate({})

    assert model.model_dump(mode="json") == expected_defaults


@pytest.mark.parametrize(
    "model_class",
    [
        CenterOutEnsembleEvalParams,
        PerturbationResponseBankEvalParams,
        FeedbackAblationEvalParams,
        WorstCaseEpsilonEvalParams,
        DelayedReachBankEvalParams,
    ],
)
def test_eval_params_models_reject_extra_fields(model_class: type[BaseModel]) -> None:
    with pytest.raises(ValidationError):
        model_class.model_validate({"unknown": True})


def test_eval_params_model_table_resolves_registered_recipes() -> None:
    _register()

    expected = {
        CENTER_OUT_ENSEMBLE_EVALUATION_TYPE: CenterOutEnsembleEvalParams,
        PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE: PerturbationResponseBankEvalParams,
        FEEDBACK_ABLATION_EVALUATION_TYPE: FeedbackAblationEvalParams,
        WORST_CASE_EPSILON_EVALUATION_TYPE: WorstCaseEpsilonEvalParams,
        DELAYED_REACH_BANK_EVALUATION_TYPE: DelayedReachBankEvalParams,
    }
    for recipe_name, model_class in expected.items():
        assert params_model_for(recipe_name) is model_class
    assert expected.items() <= registered_params_models().items()
    with pytest.raises(KeyError):
        params_model_for("rlrmp.eval.unknown")


@pytest.mark.parametrize(
    ("evaluation_type", "params_kind", "params", "product_role"),
    [
        (
            CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
            CENTER_OUT_ENSEMBLE_EVAL_PARAMS_KIND,
            {
                "task": "center_out",
                "n_directions": 8,
                "n_trials_per_direction": 2,
                "replicate_selector": {"replicates": [0, 1]},
                "perturbation": {"type": "scale_sweep", "scales": [0.0, 1.0]},
                "sisu_values": [0.0, 0.5],
                "seed": 11,
            },
            "center_out_ensemble_states",
        ),
        (
            PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
            PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
            {
                "checkpoint_bank_ref": {"kind": "CheckpointSelectionManifest", "id": "bank-a"},
                "perturbation_battery": _bank(),
                "alignment_mode": "reach_locked",
                "response_tensors": {"runs": {}},
                "legacy_payload_mode": True,
            },
            "perturbation_response_bank",
        ),
        (
            FEEDBACK_ABLATION_EVALUATION_TYPE,
            FEEDBACK_ABLATION_EVAL_PARAMS_KIND,
            {
                "ablation_masks": [{"name": "no_feedback", "channels": ["vision"]}],
                "base_task": {"task": "center_out"},
            },
            "feedback_ablation_rollouts",
        ),
        (
            WORST_CASE_EPSILON_EVALUATION_TYPE,
            WORST_CASE_EPSILON_EVAL_PARAMS_KIND,
            {
                "epsilon_budget_data_product_identity": {
                    "product_schema_id": "rlrmp.broad_epsilon_anchors",
                    "product_identity_hash": "sha256:unit",
                },
                "optimizer": {"steps": 5, "step_size": 0.1},
                "consumed_data_identities": [{"product_identity_hash": "sha256:unit"}],
            },
            "worst_case_epsilon_rollouts",
        ),
        (
            DELAYED_REACH_BANK_EVALUATION_TYPE,
            DELAYED_REACH_BANK_EVAL_PARAMS_KIND,
            {
                "bank_spec": {"task": "delayed_reach", "delays": [0, 250]},
                "selection_inputs": {"metric": "validation_loss"},
            },
            "delayed_reach_eval_bank",
        ),
    ],
)
def test_registered_eval_recipes_execute_and_reuse_states_cache(
    tmp_path: Path,
    evaluation_type: str,
    params_kind: str,
    params: dict[str, Any],
    product_role: str,
) -> None:
    _register()
    stamped = stamp_current_schema(params_kind, params)
    spec = _spec(evaluation_type, stamped)
    expected_manifest_id = evaluation_run_manifest_id(spec)
    expected_params_sha = _canonical_sha256(stamped)

    manifest, path = execute_evaluation_run_spec(spec, root=tmp_path, force=True)
    cached_manifest, cached_path = execute_evaluation_run_spec(spec, root=tmp_path)

    assert path == cached_path
    assert manifest.id == expected_manifest_id
    assert cached_manifest.id == manifest.id
    assert cached_manifest.metadata["cache"]["states_cache_hit"] is True
    assert cached_manifest.metadata["rlrmp_evaluation_recipe"] == evaluation_type
    assert cached_manifest.summary_metrics["input_training_runs"] == 1
    assert cached_manifest.summary_metrics["input_ref_count"] == 1
    assert cached_manifest.metadata["caching_identity"]["source"] == "EvaluationRunSpec"

    states_path = tmp_path / "cache" / "states"
    assert any(states_path.glob("*.pkl"))
    assert manifest.metadata["params_schema_version"] == stamped["schema_version"]
    assert manifest.metadata["params_sha256"] == expected_params_sha
    assert manifest.metadata["cache"]["states_cache_key"] == manifest.id
    assert manifest.summary_metrics["consumed_data_identity_count"] == len(
        stamped.get("consumed_data_identities", [])
    )
    assert manifest.metadata["params_sha256"]
    assert manifest.metadata["caching_identity"]["manifest_id"] == manifest.id
    assert manifest.status == "completed"
    assert manifest.summary_metrics["training_run_id_count"] == 1
    assert manifest.metadata["params_schema_id"] == stamped["schema_id"]
    assert manifest.input_training_runs == [_training_ref()]
    assert manifest.summary_metrics["input_ref_count"] == len(manifest.input_training_runs)
    assert manifest.metadata["cache"]["states_cache_hit"] is False
    assert manifest.metadata["cache"]["states_cache_saved"] is True

    changed_params = _identity_changed_params(evaluation_type, stamped)
    changed_spec = _spec(evaluation_type, changed_params)
    changed_manifest, _changed_path = execute_evaluation_run_spec(
        changed_spec,
        root=tmp_path,
        force=True,
    )
    assert changed_manifest.id != manifest.id
    assert changed_manifest.summary_metrics["input_ref_count"] == 1
    assert changed_manifest.metadata["product_role"] == product_role


def test_standard_matrix_legacy_payload_requires_explicit_mode(tmp_path: Path) -> None:
    _register()
    payload = {"cells": [{"run_id": "cell-a", "summary_metrics": {"velocity_rmse": 0.1}}]}
    params = stamp_current_schema(
        STANDARD_MATRIX_EVAL_PARAMS_KIND,
        {"matrix_payload": payload},
    )
    spec = EvaluationRunSpec(
        evaluation_type=STANDARD_MATRIX_EVALUATION_TYPE,
        inputs=[_training_ref()],
        params=params,
    )

    with pytest.raises(EvaluationRecipeExecutionError) as excinfo:
        execute_evaluation_run_spec(spec, root=tmp_path, force=True)
    assert "legacy_payload_mode=true" in str(excinfo.value.__cause__)
    assert excinfo.value.manifest.status == "failed"

    explicit = spec.model_copy(
        update={"params": {**params, "legacy_payload_mode": True}},
        deep=True,
    )
    manifest, _path = execute_evaluation_run_spec(explicit, root=tmp_path, force=True)

    assert manifest.status == "completed"
    assert manifest.summary_metrics["standard_matrix_cells"] == 1
    assert manifest.metadata["legacy_payload_mode"] is True


def test_standard_matrix_model_ref_mode_does_not_need_legacy_payload(tmp_path: Path) -> None:
    _register()
    params = stamp_current_schema(
        STANDARD_MATRIX_EVAL_PARAMS_KIND,
        {"cell_metadata": {"training-run-a": {"display_name": "Cell A"}}},
    )
    spec = EvaluationRunSpec(
        evaluation_type=STANDARD_MATRIX_EVALUATION_TYPE,
        inputs=[_training_ref()],
        params=params,
    )

    manifest, _path = execute_evaluation_run_spec(spec, root=tmp_path, force=True)

    assert manifest.status == "completed"
    assert manifest.summary_metrics["standard_matrix_cells"] == 1
    assert manifest.metadata["legacy_payload_mode"] is False


def test_perturbation_response_bank_legacy_payload_requires_explicit_mode(
    tmp_path: Path,
) -> None:
    _register()
    params = stamp_current_schema(
        PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
        {
            "checkpoint_bank_ref": {"kind": "CheckpointSelectionManifest", "id": "bank-a"},
            "perturbation_battery": _bank(),
            "response_tensors": {"runs": {}},
        },
    )
    spec = _spec(PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE, params)

    with pytest.raises(EvaluationRecipeExecutionError) as excinfo:
        execute_evaluation_run_spec(spec, root=tmp_path, force=True)
    assert "legacy_payload_mode=true" in str(excinfo.value.__cause__)
    assert excinfo.value.manifest.status == "failed"

    explicit = spec.model_copy(
        update={"params": {**params, "legacy_payload_mode": True}},
        deep=True,
    )
    manifest, _path = execute_evaluation_run_spec(explicit, root=tmp_path, force=True)

    assert manifest.status == "completed"
    assert manifest.metadata["legacy_payload_mode"] is True
    assert manifest.summary_metrics["perturbation_family_count"] == 2


def test_perturbation_response_bank_model_driven_emits_class_index_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlrmp.eval.recipes as eval_recipes

    _register()

    def fake_resolve(**_kwargs: Any) -> list[Any]:
        return [SimpleNamespace(run_id="training-run-a", label="Run A")]

    def fake_evaluate(run: Any, **kwargs: Any) -> dict[str, Any]:
        assert run.run_id == "training-run-a"
        assert [row["perturbation_id"] for row in kwargs["bank"]["perturbations"]] == [
            "row-a"
        ]
        return {
            "label": "Run A",
            "status_counts": {"evaluated": 1},
            "perturbations": [{"perturbation_id": "row-a", "status": "evaluated"}],
        }

    monkeypatch.setattr(eval_recipes, "_resolve_perturbation_run_inputs", fake_resolve)
    monkeypatch.setattr(eval_recipes, "_evaluate_single_perturbation_bank_run", fake_evaluate)
    params = stamp_current_schema(
        PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
        {
            "source_experiment": "unit-exp",
            "run_ids": ["training-run-a"],
            "perturbation_battery": _bank(),
            "class_set": ["command_input_pulse"],
        },
    )
    spec = _spec(PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE, params)

    manifest, _path = execute_evaluation_run_spec(spec, root=tmp_path, force=True)
    with Path(manifest.metadata["cache"]["states_path"]).open("rb") as stream:
        states = pickle.load(stream)

    class_index = states["class_index_map"]
    assert manifest.metadata["legacy_payload_mode"] is False
    assert manifest.metadata["production_mode"] == "model_driven"
    assert manifest.summary_metrics["perturbation_family_count"] == 1
    assert manifest.summary_metrics["perturbation_row_count"] == 1
    assert manifest.summary_metrics["evaluated_run_count"] == 1
    assert states["response_tensors"]["runs"]["training-run-a"]["status_counts"] == {
        "evaluated": 1
    }
    assert list(class_index) == ["command_input_pulse"]
    assert class_index["command_input_pulse"]["perturbation_ids"] == ["row-a"]
    assert class_index["command_input_pulse"]["tensor_slices"] == {
        "axis": "perturbation",
        "indices": [0],
        "contiguous": True,
        "start": 0,
        "stop": 1,
    }
    assert class_index["command_input_pulse"]["calibration_provenance"]["row-a"][
        "calibration_mode"
    ] == "unit_test"


def test_perturbation_response_bank_stamps_eval_time_calibration_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlrmp.eval.recipes as eval_recipes

    _register()
    monkeypatch.setattr(
        eval_recipes,
        "_consumed_open_loop_calibration_identity",
        lambda: {
            "role": "perturbation_open_loop_calibration",
            "schema": "rlrmp.perturbation_open_loop_calibration.v2",
            "hash": "sha256:unit-calibration",
        },
    )
    monkeypatch.setattr(
        eval_recipes,
        "_resolve_perturbation_run_inputs",
        lambda **_kwargs: [SimpleNamespace(run_id="training-run-a", label="Run A")],
    )
    monkeypatch.setattr(
        eval_recipes,
        "_evaluate_single_perturbation_bank_run",
        lambda run, **_kwargs: {"label": run.label, "status_counts": {"evaluated": 1}},
    )
    params = stamp_current_schema(
        PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
        {
            "source_experiment": "unit-exp",
            "run_ids": ["training-run-a"],
            "bank_mode": "calibrated",
            "perturbation_battery": {"perturbations": [_perturbation_row("row-a")]},
        },
    )
    spec = _spec(PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE, params)

    manifest, _path = execute_evaluation_run_spec(spec, root=tmp_path, force=True)
    with Path(manifest.metadata["cache"]["states_path"]).open("rb") as stream:
        states = pickle.load(stream)

    assert states["consumed_data_identities"] == [
        {
            "role": "perturbation_open_loop_calibration",
            "schema": "rlrmp.perturbation_open_loop_calibration.v2",
            "hash": "sha256:unit-calibration",
        }
    ]
    assert manifest.summary_metrics["consumed_data_identity_count"] == 1


def test_perturbation_response_bank_identity_differs_by_class_set() -> None:
    params = stamp_current_schema(
        PERTURBATION_RESPONSE_BANK_EVAL_PARAMS_KIND,
        {
            "source_experiment": "unit-exp",
            "run_ids": ["training-run-a"],
            "perturbation_battery": _bank(),
            "class_set": ["command_input_pulse"],
        },
    )
    other_params = {
        **params,
        "class_set": ["target_aligned_lateral_command_load_pulse"],
    }

    assert evaluation_run_manifest_id(
        _spec(PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE, params)
    ) != evaluation_run_manifest_id(
        _spec(PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE, other_params)
    )
