from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import rlrmp
from feedbax.analysis.evaluation import EvaluationRecipeExecutionError, execute_evaluation_run_spec
from feedbax.contracts.manifest import EvaluationRunSpec, ParentRef, evaluation_run_manifest_id
from feedbax.plugins.registry import ExperimentRegistry
from rlrmp.analysis.matrix import STANDARD_MATRIX_EVALUATION_TYPE
from rlrmp.eval.recipes import (
    CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
    DELAYED_REACH_BANK_EVALUATION_TYPE,
    FEEDBACK_ABLATION_EVALUATION_TYPE,
    PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
    WORST_CASE_EPSILON_EVALUATION_TYPE,
)
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
                "perturbation_battery": {"scales": [0.25, 0.5]},
                "alignment_mode": "reach_locked",
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

    manifest, path = execute_evaluation_run_spec(spec, root=tmp_path, force=True)
    cached_manifest, cached_path = execute_evaluation_run_spec(spec, root=tmp_path)

    assert path == cached_path
    assert manifest.id == evaluation_run_manifest_id(spec)
    assert cached_manifest.id == manifest.id
    assert cached_manifest.metadata["cache"]["states_cache_hit"] is True
    assert cached_manifest.metadata["rlrmp_evaluation_recipe"] == evaluation_type
    assert cached_manifest.summary_metrics["input_training_runs"] == 1
    assert cached_manifest.summary_metrics["input_ref_count"] == 1
    assert cached_manifest.metadata["caching_identity"]["source"] == "EvaluationRunSpec"

    states_path = tmp_path / "cache" / "states"
    assert any(states_path.glob("*.pkl"))
    assert manifest.metadata["params_schema_version"] == stamped["schema_version"]
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

    changed_params = dict(stamped)
    changed_params["seed"] = 999
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
