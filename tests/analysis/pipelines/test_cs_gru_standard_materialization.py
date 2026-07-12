"""Tests for the 30f2313 C&S GRU standard materializer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from feedbax import TaskTrialSpec, WhereDict
from feedbax.objectives.loss import TargetSpec

import rlrmp.analysis.pipelines.cs_gru_standard_materialization as cs_standard
from rlrmp.analysis.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    MEASUREMENT_HISTORY_TO_ACTION_MAP_MISMATCH,
    OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
)
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import (
    RUN_IDS,
    _align_candidate_actions_to_reference_window,
    _controller_feedback_dim,
    _repeat_single_validation_trial,
    build_gru_standard_manifest_from_actions,
    gru_io_response_map_blocker,
    materialize_gru_standard_result,
    normalize_gru_hps,
    observation_history_covariance_from_net_inputs,
    render_gru_standard_markdown,
)
from rlrmp.analysis.pipelines.failure_decomposition import failure_diagnostic_from_standard_row


def _minimal_run_spec() -> dict[str, object]:
    return {
        "seed": 7,
        "stochastic_preset": "cs_stochastic",
        "fidelity_status": {"nn_hidden": "gru"},
        "hps": {
            "model": {
                "hidden_size": 8,
                "n_replicates": 2,
                "population_structure": {
                    "n_input_only": 0,
                    "n_input_readout": 8,
                    "n_readout_only": 0,
                    "n_recurrent_only": 0,
                },
            }
        },
    }


def _components(row: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        component["name"]: component
        for component in row["certificate_components"]  # type: ignore[index]
    }


def test_normalize_gru_hps_maps_serialized_gru_type_to_builder_default() -> None:
    hps = {"hidden_type": "equinox.nn._rnn.GRUCell", "model": {"hidden_size": 8}}

    normalized = normalize_gru_hps(hps)

    assert normalized["hidden_type"] is None
    assert hps["hidden_type"] == "equinox.nn._rnn.GRUCell"


def test_controller_feedback_dim_uses_h0_force_filter_context_shape() -> None:
    run_spec = {
        "model_summary": {
            "initial_hidden_encoder": {
                "enabled": True,
                "context_shape": [6],
            }
        },
        "hps": {
            "model": {
                "force_filter_feedback": True,
            }
        },
    }

    assert _controller_feedback_dim(run_spec) == 6
    blocker = gru_io_response_map_blocker(run_spec)
    assert "6D delayed position/velocity plus force-filter feedback" in blocker
    assert "6D-to-8D" in blocker


def test_delayed_candidate_actions_align_to_reference_movement_window() -> None:
    candidate = np.arange(2 * 90 * 2, dtype=np.float64).reshape(2, 90, 2)
    reference = np.zeros((60, 2), dtype=np.float64)
    metadata: dict[str, object] = {}

    aligned = _align_candidate_actions_to_reference_window(
        candidate,
        reference_actions=reference,
        run_spec={"delayed_reach": {"enabled": True}},
        evaluation_metadata=metadata,
    )

    assert aligned.shape == (2, 60, 2)
    np.testing.assert_array_equal(aligned, candidate[:, 30:, :])
    assert metadata["action_alignment"]["status"] == "aligned_to_delayed_movement_window"  # type: ignore[index]


def test_gru_manifest_keeps_same_coordinate_rows_not_applicable() -> None:
    reference = np.ones((2, 3, 2))
    candidate = np.zeros_like(reference)

    manifest = build_gru_standard_manifest_from_actions(
        run_id="cs_stochastic_gru__unit",
        run_spec=_minimal_run_spec(),
        training_summary={"completed_batches": 12},
        candidate_actions=candidate,
        reference_actions=reference,
        action_weight=np.broadcast_to(np.eye(2), (3, 2, 2)),
        source_issue_id="3b2af27",
    )
    row = manifest.to_payload()
    by_name = _components(row)

    assert row["spec"]["architecture"] == "gru"
    assert row["spec"]["parameters"]["certificate_mode"] == "empirical_nonlinear"
    assert row["spec"]["parameters"]["source_issue"] == "3b2af27"
    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH]["status"] == "available"
    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH]["summary"]["aggregate_mismatch_ratio"] == 1.0
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH]["status"] == "not_applicable"
    assert by_name[VALUE_POLICY_GAP]["status"] == "not_applicable"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL]["status"] == "not_applicable"
    assert by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["status"] == "missing"
    assert "3b2af27" in by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["reason"]
    assert "4D-to-8D" in by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["reason"]

    diagnostic = failure_diagnostic_from_standard_row(row, source_group="cs_stochastic_gru")
    assert diagnostic["classification"]["classification"] == "external_rollout_mismatch"
    assert diagnostic["certificate"]["response_map_mismatch"] is None
    assert diagnostic["gradient_diagnostics"]["status"] == "not_applicable"
    assert diagnostic["gain_error_decomposition"]["status"] == "not_applicable"


def test_gru_manifest_accepts_4d_observation_response_maps() -> None:
    actions = np.zeros((2, 3, 2))
    reference = np.ones_like(actions)
    candidate_map = np.zeros((2, 3, 2, 12))
    reference_map = np.ones_like(candidate_map)

    manifest = build_gru_standard_manifest_from_actions(
        run_id="cs_stochastic_gru__unit",
        run_spec=_minimal_run_spec(),
        training_summary={"completed_batches": 12},
        candidate_actions=actions,
        reference_actions=reference,
        action_weight=np.broadcast_to(np.eye(2), (3, 2, 2)),
        candidate_observation_to_action_map=candidate_map,
        reference_observation_to_action_map=reference_map,
    )
    row = manifest.to_payload()
    by_name = _components(row)

    assert row["metrics"]["io_response_map_status"] == "available_4d_observation_contract"
    assert row["metrics"]["io_response_map_blocker"] is None
    assert by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["status"] == "available"
    assert by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["summary"]["map_shape"] == [
        2,
        3,
        2,
        12,
    ]
    assert (
        by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["summary"]["aggregate_mismatch_ratio"]
        == 1.0
    )
    assert by_name[MEASUREMENT_HISTORY_TO_ACTION_MAP_MISMATCH]["status"] == "available"


def test_gru_manifest_adds_covariance_weighted_observation_response_map() -> None:
    actions = np.zeros((2, 3, 2))
    reference = np.ones_like(actions)
    candidate_map = np.zeros((2, 3, 2, 12))
    reference_map = np.ones_like(candidate_map)
    net_inputs = np.array(
        [
            [
                [100.0, 1.0, 2.0, 3.0, 4.0],
                [100.0, 5.0, 6.0, 7.0, 8.0],
                [100.0, 9.0, 10.0, 11.0, 12.0],
            ],
            [
                [200.0, 2.0, 3.0, 4.0, 5.0],
                [200.0, 6.0, 7.0, 8.0, 9.0],
                [200.0, 10.0, 11.0, 12.0, 13.0],
            ],
        ]
    )
    covariance, covariance_metadata = observation_history_covariance_from_net_inputs(
        net_inputs,
        feedback_dim=4,
        source="empirical_validation_observation_history",
    )

    manifest = build_gru_standard_manifest_from_actions(
        run_id="cs_stochastic_gru__unit",
        run_spec=_minimal_run_spec(),
        training_summary={"completed_batches": 12},
        candidate_actions=actions,
        reference_actions=reference,
        action_weight=np.broadcast_to(np.eye(2), (3, 2, 2)),
        candidate_observation_to_action_map=candidate_map,
        reference_observation_to_action_map=reference_map,
        observation_history_covariance=covariance,
        observation_history_covariance_metadata=covariance_metadata,
    )
    row = manifest.to_payload()
    summary = _components(row)[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["summary"]

    assert row["metrics"]["observation_history_covariance"]["status"] == "available"
    assert summary["aggregate_mismatch_ratio"] == 1.0
    assert summary["covariance_weighted_status"] == "available"
    assert summary["covariance_weighted_aggregate_mismatch_ratio"] == 1.0
    assert summary["covariance_weighting"]["source"] == ("empirical_validation_observation_history")
    assert summary["covariance_weighting"]["sample_count"] == 2
    assert summary["covariance_weighting"]["centering"] == "sample_mean_subtracted"
    assert summary["covariance_weighting"]["regularization"] == {
        "type": "none",
        "eigenvalue_floor": 0.0,
        "diagonal_jitter": 0.0,
        "ratio_denominator_floor": 1e-12,
    }
    assert summary["covariance_weighting"]["normalization"] == (
        "expected_squared_output_energy_ratio"
    )
    assert summary["covariance_weighting"]["future_lenses"]["perturbation_bank_covariance"] == (
        "blocked_pending_issue_3992394"
    )

    rendered = render_gru_standard_markdown(
        {
            "issue": "unit",
            "source_issue": "unit",
            "rows": [row],
            "summary": {},
            "failure_decomposition": {
                "rows": [
                    {
                        "run_id": row["spec"]["run_id"],
                        "classification": {"classification": "mixed"},
                    }
                ]
            },
        }
    )
    assert "cov-weighted obs-action" in rendered
    assert "| cs_stochastic_gru__unit__nominal_clean |" in rendered
    assert "| 1 | 1 |" in rendered


def test_gru_manifest_marks_covariance_weighted_observation_response_map_missing() -> None:
    actions = np.zeros((2, 3, 2))
    reference = np.ones_like(actions)
    candidate_map = np.zeros((2, 3, 2, 12))
    reference_map = np.ones_like(candidate_map)

    manifest = build_gru_standard_manifest_from_actions(
        run_id="cs_stochastic_gru__unit",
        run_spec=_minimal_run_spec(),
        training_summary={"completed_batches": 12},
        candidate_actions=actions,
        reference_actions=reference,
        action_weight=np.broadcast_to(np.eye(2), (3, 2, 2)),
        candidate_observation_to_action_map=candidate_map,
        reference_observation_to_action_map=reference_map,
    )
    row = manifest.to_payload()
    summary = _components(row)[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["summary"]

    assert row["metrics"]["observation_history_covariance"]["status"] == "missing"
    assert _components(row)[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["status"] == "available"
    assert summary["aggregate_mismatch_ratio"] == 1.0
    assert summary["covariance_weighted_status"] == "missing"
    assert summary["covariance_weighting"]["reason"] == (
        "sampled observation histories were not supplied"
    )
    assert "covariance_weighted_aggregate_mismatch_ratio" not in summary


def test_response_map_sampling_repeats_first_trial_from_multitarget_bank() -> None:
    trial_specs = TaskTrialSpec(
        inits=WhereDict({"mechanics.vector": jnp.arange(20 * 48).reshape(20, 48)}),
        targets=WhereDict(
            {"mechanics.effector.pos": TargetSpec(value=jnp.arange(20 * 60 * 2).reshape(20, 60, 2))}
        ),
        inputs={"target": jnp.arange(20 * 60 * 2).reshape(20, 60, 2)},
    )

    repeated = _repeat_single_validation_trial(trial_specs, 16)

    assert repeated.inits["mechanics.vector"].shape == (16, 48)
    assert repeated.targets["mechanics.effector.pos"].value.shape == (16, 60, 2)
    assert repeated.inputs["target"].shape == (16, 60, 2)
    assert jnp.all(repeated.inits["mechanics.vector"] == trial_specs.inits["mechanics.vector"][0])
    assert jnp.all(
        repeated.targets["mechanics.effector.pos"].value
        == trial_specs.targets["mechanics.effector.pos"].value[0]
    )


def test_gru_materializer_does_not_claim_action_evidence_when_models_are_not_loaded(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cs_standard,
        "resolve_run_record",
        lambda _experiment, _run_id, *, repo_root: _minimal_run_spec(),
    )

    result = materialize_gru_standard_result(run_ids=(RUN_IDS[0],), load_models=False)
    row = result["rows"][0]
    by_name = _components(row)

    assert row["status"] == "standard_certificate_missing_action_evidence"
    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH]["status"] == "missing"
    assert "not evaluated" in by_name[STATE_WEIGHTED_ACTION_MISMATCH]["reason"]
    assert "aggregate_mismatch_ratio" not in by_name[STATE_WEIGHTED_ACTION_MISMATCH]["summary"]
    assert result["failure_decomposition"]["rows"][0]["classification"]["classification"] == (
        "uncertain"
    )
