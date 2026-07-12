"""Smoke tests for bridge standard-certificate adapters."""

from __future__ import annotations

import warnings

import numpy as np

from rlrmp.analysis.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    DISTURBANCE_HISTORY_TO_COST_QUADRATIC,
    DISTURBANCE_HISTORY_TO_ACTION_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_OUTPUT_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_STATE_MAP_MISMATCH,
    MEASUREMENT_HISTORY_TO_ACTION_MAP_MISMATCH,
    MEASUREMENT_HISTORY_TO_OUTPUT_MAP_MISMATCH,
    OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH,
    OPTIMIZER_METADATA,
    RECURRENCE_GRU_DIAGNOSTICS,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
    VISITED_SUBSPACE_DIAGNOSTICS,
    action_energy_mismatch_summary,
    build_standard_certificate_components,
)
from rlrmp.analysis.bridge_results import (
    BridgeAnalysisResult,
    BridgeRunSpec,
    make_bridge_run_id,
)


def _spec() -> BridgeRunSpec:
    return BridgeRunSpec(
        issue_id="735c87f",
        run_id=make_bridge_run_id("certificate", "smoke"),
        objective="diagnostic",
        architecture="free_time_varying",
        controller_label="candidate",
        optimizer_label="lbfgsb",
        training_distribution="nominal",
        evaluation_lane="diagnostic",
        reference_controller="analytical_lqr",
        parameters={"horizon": 2},
    )


def _linear_fixture() -> dict[str, np.ndarray]:
    states = np.asarray(
        [
            [[1.0, 0.0], [0.5, 0.2], [0.1, 0.3]],
            [[0.0, 1.0], [0.1, 0.4], [0.2, 0.2]],
        ]
    )
    reference_gain = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]])
    candidate_gain = reference_gain + np.asarray([[[0.1, 0.0]], [[0.0, -0.1]]])
    reference_transition = np.asarray(
        [
            [[0.8, 0.0], [0.0, 0.9]],
            [[0.7, 0.1], [0.0, 0.8]],
        ]
    )
    candidate_transition = reference_transition + 0.01 * np.eye(2)[None]
    reference_value = np.asarray(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.5, 0.0], [0.0, 0.8]],
            [[1.0, 0.0], [0.0, 0.5]],
        ]
    )
    candidate_value = reference_value + 0.05 * np.eye(2)[None]
    return {
        "states": states,
        "reference_gain": reference_gain,
        "candidate_gain": candidate_gain,
        "reference_transition": reference_transition,
        "candidate_transition": candidate_transition,
        "reference_value": reference_value,
        "candidate_value": candidate_value,
        "action_weight": np.asarray([[2.0]]),
        "bellman_hessian": np.asarray([[[3.0]], [[2.0]]]),
    }


def test_linear_components_are_available_and_result_compatible() -> None:
    fixture = _linear_fixture()

    components = build_standard_certificate_components(
        architecture="free_time_varying",
        states=fixture["states"],
        candidate_gain=fixture["candidate_gain"],
        reference_gain=fixture["reference_gain"],
        action_weight=fixture["action_weight"],
        candidate_transition=fixture["candidate_transition"],
        reference_transition=fixture["reference_transition"],
        candidate_value_matrices=fixture["candidate_value"],
        reference_value_matrices=fixture["reference_value"],
        bellman_hessian=fixture["bellman_hessian"],
        optimizer_metadata={"final_gradient_norm": 0.01, "steps": 12},
    )
    by_name = {component.name: component for component in components}

    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH].status == "available"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].status == "available"
    assert by_name[VALUE_POLICY_GAP].status == "available"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL].status == "available"
    assert by_name[VISITED_SUBSPACE_DIAGNOSTICS].status == "available"
    assert by_name[OPTIMIZER_METADATA].summary["final_gradient_norm"] == 0.01
    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH].summary["mismatch_ratio_mean"] > 0.0
    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH].summary["aggregate_mismatch_ratio"] > 0.0

    result = BridgeAnalysisResult(
        spec=_spec(),
        status="smoke",
        certificate_components=components,
    )
    assert len(result.to_payload()["certificate_components"]) == len(components)


def test_missing_rows_are_explicit_when_inputs_are_absent() -> None:
    components = build_standard_certificate_components(architecture="free_time_varying")

    assert {component.name for component in components} == {
        STATE_WEIGHTED_ACTION_MISMATCH,
        VISITED_SUBSPACE_DIAGNOSTICS,
        OPTIMIZER_METADATA,
        CLOSED_LOOP_TRANSITION_MISMATCH,
        VALUE_POLICY_GAP,
        BELLMAN_HESSIAN_RESIDUAL,
    }
    assert {component.status for component in components} == {"missing"}


def test_gru_bundle_marks_formal_linear_components_not_applicable() -> None:
    fixture = _linear_fixture()
    candidate_actions = np.zeros((2, 2, 1))
    reference_actions = np.ones((2, 2, 1))

    components = build_standard_certificate_components(
        architecture="gru",
        states=fixture["states"],
        candidate_actions=candidate_actions,
        reference_actions=reference_actions,
        recurrence_diagnostics={"hidden_state_rms": 0.4},
    )
    by_name = {component.name: component for component in components}

    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH].status == "available"
    assert by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH].status == "missing"
    assert by_name[DISTURBANCE_HISTORY_TO_ACTION_MAP_MISMATCH].status == "missing"
    assert by_name[DISTURBANCE_HISTORY_TO_STATE_MAP_MISMATCH].status == "missing"
    assert by_name[VISITED_SUBSPACE_DIAGNOSTICS].status == "available"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].status == "not_applicable"
    assert by_name[VALUE_POLICY_GAP].status == "not_applicable"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL].status == "not_applicable"
    assert by_name[RECURRENCE_GRU_DIAGNOSTICS].summary == {
        "architecture": "gru",
        "hidden_state_rms": 0.4,
    }


def test_linear_recurrence_uses_augmented_certificate_when_inputs_are_available() -> None:
    fixture = _linear_fixture()
    hidden = 0.25 * fixture["states"]
    augmented_states = np.concatenate([fixture["states"], hidden], axis=-1)
    reference_gain = np.concatenate(
        [fixture["reference_gain"], np.zeros((2, 1, 2))],
        axis=-1,
    )
    candidate_gain = reference_gain.copy()
    candidate_gain[:, :, 2:] = 0.05
    reference_transition = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    candidate_transition = reference_transition.copy()
    candidate_transition[:, 0, 2] = 0.1
    reference_value = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    candidate_value = reference_value + 0.01 * np.eye(4)[None]
    bellman_hessian = np.broadcast_to(np.eye(1), (2, 1, 1)).copy()

    components = build_standard_certificate_components(
        architecture="linear_recurrence",
        augmented_states=augmented_states,
        candidate_augmented_action_sensitivity=candidate_gain,
        reference_augmented_action_sensitivity=reference_gain,
        candidate_transition=candidate_transition,
        reference_transition=reference_transition,
        candidate_value_matrices=candidate_value,
        reference_value_matrices=reference_value,
        bellman_hessian=bellman_hessian,
        recurrence_diagnostics={"linear_recurrence": True},
    )
    by_name = {component.name: component for component in components}

    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH].status == "available"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].status == "available"
    assert by_name[VALUE_POLICY_GAP].status == "available"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL].status == "available"
    assert by_name[RECURRENCE_GRU_DIAGNOSTICS].summary["certificate_mode"] == "augmented_linear"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].summary["state_label"] == "augmented_state"


def test_linear_recurrence_without_augmented_inputs_keeps_static_rows_not_applicable() -> None:
    fixture = _linear_fixture()

    components = build_standard_certificate_components(
        architecture="linear_recurrence",
        states=fixture["states"],
        candidate_gain=fixture["candidate_gain"],
        reference_gain=fixture["reference_gain"],
    )
    by_name = {component.name: component for component in components}

    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH].status == "available"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].status == "not_applicable"
    assert by_name[VALUE_POLICY_GAP].status == "not_applicable"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL].status == "not_applicable"


def test_recurrent_rows_report_available_io_map_components() -> None:
    fixture = _linear_fixture()
    reference_observation_action = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.5, 0.5, 0.0, 0.0]],
        ]
    )
    candidate_observation_action = reference_observation_action.copy()
    candidate_observation_action[1, 0, 1] += 0.5
    reference_disturbance_action = np.asarray([[[0.2, 0.0]], [[0.0, 0.1]]])
    candidate_disturbance_action = reference_disturbance_action + 0.05
    reference_disturbance_state = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.1], [0.0, 0.5]],
        ]
    )
    candidate_disturbance_state = reference_disturbance_state.copy()
    candidate_disturbance_state[:, 0, 1] += 0.1

    components = build_standard_certificate_components(
        architecture="gru",
        states=fixture["states"],
        candidate_actions=np.zeros((2, 2, 1)),
        reference_actions=np.zeros((2, 2, 1)),
        candidate_observation_to_action_map=candidate_observation_action,
        reference_observation_to_action_map=reference_observation_action,
        observation_history_covariance=np.eye(4),
        candidate_disturbance_to_action_map=candidate_disturbance_action,
        reference_disturbance_to_action_map=reference_disturbance_action,
        candidate_disturbance_to_state_map=candidate_disturbance_state,
        reference_disturbance_to_state_map=reference_disturbance_state,
        disturbance_history_covariance=np.eye(2),
    )
    by_name = {component.name: component for component in components}

    observation_map = by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]
    disturbance_action_map = by_name[DISTURBANCE_HISTORY_TO_ACTION_MAP_MISMATCH]
    disturbance_state_map = by_name[DISTURBANCE_HISTORY_TO_STATE_MAP_MISMATCH]
    assert observation_map.status == "available"
    assert observation_map.summary["input_label"] == "observation_history"
    assert observation_map.summary["output_label"] == "action"
    assert observation_map.summary["map_shape"] == [2, 1, 4]
    assert observation_map.summary["covariance_weighted_mismatch_ratio_mean"] > 0.0
    assert disturbance_action_map.status == "available"
    assert disturbance_action_map.summary["input_label"] == "disturbance_history"
    assert disturbance_state_map.status == "available"
    assert disturbance_state_map.summary["output_label"] == "state"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].status == "not_applicable"
    assert by_name[VALUE_POLICY_GAP].status == "not_applicable"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL].status == "not_applicable"


def test_recurrent_rows_report_measurement_history_input_output_maps() -> None:
    fixture = _linear_fixture()
    reference_measurement_action = np.asarray([[[1.0, 0.0]], [[0.25, 0.75]]])
    candidate_measurement_action = reference_measurement_action.copy()
    candidate_measurement_action[1, 0, 1] += 0.2
    reference_measurement_output = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.5]],
        ]
    )
    candidate_measurement_output = reference_measurement_output.copy()
    candidate_measurement_output[:, 0, 1] += 0.1
    reference_disturbance_output = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]])
    candidate_disturbance_output = reference_disturbance_output.copy()
    candidate_disturbance_output[0, 0, 1] = 0.25

    components = build_standard_certificate_components(
        architecture="gru",
        states=fixture["states"],
        candidate_actions=np.zeros((2, 2, 1)),
        reference_actions=np.zeros((2, 2, 1)),
        candidate_measurement_to_action_map=candidate_measurement_action,
        reference_measurement_to_action_map=reference_measurement_action,
        candidate_measurement_to_output_map=candidate_measurement_output,
        reference_measurement_to_output_map=reference_measurement_output,
        measurement_history_covariance=np.eye(2),
        candidate_disturbance_to_output_map=candidate_disturbance_output,
        reference_disturbance_to_output_map=reference_disturbance_output,
        disturbance_history_covariance=np.eye(2),
    )
    by_name = {component.name: component for component in components}

    measurement_action = by_name[MEASUREMENT_HISTORY_TO_ACTION_MAP_MISMATCH]
    measurement_output = by_name[MEASUREMENT_HISTORY_TO_OUTPUT_MAP_MISMATCH]
    disturbance_output = by_name[DISTURBANCE_HISTORY_TO_OUTPUT_MAP_MISMATCH]
    assert measurement_action.status == "available"
    assert measurement_action.summary["input_label"] == "measurement_history"
    assert measurement_action.summary["output_label"] == "action"
    assert measurement_action.summary["response_map_schema"] == "finite_horizon_linear_v1"
    assert measurement_action.summary["aggregate_mismatch_ratio"] > 0.0
    assert measurement_output.status == "available"
    assert measurement_output.summary["output_label"] == "external_output"
    assert disturbance_output.status == "available"
    assert disturbance_output.summary["input_label"] == "disturbance_history"


def test_augmented_recurrent_reports_disturbance_to_cost_quadratic_sidecar() -> None:
    fixture = _linear_fixture()
    hidden = 0.25 * fixture["states"]
    augmented_states = np.concatenate([fixture["states"], hidden], axis=-1)
    reference_disturbance_action = np.asarray([[[0.0, 1.0]]])
    candidate_disturbance_action = reference_disturbance_action.copy()
    reference_disturbance_state = np.asarray([[[0.0, 0.0]], [[1.0, 0.0]]])
    candidate_disturbance_state = np.asarray([[[0.0, 0.0]], [[2.0, 0.0]]])

    components = build_standard_certificate_components(
        architecture="linear_recurrence",
        certificate_mode="augmented_linear",
        augmented_states=augmented_states[:, :2],
        candidate_actions=np.zeros((2, 1, 1)),
        reference_actions=np.zeros((2, 1, 1)),
        candidate_disturbance_to_action_map=candidate_disturbance_action,
        reference_disturbance_to_action_map=reference_disturbance_action,
        candidate_disturbance_to_state_map=candidate_disturbance_state,
        reference_disturbance_to_state_map=reference_disturbance_state,
        disturbance_state_cost=np.eye(1),
        disturbance_action_cost=np.eye(1),
        disturbance_history_covariance=np.eye(2),
    )
    by_name = {component.name: component for component in components}

    sidecar = by_name[DISTURBANCE_HISTORY_TO_COST_QUADRATIC]
    assert sidecar.status == "available"
    assert sidecar.summary["sidecar_type"] == "disturbance_to_cost_quadratic"
    assert sidecar.summary["quadratic_map_shape"] == [2, 2]
    assert sidecar.summary["candidate_expected_cost"] == 5.0
    assert sidecar.summary["reference_expected_cost"] == 2.0
    assert sidecar.summary["aggregate_mismatch_ratio"] > 0.0


def test_action_energy_summary_reports_aggregate_ratio_separately() -> None:
    candidate = np.asarray([[[2.0]], [[1.0]]])
    reference = np.asarray([[[1.0]], [[1.0]]])

    summary = action_energy_mismatch_summary(
        candidate_actions=candidate,
        reference_actions=reference,
    )

    assert summary["mismatch_ratio_mean"] == 0.5
    assert summary["aggregate_mismatch_ratio"] == 0.5
    assert summary["aggregate_delta_energy"] == 1.0
    assert summary["aggregate_reference_energy"] == 2.0


def test_output_feedback_linear_core_uses_coupled_and_estimated_state_labels() -> None:
    fixture = _linear_fixture()
    xhat = fixture["states"]
    coupled_states = np.concatenate([fixture["states"], 0.5 * fixture["states"]], axis=-1)
    reference_transition = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    candidate_transition = reference_transition.copy()
    candidate_transition[:, 0, 2] = 0.05
    reference_value = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    candidate_value = reference_value + 0.02 * np.eye(4)[None]

    components = build_standard_certificate_components(
        architecture="time_constrained_free_gain",
        states=coupled_states,
        action_states=xhat,
        candidate_gain=fixture["candidate_gain"],
        reference_gain=fixture["reference_gain"],
        candidate_transition=candidate_transition,
        reference_transition=reference_transition,
        candidate_value_matrices=candidate_value,
        reference_value_matrices=reference_value,
        bellman_hessian=fixture["bellman_hessian"],
        state_label="coupled_state",
        action_state_label="estimated_state",
    )
    by_name = {component.name: component for component in components}

    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH].summary["state_label"] == "estimated_state"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].summary["state_label"] == "coupled_state"
    assert by_name[VALUE_POLICY_GAP].summary["state_label"] == "coupled_state"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL].summary["state_label"] == "estimated_state"
    assert all(component.status == "available" for component in components[:2])


def test_visited_subspace_handles_zero_singular_values_without_warning() -> None:
    states = np.zeros((1, 3, 4))
    states[0, :, 0] = np.asarray([1.0, 0.5, 0.25])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        components = build_standard_certificate_components(
            architecture="free_time_varying",
            states=states,
        )

    by_name = {component.name: component for component in components}
    assert by_name[VISITED_SUBSPACE_DIAGNOSTICS].status == "available"
    assert by_name[VISITED_SUBSPACE_DIAGNOSTICS].summary["mean_effective_rank"] == 1.0
