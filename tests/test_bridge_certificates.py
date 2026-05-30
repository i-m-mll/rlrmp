"""Smoke tests for bridge standard-certificate adapters."""

from __future__ import annotations

import warnings

import numpy as np

from rlrmp.analysis.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    OPTIMIZER_METADATA,
    RECURRENCE_GRU_DIAGNOSTICS,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
    VISITED_SUBSPACE_DIAGNOSTICS,
    build_standard_certificate_components,
)
from rlrmp.analysis.bridge_contracts import (
    BridgeRunManifest,
    BridgeRunSpec,
    make_bridge_run_id,
    read_bridge_manifest,
    write_bridge_manifest,
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


def test_linear_components_are_available_and_manifest_compatible(tmp_path) -> None:
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

    manifest = BridgeRunManifest(
        spec=_spec(),
        status="smoke",
        certificate_components=components,
    )
    path = tmp_path / "manifest.json"
    write_bridge_manifest(manifest, path)

    assert read_bridge_manifest(path) == manifest


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
    assert by_name[VISITED_SUBSPACE_DIAGNOSTICS].status == "available"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].status == "not_applicable"
    assert by_name[VALUE_POLICY_GAP].status == "not_applicable"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL].status == "not_applicable"
    assert by_name[RECURRENCE_GRU_DIAGNOSTICS].summary == {
        "architecture": "gru",
        "hidden_state_rms": 0.4,
    }


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
