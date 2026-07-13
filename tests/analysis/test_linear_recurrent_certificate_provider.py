"""Focused tests for the canonical linear-recurrent certificate provider."""

from __future__ import annotations

import numpy as np
import pytest
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.contracts.graphs.serialization import spec_to_graph
from feedbax.runtime.iteration import run_component

from rlrmp.analysis.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
)
from rlrmp.analysis.standard_certificate import (
    component_by_name,
    materialize_evaluation_standard_certificate_rows,
)
from rlrmp.eval.linear_recurrent_certificate import (
    LINEAR_RECURRENT_AUGMENTED_PROVIDER,
    linear_recurrent_augmented_component_kwargs,
)
from rlrmp.eval.recipes import register_rlrmp_evaluation_recipes


def _provider_inputs() -> dict[str, object]:
    identity_3 = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    return {
        "cached_evaluation": {
            "controller_visible_coupled_states": [[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]],
            "target_coupled_states": [1.0, 2.0],
            "hidden_states": [[[0.0], [0.25], [0.5]]],
            "action_weight": [[2.0]],
        },
        "trained_controller": {
            "input_weight": [[2.0, 3.0]],
            "recurrent_weight": [[0.5]],
            "readout_weight": [[4.0]],
            "use_bias": False,
            "readout_use_bias": False,
            "use_noise": False,
            "dt": 0.1,
            "tau": 0.2,
            "architecture": "linear_recurrence",
            "component_type": "VanillaRNN",
            "activation": "identity",
        },
        "dynamics": {
            "state_transition": [[1.0, 0.1], [0.0, 1.0]],
            "action_input": [[0.0], [0.2]],
            "controller_observation_map": [[1.0, 0.0], [0.0, 1.0]],
        },
        "reference": {
            "reference_augmented_action_sensitivity": [
                [[0.0, 0.0, 3.5]],
                [[0.0, 0.0, 3.5]],
            ],
            "reference_transition": identity_3,
            "candidate_value_matrices": 1.1 * identity_3,
            "reference_value_matrices": identity_3,
            "bellman_hessian": np.ones((2, 1, 1)),
        },
        "provenance": {
            "evaluation_manifest_id": "eval-manifest-unit",
            "training_manifest_id": "training-manifest-unit",
            "reference_id": "reference-unit",
        },
    }


def _row_payload(inputs: dict[str, object]) -> dict[str, object]:
    return {
        "spec": {
            "issue_id": "427d0d8",
            "run_id": "unit__linear_recurrent",
            "objective": "diagnostic",
            "architecture": "linear_recurrence",
            "controller_label": "linear recurrent",
            "optimizer_label": "unit",
            "training_distribution": "nominal",
            "evaluation_lane": "diagnostic",
            "reference_controller": "unit reference",
            "parameters": {"evaluation_lens": "unit"},
        },
        "architecture": "linear_recurrence",
        "status": "unit",
        "certificate_mode": "augmented_linear",
        "component_provider": {
            "name": LINEAR_RECURRENT_AUGMENTED_PROVIDER,
            "inputs": inputs,
        },
    }


def test_provider_constructs_target_relative_augmented_basis_and_leaky_recurrence() -> None:
    component_kwargs = linear_recurrent_augmented_component_kwargs(_provider_inputs())

    np.testing.assert_allclose(
        component_kwargs["augmented_states"],
        [[[0.0, 0.0, 0.0], [1.0, 2.0, 0.25], [2.0, 4.0, 0.5]]],
    )
    np.testing.assert_allclose(
        component_kwargs["candidate_augmented_action_sensitivity"],
        [[[4.0, 6.0, 3.0]], [[4.0, 6.0, 3.0]]],
    )
    expected_transition = np.asarray(
        [
            [1.0, 0.1, 0.0],
            [0.8, 2.2, 0.6],
            [1.0, 1.5, 0.75],
        ]
    )
    np.testing.assert_allclose(
        component_kwargs["candidate_transition"],
        np.broadcast_to(expected_transition, (2, 3, 3)),
    )
    assert component_kwargs["recurrence_diagnostics"]["zero_affine"] is True
    assert component_kwargs["recurrence_diagnostics"]["alpha"] == 0.5
    assert component_kwargs["recurrence_diagnostics"]["verified_source_ids"] == {
        "evaluation_manifest_id": "eval-manifest-unit",
    }
    assert component_kwargs["recurrence_diagnostics"]["caller_asserted_source_ids"] == {
        "training_manifest_id": "training-manifest-unit",
        "reference_id": "reference-unit",
    }


def test_provider_matches_real_feedbax_post_step_state_history() -> None:
    graph = spec_to_graph(_tiny_recurrent_graph_spec())
    graph = eqx.tree_at(
        lambda model: (
            model.nodes["net"].cell.weight_ih,
            model.nodes["net"].cell.weight_hh,
            model.nodes["readout"].layer.weight,
        ),
        graph,
        (jnp.asarray([[2.0, 3.0]]), jnp.asarray([[0.5]]), jnp.asarray([[4.0]])),
    )
    outputs, _final_state, history = run_component(
        graph,
        {},
        graph.init_state(key=jr.PRNGKey(0)),
        key=jr.PRNGKey(1),
        n_steps=2,
    )
    cached_mechanics = np.asarray(history.mechanics.vector[1:])[None, ...]
    cached_hidden = np.asarray(history.net.hidden[1:])[None, ...]
    inputs = _provider_inputs()
    inputs["cached_evaluation"] = {
        "controller_visible_coupled_states": cached_mechanics,
        "target_coupled_states": np.zeros(2),
        "hidden_states": cached_hidden,
    }
    inputs["trained_controller"] = {
        **dict(inputs["trained_controller"]),
        "dt": 1.0,
        "tau": 1.0,
    }
    component_kwargs = linear_recurrent_augmented_component_kwargs(inputs)
    z_t = component_kwargs["augmented_states"][0, 0]

    predicted_action = component_kwargs["candidate_augmented_action_sensitivity"][0] @ z_t
    predicted_next = component_kwargs["candidate_transition"][0] @ z_t

    np.testing.assert_allclose(predicted_action, np.asarray(outputs["action"])[1], rtol=1e-6)
    np.testing.assert_allclose(
        predicted_next,
        component_kwargs["augmented_states"][0, 1],
        rtol=1e-6,
    )


def test_registered_provider_executes_standard_certificate_request() -> None:
    register_rlrmp_evaluation_recipes(replace=True)
    payload = materialize_evaluation_standard_certificate_rows(
        [("eval-manifest-unit", [_row_payload(_provider_inputs())])],
        issue_id="427d0d8",
    )
    by_name = component_by_name(payload["rows"][0])

    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH]["status"] == "available"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH]["status"] == "available"
    assert by_name[VALUE_POLICY_GAP]["status"] == "available"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL]["status"] == "available"


def test_provider_fails_closed_without_reference_certificate_arrays() -> None:
    inputs = _provider_inputs()
    reference = dict(inputs["reference"])
    reference.pop("reference_transition")
    inputs["reference"] = reference

    with pytest.raises(ValueError, match="reference_transition"):
        linear_recurrent_augmented_component_kwargs(inputs)


@pytest.mark.parametrize("field", ["use_bias", "readout_use_bias", "use_noise"])
def test_provider_rejects_affine_or_noisy_recurrence(field: str) -> None:
    inputs = _provider_inputs()
    controller = dict(inputs["trained_controller"])
    controller[field] = True
    inputs["trained_controller"] = controller

    with pytest.raises(ValueError, match=field):
        linear_recurrent_augmented_component_kwargs(inputs)


def test_provider_rejects_nonlinear_registered_controller() -> None:
    inputs = _provider_inputs()
    controller = dict(inputs["trained_controller"])
    controller["activation"] = "tanh"
    inputs["trained_controller"] = controller

    with pytest.raises(ValueError, match="identity activation"):
        linear_recurrent_augmented_component_kwargs(inputs)


def test_grouped_request_rejects_provider_manifest_provenance_mismatch() -> None:
    register_rlrmp_evaluation_recipes(replace=True)

    with pytest.raises(ValueError, match="conflicts with grouped dependency"):
        materialize_evaluation_standard_certificate_rows(
            [("different-eval-manifest", [_row_payload(_provider_inputs())])],
            issue_id="427d0d8",
        )


def _tiny_recurrent_graph_spec() -> GraphSpec:
    return GraphSpec(
        nodes={
            "net": ComponentSpec(
                type="VanillaRNN",
                params={
                    "input_size": 2,
                    "hidden_size": 1,
                    "activation": "identity",
                    "use_bias": False,
                },
                input_ports=["input", "hidden"],
                output_ports=["output", "hidden"],
            ),
            "readout": ComponentSpec(
                type="Linear",
                params={
                    "input_size": 1,
                    "output_size": 1,
                    "use_bias": False,
                    "seed": 0,
                },
                input_ports=["input"],
                output_ports=["output"],
            ),
            "mechanics": ComponentSpec(
                type="LinearStateSpace",
                params={
                    "A": [[1.0, 0.1], [0.0, 1.0]],
                    "B": [[0.0], [0.2]],
                    "initial_state": [1.0, 2.0],
                    "pos_slice": [0, 1],
                    "vel_slice": [1, 2],
                },
                input_ports=["force"],
                output_ports=["state", "effector"],
            ),
        },
        wires=[
            WireSpec(
                source_node="mechanics",
                source_port="state",
                target_node="net",
                target_port="input",
                temporality="recurrent",
                recurrent_initializer={
                    "kind": "state_output",
                    "scope": "trial",
                    "source": "state_initializer",
                    "state_slot": "mechanics",
                },
            ),
            WireSpec(
                source_node="net",
                source_port="hidden",
                target_node="net",
                target_port="hidden",
                temporality="recurrent",
                recurrent_initializer={"kind": "zeros", "shape": [1]},
            ),
            WireSpec(
                source_node="net",
                source_port="output",
                target_node="readout",
                target_port="input",
            ),
            WireSpec(
                source_node="readout",
                source_port="output",
                target_node="mechanics",
                target_port="force",
            ),
        ],
        output_ports=["action", "mechanics_state"],
        output_bindings={
            "action": ("readout", "output"),
            "mechanics_state": ("mechanics", "state"),
        },
    )
