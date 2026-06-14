"""Tests for the C&S LinearStateSpace GRU graph path."""

from __future__ import annotations

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import feedbax.serialization_prototypes as fbx_prototypes
from feedbax.contracts.graph import GraphSpec
from feedbax.runtime.graph import init_state_from_component
from feedbax.serialization import spec_to_graph
from feedbax.runtime.state_feedback import StateFeedbackSelector
from feedbax._tree import filter_spec_leaves
from feedbax.training.trainer import get_model_parameters

from rlrmp.analysis.math.cs_game_card import build_canonical_game, build_no_integrator_game
from rlrmp.model.cs_lss_gru import (
    CS_DELAYED_POS_VEL_INDICES,
    CS_DELAYED_POS_VEL_FORCE_INDICES,
    CS_EPSILON_DIM,
    FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
    CS_LSS_DELAYED_FEEDBACK_COMPONENT,
    CS_LSS_INITIAL_HIDDEN_NET_COMPONENT,
    CS_LSS_TARGET_FEEDBACK_COMPONENT,
    CS_LSS_TARGET_PROPRIOCEPTIVE_FEEDBACK_COMPONENT,
    CS_REDUCED_EPSILON_DIM,
    InitialHiddenStagedNetwork,
    build_cs_lss_gru_graph,
    build_cs_lss_gru_graph_spec,
    cs_lss_gru_where_train,
    is_canonical_cs_lss_mechanics,
    materialize_cs_lss_gru_graph_spec,
    register_cs_lss_graph_components,
)
from rlrmp.model.feedbax_graph import graph_spec_from_model, graph_spec_payload


def test_feedback_selector_uses_oldest_delayed_position_velocity_block() -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=5,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(0),
    )
    selector = StateFeedbackSelector(**spec.nodes["feedback"].params)
    vector = jnp.arange(48, dtype=jnp.float64)
    state = init_state_from_component(selector)

    outputs, _ = selector({"state": vector}, state, key=jax.random.PRNGKey(0))

    assert outputs["feedback"].shape == (4,)
    assert spec.nodes["feedback"].type == FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT
    assert tuple(outputs["feedback"].tolist()) == CS_DELAYED_POS_VEL_INDICES


@pytest.mark.parametrize(
    ("variant", "kwargs", "expected_feedback_type", "expected_net_type"),
    [
        (
            "default",
            {"bind_epsilon_input": True},
            FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
            "RLRMPSimpleStagedNetwork",
        ),
        (
            "target_relative",
            {"target_relative_feedback": True, "bind_epsilon_input": True},
            FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
            "RLRMPSimpleStagedNetwork",
        ),
        (
            "force_filter_feedback",
            {
                "target_relative_feedback": True,
                "force_filter_feedback": True,
                "bind_epsilon_input": True,
            },
            FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
            "RLRMPSimpleStagedNetwork",
        ),
        (
            "noise_channels",
            {
                "sensory_noise_std": 0.25,
                "additive_motor_noise_std": 1e-5,
                "signal_dependent_motor_noise_std": 0.02,
                "bind_epsilon_input": True,
            },
            FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
            "RLRMPSimpleStagedNetwork",
        ),
        (
            "deterministic_no_epsilon_binding",
            {"bind_epsilon_input": False},
            FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
            "RLRMPSimpleStagedNetwork",
        ),
        (
            "no_integrator",
            {
                "target_relative_feedback": True,
                "force_filter_feedback": True,
                "bind_epsilon_input": True,
                "no_integrator_state": True,
            },
            FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
            "RLRMPSimpleStagedNetwork",
        ),
        (
            "initial_hidden_encoder",
            {
                "target_relative_feedback": True,
                "bind_epsilon_input": True,
                "initial_hidden_encoder": True,
            },
            FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT,
            CS_LSS_INITIAL_HIDDEN_NET_COMPONENT,
        ),
    ],
)
def test_cs_lss_graph_specs_round_trip_and_materialize_representative_variants(
    variant: str,
    kwargs: dict[str, object],
    expected_feedback_type: str,
    expected_net_type: str,
) -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=5,
        key=jax.random.PRNGKey(len(variant)),
        **kwargs,
    )
    payload = graph_spec_payload(spec)

    round_tripped = GraphSpec.model_validate_json(json.dumps(payload))
    graph = materialize_cs_lss_gru_graph_spec(round_tripped)

    assert graph_spec_payload(round_tripped) == payload
    assert spec.nodes["feedback"].type == expected_feedback_type
    assert spec.nodes["net"].type == expected_net_type
    assert graph.nodes["mechanics"].__class__.__name__ == "LinearStateSpace"
    assert graph.input_ports == tuple(round_tripped.input_ports)
    assert graph.input_bindings == {
        name: tuple(binding) for name, binding in round_tripped.input_bindings.items()
    }
    assert graph.output_ports == tuple(round_tripped.output_ports)
    assert graph.output_bindings == {
        name: tuple(binding) for name, binding in round_tripped.output_bindings.items()
    }
    if kwargs.get("bind_epsilon_input"):
        assert spec.input_bindings["epsilon"] == ("mechanics", "epsilon")
    else:
        assert "epsilon" not in spec.input_bindings
    if kwargs.get("no_integrator_state"):
        assert len(spec.nodes["mechanics"].params["B_w"]) == 36
        assert len(spec.nodes["mechanics"].params["B_w"][0]) == 6
        assert graph.nodes["mechanics"].B_w.shape == (36, 6)


@pytest.mark.parametrize(
    (
        "legacy_type",
        "params",
        "target_relative_feedback",
        "force_filter_feedback",
        "inputs",
        "expected_feedback",
    ),
    [
        (
            CS_LSS_DELAYED_FEEDBACK_COMPONENT,
            {
                "indices": list(CS_DELAYED_POS_VEL_INDICES),
                "expected_state_dim": 48,
                "feedback_dim": 4,
            },
            False,
            False,
            {
                "state": jnp.arange(48, dtype=jnp.float32),
            },
            jnp.asarray(CS_DELAYED_POS_VEL_INDICES, dtype=jnp.float32),
        ),
        (
            CS_LSS_TARGET_FEEDBACK_COMPONENT,
            {
                "indices": list(CS_DELAYED_POS_VEL_INDICES),
                "expected_state_dim": 48,
                "feedback_dim": 4,
            },
            True,
            False,
            {
                "state": jnp.zeros((48,), dtype=jnp.float32).at[40:44].set(
                    jnp.array([0.02, -0.03, 0.40, -0.20], dtype=jnp.float32)
                ),
                "target": jnp.array([0.15, 0.01], dtype=jnp.float32),
            },
            jnp.array([0.13, 0.04, -0.40, 0.20], dtype=jnp.float32),
        ),
        (
            CS_LSS_TARGET_PROPRIOCEPTIVE_FEEDBACK_COMPONENT,
            {
                "indices": list(CS_DELAYED_POS_VEL_FORCE_INDICES),
                "expected_state_dim": 48,
                "feedback_dim": 6,
            },
            True,
            True,
            {
                "state": jnp.zeros((48,), dtype=jnp.float32).at[40:46].set(
                    jnp.array([0.02, -0.03, 0.40, -0.20, 0.70, -0.80], dtype=jnp.float32)
                ),
                "target": jnp.array([0.15, 0.01], dtype=jnp.float32),
            },
            jnp.array([0.13, 0.04, -0.40, 0.20, 0.70, -0.80], dtype=jnp.float32),
        ),
    ],
)
def test_legacy_cs_lss_feedback_selector_ids_materialize_through_migration(
    legacy_type: str,
    params: dict[str, object],
    target_relative_feedback: bool,
    force_filter_feedback: bool,
    inputs: dict[str, jax.Array],
    expected_feedback: jax.Array,
) -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=5,
        bind_epsilon_input=True,
        target_relative_feedback=target_relative_feedback,
        force_filter_feedback=force_filter_feedback,
        key=jax.random.PRNGKey(12),
    )
    nodes = dict(spec.nodes)
    nodes["feedback"] = nodes["feedback"].model_copy(
        update={
            "type": legacy_type,
            "params": params,
        }
    )
    legacy_spec = spec.model_copy(update={"nodes": nodes})

    registry = register_cs_lss_graph_components()
    graph = materialize_cs_lss_gru_graph_spec(legacy_spec, registry)

    assert graph.nodes["feedback"].__class__.__name__ == "StateFeedbackSelector"
    definition = next(
        item for item in registry.list_all() if item.name == FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT
    )
    migration = next(
        item
        for item in definition.migrations
        if item.source_type == legacy_type
    )
    assert migration.owner == "rlrmp"
    outputs, _ = graph.nodes["feedback"](
        inputs,
        init_state_from_component(graph.nodes["feedback"]),
        key=jax.random.PRNGKey(13),
    )
    assert jnp.allclose(outputs["feedback"], expected_feedback)


def test_cs_lss_materialization_rejects_unsupported_state_diffusion_params() -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=5,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(16),
    )
    nodes = dict(spec.nodes)
    mechanics = nodes["mechanics"]
    nodes["mechanics"] = mechanics.model_copy(
        update={
            "params": {
                **mechanics.params,
                "state_diffusion_covariance": [[1.0]],
            }
        }
    )
    invalid_spec = spec.model_copy(update={"nodes": nodes})

    with pytest.raises(ValueError, match="Unsupported LinearStateSpace stochastic"):
        materialize_cs_lss_gru_graph_spec(invalid_spec)


def test_cs_lss_materialization_rejects_unsupported_stochastic_component_id() -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=5,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(17),
    )
    nodes = dict(spec.nodes)
    nodes["state_noise"] = nodes["feedback"].model_copy(
        update={
            "type": "RLRMPCsLssStateDiffusionNoise",
            "params": {"std": 0.1},
        }
    )
    invalid_spec = spec.model_copy(update={"nodes": nodes})

    with pytest.raises(ValueError, match="Unsupported C&S stochastic component"):
        materialize_cs_lss_gru_graph_spec(invalid_spec)


def test_runtime_cs_lss_graph_export_preserves_executable_component_contract() -> None:
    original_spec = build_cs_lss_gru_graph_spec(
        hidden_size=7,
        target_relative_feedback=True,
        force_filter_feedback=True,
        initial_hidden_encoder=True,
        sensory_noise_std=0.25,
        additive_motor_noise_std=1e-5,
        signal_dependent_motor_noise_std=0.02,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(101),
    )
    graph = materialize_cs_lss_gru_graph_spec(original_spec)

    exported = graph_spec_from_model(graph)
    exported_payload = graph_spec_payload(exported)
    reparsed = GraphSpec.model_validate_json(json.dumps(exported_payload))
    rematerialized = materialize_cs_lss_gru_graph_spec(reparsed)

    assert exported.nodes["feedback"].type == FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT
    assert exported.nodes["feedback"].params == original_spec.nodes["feedback"].params
    assert exported.nodes["net"].type == CS_LSS_INITIAL_HIDDEN_NET_COMPONENT
    assert "net" not in (exported.subgraphs or {})
    assert exported.nodes["mechanics"].type == "LinearStateSpace"
    assert exported.nodes["efferent"].type == "Channel"
    assert rematerialized.nodes["net"].h0_encoder.weight.shape == (7, 6)


def test_cs_lss_materialization_uses_registered_prototypes_without_global_patch() -> None:
    original_output_prototypes_for_node = fbx_prototypes.output_prototypes_for_node
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=5,
        target_relative_feedback=True,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(201),
    )
    registry = register_cs_lss_graph_components()

    graph = materialize_cs_lss_gru_graph_spec(spec, registry)

    assert fbx_prototypes.output_prototypes_for_node is original_output_prototypes_for_node
    assert registry.get(FEEDBAX_STATE_FEEDBACK_SELECTOR_COMPONENT).output_prototype_fn is not None
    assert registry.get("RLRMPSimpleStagedNetwork").output_prototype_fn is not None
    assert registry.get("Channel").output_prototype_fn is None
    assert registry.get("LinearStateSpace").output_prototype_fn is None
    assert graph.nodes["mechanics"].A.shape == (48, 48)


def test_cs_lss_runtime_state_view_is_attached_out_of_place() -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=5,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(202),
    )
    registry = register_cs_lss_graph_components()
    raw_graph = spec_to_graph(spec, registry)

    graph = materialize_cs_lss_gru_graph_spec(spec, registry)

    assert raw_graph.state_view_fn is None
    assert graph is not raw_graph
    assert graph.state_view_fn is not None
    assert graph.input_ports == raw_graph.input_ports
    assert graph.output_bindings == raw_graph.output_bindings


def test_graph_first_step_feedback_is_seeded_from_initial_lss_state() -> None:
    initial_state = jnp.arange(48, dtype=jnp.float64)
    graph = build_cs_lss_gru_graph(
        hidden_size=5,
        initial_state=initial_state,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(1),
    )
    state = init_state_from_component(graph)
    epsilon = jnp.zeros((CS_EPSILON_DIM,), dtype=jnp.float64)

    outputs, _state, _cycle = graph.step(
        {"epsilon": epsilon},
        state,
        cycle_port_values=None,
        key=jax.random.PRNGKey(2),
    )

    expected_feedback = initial_state[jnp.array(CS_DELAYED_POS_VEL_INDICES)]
    assert jnp.allclose(outputs["feedback"], expected_feedback)


def test_lss_command_updates_force_filter_without_same_step_velocity_jump() -> None:
    plant, _schedule = build_canonical_game()
    graph = build_cs_lss_gru_graph(
        hidden_size=4,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(3),
    )
    mechanics = graph.nodes["mechanics"]
    command = jnp.array([1.25, -0.5], dtype=plant.A.dtype)
    state = init_state_from_component(mechanics)

    outputs, _ = mechanics(
        {
            "force": command,
            "epsilon": jnp.zeros((plant.m_w,), dtype=plant.A.dtype),
        },
        state,
        key=jax.random.PRNGKey(4),
    )

    assert is_canonical_cs_lss_mechanics(mechanics)
    assert jnp.allclose(outputs["state"][2:4], 0.0, atol=0.0)
    assert jnp.allclose(outputs["state"][4:6], plant.B[4:6] @ command, atol=1e-14)


def test_no_integrator_graph_uses_reduced_lss_state_and_epsilon() -> None:
    plant, _schedule = build_no_integrator_game()
    initial_state = jnp.arange(36, dtype=jnp.float64)
    graph = build_cs_lss_gru_graph(
        hidden_size=4,
        initial_state=initial_state,
        bind_epsilon_input=True,
        target_relative_feedback=True,
        force_filter_feedback=True,
        no_integrator_state=True,
        key=jax.random.PRNGKey(10),
    )
    state = init_state_from_component(graph)
    mechanics = graph.nodes["mechanics"]

    outputs, _state, _cycle = graph.step(
        {
            "target": jnp.array([0.15, 0.0], dtype=jnp.float64),
            "epsilon": jnp.zeros((CS_REDUCED_EPSILON_DIM,), dtype=jnp.float64),
        },
        state,
        cycle_port_values=None,
        key=jax.random.PRNGKey(11),
    )

    assert mechanics.A.shape == (36, 36)
    assert mechanics.B_w.shape == (36, 6)
    assert jnp.allclose(mechanics.A, plant.A)
    assert jnp.allclose(outputs["clean_feedback"][-2:], initial_state[34:36])
    assert not is_canonical_cs_lss_mechanics(mechanics)


def test_graph_runs_multiple_steps_with_zero_epsilon() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=6,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(5),
    )
    state = init_state_from_component(graph)
    epsilon = jnp.zeros((3, CS_EPSILON_DIM), dtype=jnp.float64)

    outputs, final_state, state_history = graph(
        {"epsilon": epsilon},
        state,
        key=jax.random.PRNGKey(6),
        return_state_history=True,
    )

    assert outputs["state"].shape == (3, 48)
    assert outputs["feedback"].shape == (3, 4)
    assert outputs["force"].shape == (3, 2)
    assert state_history.mechanics.vector.shape == (4, 48)
    assert state_history.mechanics.effector.pos.shape == (4, 2)
    assert state_history.mechanics.effector.vel.shape == (4, 2)
    assert jnp.allclose(state_history.mechanics.effector.pos, state_history.mechanics.vector[:, :2])
    assert jnp.allclose(
        state_history.mechanics.effector.vel, state_history.mechanics.vector[:, 2:4]
    )
    assert state_history.sensory.output.shape == (4, 4)
    assert state_history.net.hidden.shape == (4, 6)
    assert final_state is not None


def test_graph_wires_sensory_and_motor_noise_channels() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=4,
        sensory_noise_std=0.25,
        additive_motor_noise_std=1e-5,
        signal_dependent_motor_noise_std=0.02,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(10),
    )

    sensory = graph.nodes["sensory"]
    efferent = graph.nodes["efferent"]

    assert sensory.add_noise is True
    assert sensory.delay == 0
    assert sensory.noise_func.std == 0.25
    assert efferent.add_noise is True
    assert efferent.delay == 0
    assert efferent.noise_func[0].noise_func.std == 0.02
    assert efferent.noise_func[1].std == 1e-5


def test_graph_omits_epsilon_binding_for_deterministic_default() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=4,
        bind_epsilon_input=False,
        key=jax.random.PRNGKey(8),
    )
    state = init_state_from_component(graph)

    outputs, _final_state = graph(
        {},
        state,
        key=jax.random.PRNGKey(9),
        n_steps=2,
    )

    assert graph.input_ports == ()
    assert outputs["state"].shape == (2, 48)
    assert outputs["feedback"].shape == (2, 4)


def test_default_graph_uses_plain_zero_h0_network() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=4,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(11),
    )
    state = init_state_from_component(graph)

    assert not isinstance(graph.nodes["net"], InitialHiddenStagedNetwork)
    assert jnp.allclose(graph.state_view(state).net.hidden, jnp.zeros((4,)))


def test_initial_hidden_encoder_uses_target_relative_feedback_context_shape() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=7,
        bind_epsilon_input=True,
        target_relative_feedback=True,
        initial_hidden_encoder=True,
        key=jax.random.PRNGKey(12),
    )
    net = graph.nodes["net"]
    context = jnp.array([0.15, 0.0, 0.0, 0.0], dtype=jnp.float32)

    assert isinstance(net, InitialHiddenStagedNetwork)
    assert net.h0_encoder.weight.shape == (7, 4)
    assert net.h0_encoder.bias.shape == (7,)
    assert net.h0_encoder(context).shape == (7,)
    assert jnp.allclose(net.h0_encoder(context), jnp.zeros((7,)))


def test_initial_hidden_encoder_dtype_matches_mechanics_dtype() -> None:
    spec = build_cs_lss_gru_graph_spec(
        hidden_size=7,
        bind_epsilon_input=True,
        target_relative_feedback=True,
        initial_hidden_encoder=True,
        key=jax.random.PRNGKey(15),
    )
    graph = materialize_cs_lss_gru_graph_spec(spec)

    expected_dtype = jnp.dtype(graph.nodes["mechanics"].A.dtype)
    assert spec.nodes["net"].params["h0_dtype"] == expected_dtype.name
    assert jnp.dtype(graph.nodes["net"].h0_encoder.weight.dtype) == expected_dtype
    assert jnp.dtype(graph.nodes["net"].h0_encoder.bias.dtype) == expected_dtype


def test_initial_hidden_encoder_requires_target_relative_context() -> None:
    with pytest.raises(ValueError, match="requires target_relative_feedback"):
        build_cs_lss_gru_graph(
            hidden_size=4,
            bind_epsilon_input=True,
            initial_hidden_encoder=True,
            key=jax.random.PRNGKey(13),
        )


def test_train_filter_excludes_canonical_lss_matrices() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=6,
        bind_epsilon_input=True,
        key=jax.random.PRNGKey(7),
    )
    where_train = cs_lss_gru_where_train()[0]
    where_train_spec = filter_spec_leaves(graph, where_train)
    trainable = get_model_parameters(graph, where_train_spec)
    trainable_arrays = [leaf for leaf in jax.tree.leaves(trainable) if eqx.is_array(leaf)]
    mechanics = graph.nodes["mechanics"]

    assert trainable.nodes["mechanics"].A is None
    assert trainable.nodes["mechanics"].B is None
    assert trainable.nodes["mechanics"].B_w is None
    assert mechanics.A is graph.nodes["mechanics"].A
    assert any(leaf.shape == (18, 4) for leaf in trainable_arrays)


def test_train_filter_includes_multiplicative_sisu_alpha() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=6,
        bind_epsilon_input=True,
        sisu_gating="multiplicative",
        key=jax.random.PRNGKey(17),
    )
    where_train = cs_lss_gru_where_train()[0]
    where_train_spec = filter_spec_leaves(graph, where_train)
    trainable = get_model_parameters(graph, where_train_spec)

    assert trainable.nodes["mechanics"].A is None
    assert trainable.nodes["net"].sisu_alpha.shape == (6,)


def test_train_filter_includes_h0_encoder_only_when_present() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=6,
        bind_epsilon_input=True,
        target_relative_feedback=True,
        initial_hidden_encoder=True,
        key=jax.random.PRNGKey(14),
    )
    where_train = cs_lss_gru_where_train()[0]
    where_train_spec = filter_spec_leaves(graph, where_train)
    trainable = get_model_parameters(graph, where_train_spec)
    trainable_arrays = [leaf for leaf in jax.tree.leaves(trainable) if eqx.is_array(leaf)]

    assert trainable.nodes["mechanics"].A is None
    assert trainable.nodes["net"].h0_encoder.weight.shape == (6, 4)
    assert trainable.nodes["net"].h0_encoder.bias.shape == (6,)
    assert any(leaf.shape == (6, 4) for leaf in trainable_arrays)


def test_train_filter_includes_h0_encoder_and_multiplicative_sisu_alpha() -> None:
    graph = build_cs_lss_gru_graph(
        hidden_size=6,
        bind_epsilon_input=True,
        target_relative_feedback=True,
        initial_hidden_encoder=True,
        sisu_gating="multiplicative",
        key=jax.random.PRNGKey(18),
    )
    where_train = cs_lss_gru_where_train()[0]
    where_train_spec = filter_spec_leaves(graph, where_train)
    trainable = get_model_parameters(graph, where_train_spec)

    assert trainable.nodes["mechanics"].A is None
    assert trainable.nodes["net"].h0_encoder.weight.shape == (6, 4)
    assert trainable.nodes["net"].sisu_alpha.shape == (6,)
