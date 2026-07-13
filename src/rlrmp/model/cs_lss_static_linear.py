"""C&S LinearStateSpace graph path for a trainable static linear controller."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from feedbax.contracts.graph import ComponentSpec, GraphMetadata, GraphSpec, WireSpec

from rlrmp.model.cs_lss_gru import (
    CS_FEEDBACK_DIM,
    CS_FORCE_DIM,
    CS_PROPRIOCEPTIVE_FEEDBACK_DIM,
    build_cs_lss_gru_graph_spec,
    materialize_cs_lss_gru_graph_spec,
)


STATIC_LINEAR_CONTROLLER_KIND = "static_linear"
STATIC_LINEAR_GRAPH_SPEC_VERSION = "1.0.0"


def build_cs_lss_static_linear_graph_spec(
    *,
    input_size: int = 0,
    sensory_noise_std: float = 0.0,
    additive_motor_noise_std: float = 0.0,
    signal_dependent_motor_noise_std: float = 0.0,
    bind_epsilon_input: bool = False,
    finite_epsilon_policy: str | None = None,
    target_relative_feedback: bool = False,
    force_filter_feedback: bool = False,
    no_integrator_state: bool = False,
    trainable_dtype: str | None = None,
    key: PRNGKeyArray,
) -> GraphSpec:
    """Build the canonical affine controller over the C&S delayed feedback basis.

    The controller is memoryless: ``u_t = K y_t``. Its only trainable leaf is
    ``nodes.net.gain``. Task-only scalar inputs such as the delayed-reach go cue
    remain explicit graph inputs but are routed to an inert sink rather than
    silently changing the controller's action basis.
    """

    base = build_cs_lss_gru_graph_spec(
        hidden_size=1,
        input_size=input_size,
        sensory_noise_std=sensory_noise_std,
        additive_motor_noise_std=additive_motor_noise_std,
        signal_dependent_motor_noise_std=signal_dependent_motor_noise_std,
        bind_epsilon_input=bind_epsilon_input,
        finite_epsilon_policy=finite_epsilon_policy,
        target_relative_feedback=target_relative_feedback,
        force_filter_feedback=force_filter_feedback,
        no_integrator_state=no_integrator_state,
        trainable_dtype=trainable_dtype,
        key=key,
    )
    dtype = jnp.dtype(trainable_dtype or "float32")
    feedback_dim = CS_PROPRIOCEPTIVE_FEEDBACK_DIM if force_filter_feedback else CS_FEEDBACK_DIM
    nodes = dict(base.nodes)
    nodes["net"] = ComponentSpec(
        type="AffineFeedbackController",
        params={
            "gain": jnp.zeros((CS_FORCE_DIM, feedback_dim), dtype=dtype).tolist(),
            "schedule_policy": "hold",
        },
        input_ports=["feedback"],
        output_ports=["command"],
    )
    input_bindings = dict(base.input_bindings)
    if input_size > 0:
        nodes["task_input_sink"] = ComponentSpec(
            type="Gain",
            params={"gain": 0.0},
            input_ports=["input"],
            output_ports=["output"],
        )
        input_bindings["input"] = ("task_input_sink", "input")
    wires = [
        WireSpec(
            source_node=wire.source_node,
            source_port="command"
            if wire.source_node == "net" and wire.source_port == "output"
            else wire.source_port,
            target_node=wire.target_node,
            target_port=wire.target_port,
            temporality=wire.temporality,
            recurrent_initializer=wire.recurrent_initializer,
        )
        for wire in base.wires
    ]
    metadata = GraphMetadata(
        name="RLRMP C&S LSS static-linear loop",
        description=(
            "Executable GraphSpec contract for a memoryless affine-feedback "
            "controller trained on the canonical C&S LinearStateSpace plant."
        ),
        created_at="1970-01-01T00:00:00",
        updated_at="1970-01-01T00:00:00",
        version=STATIC_LINEAR_GRAPH_SPEC_VERSION,
        tags=["rlrmp", "feedbax", "graphspec", "cs_lss", STATIC_LINEAR_CONTROLLER_KIND],
    )
    return base.model_copy(
        update={
            "nodes": nodes,
            "wires": wires,
            "input_bindings": input_bindings,
            "subgraphs": {},
            "metadata": metadata,
        }
    )


def build_cs_lss_static_linear_graph(**kwargs: Any) -> Any:
    """Materialize :func:`build_cs_lss_static_linear_graph_spec`."""

    return materialize_cs_lss_gru_graph_spec(build_cs_lss_static_linear_graph_spec(**kwargs))


def build_cs_lss_static_linear_ensemble(
    *,
    n_replicates: int,
    key: PRNGKeyArray,
    **kwargs: Any,
) -> Any:
    """Build a vectorized ensemble with independently keyed affine controllers."""

    models = [
        build_cs_lss_static_linear_graph(key=key_one, **kwargs)
        for key_one in jr.split(key, int(n_replicates))
    ]

    def stack_leaves(*leaves: Any) -> Any:
        first = leaves[0]
        if isinstance(first, eqx.nn.StateIndex):
            return eqx.nn.StateIndex(jax.tree.map(stack_leaves, *(leaf.init for leaf in leaves)))
        return jnp.stack(leaves) if all(eqx.is_array(leaf) for leaf in leaves) else first

    return jax.tree.map(
        stack_leaves,
        *models,
        is_leaf=lambda leaf: isinstance(leaf, eqx.nn.StateIndex),
    )


__all__ = [
    "STATIC_LINEAR_CONTROLLER_KIND",
    "build_cs_lss_static_linear_ensemble",
    "build_cs_lss_static_linear_graph",
    "build_cs_lss_static_linear_graph_spec",
]
