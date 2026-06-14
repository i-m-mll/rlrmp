"""Helpers for eager use of Feedbax additive graph-channel adapter specs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
from feedbax.runtime.components import Sum
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    ComponentSpec,
    GraphSpec,
    WireSpec,
)
from feedbax.runtime.graph import Wire
from feedbax.runtime.graph_channel_adapters import materialize_additive_channel_adapters

SUM_BASE_PORT = "a"
SUM_OFFSET_PORT = "b"
SUM_OUTPUT_PORT = "output"


def additive_channel_insertion_point(spec: AdditiveGraphChannelAdapterSpec) -> str:
    """Return a compact graph target description for a Feedbax additive spec."""

    target = spec.target
    if target.kind == "edge":
        return (
            f"{target.source_node}.{target.source_port} -> "
            f"{target.target_node}.{target.target_port}"
        )
    return f"{target.target_node}.{target.target_port}"


def additive_channel_payload_dim(spec: AdditiveGraphChannelAdapterSpec) -> int:
    """Return the trailing payload dimension declared by a Feedbax additive spec."""

    if not spec.payload_shape:
        raise ValueError(f"Additive channel spec {spec.label!r} does not declare payload_shape.")
    return int(spec.payload_shape[-1])


def additive_channel_provenance(
    spec: AdditiveGraphChannelAdapterSpec,
    *,
    adapter: str,
) -> dict[str, Any]:
    """Return JSON-safe provenance for a Feedbax additive graph-channel spec."""

    target = spec.target
    return {
        "adapter": adapter,
        "feedbax_additive_channel_adapter": spec.model_dump(mode="json", exclude_none=True),
        "label": spec.label,
        "input_key": spec.input_key,
        "insertion_point": additive_channel_insertion_point(spec),
        "target_kind": target.kind,
        "source_node": target.source_node,
        "source_port": target.source_port,
        "target_node": target.target_node,
        "target_port": target.target_port,
        "adapter_node": spec.adapter_node,
        "payload_shape": list(spec.payload_shape or []),
        "payload_dtype": spec.payload_dtype,
        "provenance_role": spec.provenance_role,
        "metadata": dict(spec.metadata),
        "controller_input_mutated": False,
        "controller_internal_state_mutated": False,
    }


def materialize_additive_channel_adapter_on_graph(
    model: Any,
    spec: AdditiveGraphChannelAdapterSpec,
) -> Any:
    """Lower one Feedbax additive channel spec onto an eager Feedbax ``Graph``."""

    return materialize_additive_channel_adapters_on_graph(model, (spec,))


def materialize_additive_channel_adapters_on_graph(
    model: Any,
    specs: Sequence[AdditiveGraphChannelAdapterSpec],
) -> Any:
    """Lower Feedbax additive channel specs onto an eager Feedbax ``Graph``.

    Feedbax owns the generic lowering algorithm on ``GraphSpec``. This eager
    bridge builds a topology-only ``GraphSpec``, lets Feedbax lower the adapter
    declarations, then applies the resulting topology back to the source graph
    without reconstructing RLRMP runtime components or hooks.
    """

    pending = [
        spec
        for spec in specs
        if spec.input_key not in getattr(model, "input_bindings", {})
        and find_materialized_additive_channel_adapter(model, spec) is None
    ]
    if not pending:
        return model

    topology = _topology_graph_spec_from_model(model, pending)
    materialized = materialize_additive_channel_adapters(topology)
    return _apply_materialized_topology(model, materialized)


def find_materialized_additive_channel_adapter(
    model: Any,
    spec: AdditiveGraphChannelAdapterSpec,
) -> AdditiveGraphChannelAdapterSpec | None:
    """Find an already materialized additive adapter for a spec target.

    This recognizes both Feedbax ``Sum``-port materialization and the older
    RLRMP temporary adapter ports so historical materialized graphs can be
    reused without adding a second adapter on the same edge.
    """

    if spec.target.kind != "edge":
        return _find_materialized_input_adapter(model, spec)
    target = spec.target
    wires = set(getattr(model, "wires", ()))
    for input_key, binding in getattr(model, "input_bindings", {}).items():
        if not isinstance(binding, tuple) or len(binding) != 2:
            continue
        node_name, offset_port = str(binding[0]), str(binding[1])
        if offset_port == SUM_OFFSET_PORT:
            base_port = SUM_BASE_PORT
            output_port = SUM_OUTPUT_PORT
        elif offset_port == "offset":
            base_port = "signal"
            output_port = "signal"
        else:
            continue
        if (
            Wire(
                target.source_node,
                target.source_port,
                node_name,
                base_port,
            )
            in wires
            and Wire(
                node_name,
                output_port,
                target.target_node,
                target.target_port,
            )
            in wires
        ):
            return spec.model_copy(
                update={
                    "input_key": str(input_key),
                    "adapter_node": node_name,
                    "metadata": {
                        **dict(spec.metadata),
                        "materialized_adapter_reused": True,
                        "materialized_offset_port": offset_port,
                    },
                }
            )
    return None


def _find_materialized_input_adapter(
    model: Any,
    spec: AdditiveGraphChannelAdapterSpec,
) -> AdditiveGraphChannelAdapterSpec | None:
    target = spec.target
    target_key = (target.target_node, target.target_port)
    binding = getattr(model, "input_bindings", {}).get(spec.input_key)
    if binding == target_key:
        return spec
    if not isinstance(binding, tuple) or len(binding) != 2 or binding[1] != SUM_OFFSET_PORT:
        return None
    node_name = str(binding[0])
    if Wire(node_name, SUM_OUTPUT_PORT, target.target_node, target.target_port) not in set(
        getattr(model, "wires", ())
    ):
        return None
    return spec.model_copy(update={"adapter_node": node_name})


def _topology_graph_spec_from_model(
    model: Any,
    specs: Sequence[AdditiveGraphChannelAdapterSpec],
) -> GraphSpec:
    return GraphSpec(
        nodes={
            node_id: ComponentSpec(
                type=type(component).__name__,
                params={},
                input_ports=list(getattr(component, "input_ports", ())),
                output_ports=list(getattr(component, "output_ports", ())),
            )
            for node_id, component in getattr(model, "nodes", {}).items()
        },
        wires=[
            WireSpec(
                source_node=wire.source_node,
                source_port=wire.source_port,
                target_node=wire.target_node,
                target_port=wire.target_port,
                temporality=wire.temporality,
                recurrent_initializer=wire.recurrent_initializer,
            )
            for wire in getattr(model, "wires", ())
        ],
        input_ports=list(getattr(model, "input_ports", ())),
        output_ports=list(getattr(model, "output_ports", ())),
        input_bindings=dict(getattr(model, "input_bindings", {})),
        output_bindings=dict(getattr(model, "output_bindings", {})),
        additive_channel_adapters=list(specs),
    )


def _apply_materialized_topology(model: Any, graph_spec: GraphSpec) -> Any:
    nodes = dict(getattr(model, "nodes", {}))
    for node_id, node_spec in graph_spec.nodes.items():
        if node_id in nodes:
            continue
        if node_spec.type != "Sum":
            raise ValueError(
                f"Feedbax additive channel materialization produced unsupported "
                f"node {node_id!r} of type {node_spec.type!r}."
            )
        nodes[node_id] = Sum()

    wires = tuple(
        Wire(
            wire.source_node,
            wire.source_port,
            wire.target_node,
            wire.target_port,
            temporality=wire.temporality,
            recurrent_initializer=wire.recurrent_initializer,
        )
        for wire in graph_spec.wires
    )
    return eqx.tree_at(
        lambda graph: (
            graph.nodes,
            graph.wires,
            graph.input_ports,
            graph.input_bindings,
        ),
        model,
        (
            nodes,
            wires,
            tuple(graph_spec.input_ports),
            dict(graph_spec.input_bindings),
        ),
    )
