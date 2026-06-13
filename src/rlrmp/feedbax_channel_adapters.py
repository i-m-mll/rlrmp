"""Helpers for eager use of Feedbax additive graph-channel adapter specs."""

from __future__ import annotations

import re
from typing import Any

import equinox as eqx
from feedbax.components import Sum
from feedbax.contracts.graph import AdditiveGraphChannelAdapterSpec
from feedbax.graph import Wire

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
    """Lower a Feedbax additive channel spec onto an eager Feedbax ``Graph``."""

    if spec.input_key in getattr(model, "input_bindings", {}):
        return model
    if spec.target.kind == "edge":
        return _materialize_edge_adapter_on_graph(model, spec)
    return _materialize_input_adapter_on_graph(model, spec)


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


def _materialize_edge_adapter_on_graph(
    model: Any,
    spec: AdditiveGraphChannelAdapterSpec,
) -> Any:
    target = spec.target
    old_wire = Wire(target.source_node, target.source_port, target.target_node, target.target_port)
    graph = model.remove_wire(old_wire)
    node_name = _adapter_node_name(spec, graph.nodes)
    graph = graph.add_node(node_name, Sum())
    graph = graph.add_wire(
        Wire(target.source_node, target.source_port, node_name, SUM_BASE_PORT)
    )
    graph = graph.add_wire(
        Wire(node_name, SUM_OUTPUT_PORT, target.target_node, target.target_port)
    )
    graph = eqx.tree_at(lambda g: g.input_ports, graph, (*graph.input_ports, spec.input_key))
    return eqx.tree_at(
        lambda g: g.input_bindings,
        graph,
        {**graph.input_bindings, spec.input_key: (node_name, SUM_OFFSET_PORT)},
    )


def _materialize_input_adapter_on_graph(
    model: Any,
    spec: AdditiveGraphChannelAdapterSpec,
) -> Any:
    target = spec.target
    target_key = (target.target_node, target.target_port)
    incoming = [
        wire
        for wire in getattr(model, "wires", ())
        if (wire.target_node, wire.target_port) == target_key
    ]
    if incoming:
        raise ValueError(
            f"Input additive channel adapter {spec.label!r} targets "
            f"{target.target_node}.{target.target_port}, which is already wired; "
            "use an edge adapter for wired targets"
        )
    base_bindings = [
        key
        for key, binding in getattr(model, "input_bindings", {}).items()
        if tuple(binding) == target_key
    ]
    if not base_bindings:
        return _add_graph_input_binding(model, spec.input_key, target_key)
    if len(base_bindings) != 1:
        raise ValueError(
            f"Input additive channel adapter {spec.label!r} found "
            f"{len(base_bindings)} base bindings for {target.target_node}.{target.target_port}"
        )
    base_key = base_bindings[0]
    node_name = _adapter_node_name(spec, model.nodes)
    graph = model.add_node(node_name, Sum())
    graph = eqx.tree_at(lambda g: g.input_ports, graph, (*graph.input_ports, spec.input_key))
    graph = eqx.tree_at(
        lambda g: g.input_bindings,
        graph,
        {
            **graph.input_bindings,
            base_key: (node_name, SUM_BASE_PORT),
            spec.input_key: (node_name, SUM_OFFSET_PORT),
        },
    )
    return graph.add_wire(Wire(node_name, SUM_OUTPUT_PORT, target.target_node, target.target_port))


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


def _add_graph_input_binding(
    model: Any,
    input_key: str,
    target_key: tuple[str, str],
) -> Any:
    graph = eqx.tree_at(lambda g: g.input_ports, model, (*model.input_ports, input_key))
    return eqx.tree_at(
        lambda g: g.input_bindings,
        graph,
        {**graph.input_bindings, input_key: target_key},
    )


def _adapter_node_name(
    spec: AdditiveGraphChannelAdapterSpec,
    nodes: dict[str, Any],
) -> str:
    base = spec.adapter_node or f"{spec.label}_additive"
    candidate = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_") or "additive_channel"
    if candidate not in nodes:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in nodes:
        suffix += 1
    return f"{candidate}_{suffix}"
