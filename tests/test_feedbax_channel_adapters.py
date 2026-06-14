"""Tests for RLRMP's eager bridge to Feedbax graph-channel adapters."""

from __future__ import annotations

from typing import Any

from feedbax.runtime.components import Constant, Gain, Sum
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
    ComponentSpec,
    GraphSpec,
    WireSpec,
)
from feedbax.runtime.graph import Component, Graph, State, Wire
from feedbax.serialization import spec_to_graph

from rlrmp.model.feedbax_channel_adapters import (
    find_materialized_additive_channel_adapter,
    materialize_additive_channel_adapter_on_graph,
)


def test_eager_edge_adapter_materialization_matches_feedbax_graphspec_helper() -> None:
    adapter = AdditiveGraphChannelAdapterSpec(
        label="command_input",
        input_key="perturbation.command",
        target=AdditiveGraphChannelTargetSpec(
            kind="edge",
            source_node="source",
            source_port="output",
            target_node="readout",
            target_port="input",
        ),
        payload_shape=[2],
        provenance_role="perturbation_input",
    )
    graph_spec = _source_to_readout_spec().model_copy(
        update={"additive_channel_adapters": [adapter]}
    )
    base_graph = spec_to_graph(_source_to_readout_spec(), {})

    materialized = materialize_additive_channel_adapter_on_graph(base_graph, adapter)
    expected = spec_to_graph(graph_spec, {})

    assert materialized.input_ports == expected.input_ports
    assert materialized.input_bindings == expected.input_bindings
    assert _wire_tuples(materialized) == _wire_tuples(expected)
    assert isinstance(materialized.nodes["command_input_additive"], Sum)


def test_eager_input_adapter_materialization_matches_feedbax_wrapped_binding() -> None:
    adapter = AdditiveGraphChannelAdapterSpec(
        label="command_input",
        input_key="perturbation.command",
        target=AdditiveGraphChannelTargetSpec(
            kind="input",
            target_node="readout",
            target_port="input",
        ),
        payload_shape=[2],
    )
    base_spec = GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type="Gain",
                params={"gain": 1.0},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        input_ports=["command"],
        output_ports=["output"],
        input_bindings={"command": ("readout", "input")},
        output_bindings={"output": ("readout", "output")},
    )
    graph_spec = base_spec.model_copy(update={"additive_channel_adapters": [adapter]})
    base_graph = spec_to_graph(base_spec, {})

    materialized = materialize_additive_channel_adapter_on_graph(base_graph, adapter)
    expected = spec_to_graph(graph_spec, {})

    assert materialized.input_bindings == expected.input_bindings
    assert _wire_tuples(materialized) == _wire_tuples(expected)
    assert materialized.input_bindings["command"] == ("command_input_additive", "a")
    assert materialized.input_bindings["perturbation.command"] == (
        "command_input_additive",
        "b",
    )


def test_eager_input_adapter_materialization_matches_feedbax_direct_binding() -> None:
    adapter = AdditiveGraphChannelAdapterSpec(
        label="process_epsilon",
        input_key="perturbation.epsilon",
        target=AdditiveGraphChannelTargetSpec(
            kind="input",
            target_node="readout",
            target_port="input",
        ),
        payload_shape=[1],
    )
    base_spec = GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type="Gain",
                params={"gain": 1.0},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("readout", "output")},
    )
    graph_spec = base_spec.model_copy(update={"additive_channel_adapters": [adapter]})
    base_graph = spec_to_graph(base_spec, {})

    materialized = materialize_additive_channel_adapter_on_graph(base_graph, adapter)
    expected = spec_to_graph(graph_spec, {})

    assert materialized.input_bindings == expected.input_bindings
    assert materialized.input_bindings["perturbation.epsilon"] == ("readout", "input")
    assert "process_epsilon_additive" not in materialized.nodes


def test_historical_temporary_edge_adapter_reuses_existing_graph_without_duplicate() -> None:
    adapter = AdditiveGraphChannelAdapterSpec(
        label="command_input",
        input_key="perturbation.command",
        target=AdditiveGraphChannelTargetSpec(
            kind="edge",
            source_node="source",
            source_port="output",
            target_node="readout",
            target_port="input",
        ),
        payload_shape=[2],
    )
    historical_graph = Graph(
        input_ports=("historical.command",),
        output_ports=("output",),
        nodes={
            "source": Constant(1.0),
            "command_input_adapter": HistoricalAdditiveAdapter(),
            "readout": Gain(1.0),
        },
        wires=(
            Wire("source", "output", "command_input_adapter", "signal"),
            Wire("command_input_adapter", "signal", "readout", "input"),
        ),
        input_bindings={"historical.command": ("command_input_adapter", "offset")},
        output_bindings={"output": ("readout", "output")},
    )

    found = find_materialized_additive_channel_adapter(historical_graph, adapter)
    materialized = materialize_additive_channel_adapter_on_graph(historical_graph, adapter)

    assert found is not None
    assert found.input_key == "historical.command"
    assert found.adapter_node == "command_input_adapter"
    assert found.metadata["materialized_adapter_reused"] is True
    assert found.metadata["materialized_offset_port"] == "offset"
    assert materialized is historical_graph
    assert tuple(materialized.nodes) == ("source", "command_input_adapter", "readout")
    assert _wire_tuples(materialized) == {
        ("source", "output", "command_input_adapter", "signal"),
        ("command_input_adapter", "signal", "readout", "input"),
    }


def _source_to_readout_spec() -> GraphSpec:
    return GraphSpec(
        nodes={
            "source": ComponentSpec(
                type="Constant",
                params={"value": 1.0},
                input_ports=[],
                output_ports=["output"],
            ),
            "readout": ComponentSpec(
                type="Gain",
                params={"gain": 2.0},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="source",
                source_port="output",
                target_node="readout",
                target_port="input",
            )
        ],
        output_ports=["output"],
        output_bindings={"output": ("readout", "output")},
    )


def _wire_tuples(graph: Graph) -> set[tuple[str, str, str, str]]:
    return {
        (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
        for wire in graph.wires
    }


class HistoricalAdditiveAdapter(Component):
    """Minimal valid component exposing RLRMP's historical adapter ports."""

    input_ports: tuple[str, ...] = ("signal", "offset")
    output_ports: tuple[str, ...] = ("signal",)

    def __call__(self, inputs: dict[str, Any], state: State, *, key: Any) -> tuple[dict[str, Any], State]:
        del key
        return {"signal": inputs["signal"] + inputs["offset"]}, state
