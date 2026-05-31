"""RLRMP Feedbax GraphSpec builders.

This module is the RLRMP-owned migration bridge for issue ``b41c940``. It
serializes the model/task surface currently built by ``setup_task_model_pair``
without pretending that Feedbax's generic worker can yet execute every RLRMP
training contract. The graph is therefore the portable construction contract;
legacy ``SimpleFeedback`` constructors remain the execution backend until the
Feedbax worker grows the required batched/minimax semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedbax.web.models.graph import (
    ComponentSpec,
    GraphMetadata,
    GraphSpec,
    RetainedObservableSpec,
    RetainedObservableTargetSpec,
    RetentionPolicySpec,
    WireSpec,
)


SCHEMA_VERSION = "rlrmp.feedbax_graph.v1"
EXECUTION_BACKEND = "rlrmp.legacy_simple_feedback_compat"
PLANT_INTERVENOR_LABEL = "plant_intervenor"


@dataclass(frozen=True)
class RLRMPFeedbaxGraphBundle:
    """Serializable graph contract plus adjacent training metadata."""

    graph_spec: GraphSpec
    task_spec: dict[str, Any]
    loss_spec: dict[str, Any]
    training_spec: dict[str, Any]
    manifest: dict[str, Any]

    def to_run_metadata(
        self,
        *,
        graph_spec_path: str = "model.graph.json",
        manifest_path: str = "model.graph.manifest.json",
    ) -> dict[str, Any]:
        """Return the compact metadata embedded into ``run.json``."""

        return {
            "schema_version": SCHEMA_VERSION,
            "graph_spec_path": graph_spec_path,
            "manifest_path": manifest_path,
            "execution_backend": EXECUTION_BACKEND,
            "component_policy": self.manifest["component_policy"],
            "legacy_loader": self.manifest["legacy_loader"],
        }


def build_rlrmp_feedbax_graph_bundle(hps: Any) -> RLRMPFeedbaxGraphBundle:
    """Build the GraphSpec/manifest bundle for the current RLRMP model setup."""

    controller_kind = _controller_kind(hps)
    graph_spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        controller_kind=controller_kind,
        intervention_type="FixedField",
    )
    task_spec = _task_spec(hps)
    loss_spec = _loss_spec(hps)
    training_spec = _training_spec(hps, controller_kind=controller_kind)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "execution_backend": EXECUTION_BACKEND,
        "component_policy": {
            "rlrmp_component_types": [
                "RLRMPFeedbackChannels",
                "RLRMPSimpleStagedNetwork",
                "RLRMPLinearController",
                "RLRMPLinearTrackerController",
                "DynamicsMatrixPerturb",
            ],
            "note": (
                "RLRMP component types are GraphSpec-addressable contracts. "
                "They are executed through the legacy RLRMP SimpleFeedback path "
                "until Feedbax exposes extension builders for these components."
            ),
        },
        "legacy_loader": {
            "setup_function": "rlrmp.modules.training.part2.setup_task_model_pair",
            "checkpoint_format": "feedbax._io.save/load_with_hyperparameters",
        },
        "task_spec": task_spec,
        "loss_spec": loss_spec,
        "training_spec": training_spec,
    }
    return RLRMPFeedbaxGraphBundle(
        graph_spec=graph_spec,
        task_spec=task_spec,
        loss_spec=loss_spec,
        training_spec=training_spec,
        manifest=manifest,
    )


def build_point_mass_sensorimotor_graph_spec(
    hps: Any,
    *,
    controller_kind: str | None = None,
    intervention_type: str | None = "FixedField",
) -> GraphSpec:
    """Serialize the point-mass feedback loop used by RLRMP training.

    The shape mirrors Feedbax ``SimpleFeedback``: feedback channel, controller,
    efferent channel, optional force filter, mechanics, and optional plant
    intervenor inserted before ``mechanics.force``.
    """

    if controller_kind is None:
        controller_kind = _controller_kind(hps)

    mechanics_params = {
        "dt": float(hps.dt),
        "mass": float(hps.model.effector_mass),
        "damping": float(hps.model.damping),
    }
    nodes: dict[str, ComponentSpec] = {
        "feedback": ComponentSpec(
            type="RLRMPFeedbackChannels",
            params={
                "where": ["plant.skeleton.pos", "plant.skeleton.vel"],
                "delay": int(hps.model.feedback_delay_steps),
                "noise_std": float(hps.model.feedback_noise_std),
            },
            input_ports=["mechanics"],
            output_ports=["feedback"],
        ),
        "net": _controller_component_spec(hps, controller_kind),
        "efferent": ComponentSpec(
            type="Channel",
            params={
                "delay": 0,
                "noise_std": float(hps.model.motor_noise_std),
                "add_noise": float(hps.model.motor_noise_std) != 0.0,
                "noise_model": "multiplicative_plus_constant",
                "constant_noise_scale": 1.8,
                "input_shape": [2],
            },
            input_ports=["input"],
            output_ports=["output"],
        ),
        "mechanics": ComponentSpec(
            type="PointMass",
            params=mechanics_params,
            input_ports=["force"],
            output_ports=["effector", "state"],
        ),
    }
    wires: list[WireSpec] = [
        WireSpec(
            source_node="feedback",
            source_port="feedback",
            target_node="net",
            target_port="feedback",
        ),
        WireSpec(
            source_node="net",
            source_port="output",
            target_node="efferent",
            target_port="input",
        ),
        WireSpec(
            source_node="mechanics",
            source_port="state",
            target_node="feedback",
            target_port="mechanics",
            temporality="recurrent",
            recurrent_initializer={
                "kind": "state_output",
                "scope": "trial",
                "source": "state_initializer",
                "state_slot": "mechanics",
            },
        ),
    ]

    force_source = ("efferent", "output")
    if float(hps.model.tau_rise) != 0.0 or float(hps.model.tau_rise) != 0.0:
        nodes["force_filter"] = ComponentSpec(
            type="FirstOrderFilter",
            params={
                "tau_rise": float(hps.model.tau_rise),
                "tau_decay": float(hps.model.tau_rise),
                "dt": float(hps.dt),
                "init_value": 0.0,
                "input_shape": [2],
            },
            input_ports=["input"],
            output_ports=["output"],
        )
        wires.append(
            WireSpec(
                source_node="efferent",
                source_port="output",
                target_node="force_filter",
                target_port="input",
            )
        )
        force_source = ("force_filter", "output")

    input_bindings: dict[str, tuple[str, str]] = {"input": ("net", "input")}
    if intervention_type is None:
        wires.append(
            WireSpec(
                source_node=force_source[0],
                source_port=force_source[1],
                target_node="mechanics",
                target_port="force",
            )
        )
    else:
        nodes[PLANT_INTERVENOR_LABEL] = _intervention_component_spec(intervention_type, hps)
        wires.append(
            WireSpec(
                source_node=force_source[0],
                source_port=force_source[1],
                target_node=PLANT_INTERVENOR_LABEL,
                target_port="force",
            )
        )
        if intervention_type in {"CurlField", "DynamicsMatrixPerturb"}:
            wires.append(
                WireSpec(
                    source_node="mechanics",
                    source_port="effector",
                    target_node=PLANT_INTERVENOR_LABEL,
                    target_port="effector",
                    temporality="recurrent",
                    recurrent_initializer={
                        "kind": "state_output",
                        "scope": "trial",
                        "source": "state_initializer",
                        "state_slot": "effector",
                    },
                )
            )
        wires.append(
            WireSpec(
                source_node=PLANT_INTERVENOR_LABEL,
                source_port="force",
                target_node="mechanics",
                target_port="force",
            )
        )
        input_bindings[f"intervene:{PLANT_INTERVENOR_LABEL}"] = (
            PLANT_INTERVENOR_LABEL,
            "params_override",
        )

    return GraphSpec(
        nodes=nodes,
        wires=wires,
        input_ports=list(input_bindings),
        output_ports=["effector"],
        input_bindings=input_bindings,
        output_bindings={"effector": ("mechanics", "effector")},
        retained_observables=_retained_observables(),
        metadata=GraphMetadata(
            name="RLRMP point-mass sensorimotor loop",
            description=(
                "GraphSpec contract for RLRMP minimax training. Executed via "
                "legacy SimpleFeedback compatibility until Feedbax worker "
                "supports batched/ensembled minimax training."
            ),
            created_at="1970-01-01T00:00:00",
            updated_at="1970-01-01T00:00:00",
            version=SCHEMA_VERSION,
            tags=["rlrmp", "feedbax", "graphspec", "minimax"],
        ),
    )


def graph_spec_payload(graph_spec: GraphSpec) -> dict[str, Any]:
    """Return a JSON-serializable GraphSpec payload."""

    return graph_spec.model_dump(mode="json", exclude_none=True)


def write_graph_spec_bundle(bundle: RLRMPFeedbaxGraphBundle, spec_dir: Path) -> Path:
    """Write ``model.graph.json`` beside a run spec and return its path."""

    graph_path = spec_dir / "model.graph.json"
    graph_path.write_text(
        _json_dumps(graph_spec_payload(bundle.graph_spec)),
        encoding="utf-8",
    )
    manifest_path = spec_dir / "model.graph.manifest.json"
    manifest_path.write_text(_json_dumps(bundle.manifest), encoding="utf-8")
    return graph_path


def _controller_kind(hps: Any) -> str:
    hidden_type = getattr(hps, "hidden_type", None)
    if isinstance(hidden_type, str) and hidden_type in {"linear", "linear_tracker"}:
        return hidden_type
    name = getattr(hidden_type, "__name__", None)
    if name is None and hasattr(hidden_type, "func"):
        name = getattr(hidden_type.func, "__name__", None)
    return "vanilla_rnn" if name == "VanillaRNNCell" else "gru"


def _controller_component_spec(hps: Any, controller_kind: str) -> ComponentSpec:
    if controller_kind == "linear":
        return ComponentSpec(
            type="RLRMPLinearController",
            params=_linear_controller_params(hps),
            input_ports=["input", "feedback"],
            output_ports=["output", "hidden"],
        )
    if controller_kind == "linear_tracker":
        return ComponentSpec(
            type="RLRMPLinearTrackerController",
            params=_linear_controller_params(hps),
            input_ports=["input", "feedback"],
            output_ports=["output", "hidden"],
        )
    return ComponentSpec(
        type="RLRMPSimpleStagedNetwork",
        params={
            "controller_kind": controller_kind,
            "input_size": None,
            "input_size_source": (
                "SimpleFeedback.get_nn_input_size(task, mechanics, feedback_spec) + n_extra_inputs"
            ),
            "hidden_size": int(hps.model.hidden_size),
            "out_size": 2,
            "encoding_size": None,
            "sisu_gating": str(getattr(hps, "sisu_gating", "additive")),
            "n_extra_inputs": 1,
            "population_structure": _population_structure(hps),
        },
        input_ports=["input", "feedback"],
        output_ports=["output", "hidden"],
    )


def _linear_controller_params(hps: Any) -> dict[str, Any]:
    return {
        "n_steps": int(hps.task.n_steps) - 1,
        "n_controls": 2,
        "n_states": 4,
        "target_source": "input.task.effector_target.pos",
        "feedback_order": ["pos_x", "pos_y", "vel_x", "vel_y"],
    }


def _intervention_component_spec(intervention_type: str, hps: Any) -> ComponentSpec:
    input_ports = ["force", "params_override"]
    if intervention_type in {"CurlField", "DynamicsMatrixPerturb"}:
        input_ports = ["effector", *input_ports]
    params: dict[str, Any] = {"active": False, "scale": 1.0}
    if intervention_type == "FixedField":
        params.update({"amplitude": 1.0, "field": [0.0, 0.0]})
    elif intervention_type == "DynamicsMatrixPerturb":
        params.update(
            {
                "delta_A_shape": [2, 4],
                "mass": float(hps.model.effector_mass),
            }
        )
    return ComponentSpec(
        type=intervention_type,
        params=params,
        input_ports=input_ports,
        output_ports=["force"],
    )


def _retained_observables() -> list[RetainedObservableSpec]:
    return [
        _port_observable("mechanics.effector", "mechanics", "effector"),
        _port_observable("mechanics.state", "mechanics", "state"),
        _port_observable("net.output", "net", "output"),
        _port_observable("net.hidden", "net", "hidden"),
        _port_observable("efferent.output", "efferent", "output"),
        _port_observable(f"{PLANT_INTERVENOR_LABEL}.force", PLANT_INTERVENOR_LABEL, "force"),
    ]


def _port_observable(selector: str, node_id: str, port: str) -> RetainedObservableSpec:
    return RetainedObservableSpec(
        id=f"observable:{selector}",
        label=selector,
        selector=selector,
        target=RetainedObservableTargetSpec(
            kind="port",
            selector=selector,
            node_id=node_id,
            port=port,
            timing="output",
        ),
        retention=RetentionPolicySpec(mode="trajectory"),
    )


def _task_spec(hps: Any) -> dict[str, Any]:
    return {
        "type": str(hps.task.type),
        "n_steps": int(hps.task.n_steps),
        "workspace": _plain(hps.task.workspace),
        "eval_grid_n": int(hps.task.eval_grid_n),
        "eval_n_directions": int(hps.task.eval_n_directions),
        "eval_reach_length": float(hps.task.eval_reach_length),
        "epoch_len_ranges": _plain(hps.task.epoch_len_ranges),
        "target_on_epochs": _plain(hps.task.target_on_epochs),
        "hold_epochs": _plain(hps.task.hold_epochs),
        "move_epochs": _plain(hps.task.move_epochs),
        "p_catch_trial": float(hps.task.p_catch_trial),
        "extra_inputs": ["sisu", f"intervene:{PLANT_INTERVENOR_LABEL}"],
    }


def _loss_spec(hps: Any) -> dict[str, Any]:
    return {
        "weights": _plain(hps.loss.weights),
        "effector_pos_late": _plain(hps.loss.effector_pos_late),
        "effector_vel_late": _plain(hps.loss.effector_vel_late),
        "effector_pos_running_schedule": str(hps.loss.effector_pos_running_schedule),
        "effector_hold_pos_schedule": str(hps.loss.effector_hold_pos_schedule),
        "position_powerlaw_power": float(hps.loss.position_powerlaw_power),
        "movement_ramp_shape": str(hps.loss.movement_ramp_shape),
        "movement_ramp_duration_steps": int(hps.loss.movement_ramp_duration_steps),
        "movement_ramp_power": float(hps.loss.movement_ramp_power),
    }


def _training_spec(hps: Any, *, controller_kind: str) -> dict[str, Any]:
    trainable = (
        ["nodes.net.K"] if controller_kind == "linear" else ["nodes.net.K", "nodes.net.u_ff"]
    )
    if controller_kind not in {"linear", "linear_tracker"}:
        trainable = ["nodes.net.hidden", "nodes.net.readout"]
        if str(getattr(hps, "sisu_gating", "additive")) == "multiplicative":
            trainable.append("nodes.net.sisu_alpha")
    return {
        "dt": float(hps.dt),
        "batch_size": int(hps.batch_size),
        "n_replicates": int(hps.model.n_replicates),
        "controller_kind": controller_kind,
        "trainable": trainable,
        "method": str(hps.method),
        "loss_update": _plain(hps.loss_update),
    }


def _population_structure(hps: Any) -> dict[str, int]:
    pop = getattr(hps.model, "population_structure", None)
    if pop is None:
        return {}
    return {
        "n_input_only": int(getattr(pop, "n_input_only", 0) or 0),
        "n_readout_only": int(getattr(pop, "n_readout_only", 0) or 0),
        "n_recurrent_only": int(getattr(pop, "n_recurrent_only", 0) or 0),
        "n_input_readout": int(getattr(pop, "n_input_readout", 0) or 0),
    }


def _plain(value: Any) -> Any:
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
