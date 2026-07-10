"""Unified stochastic-noise wiring for Feedbax-backed RLRMP models.

The Feedbax ``SimpleFeedback`` runtime already has the right hooks for the
command-channel pieces of the C&S-style stochastic contract: feedback channels
sit before the controller, and the efferent command channel sits before the
optional force filter. RLRMP owns the compatibility layer that splits the old
two-knob training config into explicit sensory, additive command,
signal-dependent command, and plant/load force noise components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
from feedbax.runtime.channel import Channel
from equinox import field
from equinox.nn import State
from feedbax.runtime.graph import Component, Wire
from feedbax.runtime.noise import Multiplicative, Normal
from jaxtyping import PRNGKeyArray, PyTree


PLANT_PROCESS_FORCE_NOISE_LABEL = "plant_process_force_noise"


@dataclass(frozen=True)
class StochasticRuntimeConfig:
    """Explicit Feedbax-runtime stochastic noise parameters.

    Attributes:
        sensory_noise_std: Standard deviation for feedback-channel additive
            Gaussian noise before the controller.
        additive_motor_noise_std: Standard deviation for command-channel
            additive Gaussian noise before any force filter.
        signal_dependent_motor_noise_std: Standard deviation for
            command-channel multiplicative Gaussian noise before any force
            filter.
        plant_process_force_noise_std: Standard deviation for additive
            plant/load force noise immediately upstream of mechanics, after
            the force filter and after deterministic/adversarial force fields.
    """

    sensory_noise_std: float = 0.0
    additive_motor_noise_std: float = 0.0
    signal_dependent_motor_noise_std: float = 0.0
    plant_process_force_noise_std: float = 0.0

    @property
    def has_command_noise(self) -> bool:
        """Whether either command-channel motor noise term is active."""

        return (
            self.additive_motor_noise_std != 0.0
            or self.signal_dependent_motor_noise_std != 0.0
        )

    @property
    def has_plant_process_force_noise(self) -> bool:
        """Whether plant/load force noise should be inserted before mechanics."""

        return self.plant_process_force_noise_std != 0.0

    def as_dict(self) -> dict[str, float]:
        """Return JSON-serializable config values."""

        return {
            "sensory_noise_std": self.sensory_noise_std,
            "additive_motor_noise_std": self.additive_motor_noise_std,
            "signal_dependent_motor_noise_std": self.signal_dependent_motor_noise_std,
            "plant_process_force_noise_std": self.plant_process_force_noise_std,
        }


class PlantProcessForceNoise(Component):
    """Additive force/load noise at the plant boundary.

    This component deliberately operates on the force port, not on arbitrary
    position/velocity state coordinates. When a ``SimpleFeedback`` graph has a
    force filter, this node is inserted downstream of that filter, so the
    sampled disturbance bypasses command filtering as a plant/load disturbance.
    """

    input_ports = ("force",)
    output_ports = ("force",)

    noise_func: Normal
    label: str = field(default=PLANT_PROCESS_FORCE_NOISE_LABEL, static=True)

    def __init__(
        self,
        std: float,
        *,
        label: str = PLANT_PROCESS_FORCE_NOISE_LABEL,
    ):
        self.noise_func = Normal(std=float(std))
        self.label = label

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        force = inputs["force"]
        noise = self.noise_func(key, force)
        return {"force": force + noise}, state


def stochastic_runtime_config_from_model(model_hps: Any) -> StochasticRuntimeConfig:
    """Build explicit stochastic runtime config from model hyperparameters.

    Backwards compatibility:
    - ``feedback_noise_std`` aliases to ``sensory_noise_std``.
    - ``motor_noise_std`` aliases to the legacy Feedbax command model:
      signal-dependent std ``motor_noise_std`` plus additive std
      ``1.8 * motor_noise_std``.

    Explicit new fields win when present.
    """

    legacy_feedback = float(getattr(model_hps, "feedback_noise_std", 0.0) or 0.0)
    legacy_motor = float(getattr(model_hps, "motor_noise_std", 0.0) or 0.0)
    sensory = _float_attr(model_hps, "sensory_noise_std", legacy_feedback)
    signal_dependent = _float_attr(
        model_hps,
        "signal_dependent_motor_noise_std",
        legacy_motor,
    )
    additive = _float_attr(
        model_hps,
        "additive_motor_noise_std",
        1.8 * legacy_motor,
    )
    process = _float_attr(model_hps, "plant_process_force_noise_std", 0.0)
    return StochasticRuntimeConfig(
        sensory_noise_std=sensory,
        additive_motor_noise_std=additive,
        signal_dependent_motor_noise_std=signal_dependent,
        plant_process_force_noise_std=process,
    )


def command_motor_noise_func(config: StochasticRuntimeConfig):
    """Return the Feedbax command-channel noise function for ``config``."""

    return Multiplicative(Normal(std=config.signal_dependent_motor_noise_std)) + Normal(
        std=config.additive_motor_noise_std
    )


def apply_stochastic_runtime_to_model(
    model: PyTree,
    config: StochasticRuntimeConfig,
    *,
    include_plant_process_force_noise: bool = True,
) -> PyTree:
    """Patch a Feedbax ``SimpleFeedback`` graph with explicit noise semantics."""

    efferent = model.nodes["efferent"]
    if not isinstance(efferent, Channel):
        raise TypeError(
            "Expected the Feedbax efferent node to be a Channel before applying "
            f"stochastic runtime wiring; got {type(efferent).__name__}"
        )
    efferent = Channel(
        delay=efferent.delay,
        noise_func=command_motor_noise_func(config),
        add_noise=config.has_command_noise,
        input_proto=efferent.input_proto,
        init_value=efferent.init_value,
    )
    model = eqx.tree_at(
        lambda graph: graph.nodes["efferent"],
        model,
        efferent,
    )
    if include_plant_process_force_noise and config.has_plant_process_force_noise:
        model = add_plant_process_force_noise(model, config.plant_process_force_noise_std)
    return model


def add_plant_process_force_noise(
    model: PyTree,
    std: float,
    *,
    label: str = PLANT_PROCESS_FORCE_NOISE_LABEL,
) -> PyTree:
    """Insert additive plant/load force noise immediately upstream of mechanics."""

    if label in getattr(model, "nodes", {}):
        return eqx.tree_at(
            lambda graph: graph.nodes[label].noise_func,
            model,
            Normal(std=float(std)),
        )

    incoming = [
        wire
        for wire in getattr(model, "wires", ())
        if wire.target_node == "mechanics" and wire.target_port == "force"
    ]
    if len(incoming) != 1:
        raise ValueError(
            "Expected exactly one force wire into mechanics before inserting "
            f"{label!r}; found {len(incoming)}"
        )
    wire = incoming[0]
    graph = model.remove_wire(wire)
    graph = graph.add_node(label, PlantProcessForceNoise(std=std, label=label))
    graph = graph.add_wire(Wire(wire.source_node, wire.source_port, label, "force"))
    graph = graph.add_wire(Wire(label, "force", "mechanics", "force"))
    return graph


def graphspec_noise_contract(config: StochasticRuntimeConfig) -> dict[str, Any]:
    """Return the GraphSpec/manifest stochastic-runtime contract block."""

    return {
        **config.as_dict(),
        "sensory_runtime": "Feedbax feedback Channel before controller",
        "command_runtime": (
            "Feedbax efferent Channel before force_filter; additive and "
            "signal-dependent command noise are both pre-filter"
        ),
        "plant_process_runtime": (
            "RLRMP force-channel node immediately upstream of mechanics; "
            "inserted after force_filter and after plant_intervenor when present"
        ),
        "state_diffusion": "not_used",
    }


def _float_attr(obj: Any, attr: str, default: float) -> float:
    value = getattr(obj, attr, default)
    if value is None:
        return float(default)
    return float(value)
