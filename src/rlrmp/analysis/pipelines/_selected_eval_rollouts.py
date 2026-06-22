"""Private JAX-backed rollout products for selected-eval materializers.

This module is an internal optimization layer for the current RLRMP post-run
diagnostic bundle implementations. It is deliberately not a durable Feedbax or
RLRMP data-product contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.pipelines.gru_pilot_figures import (
    initial_effector_velocity,
    trial_effector_target_position,
)


@dataclass(frozen=True)
class SelectedEvalRolloutProduct:
    """Device-backed rollout leaves plus explicit host materialization methods.

    Core rollout arrays have shape ``(replicate, trial, time, feature)``. Initial
    and target arrays follow the selected validation-trial batch shape.
    """

    position: Any
    velocity: Any
    command: Any
    hidden: Any
    gru_input: Any
    initial_position: Any
    initial_velocity: Any
    target_position: Any
    dt: float
    checkpoint_selection: tuple[Any, ...] = ()
    mechanics_vector: Any | None = None
    feedback: Any | None = None

    @classmethod
    def from_states(
        cls,
        states: Any,
        trial_specs: Any,
        *,
        dt: float,
        checkpoint_selection: tuple[Any, ...] = (),
        include_mechanics_vector: bool = False,
        include_feedback: bool = False,
    ) -> "SelectedEvalRolloutProduct":
        """Build a private rollout product without copying rollout leaves to host."""

        mechanics_vector = (
            jnp.asarray(states.mechanics.vector, dtype=jnp.float64)
            if include_mechanics_vector
            else None
        )
        feedback = (
            jnp.asarray(states.sensory.output, dtype=jnp.float64)
            if include_feedback
            else None
        )
        return cls(
            position=jnp.asarray(states.mechanics.effector.pos, dtype=jnp.float64),
            velocity=jnp.asarray(states.mechanics.effector.vel, dtype=jnp.float64),
            command=jnp.asarray(states.net.output, dtype=jnp.float64),
            hidden=jnp.asarray(states.net.hidden, dtype=jnp.float64),
            gru_input=jnp.asarray(states.net.input, dtype=jnp.float64),
            initial_position=jnp.asarray(initial_effector_position(trial_specs), dtype=jnp.float64),
            initial_velocity=jnp.asarray(initial_effector_velocity(trial_specs), dtype=jnp.float64),
            target_position=jnp.asarray(trial_effector_target_position(trial_specs), dtype=jnp.float64),
            dt=float(dt),
            checkpoint_selection=tuple(checkpoint_selection),
            mechanics_vector=mechanics_vector,
            feedback=feedback,
        )

    def host_rollout_kwargs(self) -> dict[str, Any]:
        """Return kwargs for the existing public ``RolloutEvaluation`` dataclass."""

        return {
            "position": self.host_array(self.position),
            "velocity": self.host_array(self.velocity),
            "command": self.host_array(self.command),
            "hidden": self.host_array(self.hidden),
            "gru_input": self.host_array(self.gru_input),
            "initial_position": self.host_array(self.initial_position),
            "initial_velocity": self.host_array(self.initial_velocity),
            "target_position": self.host_array(self.target_position),
            "dt": float(self.dt),
            "checkpoint_selection": tuple(self.checkpoint_selection),
        }

    def to_rollout_evaluation(self, rollout_type: type[Any]) -> Any:
        """Materialize the existing host-backed public rollout object."""

        rollout = rollout_type(**self.host_rollout_kwargs())
        if self.mechanics_vector is not None:
            object.__setattr__(
                rollout,
                "mechanics_vector",
                self.host_array(self.mechanics_vector),
            )
        return rollout

    @staticmethod
    def host_array(value: Any) -> np.ndarray:
        """Copy one array-like value to host NumPy at an explicit boundary."""

        return np.asarray(value, dtype=np.float64)


def initial_effector_position(trial_specs: Any) -> Any:
    """Return the selected trial batch's effector initial position."""

    for init_state in trial_specs.inits.values():
        position = getattr(init_state, "pos", None)
        if position is not None:
            return position
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 2:
            return jnp.asarray(init_state)[..., 0:2]
    raise ValueError("Trial spec does not include an effector position initial state")
