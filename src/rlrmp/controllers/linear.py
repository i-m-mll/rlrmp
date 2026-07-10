"""LEGACY (frozen 2026-07-04, issue fac28cb).

Retained only for the archived-recipe conversion path (issue ae15851) and the
sign-convention regression test. Superseded by feedbax's
AffineFeedbackController; not a pattern for new work.

Linear time-varying (LTV) controllers for the decoupling acid test (MVP).

Implements two architectures sharing the same plant + feedback wiring as
``feedbax.xabdeef.models.point_mass_nn`` but replacing ``SimpleStagedNetwork``
with explicit LTV control laws:

- ``LinearController`` — pure regulator on the target-relative error state.
  ``u_t = -K_t · e_t`` where ``e_t = (pos - target_pos, vel)``.

- ``LinearTrackerController`` — adds a separately parameterised time-varying
  feedforward term: ``u_t = u_ff(t) - K_t · e_t``.

The discriminator hypothesis (Bug: 410d7ac) is that the tracker parameterisation
gives the optimiser freedom to decouple the open-loop feedforward drive from
the closed-loop feedback stiffness, suppressing the velocity-inflation signature
Δv > 0 that linear regulators exhibit.

Design notes
------------
- ``x_nom(t) ≡ 0`` in the target-relative frame, so the regulator and tracker
  differ purely by the additive ``u_ff(t)`` channel. The algebraic identity
  noted in d448c9d comment d444498 (tracker collapses to regulator when
  ``u_ff = -K·x_nom``) does NOT collapse here because ``u_ff`` is freely
  parameterised, not constrained to ``-K·x_nom``.
- The per-trial timestep counter lives in ``state.hidden`` (a scalar). It is
  reset to 0 at every trial boundary by feedbax's ``init_state_from_component``.
- The controllers satisfy the feedbax ``Component`` interface
  (input_ports=("input", "feedback"), output_ports=("output", "hidden"),
  state-bearing ``StateIndex`` with a ``NetworkState`` carrying ``hidden`` and
  ``output``), so the existing rlrmp loss functions and training loop work
  unchanged — only ``_get_trainable`` needs to know about the new parameter
  shapes.

Bug: 410d7ac
"""

import jax.numpy as jnp
import jax.random as jr
from equinox import field
from equinox.nn import State, StateIndex
from feedbax.models.networks import NetworkState
from feedbax.runtime.graph import Component
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_target_pos(input_value: PyTree) -> Float[Array, "2"]:
    """Extract the per-step target position ``(2,)`` from the task input pytree.

    The ``input_value`` is one timestep slice of ``trial_spec.inputs`` —
    typically a dict like ``{"task": DelayedReachTaskInputs(...), "sisu": ...,
    "intervene:plant_intervenor": ...}``. The target position is nested at
    ``task.effector_target.pos`` (shape ``(2,)`` after the per-step slice).

    Args:
        input_value: One timestep of the trial inputs.

    Returns:
        ``(2,)`` array of target position in workspace coordinates.
    """
    # Walk the structure tolerantly: support both dict-of-modules and the
    # bare DelayedReachTaskInputs case used during testing.
    if isinstance(input_value, dict):
        task_input = input_value.get("task", None)
        if task_input is None:
            # Fall back to scanning for a CartesianState-like child.
            for v in input_value.values():
                if hasattr(v, "effector_target"):
                    task_input = v
                    break
    else:
        task_input = input_value
    if task_input is None or not hasattr(task_input, "effector_target"):
        raise ValueError(
            "LinearController could not locate effector_target.pos in input pytree"
        )
    return task_input.effector_target.pos


def _flatten_feedback(feedback_value: PyTree) -> Array:
    """Flatten the feedback pytree into a 1-D array. For point-mass feedback
    this yields ``(pos_x, pos_y, vel_x, vel_y)`` of length 4."""
    flat, _ = ravel_pytree(feedback_value)
    return flat


# ---------------------------------------------------------------------------
# LinearController — pure LTV regulator on target-relative state
# ---------------------------------------------------------------------------


class LinearController(Component):
    """LTV regulator ``u_t = -K_t · e_t`` with ``e_t = (pos - target, vel)``.

    The single learnable parameter is the gain tensor ``K_t`` of shape
    ``(T, n_controls, n_states) = (n_steps - 1, 2, 4)``. The state-relative
    framing (subtracting ``target_pos`` from the position channels) is what
    makes a single time-varying gain capable of solving a random-target reach
    task; without it the regulator could only drive the plant to the origin.

    Attributes:
        K: ``(T, n_controls, n_states)`` gain tensor.
        n_steps: Number of simulation steps (``T``).
        n_controls: Dimensionality of the control output (2 for point-mass).
        n_states: Dimensionality of the state vector (4 = pos_xy, vel_xy).
        state_index: StateIndex carrying a ``NetworkState`` with a scalar
            ``hidden`` step counter and the output force.
    """

    K: Float[Array, "T n_controls n_states"]
    n_steps: int = field(static=True)
    n_controls: int = field(static=True)
    n_states: int = field(static=True)
    out_size: int = field(static=True)  # required by SimpleFeedback (force_filter sizing)
    state_index: StateIndex
    _initial_state: NetworkState = field(static=True)

    input_ports = ("input", "feedback")
    output_ports = ("output", "hidden")

    def __init__(
        self,
        n_steps: int,
        n_controls: int = 2,
        n_states: int = 4,
        K_init_scale: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialise an LTV regulator.

        Args:
            n_steps: Number of simulation timesteps (the K tensor's first dim).
            n_controls: Number of control output channels.
            n_states: Number of (target-relative) state features.
            K_init_scale: Standard deviation for Gaussian K initialisation.
                Default 0.0 — start from zero gains so the controller initially
                emits no force; the loss-driven gradient then carves out a
                non-trivial gain schedule. This is the simplest baseline; a
                tiny non-zero init was tried but did not change the qualitative
                outcome.
            key: PRNG key for initialisation.
        """
        if K_init_scale > 0.0:
            self.K = K_init_scale * jr.normal(key, (n_steps, n_controls, n_states))
        else:
            self.K = jnp.zeros((n_steps, n_controls, n_states))
        self.n_steps = int(n_steps)
        self.n_controls = int(n_controls)
        self.n_states = int(n_states)
        self.out_size = int(n_controls)

        # NetworkState compatibility: loss functions read state.net.hidden and
        # state.net.output. The "hidden" channel here doubles as the timestep
        # counter (a scalar), and "output" carries the most recent control.
        init_state = NetworkState(
            input=jnp.zeros(0),
            hidden=jnp.zeros((1,)),  # scalar counter wrapped as (1,) for nn_hidden L2
            output=jnp.zeros(n_controls),
            encoding=None,
        )
        self._initial_state = init_state
        self.state_index = StateIndex(init_state)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        net_state: NetworkState = state.get(self.state_index)

        # Extract step counter (scalar) from hidden state. We store the step
        # index as net_state.hidden[0] and increment it each call.
        t = jnp.asarray(net_state.hidden[0], dtype=jnp.int32)
        t = jnp.clip(t, 0, self.n_steps - 1)

        target_pos = _extract_target_pos(inputs.get("input", None))
        feedback_flat = _flatten_feedback(inputs.get("feedback", None))
        # feedback_flat is (pos_x, pos_y, vel_x, vel_y) for point-mass with
        # ChannelSpec returning (pos, vel) tuple. Build the target-relative
        # error vector.
        e_pos = feedback_flat[:2] - target_pos
        e_vel = feedback_flat[2:]
        e = jnp.concatenate([e_pos, e_vel], axis=-1)  # (4,)

        u = self._control(t, e)

        new_hidden = jnp.array([t + 1], dtype=jnp.float32)
        new_state = NetworkState(
            input=jnp.zeros(0),
            hidden=new_hidden,
            output=u,
            encoding=None,
        )
        state = state.set(self.state_index, new_state)
        return {"output": u, "hidden": new_hidden}, state

    def _control(self, t: Array, error: Array) -> Array:
        """Return the control command for one step and target-relative error."""

        return -jnp.dot(self.K[t], error)


# ---------------------------------------------------------------------------
# LinearTrackerController — LTV regulator + independent LTV feedforward
# ---------------------------------------------------------------------------


class LinearTrackerController(LinearController):
    """LTV tracker ``u_t = u_ff(t) - K_t · e_t`` with ``e_t = (pos - target, vel)``.

    Adds a freely-parameterised per-step feedforward vector ``u_ff(t)`` of shape
    ``(T, n_controls)`` to the regulator. Because ``u_ff`` is independent of
    ``K_t``, the optimiser is free to find a decomposition that decouples the
    open-loop drive from the closed-loop stiffness (discriminator hypothesis,
    Bug: 410d7ac).

    Attributes:
        K: ``(T, n_controls, n_states)`` LTV gain tensor.
        u_ff: ``(T, n_controls)`` LTV feedforward signal.
        n_steps, n_controls, n_states: Static dimensions.
        state_index: StateIndex for the per-step counter + output cache.
    """

    u_ff: Float[Array, "T n_controls"]

    def __init__(
        self,
        n_steps: int,
        n_controls: int = 2,
        n_states: int = 4,
        K_init_scale: float = 0.0,
        u_ff_init_scale: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialise an LTV tracker.

        Args:
            n_steps: Number of simulation timesteps.
            n_controls: Number of control output channels.
            n_states: Number of (target-relative) state features.
            K_init_scale: Std for Gaussian K initialisation (default 0.0).
            u_ff_init_scale: Std for Gaussian u_ff initialisation (default 0.0).
            key: PRNG key for initialisation.
        """
        key_k, key_uff = jr.split(key, 2)
        super().__init__(
            n_steps=n_steps,
            n_controls=n_controls,
            n_states=n_states,
            K_init_scale=K_init_scale,
            key=key_k,
        )
        if u_ff_init_scale > 0.0:
            self.u_ff = u_ff_init_scale * jr.normal(key_uff, (n_steps, n_controls))
        else:
            self.u_ff = jnp.zeros((n_steps, n_controls))

    def _control(self, t: Array, error: Array) -> Array:
        """Return the regulator command plus the independent feedforward term."""

        return self.u_ff[t] + super()._control(t, error)
