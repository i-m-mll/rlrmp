"""Induced gain ``||T_{w -> z}||_inf`` analyser for trained checkpoints.

This module computes the closed-loop disturbance-to-error H-infinity norm for
an arbitrary controller (Riccati LTI, LQR LTI, or trained nonlinear RNN) on a
given linearised plant, over a finite-horizon reach. The result is the
empirical ``gamma`` of the controller -- directly comparable across
checkpoints and to the H-infinity Riccati ``gamma_star`` for sanity round-trip.

Bug: 74bfd86 -- production induced-gain analyser, supports the synthesis-fix
phase (``b557d4e``) by giving a single scalar comparable across all existing
trained networks. See the issue body and the 2026-05-07 settled-design comment
for the spec.

Three orthogonal axes
---------------------

- **Linearisation regime.** Trajectory (LTV, primary) and fixed-point (LTI,
  auxiliary). Both code paths reuse this module's linearisers; the chosen
  algorithm decides which is needed.
- **Disturbance channel ``w``.** ``additive_force`` (flavor-(a) point-mass
  force, matches APT/minimax adversary), ``sensory_perturbation``
  (deterministic L2-bounded perturbation on the RNN's observation pathway —
  the H-inf framing makes no distinction between this and stochastic noise),
  and ``structural_da`` (small-gain framing for flavor-(b) ``Delta A`` plant
  uncertainty).
- **Performance channel ``z``.** ``qr_cost`` (cost-matched to Riccati,
  literal-comparable to ``gamma_star``), ``control`` (gain to control effort),
  ``state_error`` (gain to state deviation), ``peak_velocity`` (behavioural,
  reports peak forward + peak lateral velocity under the worst-case ``w``).

Algorithms
----------

- **Power iteration on the LTV trajectory operator** (primary). Forward sweep
  computes ``z = T w`` via ``jax.lax.scan``; adjoint sweep is generated with
  ``jax.linear_transpose`` (avoids manually implementing the backward time
  recursion and the sign-flip pitfalls that come with it). Iterates with
  random restarts; reports the largest singular value of the operator.
- **Hamiltonian bisection on the LTI fixed-point** (auxiliary). Bisects on
  ``gamma`` and tests admissibility via the discrete-time bounded-real lemma
  Riccati (the same admissibility primitive used in ``hinf_riccati``). This
  is mathematically equivalent to the symplectic-Hamiltonian eigenvalue test
  but reuses the well-tested production primitive.

The two algorithms agree on a long-hold trajectory (the trajectory tail
linearisation matches the fixed-point linearisation by construction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from rlrmp.analysis.hinf_riccati import (
    CostSchedule,
    PlantLinearization,
    make_reach_initial_state,
)

# Enable x64. Idempotent. The induced-gain power iteration accumulates many
# matrix-vector products and can drift in float32 on long horizons.
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


# =============================================================================
# Channel sentinels
# =============================================================================

# String-keyed dispatch. Defined as module-level constants for IDE-discoverable
# autocompletion; runtime checks accept either the constant or the literal
# string. Keeps the dispatch logic disciplined without adding an enum-shaped
# layer of indirection.

W_ADDITIVE_FORCE = "additive_force"
W_SENSORY_PERTURBATION = "sensory_perturbation"
W_STRUCTURAL_DA = "structural_da"

# Deprecated alias for one cycle. Bug: ec7710f — H-inf doesn't distinguish
# stochastic noise from L2-bounded perturbation. New code should use
# ``W_SENSORY_PERTURBATION`` / the literal ``"sensory_perturbation"``.
W_SENSORY_NOISE = W_SENSORY_PERTURBATION  # deprecated alias

Z_QR_COST = "qr_cost"
Z_CONTROL = "control"
Z_STATE_ERROR = "state_error"
Z_PEAK_VELOCITY = "peak_velocity"

_W_CHANNELS = frozenset({W_ADDITIVE_FORCE, W_SENSORY_PERTURBATION, W_STRUCTURAL_DA})
_Z_CHANNELS = frozenset({Z_QR_COST, Z_CONTROL, Z_STATE_ERROR, Z_PEAK_VELOCITY})


def _validate_w_channel(name: str) -> str:
    # Deprecated alias: "sensory_noise" -> "sensory_perturbation". Bug: ec7710f.
    if name == "sensory_noise":
        name = W_SENSORY_PERTURBATION
    if name not in _W_CHANNELS:
        raise ValueError(
            f"Unknown w channel {name!r}. Valid: {sorted(_W_CHANNELS)}."
        )
    return name


def _validate_z_channel(name: str) -> str:
    if name not in _Z_CHANNELS:
        raise ValueError(
            f"Unknown z channel {name!r}. Valid: {sorted(_Z_CHANNELS)}."
        )
    return name


# =============================================================================
# Controller protocol and adapters
# =============================================================================


class Controller(Protocol):
    """Closed-loop controller interface.

    The induced-gain analyser is controller-agnostic. Both LTI controllers
    (Riccati / LQR feedback gains ``K[t]``) and nonlinear RNN controllers
    (feedbax models) are handled by this protocol.

    Implementations must be pure: ``step`` is a deterministic function of
    ``(h, sensory_obs, t)``.

    Methods:
        initial_state: Build the initial controller hidden state.
        step: Advance the controller by one timestep.
    """

    def initial_state(self) -> Float[Array, "n_ctrl"]:
        """Return the initial hidden state.

        Returns:
            Array of shape ``(n_ctrl,)``. ``n_ctrl == 0`` for stateless
            controllers (e.g. LTI feedback) is allowed; the augmented closed
            loop reduces to plant dimension in that case.
        """
        ...

    def step(
        self,
        h: Float[Array, "n_ctrl"],
        sensory_obs: Float[Array, "n_obs"],
        t: int,
    ) -> Tuple[Float[Array, "n_ctrl"], Float[Array, "m_u"]]:
        """Advance one timestep.

        Args:
            h: Controller hidden state, shape ``(n_ctrl,)``.
            sensory_obs: Observation passed to the controller, shape
                ``(n_obs,)``. For LTI controllers this is the plant state.
            t: Discrete time index (used by time-varying gains).

        Returns:
            ``(h_next, u)`` where ``h_next`` has shape ``(n_ctrl,)`` and
            ``u`` has shape ``(m_u,)``.
        """
        ...


@dataclass(frozen=True)
class _LTIController:
    """LTI controller wrapping a time-varying feedback gain ``K[t]``.

    Stateless: ``n_ctrl == 0``. The augmented state reduces to the plant
    state. The controller observes the full plant state (full-state
    feedback).

    Attributes:
        K: Feedback gain sequence, shape ``(T, m_u, n)``. Control is
            ``u_t = -K[t] x_t``.
    """

    K: Float[Array, "T m_u n"]

    def initial_state(self) -> Float[Array, "0"]:
        return jnp.zeros((0,), dtype=jnp.float64)

    def step(
        self,
        h: Float[Array, "0"],
        sensory_obs: Float[Array, "n"],
        t: int,
    ) -> Tuple[Float[Array, "0"], Float[Array, "m_u"]]:
        u = -self.K[t] @ sensory_obs
        return h, u


def lti_controller(K: Float[Array, "T m_u n"]) -> Controller:
    """Build a stateless LTI controller from a time-varying feedback gain.

    Use to drop a Riccati (`hinf_riccati.solve_hinf_riccati`) or LQR
    (`hinf_riccati.solve_lqr`) gain into the analyser for the round-trip
    sanity check ``gamma == gamma_star``.

    Args:
        K: Time-varying feedback gain, shape ``(T, m_u, n)``.

    Returns:
        A ``Controller`` whose ``step`` returns ``u_t = -K[t] x_t`` and
        carries no hidden state (``n_ctrl == 0``).
    """
    return _LTIController(K=jnp.asarray(K, dtype=jnp.float64))


@dataclass(frozen=True)
class _FeedbaxGraphController:
    """Adapter wrapping a feedbax ``Graph`` as a ``Controller``.

    The graph's state is an opaque ``equinox.nn.State`` pytree. The analyser's
    ``Controller`` protocol expects ``h`` to be a 1D array (so it can stack
    with ``x_plant`` into ``x_aug`` and feed Jacobians through it). This
    adapter handles flatten/unflatten between the two representations:

    - ``initial_state()`` calls ``graph.init_state(key=...)``, then flattens
      the resulting State pytree's array leaves into a single 1D float64
      array, captures the treedef + leaf shapes for round-tripping.
    - ``step(h, obs, t)`` unflattens ``h`` back into a State, calls
      ``graph._call_single_step(inputs={"input": obs}, state=state, key=key_t)``,
      extracts ``u`` from outputs (default port ``"output"``), reflattens the
      next State to ``h_next``.

    Bug: b131510 -- replaces the legacy ``_FeedbaxRNNController`` adapter
    which assumed the caller had pre-extracted a step closure (an old
    staged-model API pattern). The new graph API exposes per-step evaluation
    directly on the Graph, so no manual extraction is needed.

    Attributes:
        graph: The feedbax ``Graph`` to wrap. Must accept an ``"input"``
            external input binding (the sensory observation channel) and emit
            an ``"output"`` external output binding (the control vector).
        h0_flat: Flattened initial state, shape ``(n_ctrl,)``.
        treedef: PyTree treedef of the original ``State`` for round-tripping.
        leaf_shapes: Per-leaf shapes for unflatten reshape.
        leaf_sizes: Per-leaf flat sizes for slicing the 1D ``h``.
        key: Base PRNGKey; per-step keys are derived by folding-in ``t``.
        input_port: External input binding name (default ``"input"``).
        output_port: External output binding name (default ``"output"``).
    """

    graph: object  # feedbax.graph.Graph
    h0_flat: Float[Array, "n_ctrl"]
    treedef: object  # jax tree treedef
    leaf_shapes: tuple
    leaf_sizes: tuple
    key: Array
    input_port: str = "input"
    output_port: str = "output"

    def initial_state(self) -> Float[Array, "n_ctrl"]:
        return self.h0_flat

    def _unflatten(self, h: Float[Array, "n_ctrl"]):
        # Slice flat h back into per-leaf arrays of original shapes.
        leaves = []
        offset = 0
        for shape, size in zip(self.leaf_shapes, self.leaf_sizes):
            leaves.append(h[offset:offset + size].reshape(shape))
            offset += size
        return jax.tree.unflatten(self.treedef, leaves)

    def _flatten(self, state) -> Float[Array, "n_ctrl"]:
        leaves = jax.tree.leaves(state)
        # Filter to array leaves matching our schema (size-keyed).
        flat_parts = [jnp.asarray(leaf, dtype=jnp.float64).reshape(-1) for leaf in leaves]
        if not flat_parts:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.concatenate(flat_parts, axis=0)

    def step(
        self,
        h: Float[Array, "n_ctrl"],
        sensory_obs: Float[Array, "n_obs"],
        t: int,
    ) -> Tuple[Float[Array, "n_ctrl"], Float[Array, "m_u"]]:
        state = self._unflatten(h)
        # Per-step key: fold t in for determinism without leaking randomness
        # across timesteps (analyser is deterministic given the key).
        key_t = jax.random.fold_in(self.key, t)
        outputs, state_next = self.graph._call_single_step(
            inputs={self.input_port: sensory_obs},
            state=state,
            key=key_t,
        )
        u = outputs[self.output_port]
        h_next = self._flatten(state_next)
        return h_next, u


def feedbax_graph_controller(
    graph,  # feedbax.graph.Graph
    *,
    key: Array,
    input_port: str = "input",
    output_port: str = "output",
) -> Controller:
    """Wrap a feedbax ``Graph`` as a ``Controller`` for the induced-gain analyser.

    The new feedbax eager-graph API (post-cutover, on ``develop``) exposes
    per-step evaluation directly on ``Graph`` via ``_call_single_step``. The
    induced-gain analyser only needs that single-step view; this adapter
    bridges it to the analyser's ``(h, obs, t) -> (h_next, u)`` protocol.

    The graph's opaque ``State`` pytree is flattened to a 1D float64 array
    (the adapter's ``h``), threading per-step keys via ``jax.random.fold_in``
    on the supplied base ``key``.

    Bug: b131510 -- replaces the legacy ``feedbax_rnn_controller(step_fn, h0)``
    helper, which required the caller to manually extract a step closure
    from staged-API plumbing. The graph API makes that extraction trivial,
    so the adapter takes the graph directly.

    Args:
        graph: A feedbax ``Graph``. Should expose an ``input_port`` external
            input binding (the sensory observation) and an ``output_port``
            external output binding (the control vector). For the rlrmp
            ``SimpleFeedback`` family of trained models, the relevant
            sub-graph is typically the network alone — see the test for a
            minimal pass-through example.
        key: Base PRNGKey. The adapter folds-in the discrete time index ``t``
            to derive a per-step key, so the controller is deterministic
            given ``key`` and ``t``.
        input_port: External input binding name. Default ``"input"``.
        output_port: External output binding name. Default ``"output"``.

    Returns:
        A ``Controller`` whose ``initial_state`` and ``step`` route through
        ``graph.init_state`` and ``graph._call_single_step`` respectively.
    """
    init_state = graph.init_state(key=key)
    leaves = jax.tree.leaves(init_state)
    leaf_shapes = tuple(tuple(jnp.asarray(leaf).shape) for leaf in leaves)
    leaf_sizes = tuple(int(jnp.asarray(leaf).size) for leaf in leaves)
    treedef = jax.tree.structure(init_state)
    if leaves:
        h0_flat = jnp.concatenate(
            [jnp.asarray(leaf, dtype=jnp.float64).reshape(-1) for leaf in leaves],
            axis=0,
        )
    else:
        h0_flat = jnp.zeros((0,), dtype=jnp.float64)
    return _FeedbaxGraphController(
        graph=graph,
        h0_flat=h0_flat,
        treedef=treedef,
        leaf_shapes=leaf_shapes,
        leaf_sizes=leaf_sizes,
        key=key,
        input_port=input_port,
        output_port=output_port,
    )


# Deprecated alias kept for one cycle. Bug: b131510. The old signature
# (step_fn, h0) doesn't match the graph API; new code must use
# ``feedbax_graph_controller(graph, key=...)``.
def feedbax_rnn_controller(*args, **kwargs):
    """Deprecated. Use ``feedbax_graph_controller(graph, key=...)`` instead.

    Bug: b131510 -- the old ``feedbax_rnn_controller(step_fn, h0)`` adapter
    implicitly assumed feedbax's pre-eager-graph staged-model API. The new
    graph API exposes per-step evaluation directly on ``Graph``; pass the
    graph object instead of a hand-extracted step closure.
    """
    raise NotImplementedError(
        "feedbax_rnn_controller is deprecated (Bug: b131510). "
        "Use feedbax_graph_controller(graph, key=...) — see its docstring."
    )


# =============================================================================
# Linearisation value types
# =============================================================================


@dataclass(frozen=True)
class TrajectoryLinearisation:
    """LTV linearisation of the closed loop along a nominal reach.

    The closed-loop state is augmented with the controller's hidden state:

    .. math::

        x_{aug,t} = (x_{plant,t},\\; h_{ctrl,t}),

    so that ``A_t``, ``B_{w,t}``, ``C_{z,t}`` are Jacobians of the augmented
    closed-loop step. For LTI controllers ``n_ctrl == 0`` and the augmented
    dimension reduces to the plant dimension.

    Attributes:
        A_t: Augmented closed-loop dynamics, shape ``(T, n_aug, n_aug)``.
        Bw_t: Disturbance-input matrix (channel-specific), shape
            ``(T, n_aug, n_w)``.
        Cz_t: Performance-output matrix (channel-specific), shape
            ``(T, n_z, n_aug)``.
        D_t: Feedthrough, shape ``(T, n_z, n_w)``.
        x_nominal: Nominal augmented trajectory, shape ``(T+1, n_aug)``.
        u_nominal: Nominal control, shape ``(T, m_u)``.
        dt: Timestep (s).
        n_plant: Plant state dimension.
        n_ctrl: Controller hidden-state dimension (0 for LTI).
        w_channel: Disturbance channel name.
        z_channel: Performance channel name.
    """

    A_t: Float[Array, "T n_aug n_aug"]
    Bw_t: Float[Array, "T n_aug n_w"]
    Cz_t: Float[Array, "T n_z n_aug"]
    D_t: Float[Array, "T n_z n_w"]
    x_nominal: Float[Array, "T_plus_1 n_aug"]
    u_nominal: Float[Array, "T m_u"]
    dt: float
    n_plant: int
    n_ctrl: int
    w_channel: str
    z_channel: str

    @property
    def T(self) -> int:
        return int(self.A_t.shape[0])

    @property
    def n_aug(self) -> int:
        return int(self.A_t.shape[1])

    @property
    def n_w(self) -> int:
        return int(self.Bw_t.shape[2])

    @property
    def n_z(self) -> int:
        return int(self.Cz_t.shape[1])


@dataclass(frozen=True)
class FixedPointLinearisation:
    """LTI linearisation of the closed loop at the held-target equilibrium.

    Attributes:
        A: Augmented dynamics, shape ``(n_aug, n_aug)``.
        Bw: Disturbance-input matrix, shape ``(n_aug, n_w)``.
        Cz: Performance-output matrix, shape ``(n_z, n_aug)``.
        D: Feedthrough, shape ``(n_z, n_w)``.
        x_star: Equilibrium augmented state, shape ``(n_aug,)``.
        n_plant: Plant state dimension.
        n_ctrl: Controller hidden-state dimension.
        w_channel: Disturbance channel name.
        z_channel: Performance channel name.
    """

    A: Float[Array, "n_aug n_aug"]
    Bw: Float[Array, "n_aug n_w"]
    Cz: Float[Array, "n_z n_aug"]
    D: Float[Array, "n_z n_w"]
    x_star: Float[Array, "n_aug"]
    n_plant: int
    n_ctrl: int
    w_channel: str
    z_channel: str


@dataclass(frozen=True)
class InducedGainResult:
    """Induced gain ``||T_{w->z}||_inf`` with channel and method breakdown.

    Attributes:
        gamma: The induced gain. ``inf`` if the algorithm reported instability.
        method: Algorithm used: ``"power_iteration"`` or ``"hamiltonian"``.
        w_channel: Disturbance channel.
        z_channel: Performance channel.
        worst_case_w: Worst-case disturbance trajectory recovered from the
            algorithm. For power iteration: shape ``(T, n_w)``. For
            Hamiltonian: ``None`` (LTI fixed-point analysis returns a single
            number, not a trajectory).
        worst_case_trajectory: Closed-loop state trajectory under
            ``worst_case_w`` (only populated for power iteration, optional).
        peak_forward_velocity: Peak signed velocity along the reach axis under
            ``worst_case_w`` (only meaningful for ``z_channel == "peak_velocity"``;
            ``None`` otherwise).
        peak_lateral_velocity: Peak velocity perpendicular to the reach axis
            (only meaningful for ``z_channel == "peak_velocity"``).
        iterations: Power iteration step count (0 for Hamiltonian).
        converged: Whether the algorithm reported convergence.
        diagnostics: Free-form dict for algorithm-specific extras (per-restart
            estimates, bisection bracket, etc.).
    """

    gamma: float
    method: str
    w_channel: str
    z_channel: str
    worst_case_w: Optional[Float[Array, "T n_w"]] = None
    worst_case_trajectory: Optional[Float[Array, "T_plus_1 n_aug"]] = None
    peak_forward_velocity: Optional[float] = None
    peak_lateral_velocity: Optional[float] = None
    iterations: int = 0
    converged: bool = True
    diagnostics: dict = field(default_factory=dict)


# =============================================================================
# Linearisation: trajectory (LTV) and fixed-point (LTI)
# =============================================================================


def _augmented_step(
    plant: PlantLinearization,
    controller: Controller,
    sensory_map: Callable[[Float[Array, "n"]], Float[Array, "n_obs"]],
) -> Callable[
    [Float[Array, "n_aug"], Float[Array, "m_w"], int],
    Tuple[Float[Array, "n_aug"], Float[Array, "m_u"]],
]:
    """Build the augmented closed-loop step ``f(x_aug, w, t) -> (x_aug', u)``.

    The augmented state stacks plant state and controller hidden state. For
    LTI controllers (``n_ctrl == 0``) the controller-half is empty and the
    closed loop reduces to plant-only dynamics.

    Args:
        plant: Linearised plant.
        controller: Closed-loop controller.
        sensory_map: Map ``x_plant -> sensory_obs`` for the controller. For
            full-state feedback this is the identity.

    Returns:
        A function ``(x_aug, w, t) -> (x_aug_next, u_t)``.
    """
    n_plant = plant.n
    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    Bw = plant.Bw.astype(jnp.float64)

    def step(x_aug: Float[Array, "n_aug"], w: Float[Array, "m_w"], t: int):
        x_plant = x_aug[:n_plant]
        h = x_aug[n_plant:]
        obs = sensory_map(x_plant)
        h_next, u = controller.step(h, obs, t)
        x_plant_next = A @ x_plant + B @ u + Bw @ w
        x_aug_next = jnp.concatenate([x_plant_next, h_next], axis=0)
        return x_aug_next, u

    return step


def _full_state_observer(x: Float[Array, "n"]) -> Float[Array, "n"]:
    """Identity sensory map: controller observes the full plant state."""
    return x


def linearise_trajectory(
    plant: PlantLinearization,
    controller: Controller,
    *,
    init_pos: Float[Array, "2"],
    target_pos: Float[Array, "2"],
    horizon: int,
    w_channel: str = W_ADDITIVE_FORCE,
    z_channel: str = Z_QR_COST,
    schedule: Optional[CostSchedule] = None,
    sensory_map: Optional[Callable[[Float[Array, "n"]], Float[Array, "n_obs"]]] = None,
) -> TrajectoryLinearisation:
    """Linearise the closed loop along an undisturbed nominal reach.

    Rolls a nominal reach (``w == 0``) from ``init_pos`` to ``target_pos`` and
    linearises the augmented closed-loop step at every timestep via
    ``jax.jacobian``. Yields ``(A_t, B_{w,t}, C_{z,t}, D_t)`` for the requested
    channels.

    Plant state is stored in goal-centred coordinates (matches
    ``simulate_closed_loop``): ``x_plant[0:2] = pos - target_pos``. The
    controller observes the goal-centred plant state via ``sensory_map``.

    Args:
        plant: Linearised plant.
        controller: Closed-loop controller.
        init_pos: Reach starting position, shape ``(2,)``.
        target_pos: Reach target position, shape ``(2,)``.
        horizon: Number of LTV stages T. Each stage produces one
            ``(A_t, B_{w,t}, C_{z,t}, D_t)`` tuple.
        w_channel: Disturbance channel.
        z_channel: Performance channel.
        schedule: Cost schedule (required for ``z_channel == "qr_cost"``).
        sensory_map: Map ``x_plant -> sensory_obs``. Defaults to full-state.

    Returns:
        A ``TrajectoryLinearisation``.
    """
    w_channel = _validate_w_channel(w_channel)
    z_channel = _validate_z_channel(z_channel)
    if z_channel == Z_QR_COST and schedule is None:
        raise ValueError("z_channel='qr_cost' requires a CostSchedule.")
    if schedule is not None and schedule.T < horizon:
        raise ValueError(
            f"schedule.T={schedule.T} must be >= horizon={horizon}."
        )
    if sensory_map is None:
        sensory_map = _full_state_observer

    init_pos = jnp.asarray(init_pos, dtype=jnp.float64)
    target_pos = jnp.asarray(target_pos, dtype=jnp.float64)

    # Build augmented step.
    step = _augmented_step(plant, controller, sensory_map)

    # Initial augmented state.
    x_plant_0 = make_reach_initial_state(plant, init_pos=init_pos, target_pos=target_pos)
    h_0 = controller.initial_state()
    x_aug_0 = jnp.concatenate([x_plant_0, h_0], axis=0)

    n_plant = plant.n
    n_ctrl = int(h_0.shape[0])
    n_aug = n_plant + n_ctrl
    m_w = plant.m_w

    # Roll nominal trajectory (w == 0).
    x_traj = [x_aug_0]
    u_traj: list[jnp.ndarray] = []
    w_zero = jnp.zeros((m_w,), dtype=jnp.float64)
    for t in range(horizon):
        x_next, u_t = step(x_traj[-1], w_zero, t)
        x_traj.append(x_next)
        u_traj.append(u_t)
    x_nominal = jnp.stack(x_traj, axis=0)
    u_nominal = jnp.stack(u_traj, axis=0)

    # Linearise at every step. step(x, w, t) -> (x_next, u). We need:
    #   A_t = d x_next / d x  at (x_t, 0, t)
    #   Bw_t (raw plant) = d x_next / d w at (x_t, 0, t)
    # Then we lift to the chosen w/z channel below.
    jac_x_fn = jax.jacobian(lambda x, w, t: step(x, w, t)[0], argnums=0)
    jac_w_fn = jax.jacobian(lambda x, w, t: step(x, w, t)[0], argnums=1)
    jac_xu_fn = jax.jacobian(lambda x, w, t: step(x, w, t)[1], argnums=0)
    jac_wu_fn = jax.jacobian(lambda x, w, t: step(x, w, t)[1], argnums=1)

    A_list = []
    Bw_raw_list = []
    Du_list = []  # d u / d w
    Cu_list = []  # d u / d x
    for t in range(horizon):
        A_t = jac_x_fn(x_nominal[t], w_zero, t)
        Bw_t = jac_w_fn(x_nominal[t], w_zero, t)
        Cu_t = jac_xu_fn(x_nominal[t], w_zero, t)
        Du_t = jac_wu_fn(x_nominal[t], w_zero, t)
        A_list.append(A_t)
        Bw_raw_list.append(Bw_t)
        Cu_list.append(Cu_t)
        Du_list.append(Du_t)

    A_arr = jnp.stack(A_list, axis=0)  # (T, n_aug, n_aug)
    Bw_raw = jnp.stack(Bw_raw_list, axis=0)  # (T, n_aug, m_w)  — additive_force B_w
    Cu_arr = jnp.stack(Cu_list, axis=0)  # (T, m_u, n_aug)
    Du_arr = jnp.stack(Du_list, axis=0)  # (T, m_u, m_w)

    # === w channel ===
    # Also compute Du_in_w_basis = d u / d w in the lifted w basis. This is
    # used by every z-channel branch below for the feedthrough D_z.
    # Bug: ec7710f — D_u was previously dropped on the qr_cost path.
    if w_channel == W_ADDITIVE_FORCE:
        Bw_aug = Bw_raw  # (T, n_aug, m_w)
        n_w = m_w
        # Force enters next-step state, not current control. Du_arr was
        # computed against the force-w argument; it is zero by construction
        # for any controller whose step does not read w directly. Pass it
        # through unchanged (its computation already exercises autodiff).
        Du_in_w_basis = Du_arr
    elif w_channel == W_SENSORY_PERTURBATION:
        Bw_aug, n_w, Du_in_w_basis = _sensory_perturbation_Bw(
            plant, controller, sensory_map, x_nominal, horizon
        )
    elif w_channel == W_STRUCTURAL_DA:
        # Small-gain framing: w_struct in R^{n_aug}, B_w_struct = I_n_aug, so
        # w enters as +w on the augmented state (per-component multiplier on
        # ΔA·x is bounded above by ||ΔA||_op * ||x||_2; the small-gain norm
        # we want is ||T_{w_struct -> z_struct}||_∞ with z_struct = x).
        Bw_aug = jnp.tile(jnp.eye(n_aug, dtype=jnp.float64)[None], (horizon, 1, 1))
        n_w = n_aug
        # w_struct adds to next-step augmented state. Under full-state
        # feedback at time t, u_t does not depend on w_t. D_u == 0 by
        # construction. (We could autodiff to verify, but the structural-da
        # step intentionally adds w only after the u computation.)
        Du_in_w_basis = jnp.zeros((horizon, plant.m_u, n_w), dtype=jnp.float64)
    else:  # pragma: no cover -- already validated
        raise AssertionError(w_channel)

    # === z channel ===
    if z_channel == Z_QR_COST:
        Cz, Dz = _qr_cost_Cz_Dz(
            plant, schedule, Cu_arr, Du_in_w_basis, n_aug, n_w, n_plant, horizon
        )
    elif z_channel == Z_CONTROL:
        # z = u_t = Cu_t @ x_aug + Du_t @ w. Same-time feedthrough is
        # exactly Du_in_w_basis (by definition of w basis).
        Cz = Cu_arr
        Dz = Du_in_w_basis
    elif z_channel == Z_STATE_ERROR:
        # z = x_aug[plant slice]  (we measure plant state deviation only;
        # controller hidden state is internal)
        I_plant = jnp.eye(n_plant, dtype=jnp.float64)
        Cz_block = jnp.concatenate(
            [I_plant, jnp.zeros((n_plant, n_ctrl), dtype=jnp.float64)], axis=1
        )
        Cz = jnp.tile(Cz_block[None], (horizon, 1, 1))
        Dz = jnp.zeros((horizon, n_plant, n_w), dtype=jnp.float64)
    elif z_channel == Z_PEAK_VELOCITY:
        # z = velocity components only (plant velocity slice).
        # The induced gain is computed on the L2 velocity norm; the headline
        # peak metrics are reported separately by probing the worst-case w*.
        vel_lo, vel_hi = plant.vel_slice
        n_vel = vel_hi - vel_lo
        Cz_block = jnp.zeros((n_vel, n_aug), dtype=jnp.float64)
        # Set identity on the velocity sub-block.
        for i in range(n_vel):
            Cz_block = Cz_block.at[i, vel_lo + i].set(1.0)
        Cz = jnp.tile(Cz_block[None], (horizon, 1, 1))
        Dz = jnp.zeros((horizon, n_vel, n_w), dtype=jnp.float64)
    else:  # pragma: no cover
        raise AssertionError(z_channel)

    return TrajectoryLinearisation(
        A_t=A_arr,
        Bw_t=Bw_aug,
        Cz_t=Cz,
        D_t=Dz,
        x_nominal=x_nominal,
        u_nominal=u_nominal,
        dt=plant.dt,
        n_plant=n_plant,
        n_ctrl=n_ctrl,
        w_channel=w_channel,
        z_channel=z_channel,
    )


def _sensory_perturbation_Bw(
    plant: PlantLinearization,
    controller: Controller,
    sensory_map: Callable[[Float[Array, "n"]], Float[Array, "n_obs"]],
    x_nominal: Float[Array, "T_plus_1 n_aug"],
    horizon: int,
) -> Tuple[Float[Array, "T n_aug n_w"], int, Float[Array, "T m_u n_w"]]:
    """Build B_w and D_u for the sensory-perturbation channel.

    The H-infinity sensory channel takes a deterministic L2-bounded input
    ``w_obs`` that enters as ``obs = sensory_map(x_plant) + w_obs``. (Note:
    H-inf does not distinguish stochastic noise from deterministic
    perturbation; the previous ``sensory_noise`` naming is a deprecated
    alias. Bug: ec7710f.)

    Lifts to the augmented step's ``w`` via:

    .. math::

        x_{plant,t+1} = A x_{plant,t} + B u_t,
        \\quad u_t = \\text{ctrl}(h_t, \\text{obs} + w_t),

    so ``d x_{aug,t+1}/dw_t = (d x_{plant,t+1}/du)(du/dw) + (d h_{t+1}/dw)``,
    which we obtain by rebuilding a step that takes ``w`` as a sensor
    perturbation rather than a force.

    For LTI controllers (``n_ctrl == 0``) the sensory channel is well-defined
    only if the controller's gain is full-state; we pass it through unchanged
    since ``obs == x_plant`` and noise on obs equals noise on the controller's
    feedback signal.

    Args:
        plant: Linearised plant.
        controller: Closed-loop controller.
        sensory_map: Plant-state to observation map.
        x_nominal: Nominal augmented trajectory.
        horizon: Number of stages.

    Returns:
        ``(Bw_aug, n_w, Du_sensory)`` where ``Bw_aug`` has shape
        ``(T, n_aug, n_w)``, ``n_w`` is the observation dimension, and
        ``Du_sensory`` has shape ``(T, m_u, n_w)``.
    """
    n_plant = plant.n
    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    Bw_force = plant.Bw.astype(jnp.float64)
    m_w_force = plant.m_w
    del Bw_force, m_w_force  # unused (sensory channel doesn't share force B_w)

    # Probe observation dimension once.
    x_plant_0 = x_nominal[0, :n_plant]
    obs_probe = sensory_map(x_plant_0)
    n_obs = int(obs_probe.shape[0])

    def sensory_step(
        x_aug: Float[Array, "n_aug"],
        w_obs: Float[Array, "n_obs"],
        t: int,
    ) -> Float[Array, "n_aug"]:
        x_plant = x_aug[:n_plant]
        h = x_aug[n_plant:]
        obs = sensory_map(x_plant) + w_obs
        h_next, u = controller.step(h, obs, t)
        x_plant_next = A @ x_plant + B @ u  # no force disturbance on this channel
        return jnp.concatenate([x_plant_next, h_next], axis=0)

    def sensory_step_u(
        x_aug: Float[Array, "n_aug"],
        w_obs: Float[Array, "n_obs"],
        t: int,
    ) -> Float[Array, "m_u"]:
        x_plant = x_aug[:n_plant]
        h = x_aug[n_plant:]
        obs = sensory_map(x_plant) + w_obs
        _, u = controller.step(h, obs, t)
        return u

    jac_w = jax.jacobian(sensory_step, argnums=1)
    jac_wu = jax.jacobian(sensory_step_u, argnums=1)

    Bw_aug_list = []
    Du_list = []
    w_zero = jnp.zeros((n_obs,), dtype=jnp.float64)
    for t in range(horizon):
        Bw_aug_list.append(jac_w(x_nominal[t], w_zero, t))
        Du_list.append(jac_wu(x_nominal[t], w_zero, t))
    return (
        jnp.stack(Bw_aug_list, axis=0),
        n_obs,
        jnp.stack(Du_list, axis=0),
    )


def _qr_cost_Cz_Dz(
    plant: PlantLinearization,
    schedule: CostSchedule,
    Cu_arr: Float[Array, "T m_u n_aug"],
    Du_arr: Float[Array, "T m_u n_w"],
    n_aug: int,
    n_w: int,
    n_plant: int,
    horizon: int,
) -> Tuple[Float[Array, "T n_z n_aug"], Float[Array, "T n_z n_w"]]:
    """Build C_z, D_z for the Q+R cost channel.

    Defines the per-step performance output

    .. math::

        z_t = \\begin{pmatrix} \\sqrt{Q_t} x_{plant,t} \\\\ \\sqrt{R_t} u_t \\end{pmatrix},

    so that ``||z||_2^2 = sum_t (x^T Q_t x + u^T R_t u)`` -- the running
    Riccati cost. The state half acts only on the plant slice of the
    augmented state and has zero feedthrough (``x_t`` does not depend on
    ``w_t``). The control half feeds through ``sqrt(R_t) @ D_u``, where
    ``D_u = du/dw`` in the *lifted* ``w`` basis is provided by the caller.

    Per-channel feedthrough ``D_u`` (computed by the caller via JAX autodiff):

    - ``additive_force``: ``D_u == 0`` (force enters at the next state, not
      the current control).
    - ``sensory_perturbation``: ``D_u = ∂u/∂w_obs`` (controller observes
      ``obs + w_obs`` and emits ``u_t`` at the same step).
    - ``structural_da``: ``D_u == 0`` (``w_struct`` adds to the next-step
      augmented state; with full-state feedback, ``u_t`` does not depend on
      ``w_t`` at the same step).

    Bug: ec7710f -- the previous implementation conservatively zeroed ``D_z``
    for non-force channels; this was a strict underestimate of the induced
    gain. Now ``D_z`` is composed exactly from the supplied ``D_u``.

    Args:
        plant: Linearised plant.
        schedule: Cost schedule.
        Cu_arr: ``d u / d x_aug``, shape ``(T, m_u, n_aug)``.
        Du_arr: ``d u / d w`` in the lifted ``w`` basis, shape
            ``(T, m_u, n_w)``. May be all zeros (e.g. ``additive_force``,
            ``structural_da``).
        n_aug: Augmented state dimension.
        n_w: Disturbance dimension (after channel lift).
        n_plant: Plant state dimension.
        horizon: Number of stages.

    Returns:
        ``(Cz, Dz)`` with shapes ``(T, n_z, n_aug)`` and ``(T, n_z, n_w)``.
    """
    Q = schedule.Q[:horizon].astype(jnp.float64)  # (T, n_plant, n_plant)
    R = schedule.R[:horizon].astype(jnp.float64)  # (T, m_u, m_u)

    # sqrt of symmetric PSD via eigendecomposition (Q can be rank-deficient)
    def _psd_sqrt(M: jnp.ndarray) -> jnp.ndarray:
        # Symmetrise then eigh for numerical stability.
        M = 0.5 * (M + M.swapaxes(-1, -2))
        eigvals, eigvecs = jnp.linalg.eigh(M)
        # Clip negatives (numerical noise).
        eigvals = jnp.maximum(eigvals, 0.0)
        sqrt_eig = jnp.sqrt(eigvals)
        return eigvecs @ (sqrt_eig[..., None] * eigvecs.swapaxes(-1, -2))

    Q_sqrt = jax.vmap(_psd_sqrt)(Q)  # (T, n_plant, n_plant)
    R_sqrt = jax.vmap(_psd_sqrt)(R)  # (T, m_u, m_u)

    # State half: top block of C_z is sqrt(Q_t) on plant slice, zero on ctrl.
    n_ctrl = n_aug - n_plant
    Cz_state = jnp.concatenate(
        [Q_sqrt, jnp.zeros((horizon, n_plant, n_ctrl), dtype=jnp.float64)],
        axis=2,
    )  # (T, n_plant, n_aug)

    # Control half: sqrt(R_t) @ Cu_t  (= sqrt(R) * d u / d x_aug)
    Cz_ctrl = jnp.einsum("tij,tjk->tik", R_sqrt, Cu_arr)  # (T, m_u, n_aug)

    Cz = jnp.concatenate([Cz_state, Cz_ctrl], axis=1)  # (T, n_plant + m_u, n_aug)

    # D_z: state half has no feedthrough (x_t does not depend on w_t).
    # Control half: D_z_ctrl = sqrt(R_t) @ D_u (in the lifted w basis).
    Dz_state = jnp.zeros((horizon, n_plant, n_w), dtype=jnp.float64)
    Dz_ctrl = jnp.einsum("tij,tjk->tik", R_sqrt, Du_arr)  # (T, m_u, n_w)
    Dz = jnp.concatenate([Dz_state, Dz_ctrl], axis=1)  # (T, n_plant + m_u, n_w)
    return Cz, Dz


def linearise_fixed_point(
    plant: PlantLinearization,
    controller: Controller,
    *,
    target_pos: Float[Array, "2"],
    w_channel: str = W_ADDITIVE_FORCE,
    z_channel: str = Z_STATE_ERROR,
    schedule: Optional[CostSchedule] = None,
    sensory_map: Optional[Callable[[Float[Array, "n"]], Float[Array, "n_obs"]]] = None,
    t_idx: int = -1,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 50,
) -> FixedPointLinearisation:
    """LTI linearisation of the closed loop at the held-target equilibrium.

    Newton-solves for ``x_aug*`` such that ``f(x_aug*, 0, t_idx) == x_aug*``
    where ``f`` is the augmented closed-loop step. For LTI controllers with
    target-centred feedback, the fixed point is at the origin (target reached,
    zero velocity, zero hidden state) and Newton converges in one step. For
    nonlinear controllers, multiple iterations may be needed.

    Args:
        plant: Linearised plant.
        controller: Closed-loop controller.
        target_pos: Target position in workspace coordinates, shape ``(2,)``.
        w_channel: Disturbance channel.
        z_channel: Performance channel. ``qr_cost`` requires ``schedule``.
        schedule: Cost schedule (used to pick ``Q_{t_idx}``, ``R_{t_idx}``).
        sensory_map: Plant-state to observation map; defaults to identity.
        t_idx: Time index for the time-varying gain. ``-1`` (default) selects
            a mid-trajectory step (most representative of the controller's
            steady-state behaviour for finite-horizon LQ-game Riccati gains,
            where ``K[T-1]`` is dictated by ``Q_f`` and not representative).
            Pass an explicit non-negative index to override.
        newton_tol: Newton convergence tolerance on ``||f(x*) - x*||``.
        newton_max_iter: Maximum Newton iterations.

    Returns:
        A ``FixedPointLinearisation``.
    """
    w_channel = _validate_w_channel(w_channel)
    z_channel = _validate_z_channel(z_channel)
    if z_channel == Z_QR_COST and schedule is None:
        raise ValueError("z_channel='qr_cost' requires a CostSchedule.")
    if sensory_map is None:
        sensory_map = _full_state_observer

    n_plant = plant.n
    h_0 = controller.initial_state()
    n_ctrl = int(h_0.shape[0])
    n_aug = n_plant + n_ctrl
    m_w_force = plant.m_w

    # Resolve t_idx (allow negative). Default -1 -> mid-trajectory.
    if t_idx < 0:
        if schedule is not None:
            # Default to mid-horizon as a representative steady-state-like K
            # for finite-horizon Riccati gains. The terminal index is
            # dictated by Q_f and is rarely representative.
            t_idx = schedule.T // 2
        else:
            t_idx = 0  # fallback for stateless time-invariant controllers

    step = _augmented_step(plant, controller, sensory_map)

    def residual(x_aug: jnp.ndarray) -> jnp.ndarray:
        x_next, _ = step(x_aug, jnp.zeros((m_w_force,), dtype=jnp.float64), t_idx)
        return x_next - x_aug

    # Newton iterations.
    x_aug = jnp.concatenate(
        [jnp.zeros((n_plant,), dtype=jnp.float64), h_0], axis=0
    )
    for _ in range(newton_max_iter):
        r = residual(x_aug)
        if float(jnp.linalg.norm(r)) < newton_tol:
            break
        J = jax.jacobian(residual)(x_aug)
        # Solve J @ delta = -r
        delta = jnp.linalg.solve(J + 1e-12 * jnp.eye(n_aug, dtype=jnp.float64), -r)
        x_aug = x_aug + delta
    x_star = x_aug

    # Linearise step at the fixed point.
    w_zero = jnp.zeros((m_w_force,), dtype=jnp.float64)
    A = jax.jacobian(lambda x: step(x, w_zero, t_idx)[0])(x_star)  # (n_aug, n_aug)
    Bw_force = jax.jacobian(lambda w: step(x_star, w, t_idx)[0])(w_zero)
    # u-Jacobians (reused for qr_cost / control z channels).
    Cu = jax.jacobian(lambda x: step(x, w_zero, t_idx)[1])(x_star)
    Du_force = jax.jacobian(lambda w: step(x_star, w, t_idx)[1])(w_zero)

    # === w channel ===
    if w_channel == W_ADDITIVE_FORCE:
        Bw = Bw_force
        n_w = m_w_force
        Du_in_w_basis = Du_force
    elif w_channel == W_SENSORY_PERTURBATION:
        Bw, n_w, Du_in_w_basis = _sensory_perturbation_fp_Bw(
            plant, controller, sensory_map, x_star, t_idx
        )
    elif w_channel == W_STRUCTURAL_DA:
        Bw = jnp.eye(n_aug, dtype=jnp.float64)
        n_w = n_aug
        Du_in_w_basis = jnp.zeros((plant.m_u, n_w), dtype=jnp.float64)
    else:  # pragma: no cover
        raise AssertionError(w_channel)

    # === z channel ===
    if z_channel == Z_QR_COST:
        Q = schedule.Q[t_idx].astype(jnp.float64)
        R = schedule.R[t_idx].astype(jnp.float64)

        def _psd_sqrt(M: jnp.ndarray) -> jnp.ndarray:
            M = 0.5 * (M + M.T)
            eigvals, eigvecs = jnp.linalg.eigh(M)
            eigvals = jnp.maximum(eigvals, 0.0)
            return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T

        Q_sqrt = _psd_sqrt(Q)
        R_sqrt = _psd_sqrt(R)
        Cz_state = jnp.concatenate(
            [Q_sqrt, jnp.zeros((n_plant, n_ctrl), dtype=jnp.float64)], axis=1
        )
        Cz_ctrl = R_sqrt @ Cu
        Cz = jnp.concatenate([Cz_state, Cz_ctrl], axis=0)
        # D_z: state half is zero (state at fp does not depend on w),
        # control half = sqrt(R) @ D_u_in_w_basis. Bug: ec7710f.
        D_state = jnp.zeros((n_plant, n_w), dtype=jnp.float64)
        D_ctrl = R_sqrt @ Du_in_w_basis
        D = jnp.concatenate([D_state, D_ctrl], axis=0)
    elif z_channel == Z_CONTROL:
        Cz = Cu
        D = Du_in_w_basis
    elif z_channel == Z_STATE_ERROR:
        I_plant = jnp.eye(n_plant, dtype=jnp.float64)
        Cz = jnp.concatenate(
            [I_plant, jnp.zeros((n_plant, n_ctrl), dtype=jnp.float64)], axis=1
        )
        D = jnp.zeros((n_plant, n_w), dtype=jnp.float64)
    elif z_channel == Z_PEAK_VELOCITY:
        vel_lo, vel_hi = plant.vel_slice
        n_vel = vel_hi - vel_lo
        Cz_block = jnp.zeros((n_vel, n_aug), dtype=jnp.float64)
        for i in range(n_vel):
            Cz_block = Cz_block.at[i, vel_lo + i].set(1.0)
        Cz = Cz_block
        D = jnp.zeros((n_vel, n_w), dtype=jnp.float64)
    else:  # pragma: no cover
        raise AssertionError(z_channel)

    return FixedPointLinearisation(
        A=A,
        Bw=Bw,
        Cz=Cz,
        D=D,
        x_star=x_star,
        n_plant=n_plant,
        n_ctrl=n_ctrl,
        w_channel=w_channel,
        z_channel=z_channel,
    )


def _sensory_perturbation_fp_Bw(
    plant: PlantLinearization,
    controller: Controller,
    sensory_map: Callable[[Float[Array, "n"]], Float[Array, "n_obs"]],
    x_star: Float[Array, "n_aug"],
    t_idx: int,
) -> Tuple[Float[Array, "n_aug n_obs"], int, Float[Array, "m_u n_obs"]]:
    """Sensory-perturbation B_w and D_u at a fixed point. See ``_sensory_perturbation_Bw``."""
    n_plant = plant.n
    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)

    x_plant_star = x_star[:n_plant]
    obs_probe = sensory_map(x_plant_star)
    n_obs = int(obs_probe.shape[0])

    def sensory_step(x_aug: jnp.ndarray, w_obs: jnp.ndarray) -> jnp.ndarray:
        x_plant = x_aug[:n_plant]
        h = x_aug[n_plant:]
        obs = sensory_map(x_plant) + w_obs
        h_next, u = controller.step(h, obs, t_idx)
        x_plant_next = A @ x_plant + B @ u
        return jnp.concatenate([x_plant_next, h_next], axis=0)

    def sensory_step_u(x_aug: jnp.ndarray, w_obs: jnp.ndarray) -> jnp.ndarray:
        x_plant = x_aug[:n_plant]
        h = x_aug[n_plant:]
        obs = sensory_map(x_plant) + w_obs
        _, u = controller.step(h, obs, t_idx)
        return u

    w_zero = jnp.zeros((n_obs,), dtype=jnp.float64)
    Bw = jax.jacobian(lambda w: sensory_step(x_star, w))(w_zero)
    Du = jax.jacobian(lambda w: sensory_step_u(x_star, w))(w_zero)
    return Bw, n_obs, Du


# =============================================================================
# Power iteration on LTV operator
# =============================================================================


def _ltv_forward_sweep(
    A_t: Float[Array, "T n_aug n_aug"],
    Bw_t: Float[Array, "T n_aug n_w"],
    Cz_t: Float[Array, "T n_z n_aug"],
    D_t: Float[Array, "T n_z n_w"],
    w_seq: Float[Array, "T n_w"],
) -> Float[Array, "T n_z"]:
    """Apply the LTV operator ``T: w -> z`` via a forward time sweep.

    The closed loop is

    .. math::

        x_{t+1} = A_t x_t + B_{w,t} w_t,
        \\quad z_t = C_{z,t} x_t + D_t w_t,

    with ``x_0 = 0`` (the operator is the linear map from disturbances to
    perturbation outputs around the nominal trajectory). The induced gain
    ``L^2 -> L^2`` over ``[0, T)`` is the largest singular value of this
    linear operator on length-T signals.

    Args:
        A_t: Augmented dynamics, shape ``(T, n_aug, n_aug)``.
        Bw_t: Disturbance-input matrix, shape ``(T, n_aug, n_w)``.
        Cz_t: Performance-output matrix, shape ``(T, n_z, n_aug)``.
        D_t: Feedthrough, shape ``(T, n_z, n_w)``.
        w_seq: Disturbance trajectory, shape ``(T, n_w)``.

    Returns:
        Performance trajectory ``z_seq`` of shape ``(T, n_z)``.
    """
    n_aug = A_t.shape[1]

    def body(x_t, inputs):
        A, Bw, Cz, D, w_t = inputs
        z = Cz @ x_t + D @ w_t
        x_next = A @ x_t + Bw @ w_t
        return x_next, z

    x0 = jnp.zeros((n_aug,), dtype=jnp.float64)
    _, z_seq = jax.lax.scan(body, x0, (A_t, Bw_t, Cz_t, D_t, w_seq))
    return z_seq  # shape (T, n_z)


def induced_gain_power_iteration(
    lin: TrajectoryLinearisation,
    *,
    n_restarts: int = 5,
    rtol: float = 1e-6,
    max_iter: int = 500,
    seed: int = 0,
    return_trajectory: bool = True,
) -> InducedGainResult:
    """Compute the LTV induced gain ``||T||_2`` by power iteration.

    The forward sweep is ``z = T w``; the adjoint sweep is
    ``T^* z`` (computed via ``jax.linear_transpose``). Power iteration
    alternates these and renormalises ``w`` to recover the largest singular
    value of ``T`` (the induced ``L2 -> L2`` gain).

    The linear-transpose route is preferred over a hand-coded backward
    recursion: the transpose is exact by construction (no risk of sign flips
    or off-by-one indexing) and it works uniformly across all channels
    without per-channel adjoint code.

    Round-trip band for an LTI Riccati controller designed at ``gamma_design``:
        ``||T||_PI`` lies in ``(gamma_star, gamma_design]``. ``gamma_star`` is
        the H-infinity *infimum* (no admissible LTI controller can hit it
        exactly on a finite horizon), so the analyser gain is always strictly
        above ``gamma_star`` and bounded above by the design level. Empirically
        this band is regime-dependent: for the rlrmp point-mass regime
        (mass=1, damping=10, tau=0.05, dt=0.01), with ``gamma_design = 1.5 *
        gamma_star``, the long-horizon plateau is around ``1.21 * gamma_star``.
        See ``scripts/probe_round_trip_ratio.py`` for the diagnostic sweep.

    Note:
        ``max_iter`` defaults to 500. Long horizons (T >= 200) frequently need
        more than 200 iterations to satisfy ``rtol = 1e-6`` because the
        operator's leading-singular-value gap shrinks; bumping the default is
        cheaper than per-call surgery. See bug ``3c74e3b``.

    Args:
        lin: Trajectory linearisation.
        n_restarts: Number of random initial ``w`` vectors. The reported
            ``gamma`` is the maximum estimate across restarts; this guards
            against converging to a non-leading singular value.
        rtol: Convergence tolerance: relative change in ``gamma`` below this
            for two consecutive iterations triggers convergence.
        max_iter: Maximum iterations per restart. Default 500 covers long
            horizons; tighten for short horizons if needed.
        seed: Base RNG seed for restart initialisations.
        return_trajectory: If True, simulate and store the closed-loop
            augmented trajectory under the worst-case ``w``.

    Returns:
        An ``InducedGainResult``.
    """
    A_t = lin.A_t
    Bw_t = lin.Bw_t
    Cz_t = lin.Cz_t
    D_t = lin.D_t
    T = lin.T
    n_w = lin.n_w

    def forward(w_seq: jnp.ndarray) -> jnp.ndarray:
        return _ltv_forward_sweep(A_t, Bw_t, Cz_t, D_t, w_seq)

    # Use VJP rather than ``jax.linear_transpose`` -- the latter fails inside
    # ``lax.scan`` when integer carry indices are present (a known JAX
    # limitation). VJP at the zero input is mathematically equivalent for a
    # linear operator: T^*(z) = (d (z^T T(w)) / d w) |_{w=0}.
    w_dummy = jnp.zeros((T, n_w), dtype=jnp.float64)
    _, vjp_fn = jax.vjp(forward, w_dummy)

    def adjoint(z_seq: jnp.ndarray) -> jnp.ndarray:
        out = vjp_fn(z_seq)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    best_gamma = 0.0
    best_w: Optional[jnp.ndarray] = None
    total_iters = 0
    converged_any = False
    per_restart_gammas = []

    rng = jax.random.PRNGKey(int(seed))
    for r in range(int(n_restarts)):
        rng, sub = jax.random.split(rng)
        w = jax.random.normal(sub, (T, n_w), dtype=jnp.float64)
        w = w / (jnp.linalg.norm(w) + 1e-30)
        prev_gamma = 0.0
        prev_prev_gamma = -1.0
        gamma_est = 0.0
        consec_close = 0
        converged = False
        for it in range(int(max_iter)):
            z = forward(w)
            z_norm = float(jnp.linalg.norm(z))
            w_norm = float(jnp.linalg.norm(w))
            if w_norm < 1e-30:
                break
            gamma_est = z_norm / w_norm
            # Adjoint and renormalise.
            w_new = adjoint(z)
            w_new_norm = float(jnp.linalg.norm(w_new))
            if w_new_norm < 1e-30:
                break
            w = w_new / w_new_norm
            # Convergence test: relative change in gamma across two
            # consecutive iterations stays below rtol.
            if prev_gamma > 0:
                rel_change = abs(gamma_est - prev_gamma) / max(prev_gamma, 1e-30)
                if rel_change < rtol:
                    consec_close += 1
                else:
                    consec_close = 0
                if consec_close >= 2:
                    converged = True
                    total_iters += it + 1
                    break
            prev_prev_gamma = prev_gamma
            prev_gamma = gamma_est
        else:
            total_iters += int(max_iter)
        del prev_prev_gamma  # debug-only
        per_restart_gammas.append(float(gamma_est))
        converged_any = converged_any or converged
        if gamma_est > best_gamma:
            best_gamma = float(gamma_est)
            best_w = w  # the current w is the right-singular-vector estimate

    # Recover the worst-case w*: re-normalise so its L2 norm matches the
    # canonical "unit input" interpretation, i.e. ||w*||_2 == 1.
    if best_w is not None:
        worst_w = best_w / (jnp.linalg.norm(best_w) + 1e-30)
    else:
        worst_w = jnp.zeros((T, n_w), dtype=jnp.float64)

    # Optionally simulate the augmented closed loop under worst_w to recover
    # the worst-case state trajectory (used for the peak_velocity z channel).
    worst_traj = None
    pf_v = None
    pl_v = None
    if return_trajectory:
        # Forward sweep with arbitrary w_seq, but we want x_t (not z_t).
        n_aug = lin.n_aug

        def body_x(x_t, inputs):
            A, Bw, w = inputs
            x_next = A @ x_t + Bw @ w
            return x_next, x_next

        x0 = jnp.zeros((n_aug,), dtype=jnp.float64)
        _, x_seq_perturb = jax.lax.scan(
            body_x, x0, (A_t, Bw_t, worst_w)
        )  # (T, n_aug) — perturbation around nominal
        # Add nominal: x_aug = x_nominal + x_perturb; but we're operating in
        # goal-centred coordinates and the linearisation is around the
        # nominal trajectory, so the worst-case absolute trajectory is
        # x_nominal[1:] + x_seq_perturb. Concatenate with x_nominal[0] for
        # the (T+1, n_aug) shape.
        worst_traj = jnp.concatenate(
            [lin.x_nominal[0:1] + 0.0, lin.x_nominal[1:] + x_seq_perturb], axis=0
        )

        # If z_channel == peak_velocity, also report peak forward / lateral
        # velocity along this trajectory. Use the same projection logic as
        # simulate_closed_loop: project velocity onto the reach axis recovered
        # from x_nominal[0] (initial position relative to target == zero, so
        # we need init_pos and target_pos; recoverable via the nominal x_aug,
        # plant slice convention).
        if lin.z_channel == Z_PEAK_VELOCITY:
            # Pull plant velocity slice. We need plant.vel_slice but only have
            # n_plant here -- defer this computation to a helper that takes
            # the plant explicitly. Skipped: caller of induced_gain (the
            # high-level wrapper) recomputes peak_forward_velocity /
            # peak_lateral_velocity by calling simulate_closed_loop with
            # the worst-case w restricted to the additive-force basis.
            pass

    diagnostics = {
        "per_restart_gammas": per_restart_gammas,
        "n_restarts": int(n_restarts),
        "best_restart_idx": int(jnp.argmax(jnp.asarray(per_restart_gammas)))
        if per_restart_gammas
        else 0,
    }

    return InducedGainResult(
        gamma=best_gamma,
        method="power_iteration",
        w_channel=lin.w_channel,
        z_channel=lin.z_channel,
        worst_case_w=worst_w,
        worst_case_trajectory=worst_traj,
        peak_forward_velocity=pf_v,
        peak_lateral_velocity=pl_v,
        iterations=total_iters,
        converged=converged_any,
        diagnostics=diagnostics,
    )


# =============================================================================
# Hamiltonian / bounded-real-Riccati bisection on LTI fixed-point
# =============================================================================


def _bounded_real_admissible(
    A: Float[Array, "n n"],
    Bw: Float[Array, "n n_w"],
    Cz: Float[Array, "n_z n"],
    D: Float[Array, "n_z n_w"],
    gamma: float,
    horizon: int,
    rho_tol: float = 1e-6,
    cond_tol: float = 1e12,
) -> bool:
    """Test whether ``||T||_inf < gamma`` via the discrete-time bounded-real
    Riccati.

    For an LTI system ``(A, Bw, Cz, D)``, the discrete-time bounded-real lemma
    states that ``||T_{w->z}||_inf < gamma`` iff the Riccati

    .. math::

        P = C_z^\\top C_z + A^\\top P A
            + (A^\\top P B_w + C_z^\\top D)
              (\\gamma^2 I - D^\\top D - B_w^\\top P B_w)^{-1}
              (A^\\top P B_w + C_z^\\top D)^\\top

    has a stabilising PSD solution. We use a finite-horizon analogue: run
    the recursion forward (``P_0 = 0`` initially) for ``horizon`` steps and
    check the bracket ``gamma^2 I - D^T D - B_w^T P B_w`` stays positive
    definite throughout. This is the canonical finite-horizon admissibility
    primitive (cf. ``solve_hinf_riccati``'s ``rho < 1`` guard, lifted to LTI).

    Args:
        A, Bw, Cz, D: LTI matrices.
        gamma: Candidate disturbance-attenuation level.
        horizon: Recursion horizon.
        rho_tol: Margin below 1 for the spectral radius
            ``gamma^{-2} (D^T D + B_w^T P B_w)``.
        cond_tol: Maximum acceptable bracket condition number.

    Returns:
        ``True`` if ``gamma`` is admissible (above the induced gain).
    """
    if gamma <= 0:
        return False

    n_w = Bw.shape[1]
    P = jnp.zeros((A.shape[0], A.shape[0]), dtype=jnp.float64)
    DTD = D.T @ D  # (n_w, n_w)

    for _ in range(int(horizon)):
        # bracket = gamma^2 I - D^T D - B_w^T P B_w
        bracket = gamma * gamma * jnp.eye(n_w, dtype=jnp.float64) - DTD - Bw.T @ P @ Bw
        # Admissibility: bracket > 0 (smallest eigenvalue > 0).
        eigvals = jnp.linalg.eigvalsh(0.5 * (bracket + bracket.T))
        smallest = float(jnp.min(jnp.real(eigvals)))
        if not jnp.isfinite(smallest) or smallest <= rho_tol * gamma * gamma:
            return False
        try:
            sing_vals = jnp.linalg.svd(bracket, compute_uv=False)
            cond = float(jnp.max(sing_vals) / max(float(jnp.min(sing_vals)), 1e-30))
        except Exception:
            return False
        if not jnp.isfinite(cond) or cond > cond_tol:
            return False
        # Update P:
        ATPBw_plus_CzTD = A.T @ P @ Bw + Cz.T @ D
        P_next = (
            Cz.T @ Cz
            + A.T @ P @ A
            + ATPBw_plus_CzTD @ jnp.linalg.solve(bracket, ATPBw_plus_CzTD.T)
        )
        P = 0.5 * (P_next + P_next.T)
        if not jnp.all(jnp.isfinite(P)):
            return False
    return True


def induced_gain_hamiltonian(
    lin: FixedPointLinearisation,
    *,
    horizon: int = 200,
    gamma_lo: float = 1e-3,
    gamma_hi: float = 1e6,
    rtol: float = 1e-6,
    max_bisect: int = 60,
) -> InducedGainResult:
    """Compute the LTI induced gain by bracketed bisection.

    Bisects ``gamma`` on a log scale and tests admissibility via a
    finite-horizon discrete-time bounded-real Riccati (see
    ``_bounded_real_admissible``). The smallest admissible ``gamma`` is the
    induced gain ``||T||_inf``.

    The Hamiltonian/symplectic eigenvalue formulation is mathematically
    equivalent for stationary LTI systems; the Riccati admissibility test is
    used here because it shares numerical primitives with ``solve_hinf_riccati``
    (well-tested) and avoids materialising the symplectic pencil.

    Args:
        lin: Fixed-point linearisation.
        horizon: Recursion horizon for the finite-horizon admissibility test.
            Should be long enough that the Riccati has reached steady state.
        gamma_lo: Initial lower bracket. Expanded downward by powers of 10
            if admissible.
        gamma_hi: Initial upper bracket. Expanded upward if inadmissible.
        rtol: Bisection convergence tolerance on ``(hi - lo) / hi``.
        max_bisect: Maximum bisection iterations.

    Returns:
        An ``InducedGainResult``.
    """
    A = lin.A.astype(jnp.float64)
    Bw = lin.Bw.astype(jnp.float64)
    Cz = lin.Cz.astype(jnp.float64)
    D = lin.D.astype(jnp.float64)

    # Expand gamma_hi upward until admissible (= induced gain is finite at
    # least at this gamma).
    expand_iter = 0
    while expand_iter < 30:
        if _bounded_real_admissible(A, Bw, Cz, D, gamma_hi, horizon):
            break
        gamma_hi *= 10.0
        expand_iter += 1
    else:
        return InducedGainResult(
            gamma=float("inf"),
            method="hamiltonian",
            w_channel=lin.w_channel,
            z_channel=lin.z_channel,
            converged=False,
            diagnostics={"reason": "could_not_find_admissible_upper_bracket"},
        )

    # Expand gamma_lo downward until inadmissible.
    expand_iter = 0
    while expand_iter < 30:
        if not _bounded_real_admissible(A, Bw, Cz, D, gamma_lo, horizon):
            break
        gamma_lo *= 0.1
        expand_iter += 1
    else:
        # Even very small gamma is admissible — induced gain is essentially 0.
        return InducedGainResult(
            gamma=float(gamma_lo),
            method="hamiltonian",
            w_channel=lin.w_channel,
            z_channel=lin.z_channel,
            converged=True,
            diagnostics={"reason": "induced_gain_below_lower_bracket"},
        )

    # Bisection on log-scale (geometric mean).
    for _ in range(int(max_bisect)):
        gamma_mid = float(jnp.sqrt(gamma_lo * gamma_hi))
        if _bounded_real_admissible(A, Bw, Cz, D, gamma_mid, horizon):
            gamma_hi = gamma_mid
        else:
            gamma_lo = gamma_mid
        if (gamma_hi - gamma_lo) / max(gamma_hi, 1e-30) < rtol:
            break

    return InducedGainResult(
        gamma=float(gamma_hi),
        method="hamiltonian",
        w_channel=lin.w_channel,
        z_channel=lin.z_channel,
        converged=True,
        iterations=0,
        diagnostics={
            "bracket_lo": float(gamma_lo),
            "bracket_hi": float(gamma_hi),
            "horizon": int(horizon),
        },
    )


# =============================================================================
# High-level entry point
# =============================================================================


def induced_gain(
    plant: PlantLinearization,
    controller: Controller,
    *,
    init_pos: Float[Array, "2"],
    target_pos: Float[Array, "2"],
    horizon: int,
    w_channel: str = W_ADDITIVE_FORCE,
    z_channel: str = Z_QR_COST,
    methods: Tuple[str, ...] = ("power_iteration", "hamiltonian"),
    schedule: Optional[CostSchedule] = None,
    sensory_map: Optional[Callable[[Float[Array, "n"]], Float[Array, "n_obs"]]] = None,
    n_restarts: int = 5,
    rtol: float = 1e-6,
    max_iter: int = 500,
    seed: int = 0,
) -> dict:
    """High-level induced-gain analyser combining linearise + algorithm.

    Returns a dict with one ``InducedGainResult`` per method requested. The
    keys are the method names (``"power_iteration"``, ``"hamiltonian"``).

    Round-trip band for an LTI controller designed at ``gamma_design``:
        ``||T||_PI in (gamma_star, gamma_design]`` (the H-inf optimum
        ``gamma_star`` is an infimum, never reached by a finite-horizon
        LTI controller; the actual closed-loop gain is strictly larger but
        bounded above by the design level). Regime-dependent in practice;
        see ``scripts/probe_round_trip_ratio.py`` and bug ``3c74e3b``.

    For the ``peak_velocity`` z channel: the power-iteration result on the
    velocity-norm channel is returned in ``InducedGainResult.gamma``, and the
    headline scalars ``peak_forward_velocity`` / ``peak_lateral_velocity`` are
    computed by simulating the worst-case ``w*`` (recovered from power
    iteration) under the closed loop and projecting velocity onto the reach
    axis. If ``additive_force`` is the w channel, the simulation reuses
    ``simulate_closed_loop`` for the LTI case; for nonlinear controllers we
    use the nominal-trajectory linearisation's worst-case state path.

    Args:
        plant: Linearised plant.
        controller: Closed-loop controller.
        init_pos: Reach starting position, shape ``(2,)``.
        target_pos: Reach target position, shape ``(2,)``.
        horizon: Number of LTV stages T.
        w_channel: Disturbance channel.
        z_channel: Performance channel.
        methods: Subset of ``("power_iteration", "hamiltonian")``.
        schedule: Cost schedule (required for ``z_channel == "qr_cost"``).
        sensory_map: Plant-state to observation map; defaults to identity.
        n_restarts, rtol, max_iter, seed: Power-iteration parameters.

    Returns:
        A dict mapping method name to ``InducedGainResult``.
    """
    w_channel = _validate_w_channel(w_channel)
    z_channel = _validate_z_channel(z_channel)

    out: dict = {}

    if "power_iteration" in methods:
        lin_traj = linearise_trajectory(
            plant,
            controller,
            init_pos=init_pos,
            target_pos=target_pos,
            horizon=horizon,
            w_channel=w_channel,
            z_channel=z_channel,
            schedule=schedule,
            sensory_map=sensory_map,
        )
        result = induced_gain_power_iteration(
            lin_traj,
            n_restarts=n_restarts,
            rtol=rtol,
            max_iter=max_iter,
            seed=seed,
        )
        # If z_channel is peak_velocity, decorate with peak forward/lateral.
        if z_channel == Z_PEAK_VELOCITY:
            result = _decorate_peak_velocity(
                result, lin_traj, plant, init_pos=init_pos, target_pos=target_pos
            )
        out["power_iteration"] = result

    if "hamiltonian" in methods:
        lin_fp = linearise_fixed_point(
            plant,
            controller,
            target_pos=target_pos,
            w_channel=w_channel,
            z_channel=z_channel,
            schedule=schedule,
            sensory_map=sensory_map,
            # Default t_idx=-1 picks mid-horizon (most representative of
            # steady-state K for finite-horizon Riccati gains).
        )
        result = induced_gain_hamiltonian(lin_fp, horizon=horizon)
        out["hamiltonian"] = result

    return out


def _decorate_peak_velocity(
    result: InducedGainResult,
    lin: TrajectoryLinearisation,
    plant: PlantLinearization,
    *,
    init_pos: Float[Array, "2"],
    target_pos: Float[Array, "2"],
) -> InducedGainResult:
    """Add peak forward / lateral velocity to a power_iteration result.

    Projects the worst-case velocity trajectory onto the reach axis (initial
    -> target) in absolute coordinates. Mirrors the projection logic in
    ``hinf_riccati.simulate_closed_loop``.
    """
    if result.worst_case_trajectory is None:
        return result
    pos_lo, pos_hi = plant.pos_slice
    vel_lo, vel_hi = plant.vel_slice
    # Augmented worst-case trajectory; restrict to plant slice.
    x_traj = result.worst_case_trajectory[:, : plant.n]
    pos_gc = x_traj[:, pos_lo:pos_hi]
    vel_t = x_traj[:, vel_lo:vel_hi]

    init_pos_arr = jnp.asarray(init_pos, dtype=jnp.float64)
    target_pos_arr = jnp.asarray(target_pos, dtype=jnp.float64)
    line_vec = target_pos_arr - init_pos_arr
    line_len = float(jnp.linalg.norm(line_vec))
    if line_len < 1e-12:
        return result
    line_dir = line_vec / line_len
    v_forward = vel_t @ line_dir  # (T+1,)
    v_lat_vec = vel_t - v_forward[:, None] * line_dir[None, :]
    v_lat = jnp.linalg.norm(v_lat_vec, axis=-1)

    # Frozen dataclass: rebuild rather than mutate.
    del pos_gc  # not used here; kept for shape parity with ClosedLoopRollout
    return InducedGainResult(
        gamma=result.gamma,
        method=result.method,
        w_channel=result.w_channel,
        z_channel=result.z_channel,
        worst_case_w=result.worst_case_w,
        worst_case_trajectory=result.worst_case_trajectory,
        peak_forward_velocity=float(jnp.max(v_forward)),
        peak_lateral_velocity=float(jnp.max(v_lat)),
        iterations=result.iterations,
        converged=result.converged,
        diagnostics=result.diagnostics,
    )


# =============================================================================
# Public exports
# =============================================================================

__all__ = [
    # Channel sentinels
    "W_ADDITIVE_FORCE",
    "W_SENSORY_PERTURBATION",
    "W_SENSORY_NOISE",  # deprecated alias for W_SENSORY_PERTURBATION
    "W_STRUCTURAL_DA",
    "Z_QR_COST",
    "Z_CONTROL",
    "Z_STATE_ERROR",
    "Z_PEAK_VELOCITY",
    # Value types
    "Controller",
    "TrajectoryLinearisation",
    "FixedPointLinearisation",
    "InducedGainResult",
    # Controller adapters
    "lti_controller",
    "feedbax_graph_controller",
    "feedbax_rnn_controller",  # deprecated; raises NotImplementedError
    # Linearisation
    "linearise_trajectory",
    "linearise_fixed_point",
    # Algorithms
    "induced_gain_power_iteration",
    "induced_gain_hamiltonian",
    "induced_gain",
]
