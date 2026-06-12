"""Finite-horizon discrete-time H-infinity LQ-game Riccati on the rlrmp plant.

This module implements the finite-horizon, discrete-time H-infinity Riccati
recursion (Crevecoeur & Scott 2019 / Basar-Bernhard form) on the linearised
rlrmp point-mass plant with first-order force filter. It computes the smallest
admissible disturbance-attenuation level gamma_star, the H-infinity feedback
gain K_t, and a closed-loop simulator for comparing peak velocity against the
LQR baseline (gamma -> infinity).

Bug: 5a44bd3 - production reimplementation of the H-infinity Riccati sanity
check. See ``results/b557d4e/synthesis_review.md`` section 2 for the
mathematical spec, the Q,R schedule, and the predicted velocity inflation
magnitude on the rlrmp parameter regime.

Math (synthesis_review section 2 spec)
--------------------------------------
The discrete-time finite-horizon LQ-game value function is

.. math::

    V_t(x) = x^\\top P_t\\, x

with the backward recursion

.. math::

    P_t = Q_t + A^\\top P_{t+1}
        \\bigl(I + (B R_t^{-1} B^\\top - \\gamma^{-2} B_w B_w^\\top) P_{t+1}\\bigr)^{-1} A,

initialised at :math:`P_T = Q_f`. The H-infinity feedback gain is

.. math::

    K_t = (R_t + B^\\top \\Lambda_t B)^{-1} B^\\top \\Lambda_t A,
    \\qquad
    \\Lambda_t = (I - \\gamma^{-2} P_{t+1} B_w B_w^\\top)^{-1} P_{t+1},

and the closed-loop dynamics with the worst-case disturbance are

.. math::

    x_{t+1} = (A - B K_t)\\,x_t + B_w w_t^\\star,

where :math:`w_t^\\star = \\gamma^{-2} B_w^\\top \\Lambda_t (A - B K_t) x_t`.
We use both as needed.

Solvability of the Riccati at gamma is encoded by the requirement that, at
every step, :math:`I - \\gamma^{-2} B_w^\\top P_{t+1} B_w` is positive definite
(equivalently, all eigenvalues of :math:`\\gamma^{-2} B_w^\\top P_{t+1} B_w` lie
strictly below 1). Below this threshold, the recursion's bracket inverse is
ill-posed and the disturbance "wins".

Numerical strategy
------------------
Phase 2's quick standalone implementation blew up near the Riccati boundary.
This implementation uses three guards:

1. **Double precision.** All matrices are cast to ``float64`` before
   recursion. The module's public API also enables ``jax_enable_x64`` once at
   import time (idempotent).
2. **Linear solve, not explicit inverse.** The bracket inversion is done via
   ``jnp.linalg.solve`` on a 6x6 system at each timestep (``B R^{-1} B^T -
   gamma^{-2} B_w B_w^T`` premultiplied by ``P_{t+1}``, plus identity).
3. **Per-step admissibility check.** At each backward step we compute the
   spectral radius :math:`\\rho_t = \\lambda_{\\max}(\\gamma^{-2} B_w^\\top
   P_{t+1} B_w)` and the condition number of the bracket. If
   :math:`\\rho_t \\ge 1 - \\epsilon` or the bracket is severely
   ill-conditioned, the recursion flags the gamma as inadmissible and
   short-circuits. The ``RiccatiSolution`` carries these diagnostics for every
   step.

The bisection for ``gamma_star`` uses the admissibility flag as the predicate.

Linearisation
-------------
The rlrmp plant is a continuous-time LTI system: a 4-state point mass
(position, velocity) plus a 2-state first-order force filter (acting on the
commanded force ``u`` with rise/decay time constant ``tau``). Stacked state
:math:`x_c = [pos(2), vel(2), F(2)]`. Continuous-time matrices

.. math::

    A_c = \\begin{pmatrix} 0 & I & 0 \\\\ 0 & -k/m\\,I & I/m \\\\ 0 & 0 & -I/\\tau \\end{pmatrix},
    \\quad
    B_c = \\begin{pmatrix} 0 \\\\ 0 \\\\ I/\\tau \\end{pmatrix},
    \\quad
    B_{w,c} = \\begin{pmatrix} 0 \\\\ I/m \\\\ 0 \\end{pmatrix},

where the disturbance ``w`` enters as an additive force on the velocity row
(matching feedbax's effector-force intervenor channel; see
``rlrmp.disturbance``). Generic rlrmp plants use zero-order hold by default,
while the C&S released-code card uses forward Euler to match the ModelDB
MATLAB update ``A_d = I + dt A_c``, ``B_d = dt B_c``. The ZOH construction uses
``jax.scipy.linalg.expm`` on the augmented 9x9 block

.. math::

    \\exp\\!\\left(
        \\begin{bmatrix} A_c & B_c & B_{w,c} \\\\
        0 & 0 & 0 \\\\ 0 & 0 & 0
        \\end{bmatrix} dt
    \\right)
    = \\begin{bmatrix} A_d & B_d & B_{w,d} \\\\ 0 & I & 0 \\\\ 0 & 0 & I \\end{bmatrix}.

For ``tau == 0`` (no filter) we bypass the filter rows and return the bare
4-state plant.

Relationship to flavor (b) :math:`\\max_{\\Delta A}`
---------------------------------------------------
The synthesis (``synthesis_review.md`` section 2 / 4.2) distinguishes:

- **Flavor (a)**: additive force-trajectory disturbance ``w``. This is what
  the base ``solve_hinf_riccati`` directly computes -- ``B_w`` enters the
  bracket with weight :math:`\\gamma^{-2}`.
- **Flavor (b)**: structural perturbation :math:`\\Delta A` to the dynamics
  matrix, constant over the trial. ``\\dot v \\mathrel{+}= \\Delta A\\,
  [p, v]``, matching the feedbax ``DynamicsMatrixPerturb`` intervenor and
  ``LinearDynamicsAdversary`` in ``rlrmp.adversary``.

The flavor-(b) extension is implemented by ``solve_hinf_riccati_modelclass``
and ``find_gamma_star_modelclass``. It uses the **S-procedure / quadratic-
stability** reduction: a structured perturbation ``\\Delta A`` with
:math:`\\|\\Delta A\\|_F \\le \\eta` produces a closed-loop disturbance
:math:`w_t = m \\, \\Delta A \\, C_q \\, x_t` (where ``m`` is the mass and
:math:`C_q` selects ``[pos, vel]`` from the augmented state). By
Cauchy-Schwarz on the Frobenius-induced operator norm,
:math:`\\|w_t\\|_2 \\le m\\eta \\, \\|C_q x_t\\|_2`. A sufficient condition
for robust stability of the LQ-game value function under any such
``\\Delta A`` is that the H-infinity Riccati on the **same** force channel
``B_w`` with **augmented** state cost
:math:`Q_t \\to Q_t + (m\\eta)^2 \\, C_q^\\top C_q` is admissible. The
smallest such :math:`\\gamma` is :math:`\\gamma_*^{(b)}(\\eta)`.

This reduction is conservative: the true achievable robust-controller
:math:`\\gamma` against the structured ``\\Delta A`` ball is bounded above
by :math:`\\gamma_*^{(b)}(\\eta)` (full structured-singular-value /
:math:`\\mu`-synthesis would give a tighter bound, but is not implemented).
However, three reasons make quadratic stability the right starting point
here:

1. ``LinearDynamicsAdversary`` constrains :math:`\\|\\Delta A\\|_F \\le
   \\eta` and holds it constant across the trial -- exactly the
   parametric-uncertainty class quadratic stability handles.
2. The induced-gain analyser's ``structural_da`` channel uses the same
   small-gain framing, so empirical :math:`\\gamma_{sd}` (from the analyser)
   and analytical :math:`\\gamma_*^{(b)}` (from this synthesis) live in the
   same comparison space.
3. The reduction preserves the LTI Riccati primitive -- bisection on
   :math:`\\gamma` proceeds exactly as in flavor-(a); only the cost matrix
   :math:`Q` changes.

A tighter, time-varying ``B_w(x_t)`` formulation (where the disturbance
literally tracks the closed-loop state along a nominal reach trajectory)
is left as future work: it requires a finite-horizon differential Riccati
along a closed-loop trajectory and is not directly comparable to a single
:math:`\\gamma` scalar.

Usage
-----
Three public uses:

1. **Sanity check.** ``compute_velocity_inflation`` returns the LQR vs
   H-infinity peak-velocity comparison. Pre-registered prediction
   (synthesis_review section 11 step 1): +10 to +25% at :math:`\\gamma \\sim
   1.5\\gamma_*` on rlrmp parameters; <1% on C&S parameters.
2. **Distillation teacher** (synthesis_review section 4.3 phase A). The
   ``RiccatiSolution.K`` array provides time-varying linear feedback gains
   that can be fed into a behavioural-cloning loop to teach the GRU policy
   the H-infinity controller.
3. **Adversary baseline** (synthesis_review section 4.2). The Riccati gain
   provides a reference closed-loop for evaluating
   ``LinearDynamicsAdversary``-trained networks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import jax.tree as jt
from feedbax.mechanics.skeleton.pointmass import PointMass
from jaxtyping import Array, Float

# Enable x64 once on import. Idempotent and only triggers if x64 isn't already
# on. The Riccati recursion is numerically sensitive near the gamma boundary
# and double precision is non-negotiable here.
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================


Discretization = Literal["zoh", "euler"]


class PlantLinearization(eqx.Module):
    """Continuous- and discrete-time LTI matrices for the linearised plant.

    Attributes:
        A_c: Continuous-time state-evolution matrix, shape (n, n).
        B_c: Continuous-time control matrix, shape (n, m_u).
        Bw_c: Continuous-time disturbance matrix, shape (n, m_w).
        A: Discrete-time state-evolution matrix at time step ``dt``, shape (n, n).
        B: Discrete-time control matrix, shape (n, m_u).
        Bw: Discrete-time disturbance matrix, shape (n, m_w).
        dt: Time step (seconds).
        discretization: Named discretization used for ``A`` and ``B``.
        state_indices: Dict mapping subspace name to slice (``"pos"``, ``"vel"``,
            ``"force"``).
        n: Total state dimension.
        m_u: Control dimension.
        m_w: Disturbance dimension.
    """

    A_c: Float[Array, "n n"]
    B_c: Float[Array, "n m_u"]
    Bw_c: Float[Array, "n m_w"]
    A: Float[Array, "n n"]
    B: Float[Array, "n m_u"]
    Bw: Float[Array, "n m_w"]
    dt: float
    pos_slice: tuple[int, int]
    vel_slice: tuple[int, int]
    force_slice: Optional[tuple[int, int]]
    discretization: Discretization = eqx.field(default="zoh", static=True)

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def m_u(self) -> int:
        return int(self.B.shape[1])

    @property
    def m_w(self) -> int:
        return int(self.Bw.shape[1])

    def pos(self, x: Float[Array, "... n"]) -> Float[Array, "... 2"]:
        """Extract position subvector from a state vector."""
        return x[..., self.pos_slice[0] : self.pos_slice[1]]

    def vel(self, x: Float[Array, "... n"]) -> Float[Array, "... 2"]:
        """Extract velocity subvector from a state vector."""
        return x[..., self.vel_slice[0] : self.vel_slice[1]]


class CostSchedule(eqx.Module):
    """Time-varying quadratic cost matrices for the LQ recursion.

    Attributes:
        Q: Stage state cost, shape (T, n, n) for T stages.
        R: Stage control cost, shape (T, m_u, m_u).
        Q_f: Terminal state cost, shape (n, n).
        T: Number of stages.
    """

    Q: Float[Array, "T n n"]
    R: Float[Array, "T m_u m_u"]
    Q_f: Float[Array, "n n"]

    @property
    def T(self) -> int:
        return int(self.Q.shape[0])


class RiccatiSolution(eqx.Module):
    """Output of ``solve_hinf_riccati``.

    Attributes:
        P: Riccati matrix sequence, shape (T+1, n, n). ``P[t]`` is the value-
            function Hessian at stage ``t``; ``P[T] = Q_f``.
        K: Feedback gain sequence, shape (T, m_u, n). Optimal control is
            ``u_t = -K[t] x_t``.
        admissible: Whether gamma is admissible (no boundary crossing through
            the recursion).
        spectral_radii: Per-step spectral radius
            :math:`\\lambda_{\\max}(\\gamma^{-2} B_w^\\top P_{t+1} B_w)`,
            shape (T,). Must stay below 1 for admissibility.
        bracket_conditions: Per-step condition number of the bracket
            :math:`I + (B R^{-1} B^\\top - \\gamma^{-2} B_w B_w^\\top) P_{t+1}`,
            shape (T,).
        max_P_cond: Maximum condition number of P_t over t.
        gamma: The gamma value at which this Riccati was solved.
    """

    P: Float[Array, "T_plus_1 n n"]
    K: Float[Array, "T m_u n"]
    admissible: bool = eqx.field(static=True)
    spectral_radii: Float[Array, "T"]
    bracket_conditions: Float[Array, "T"]
    max_P_cond: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)


class ClosedLoopRollout(eqx.Module):
    """Closed-loop simulation result.

    Attributes:
        x: State trajectory, shape (T+1, n).
        u: Control trajectory, shape (T, m_u).
        peak_velocity: Maximum :math:`\\lVert v_t \\rVert_2` over t (peak
            *speed* — 2-norm of full 2D velocity). Retained for
            backward-compatibility; the headline scalar for C&S comparisons is
            ``peak_forward_velocity``.
        peak_velocity_idx: Time index of peak speed.
        peak_forward_velocity: Maximum signed projection of velocity onto the
            reach axis (initial→target). Positive = motion toward target.
            This is the metric reported by Crevecoeur & Scott (2019) Fig. 1e.
        peak_forward_velocity_idx: Time index of ``peak_forward_velocity``.
        peak_lateral_velocity: Maximum speed in the direction orthogonal to
            the reach axis. C&S report suppression of this channel under the
            robust controller.
        peak_lateral_deviation: Maximum perpendicular distance from the
            straight line connecting initial position to terminal target.
        control_effort: :math:`\\sum_t \\lVert u_t \\rVert_2^2 \\, dt`.
        terminal_position_error: :math:`\\lVert x_T^{pos} - x_{target} \\rVert`.
    """

    x: Float[Array, "T_plus_1 n"]
    u: Float[Array, "T m_u"]
    peak_velocity: float
    peak_velocity_idx: int
    peak_forward_velocity: float
    peak_forward_velocity_idx: int
    peak_lateral_velocity: float
    peak_lateral_deviation: float
    control_effort: float
    terminal_position_error: float


# =============================================================================
# Plant linearisation
# =============================================================================


def linearize_pointmass(
    *,
    mass: float,
    damping: float,
    tau: float,
    dt: float,
    disturbance_channel: str = "velocity_force",
    discretization: Discretization = "zoh",
) -> PlantLinearization:
    """Linearise the rlrmp point-mass-plus-filter plant.

    Builds the continuous-time LTI matrices for the augmented state
    ``[pos(2), vel(2), F(2)]`` (or ``[pos(2), vel(2)]`` if ``tau == 0``) and
    discretises via the named scheme. Generic rlrmp analyses default to
    zero-order hold; C&S released-code fidelity passes ``"euler"`` explicitly.

    The plant is the one used in ``feedbax.xabdeef.models.point_mass_nn``:

        ``dot pos = vel``
        ``dot vel = -(damping/mass) vel + F/mass``
        ``dot F   = -F/tau + u/tau``                  (omitted if tau == 0)

    Two disturbance-channel conventions are supported:

    - ``"velocity_force"`` (default): ``w`` is an additive force on the
      velocity row, ``dot vel += w / mass``. ``B_w`` is shape ``(n, 2)``.
      Matches the *physical* curl-field / fixed-field intervenor channel used
      by ``feedbax.intervene`` and ``rlrmp.disturbance``. This is "flavor (a)"
      in the synthesis-review framing.
    - ``"full_state"``: ``w`` is a free additive disturbance on every state
      coordinate, ``B_w_d = I_n``. Matches Crevecoeur & Scott (2019)
      Eq. 13 in which the lumped disturbance ``ε_t = ΔA·x_t·dt + C·dw``
      enters every component of the state. This is the natural ``B_w`` for
      reproducing the C&S H-infinity Riccati design and the +Δv reaching
      signature reported in C&S Fig. 1e. Bug: ``97c227a``.

      Because ``B_w_d = I_n`` is set directly (not constructed from a
      continuous-time ``B_w_c`` via ZOH), the corresponding ``B_w_c`` is
      stored as ``I_n`` for completeness; it is not used by the discrete-time
      Riccati recursion. ``m_w = n`` in this mode.

    For ``tau == 0`` the filter rows drop out and ``B = [0; I/mass]``.

    Args:
        mass: Effector mass (kg).
        damping: Velocity damping coefficient k (Ns/m).
        tau: First-order force-filter time constant (s). If ``0.0`` the filter
            is omitted and the plant is the bare 4-state point mass.
        dt: Discretisation time step (s).
        disturbance_channel: ``"velocity_force"`` (default, physical curl-field
            channel) or ``"full_state"`` (C&S Eq 13 lumped ε on all states).
        discretization: ``"zoh"`` for zero-order hold or ``"euler"`` for
            forward Euler ``A_d = I + dt A_c``, ``B_d = dt B_c``.

    Returns:
        A ``PlantLinearization`` with continuous- and discrete-time matrices.

    Raises:
        ValueError: If ``mass`` or ``dt`` is non-positive, ``tau`` is
            negative, or ``disturbance_channel``/``discretization`` is
            unrecognised.
    """
    if mass <= 0:
        raise ValueError(f"mass must be positive, got {mass}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    if disturbance_channel not in ("velocity_force", "full_state"):
        raise ValueError(
            f"disturbance_channel must be 'velocity_force' or 'full_state', "
            f"got {disturbance_channel!r}"
        )
    if discretization not in ("zoh", "euler"):
        raise ValueError(
            f"discretization must be 'zoh' or 'euler', got {discretization!r}"
        )

    I2 = jnp.eye(2, dtype=jnp.float64)
    Z2 = jnp.zeros((2, 2), dtype=jnp.float64)

    if tau == 0.0:
        # 4-state plant: [pos, vel]
        A_c = jnp.block(
            [
                [Z2, I2],
                [Z2, -(damping / mass) * I2],
            ]
        )
        # control acts directly on velocity through (1/mass)
        B_c = jnp.concatenate([Z2, I2 / mass], axis=0)  # (4, 2)
        # Velocity-force disturbance channel (physical curl-field)
        Bw_c_vel = jnp.concatenate([Z2, I2 / mass], axis=0)  # (4, 2)
        pos_slice = (0, 2)
        vel_slice = (2, 4)
        force_slice = None
    else:
        # 6-state plant: [pos, vel, F]
        A_c = jnp.block(
            [
                [Z2, I2, Z2],
                [Z2, -(damping / mass) * I2, I2 / mass],
                [Z2, Z2, -(1.0 / tau) * I2],
            ]
        )
        B_c = jnp.concatenate([Z2, Z2, I2 / tau], axis=0)  # (6, 2)
        # disturbance bypasses the force filter, entering as additive force on velocity
        Bw_c_vel = jnp.concatenate([Z2, I2 / mass, Z2], axis=0)  # (6, 2)
        pos_slice = (0, 2)
        vel_slice = (2, 4)
        force_slice = (4, 6)

    n_state = A_c.shape[0]

    if disturbance_channel == "velocity_force":
        Bw_c = Bw_c_vel
        A_d, B_d, Bw_d = _discretize_lti(A_c, B_c, Bw_c, dt, discretization)
    else:
        # disturbance_channel == "full_state": B_w_d = I_n directly
        # (matches C&S Eq 13 ε formulation; one disturbance per state coord).
        Bw_c = jnp.eye(n_state, dtype=jnp.float64)
        # Discretise A, B without using Bw, then set Bw_d = I_n.
        A_d, B_d, _ = _discretize_lti(
            A_c, B_c, Bw_c_vel, dt, discretization
        )  # Bw_c_vel only used as placeholder
        Bw_d = jnp.eye(n_state, dtype=jnp.float64)

    return PlantLinearization(
        A_c=A_c,
        B_c=B_c,
        Bw_c=Bw_c,
        A=A_d,
        B=B_d,
        Bw=Bw_d,
        dt=float(dt),
        pos_slice=pos_slice,
        vel_slice=vel_slice,
        force_slice=force_slice,
        discretization=discretization,
    )


def cs_faithful_pointmass(
    *,
    mass: float = 1.0,
    damping: float = 0.1,
    tau: float = 0.066,
    dt: float = 0.01,
    disturbance_integrator: bool = True,
    delay_steps: int = 5,
    discretization: Discretization = "euler",
) -> PlantLinearization:
    """Build the Crevecoeur & Scott (2019) 8-state plant with delay augmentation.

    Canonical C&S setup (matches the released ModelDB code 258846):

    - **Mass + force-filter** (6 physical states): ``[px, py, vx, vy, fx, fy]``
      with ``mass=1 kg``, ``k=0.1 Ns/m``, ``tau=0.066 s``, ``dt=0.01 s``.
    - **Disturbance-integrator coupling** (Bug: ``9a0558e``, default on):
      adds two pure-integrator states ``[eps_x_int, eps_y_int]`` driven by the
      disturbance ``ε`` (rows 7,8 of A are zero). The integrators couple back
      into the velocity rows via ``A[3,7]=A[4,8]=1``. Disturbance matrix
      ``B_w = I_8`` so each ε_i drives state i directly.

      Adversary-redundancy property: under ``B_w = I_8`` the worst-case H∞
      adversary always finds a more efficient direct-velocity attack than
      going through the integrator pathway, so the integrator columns of
      ``B_w`` (``B_w[:, 6:8]``) do not change γ\\* or ``K_phys``. The
      integrator states still affect the closed-loop response *because the
      Riccati P couples them to velocity*, but they are dynamically inert
      from the adversary's perspective. This is documented in the audit
      ``/tmp/flavor_ab_review/findings/cs_alignment_audit.md`` and is the
      correct C&S H∞ game behaviour, not a bug.
    - **50 ms (5-step) full-state delay augmentation** (Bug: ``9a0558e``,
      default on): appends ``delay_steps`` blocks of ``n_phys = 8`` lag
      states implementing a tap-delay shift register on the *full* physical
      state. The augmented-state vector is

      ``x_{aug, t} = [x_t; x_{t-1}; x_{t-2}; …; x_{t-h}]``

      Faithful port of C&S's ``AugRobustControl.m``. Combined with the
      C&S-faithful Q distribution (``apply_delay_distribution_to_schedule``),
      the lag chain becomes cost-relevant — each block carries a
      time-shifted, ``1/(h+1)``-weighted copy of the physical Q. Without that
      distribution the lag chain is mathematically inert.
    - **Cost schedule**: Use ``cs_eq15_cost_schedule`` to build the *physical*
      Q (state_dim=8), then apply ``apply_delay_distribution_to_schedule`` to
      distribute it across the lag chain. The canonical C&S horizon is
      ``n_steps = 60`` (0.6 s at dt=0.01), set explicitly when constructing
      the schedule.
    - **Discretization** (Bug: ``dd232cd``): canonical C&S released-code
      fidelity uses forward Euler, matching the ModelDB MATLAB formulas
      ``A = I + dt*A_c`` and ``B = dt*B_c``. Pass ``discretization="zoh"``
      for the named higher-order sensitivity variant; it is not the canonical
      released-code path.

    The total state dimension is:

        n_phys = 6 (no integrators) or 8 (with integrators)
        n_aug  = (delay_steps + 1) · n_phys
                = 6 · 8 = 48 (canonical defaults: 8 phys + 5·8 lag)

    Args:
        mass: Effector mass (kg).
        damping: Velocity damping coefficient k (Ns/m).
        tau: First-order force-filter time constant (s). C&S use 0.066 s
            (``minmaxfc_pointMass.m`` line 23). Earlier rlrmp default was
            0.06 — corrected per the audit.
        dt: Discretisation time step (s).
        disturbance_integrator: If True (default), add two integrator states
            7,8 with the C&S coupling pattern. If False, fall back to the
            6-state plant with ``B_w = I_6`` (legacy behaviour).
        delay_steps: Number of delay taps. ``5`` (default) gives a 50 ms
            sensorimotor delay at ``dt=0.01``. ``0`` disables delay
            augmentation (legacy behaviour, same as pre-`9a0558e`).
        discretization: ``"euler"`` (default) for C&S released-code fidelity,
            or ``"zoh"`` for a named sensitivity variant.

    Returns:
        A ``PlantLinearization`` with ``A``, ``B``, ``B_w`` sized for the
        augmented system.

    Backward compatibility:
        ``cs_faithful_pointmass(disturbance_integrator=False, delay_steps=0,
        tau=0.06)`` retains the pre-9a0558e 6-state, no-delay shape and
        physical parameters under the current default Euler discretization.
        Add ``discretization="zoh"`` when comparing to the older higher-order
        sensitivity dynamics.

    Bug: ``9a0558e`` — structural lift to 8-state plant + full-state delay
    augmentation. The post-recipe-audit fix (full-state lag, tau=0.066,
    distributed Q via ``apply_delay_distribution_to_schedule``) brings
    analytical Δv on the canonical 15 cm reach to ~+7-8% (matches the
    MATLAB-faithful Python port at
    ``/tmp/flavor_ab_review/cs_alignment/cs_matlab_port.py`` and Fig 1e
    measurement ~+7.76%).
    Bug: ``97c227a`` — the prior 6-state full-state-B_w version (now reachable
    via ``disturbance_integrator=False, delay_steps=0``).
    """
    if delay_steps < 0:
        raise ValueError(f"delay_steps must be non-negative, got {delay_steps}")
    if discretization not in ("zoh", "euler"):
        raise ValueError(
            f"discretization must be 'zoh' or 'euler', got {discretization!r}"
        )

    if not disturbance_integrator:
        plant = linearize_pointmass(
            mass=mass, damping=damping, tau=tau, dt=dt,
            disturbance_channel="full_state", discretization=discretization,
        )
    else:
        plant = _build_cs_8state_pointmass(
            mass=mass, damping=damping, tau=tau, dt=dt,
            discretization=discretization,
        )

    if delay_steps > 0:
        plant = _apply_delay_augmentation(plant, delay_steps=delay_steps)

    return plant


def _build_cs_8state_pointmass(
    *,
    mass: float,
    damping: float,
    tau: float,
    dt: float,
    discretization: Discretization,
) -> PlantLinearization:
    """Build the C&S 8-state plant with disturbance-integrator coupling.

    State vector: ``[px, py, vx, vy, fx, fy, eps_x_int, eps_y_int]``.

    Continuous-time matrices follow ``minmaxfc_pointMass.m`` (ModelDB 258846).
    The novel structure relative to the 6-state form is:

    - Rows 7,8 of ``A_c`` are zero: integrator states have no intrinsic
      dynamics; they are driven only by the disturbance via ``B_w``.
    - ``A_c[3,7] = A_c[4,8] = 1``: the integrated disturbance enters the
      velocity rows as a force.
    - ``B_w = I_8`` (each ε_i drives state i directly).

    Args:
        mass: Effector mass (kg). Must be > 0.
        damping: Velocity damping (Ns/m).
        tau: Force-filter time constant (s). Must be > 0 for the 8-state form.
        dt: Discretisation time step (s).

    Returns:
        A ``PlantLinearization`` with ``n=8``, ``m_u=2``, ``m_w=8``.
    """
    if mass <= 0:
        raise ValueError(f"mass must be positive, got {mass}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if tau <= 0:
        raise ValueError(
            f"tau must be positive for the 8-state form (force filter required); "
            f"got {tau}. For tau=0 use the 6-state form."
        )

    I2 = jnp.eye(2, dtype=jnp.float64)
    Z2 = jnp.zeros((2, 2), dtype=jnp.float64)

    # Continuous-time A (8 x 8): block structure
    #   [ 0          I        0        0      ]   pos
    #   [ 0    -k/m·I    1/m·I        I/m    ]   vel  (col 7,8 = I/m couples integrators)
    #   [ 0          0    -1/τ·I       0      ]   F
    #   [ 0          0        0        0      ]   ε_int
    #
    # Per the user spec and minmaxfc_pointMass.m, the integrator-to-velocity
    # coupling is A[3,7]=A[4,8]=1 in the (mass=1) MATLAB code. With explicit
    # 1/mass scaling, A_c[vel,eps_int] = (1/mass) · I.
    A_c = jnp.block(
        [
            [Z2, I2, Z2, Z2],
            [Z2, -(damping / mass) * I2, I2 / mass, I2 / mass],
            [Z2, Z2, -(1.0 / tau) * I2, Z2],
            [Z2, Z2, Z2, Z2],
        ]
    )
    # Control enters force filter only.
    B_c = jnp.concatenate([Z2, Z2, I2 / tau, Z2], axis=0)  # (8, 2)
    # Disturbance enters all states directly (full-state B_w = I_8).
    Bw_c = jnp.eye(8, dtype=jnp.float64)

    # Discretise A and B by the selected named scheme. We then SET B_w_d = I_8
    # directly, matching the C&S released code where the disturbance D is a
    # discrete-time identity rather than a continuous-time channel integrated
    # by either Euler or ZOH.
    A_d, B_d, _ = _discretize_lti(A_c, B_c, Bw_c, dt, discretization)
    Bw_d = jnp.eye(8, dtype=jnp.float64)

    return PlantLinearization(
        A_c=A_c,
        B_c=B_c,
        Bw_c=Bw_c,
        A=A_d,
        B=B_d,
        Bw=Bw_d,
        dt=float(dt),
        pos_slice=(0, 2),
        vel_slice=(2, 4),
        force_slice=(4, 6),
        discretization=discretization,
    )


def _apply_delay_augmentation(
    plant: PlantLinearization,
    *,
    delay_steps: int,
) -> PlantLinearization:
    """Augment a plant with a full-state tap-delay shift register.

    Faithful port of Crevecoeur & Scott 2019 ``AugRobustControl.m`` (ModelDB
    258846). Lifts the input plant from ``n_phys`` states to
    ``(h+1) · n_phys`` states, where each lag block tracks a *full* delayed
    copy of the physical state ``x_{t-k}``, ``k = 0, 1, …, h``:

    .. math::

        x_{aug, t} = \\begin{bmatrix}
            x_{phys, t} \\\\
            x_{t-1} \\\\
            x_{t-2} \\\\
            \\vdots \\\\
            x_{t-h}
        \\end{bmatrix} \\in \\mathbb{R}^{(h+1) \\cdot n_{phys}}

    The augmented dynamics matrix has the block form

    .. math::

        A_{aug} = \\begin{bmatrix}
            A_{phys}      & 0       & 0       & \\cdots & 0 \\\\
            I             & 0       & 0       & \\cdots & 0 \\\\
            0             & I       & 0       & \\cdots & 0 \\\\
            \\vdots       & \\vdots & \\ddots & \\vdots & \\vdots \\\\
            0             & 0       & \\cdots & I       & 0
        \\end{bmatrix}

    i.e., MATLAB's ``A_aug(1:n, 1:n) = A_phys`` and
    ``A_aug(n+1:end, 1:end-n) = eye(h·n)``. ``B_aug`` and ``B_{w, aug}`` are
    zero-padded onto the physical block (control and disturbance affect
    physical states only). The C&S delay convention places the *oldest*
    delayed state at the bottom; the H feedback selects ``x_{t-h}`` as the
    measurement (``H(:, end-n+1:end) = H0`` in MATLAB), but the analytical
    Riccati pipeline here uses full-state feedback on the augmented state, so
    H is not stored.

    .. note::

        Earlier versions of this function (commits ``1f75d9f``, ``1c313b4``,
        Bug: ``9a0558e``) stored the lag chain on the *observation* channel
        only (``n_obs = 4``: pos + vel) instead of the full physical state.
        Combined with zero-padded Q on the lag blocks (see audit at
        ``/tmp/flavor_ab_review/findings/cs_alignment_audit.md``), this
        rendered the augmentation mathematically inert — the lag block was
        unreachable from u, undisturbed by w, and uncosted by Q, so the H∞
        Riccati found ``P[lag, lag] = 0`` identically and ``K[lag] = 0``.
        Switching to full-state lag matches ``AugRobustControl.m`` and
        couples with the distributed Q (see
        ``apply_delay_distribution_to_schedule``) to make the augmentation
        load-bearing.

    Args:
        plant: Underlying physical plant (4-, 6-, or 8-state).
        delay_steps: Number of delay taps ``h ≥ 1``.

    Returns:
        A new ``PlantLinearization`` with state dimension
        ``(h+1) · n_phys``. The augmented disturbance matrix ``B_w`` has the
        physical-block ``B_w`` in its top ``n_phys`` rows and zeros in the
        lag blocks (matching MATLAB's ``D(1:8, 1:8) = eye(8)``,
        ``D(9:end, :) = 0``).

    Bug: ``9a0558e`` — structural lift to 8-state + delay augmentation. See
    audit ``/tmp/flavor_ab_review/findings/cs_alignment_audit.md`` for the
    correctness argument behind the full-state lag.
    """
    if delay_steps < 1:
        raise ValueError(
            f"_apply_delay_augmentation requires delay_steps >= 1; got {delay_steps}"
        )

    n_phys = plant.n
    m_u = plant.m_u
    m_w = plant.m_w

    h = int(delay_steps)
    n_aug = (h + 1) * n_phys

    A_d_phys = plant.A.astype(jnp.float64)
    B_d_phys = plant.B.astype(jnp.float64)
    Bw_d_phys = plant.Bw.astype(jnp.float64)

    # MATLAB AugRobustControl.m:
    #   A(1:n, 1:n) = A0
    #   A(n+1:end, 1:end-n) = eye(h·n)
    A_aug_d = jnp.zeros((n_aug, n_aug), dtype=jnp.float64)
    A_aug_d = A_aug_d.at[:n_phys, :n_phys].set(A_d_phys)
    A_aug_d = A_aug_d.at[n_phys:, : h * n_phys].set(
        jnp.eye(h * n_phys, dtype=jnp.float64)
    )

    # B_aug, Bw_aug: zero-pad onto physical block.
    B_aug_d = jnp.zeros((n_aug, m_u), dtype=jnp.float64)
    B_aug_d = B_aug_d.at[:n_phys, :].set(B_d_phys)
    Bw_aug_d = jnp.zeros((n_aug, m_w), dtype=jnp.float64)
    Bw_aug_d = Bw_aug_d.at[:n_phys, :].set(Bw_d_phys)

    # Continuous-time matrices: representative block structure only.
    # The lag shift is intrinsically discrete — there is no clean
    # continuous-time analogue of A_aug, so we store a placeholder with the
    # physical block embedded and zeros elsewhere.
    A_aug_c = jnp.zeros((n_aug, n_aug), dtype=jnp.float64)
    A_aug_c = A_aug_c.at[:n_phys, :n_phys].set(plant.A_c.astype(jnp.float64))
    B_aug_c = jnp.zeros((n_aug, m_u), dtype=jnp.float64)
    B_aug_c = B_aug_c.at[:n_phys, :].set(plant.B_c.astype(jnp.float64))
    Bw_aug_c = jnp.zeros((n_aug, m_w), dtype=jnp.float64)
    Bw_aug_c = Bw_aug_c.at[:n_phys, :].set(plant.Bw_c.astype(jnp.float64))

    return PlantLinearization(
        A_c=A_aug_c,
        B_c=B_aug_c,
        Bw_c=Bw_aug_c,
        A=A_aug_d,
        B=B_aug_d,
        Bw=Bw_aug_d,
        dt=plant.dt,
        # pos/vel/force slices are unchanged (they refer to the physical block).
        pos_slice=plant.pos_slice,
        vel_slice=plant.vel_slice,
        force_slice=plant.force_slice,
        discretization=plant.discretization,
    )


def apply_delay_distribution_to_schedule(
    schedule: CostSchedule,
    *,
    delay_steps: int,
    n_phys: int,
) -> CostSchedule:
    """Distribute a physical-state Q schedule across a delay-augmented chain.

    Faithful port of the C&S 2019 ``AugRobustControl.m`` Q-distribution
    pattern (lines 38-51). Given a physical-state cost schedule with Q of
    shape ``(T, n_phys, n_phys)``, produces an augmented schedule for the
    state dimension ``(h+1) · n_phys`` that places a time-shifted, weighted
    copy of the physical Q on each lag block:

    .. math::

        Q_{aug}[ii \\cdot n : (ii+1) \\cdot n,\\ ii \\cdot n : (ii+1) \\cdot n,\\ t]
            = Q_0[\\,t + h - ii\\,] / (h + 1),
            \\quad ii = 0, 1, \\ldots, h.

    The "Qaug" pre-pad of length ``h`` (where indices below 0 are clamped to
    ``Q_0[0]``) implements MATLAB's

    .. code-block:: matlab

        for i = 1:h, Qaug(:,:,i) = Q0(:,:,1); end
        for i = 1:t, Qaug(:,:,i+h) = Q0(:,:,i); end

    so the lag blocks at the start of the reach see the t=0 cost, not zero.

    Why distribution matters:
        Without this distribution, the lag blocks of Q are zero, and combined
        with zero ``B[lag, :]`` and zero ``B_w[lag, :]`` the H∞ Riccati's
        Joseph recursion produces ``P[lag, lag] = 0`` identically. Lag states
        become dynamically inert — they record history the controller has
        zero incentive to react to. The distribution makes lag states
        cost-relevant (each block carries a shifted copy of the physical
        cost), and the resulting K_phys matches MATLAB's. See audit
        ``/tmp/flavor_ab_review/findings/cs_alignment_audit.md``.

    Args:
        schedule: Cost schedule with ``Q`` of shape ``(T, n_phys, n_phys)``,
            ``R`` of shape ``(T, m_u, m_u)``, ``Q_f`` of shape
            ``(n_phys, n_phys)``. Pass the *physical* schedule before delay
            augmentation.
        delay_steps: Number of delay taps ``h``. Must be ≥ 0; if 0, returns
            the input schedule unchanged.
        n_phys: Physical state dimension. Must equal ``schedule.Q.shape[-1]``.

    Returns:
        A new ``CostSchedule`` with ``Q`` of shape
        ``(T, (h+1) · n_phys, (h+1) · n_phys)`` and ``Q_f`` analogously
        distributed (using the terminal Q for all lag blocks). ``R`` is
        unchanged.

    Bug: ``9a0558e``.
    """
    if delay_steps < 0:
        raise ValueError(f"delay_steps must be non-negative, got {delay_steps}")
    if delay_steps == 0:
        return schedule

    h = int(delay_steps)
    Q_phys = schedule.Q  # (T, n_phys, n_phys)
    Q_f_phys = schedule.Q_f  # (n_phys, n_phys)
    if Q_phys.shape[1] != n_phys or Q_phys.shape[2] != n_phys:
        raise ValueError(
            f"schedule.Q has shape {Q_phys.shape}; expected (T, {n_phys}, {n_phys})"
        )
    T = Q_phys.shape[0]
    n_aug = (h + 1) * n_phys

    # Build Qaug: (n_phys, n_phys, T+h) — first h slots filled with Q_phys[0],
    # then T slots from Q_phys.
    # MATLAB indexing (1-based):
    #   Qaug(:,:,1..h) = Q0(:,:,1)   (replicate first cost h times)
    #   Qaug(:,:,h+1..h+T) = Q0(:,:,1..T)
    # In Python (0-based):
    #   Qaug[:,:,0..h-1] = Q_phys[0]
    #   Qaug[:,:,h..h+T-1] = Q_phys[0..T-1]
    Q_first = Q_phys[0]  # (n_phys, n_phys)
    Q_pad = jnp.broadcast_to(Q_first, (h, n_phys, n_phys))
    Qaug = jnp.concatenate([Q_pad, Q_phys], axis=0)  # (T+h, n_phys, n_phys)

    # Distribute: Q_aug[ii·n:(ii+1)·n, ii·n:(ii+1)·n, t] = Qaug[t+h-ii] / (h+1)
    # for ii in 0..h, t in 0..T-1.
    weight = 1.0 / float(h + 1)

    def build_Q_at(t_idx):
        # t_idx is a Python int in 0..T-1; this is unrolled at trace time.
        Q_t = jnp.zeros((n_aug, n_aug), dtype=Q_phys.dtype)
        for ii in range(h + 1):
            row = ii * n_phys
            t_shifted = t_idx + h - ii  # 0..T+h-1, valid index into Qaug
            block = Qaug[t_shifted] * weight
            Q_t = jax.lax.dynamic_update_slice(Q_t, block, (row, row))
        return Q_t

    # Build the time-stacked Q_aug. Use vmap over t_idx to leverage JAX
    # tracing efficiency, but we have to pass t_idx as an array index.
    def build_Q_at_t(t_idx):
        Q_t = jnp.zeros((n_aug, n_aug), dtype=Q_phys.dtype)
        for ii in range(h + 1):
            row = ii * n_phys
            t_shifted = t_idx + h - ii
            block = jnp.take(Qaug, t_shifted, axis=0) * weight
            Q_t = jax.lax.dynamic_update_slice(Q_t, block, (row, row))
        return Q_t

    Q_aug = jax.vmap(build_Q_at_t)(jnp.arange(T))  # (T, n_aug, n_aug)

    # Q_f distribution: at the terminal step (one beyond t=T-1), the
    # Riccati's terminal condition is M[:, :, -1] = Q_f. Per the MATLAB
    # driver `script_minmax_pointMass.m`, the terminal cost is the same
    # diagonal pattern at full ramp; we distribute Q_f across the lag chain
    # with the same 1/(h+1) weighting using Qaug[T-1+h-ii] (or Q_f for ii=0).
    # Concretely, the natural extrapolation past T-1 is Qaug[T+h-1]
    # = Q_phys[T-1] (the last physical Q). For ii=0 we want Q_f_phys; for
    # ii>0 we want Q_f shifted backwards in time: Qaug[T-ii] (still 1/(h+1)).
    # Since Q_f_phys typically equals Q_phys[T-1] (terminal-ramp = 1), we
    # simply distribute Q_f_phys uniformly across lag blocks too.
    Q_f_aug = jnp.zeros((n_aug, n_aug), dtype=Q_f_phys.dtype)
    for ii in range(h + 1):
        row = ii * n_phys
        # Use Q_f for newest lag (ii=0); use shifted Qaug for older lags.
        if ii == 0:
            block = Q_f_phys * weight
        else:
            t_shifted = (T - 1) + h - ii  # = T + h - 1 - ii
            t_shifted = max(0, min(T + h - 1, t_shifted))
            block = Qaug[t_shifted] * weight
        Q_f_aug = jax.lax.dynamic_update_slice(Q_f_aug, block, (row, row))

    return CostSchedule(Q=Q_aug, R=schedule.R, Q_f=Q_f_aug)


def linearize_from_model(model, *, dt: Optional[float] = None) -> PlantLinearization:
    """Linearise the rlrmp plant from an ``eqx.Module`` model.

    Walks the model PyTree to find the ``PointMass`` skeleton and the
    associated ``mass``, ``damping``, ``tau``, ``dt`` parameters. This
    accepts either:

    - a feedbax ``SimpleFeedback`` graph (the standard ``point_mass_nn``
      output, post eager-graph cutover; previously this would have been
      wrapped in a now-removed ``Iterator``), or
    - a ``Mechanics`` instance, or
    - a ``PointMass`` instance directly.

    Bug: b131510 — the legacy "Iterator" framing is stale; on feedbax
    ``develop`` the SimpleFeedback Graph runs its own cycles internally
    and there is no separate Iterator wrapper.

    The plant is *time-invariant* in the rlrmp setup, so this function does
    not require a nominal trajectory: linearisation around any state is
    identical (the system is exactly linear). The argument is named
    ``model`` rather than ``trajectory`` to match the spec.

    Args:
        model: A feedbax model containing a ``PointMass`` skeleton (or the
            skeleton directly).
        dt: Override the discretisation time step. If ``None``, pull from the
            model's ``Mechanics.dt`` (or the FilterState's ``dt``).

    Returns:
        A ``PlantLinearization``. See ``linearize_pointmass``.

    Raises:
        ValueError: If the model does not contain a ``PointMass`` skeleton or
            ``dt`` cannot be resolved.
    """
    pm, found_dt, found_tau = _walk_model_for_plant_params(model)
    if pm is None:
        raise ValueError(
            "Could not find a PointMass skeleton inside the model. "
            "Pass a feedbax SimpleFeedback/Mechanics/PointMass instance."
        )
    final_dt = float(dt) if dt is not None else found_dt
    if final_dt is None:
        raise ValueError(
            "Could not resolve dt from model and none was provided. "
            "Pass dt explicitly."
        )
    return linearize_pointmass(
        mass=float(pm.mass),
        damping=float(pm.damping),
        tau=float(found_tau if found_tau is not None else 0.0),
        dt=final_dt,
    )


def _walk_model_for_plant_params(
    model,
) -> Tuple[Optional[PointMass], Optional[float], Optional[float]]:
    """Find PointMass instance, dt, tau in a feedbax-shaped model.

    Returns ``(pointmass_or_none, dt_or_none, tau_or_none)``. If the model is
    a ``PointMass`` directly, returns ``(model, None, None)``.
    """
    pm: Optional[PointMass] = None
    dt: Optional[float] = None
    tau: Optional[float] = None

    # Direct PointMass case
    if isinstance(model, PointMass):
        return model, None, None

    leaves = jt.leaves(model, is_leaf=lambda x: isinstance(x, PointMass))
    for leaf in leaves:
        if isinstance(leaf, PointMass):
            pm = leaf
            break

    # Best-effort attribute walk for dt and tau. SimpleFeedback (a Graph)
    # has nodes including Mechanics with ``dt`` and an optional first-order
    # force filter with tau_rise/tau_decay. We don't depend on a specific
    # shape; we search the tree.
    def _maybe_set(node):
        nonlocal dt, tau
        if dt is None and hasattr(node, "dt"):
            try:
                cand = float(getattr(node, "dt"))
                if cand > 0:
                    dt = cand
            except (TypeError, ValueError):
                pass
        if tau is None and hasattr(node, "tau_rise"):
            try:
                tau = float(getattr(node, "tau_rise"))
            except (TypeError, ValueError):
                pass
        return node

    jt.map(_maybe_set, model, is_leaf=lambda x: hasattr(x, "dt") or hasattr(x, "tau_rise"))

    return pm, dt, tau


def _zoh_discretize(
    A_c: Float[Array, "n n"],
    B_c: Float[Array, "n m_u"],
    Bw_c: Float[Array, "n m_w"],
    dt: float,
) -> Tuple[Float[Array, "n n"], Float[Array, "n m_u"], Float[Array, "n m_w"]]:
    """Zero-order hold discretisation of (A_c, B_c, Bw_c) to step size dt.

    Uses the augmented exponential trick: stack inputs into a block and
    take the matrix exponential of a (n + m_u + m_w) square matrix.
    """
    n = A_c.shape[0]
    m_u = B_c.shape[1]
    m_w = Bw_c.shape[1]
    M = jnp.zeros((n + m_u + m_w, n + m_u + m_w), dtype=jnp.float64)
    M = M.at[:n, :n].set(A_c)
    M = M.at[:n, n : n + m_u].set(B_c)
    M = M.at[:n, n + m_u :].set(Bw_c)
    expM = jsla.expm(M * dt)
    A_d = expM[:n, :n]
    B_d = expM[:n, n : n + m_u]
    Bw_d = expM[:n, n + m_u :]
    return A_d, B_d, Bw_d


def _euler_discretize(
    A_c: Float[Array, "n n"],
    B_c: Float[Array, "n m_u"],
    Bw_c: Float[Array, "n m_w"],
    dt: float,
) -> Tuple[Float[Array, "n n"], Float[Array, "n m_u"], Float[Array, "n m_w"]]:
    """Forward Euler discretisation used by the C&S released MATLAB code."""

    n = A_c.shape[0]
    A_d = jnp.eye(n, dtype=jnp.float64) + dt * A_c
    B_d = dt * B_c
    Bw_d = dt * Bw_c
    return A_d, B_d, Bw_d


def _discretize_lti(
    A_c: Float[Array, "n n"],
    B_c: Float[Array, "n m_u"],
    Bw_c: Float[Array, "n m_w"],
    dt: float,
    discretization: Discretization,
) -> Tuple[Float[Array, "n n"], Float[Array, "n m_u"], Float[Array, "n m_w"]]:
    """Dispatch to a named LTI discretization scheme."""

    if discretization == "zoh":
        return _zoh_discretize(A_c, B_c, Bw_c, dt)
    if discretization == "euler":
        return _euler_discretize(A_c, B_c, Bw_c, dt)
    raise ValueError(
        f"discretization must be 'zoh' or 'euler', got {discretization!r}"
    )


# =============================================================================
# Cost schedule
# =============================================================================


@dataclass(frozen=True)
class CostSpec:
    """Specification for the rlrmp loss-derived Q,R schedule.

    Mirrors synthesis_review section 2: position-only state cost during the
    movement window, with a cosine ramp on the late portion. Matches the
    ``running_cost`` mode used in Phase 2 of part2_5.

    All weights are unitless multipliers on identity matrices in the
    relevant subspace.

    Attributes:
        n_steps: Total number of stages T (= ``hps.task.n_steps - 1`` in
            rlrmp; one fewer than total simulation timesteps because the
            terminal stage carries Q_f).
        go_step: Time index of the go cue (zero-based).
        pos_mid_weight: Weight on position cost during mid-period (after go,
            before late). Maps to ``loss.weights.effector_pos_mid *
            loss.effector_pos_mid.ramp_final_weight`` in the rlrmp config.
        vel_mid_weight: Weight on velocity cost during mid-period.
        pos_late_weight: Weight on position cost during late period.
        vel_late_weight: Weight on velocity cost during late period.
        pos_late_scale_factor: Cosine ramp endpoint scale factor on late
            position weight (matches ``effector_pos_late.final_scale_factor``).
        vel_late_scale_factor: Cosine ramp endpoint scale factor on late
            velocity weight.
        late_start_offset: Steps after go cue when the late period begins
            (matches ``effector_pos_late.start_step_after_go``).
        R_weight: Diagonal control cost. Matches ``loss.weights.nn_output``
            for non-adaptive runs, or the converged adaptive control weight
            (typically ~3e-5) for ``loss_update``-enabled runs.
        terminal_pos_weight: Weight on final-stage position cost (Q_f).
        terminal_vel_weight: Weight on final-stage velocity cost.

    Notes:
        - The weights are applied to position and velocity subspaces of the
          full augmented state (pos, vel, [F]). Force-state cost is zero.
        - For C&S-style sanity tests, override ``vel_late_weight`` and
          ``vel_late_scale_factor`` to match C&S's cost emphasis.
    """

    n_steps: int
    go_step: int = 30
    pos_mid_weight: float = 0.1  # rlrmp default ramp_final_weight
    vel_mid_weight: float = 0.0  # rlrmp default for vel_mid is zeroed
    pos_late_weight: float = 1.0
    vel_late_weight: float = 0.1
    pos_late_scale_factor: float = 3.0
    vel_late_scale_factor: float = 3.0
    late_start_offset: int = 80
    R_weight: float = 3.0e-5  # Adaptive-control-cost converged value (synthesis_review section 2)
    terminal_pos_weight: float = 4.0
    terminal_vel_weight: float = 0.4


def cost_schedule_from_spec(
    spec: CostSpec,
    plant: PlantLinearization,
) -> CostSchedule:
    """Build a time-varying Q,R schedule from a ``CostSpec`` and plant.

    Constructs Q_t, R_t, Q_f matching the rlrmp ``running_cost`` loss with
    the late-period cosine ramp described in synthesis_review section 2.
    Position and velocity weights apply to their respective state subspaces;
    the force subspace (if present) carries zero cost.

    The cosine ramp on the late period follows
    ``rlrmp.loss.make_late_discount_from_epoch``:

        ``frac(t) = clip((t - late_start) / (T - late_start), 0, 1)``
        ``ramp(t) = 1 + (scale - 1) * (0.5 - 0.5 cos(pi * frac))``

    Args:
        spec: The cost specification.
        plant: The plant linearisation, used to size the matrices.

    Returns:
        A ``CostSchedule`` with ``Q``, ``R``, ``Q_f``.
    """
    T = spec.n_steps
    n = plant.n
    m_u = plant.m_u

    # Per-step time index
    t = jnp.arange(T, dtype=jnp.float64)
    late_start = float(spec.go_step + spec.late_start_offset)
    denom = max(1.0, float(T) - late_start)
    frac = jnp.clip((t - late_start) / denom, 0.0, 1.0)
    cos_ramp = 0.5 - 0.5 * jnp.cos(jnp.pi * frac)  # 0 -> 0, 1 -> 1, smooth

    # Mid-period mask: from go_step until late_start
    mid_mask = jnp.where((t >= float(spec.go_step)) & (t < late_start), 1.0, 0.0)
    # Late-period mask: from late_start onwards
    late_mask = jnp.where(t >= late_start, 1.0, 0.0)

    # Time-varying scalar weights
    pos_weight_t = (
        spec.pos_mid_weight * mid_mask
        + spec.pos_late_weight * (1.0 + (spec.pos_late_scale_factor - 1.0) * cos_ramp) * late_mask
    )
    vel_weight_t = (
        spec.vel_mid_weight * mid_mask
        + spec.vel_late_weight * (1.0 + (spec.vel_late_scale_factor - 1.0) * cos_ramp) * late_mask
    )

    # Build Q_t by writing position and velocity blocks into the full n x n
    pos_lo, pos_hi = plant.pos_slice
    vel_lo, vel_hi = plant.vel_slice
    I_pos = jnp.eye(pos_hi - pos_lo, dtype=jnp.float64)
    I_vel = jnp.eye(vel_hi - vel_lo, dtype=jnp.float64)

    base_Q = jnp.zeros((n, n), dtype=jnp.float64)

    def _Q_at(pos_w, vel_w):
        Q = base_Q
        Q = jax.lax.dynamic_update_slice(
            Q, pos_w * I_pos, (pos_lo, pos_lo)
        )
        Q = jax.lax.dynamic_update_slice(
            Q, vel_w * I_vel, (vel_lo, vel_lo)
        )
        return Q

    # vmap over time
    Q = jax.vmap(_Q_at)(pos_weight_t, vel_weight_t)

    # R_t (constant in time for the rlrmp setup)
    R = jnp.tile(spec.R_weight * jnp.eye(m_u, dtype=jnp.float64)[None], (T, 1, 1))

    # Terminal Q_f
    Q_f = _Q_at(jnp.float64(spec.terminal_pos_weight), jnp.float64(spec.terminal_vel_weight))

    return CostSchedule(Q=Q, R=R, Q_f=Q_f)


def cs_eq15_cost_schedule(
    n_steps: int,
    alpha_1: float = 1.0,
    *,
    state_dim: int = 8,
) -> CostSchedule:
    """Build the Crevecoeur & Scott (2019) Eq. 15 cost schedule.

    Implements the C&S 2019 cost function on the C&S point-mass plant. Per
    ``script_minmax_pointMass.m`` lines 25-32:

    .. code-block:: matlab

        runningalpha = zeros(8, nStep);
        for i = 1:nStep
            fact = min(1, (i*delta/time)^6);
            runningalpha(:,i) = [fact*1e6  fact*1e6  fact*1e5  fact*1e5  1  1  1  1]';
        end

    The diagonal Q has the form

    .. math::

        Q_t = \\mathrm{diag}([\\mathrm{fact}_t \\cdot 10^6,\\ \\mathrm{fact}_t \\cdot 10^6,\\
                              \\mathrm{fact}_t \\cdot 10^5,\\ \\mathrm{fact}_t \\cdot 10^5,\\
                              1, 1, 1, 1]),

    with the time-varying ramp factor

    .. math::

        \\mathrm{fact}_t = \\min\\!\\Big(1, \\big((t+1) \\delta / T\\big)^6\\Big)

    (MATLAB 1-indexed: ``i = 1, …, nStep``). The crucial detail is that
    **entries 5-8 (force + integrator) are NOT scaled by ``fact``** — only
    the position and velocity entries ramp. This was a bug in the earlier
    rlrmp version, where the entire diagonal was uniformly scaled by the
    ramp; see audit at
    ``/tmp/flavor_ab_review/findings/cs_alignment_audit.md``.

    Padding for higher-dimensional plants:
        - ``6``: legacy no-integrator, no-delay form. Truncated to the first
          six entries ``[fact·1e6, fact·1e6, fact·1e5, fact·1e5, 1, 1]``.
        - ``8`` (default): canonical 8-state plant with disturbance
          integrators. Integrator states 6,7 take constant weight 1.
        - ``> 8``: extra entries are padded with zeros. For C&S-faithful
          delay augmentation, prefer building the physical schedule
          (``state_dim=8``) and applying
          ``apply_delay_distribution_to_schedule`` to distribute the cost
          across the lag chain per ``AugRobustControl.m``.

    Bug: ``9a0558e`` — recipe-bug audit. Fixes vs. the prior implementation:
    (1) entries 6,7 set to 1.0 (constant), not 0; (2) entries 4-7 not scaled
    by ramp; (3) cap-at-1 applied to (t/N)^6 (matches ``min(1, ...)`` in
    MATLAB).

    Args:
        n_steps: Total number of stages T (matches MATLAB ``simdata.nStep``,
            inclusive of the terminal stage).
        alpha_1: Multiplier on the entire Q schedule (default 1.0). Scales
            both ramped and constant entries.
        state_dim: Total state dimension. Must be ≥ 6. Default ``8`` matches
            the canonical C&S plant.

    Returns:
        A ``CostSchedule`` with shape ``Q=(T, state_dim, state_dim)``,
        ``R=(T, 2, 2)``, ``Q_f=(state_dim, state_dim)``.

    Note:
        ``Q_f`` is set to the t = T value of Q (saturated ramp), i.e.
        ``alpha_1 * diag([1e6, 1e6, 1e5, 1e5, 1, 1, 1, 1])``.
    """
    if state_dim < 6:
        raise ValueError(
            f"state_dim must be >= 6 (the C&S Eq.15 6-element diagonal); got {state_dim}"
        )
    T = n_steps

    # C&S Q diagonal split into ramped and constant components per
    # script_minmax_pointMass.m line 30:
    #   runningalpha(:,i) = [fact*1e6  fact*1e6  fact*1e5  fact*1e5  1  1  1  1]'
    # Entries 0-3 (pos, vel) are ramped; entries 4-7 (force, integrator) are
    # constant 1. Bug: 9a0558e — see audit at
    # /tmp/flavor_ab_review/findings/cs_alignment_audit.md.
    n_canonical = min(state_dim, 8)
    ramped_full = jnp.array([1e6, 1e6, 1e5, 1e5, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    constant_full = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)
    ramped_diag = jnp.zeros((state_dim,), dtype=jnp.float64).at[:n_canonical].set(
        ramped_full[:n_canonical]
    )
    constant_diag = jnp.zeros((state_dim,), dtype=jnp.float64).at[:n_canonical].set(
        constant_full[:n_canonical]
    )

    # Time-varying ramp matching MATLAB `script_minmax_pointMass.m` line 28:
    #     fact = min(1, (i*delta/time)^6)
    # where i = 1..nStep, delta is the timestep, and time is the reach
    # duration. The C&S setup uses time = nStep * delta (the cost schedule
    # has nStep entries from i=1 to i=nStep, with i=nStep being the terminal
    # saturation point: fact_nStep = (nStep*delta/time)^6 = 1). Equivalently:
    #     fact_i = min(1, (i / nStep)^6)
    # In Python (0-indexed t = 0..T-1) this becomes
    #     fact_t = min(1, ((t+1) / T)^6)
    # so the terminal stage t = T-1 has fact = (T/T)^6 = 1 (full saturation).
    # This matches MATLAB's `runningalpha(:, nStep)` at the terminal index.
    i_plus_1 = jnp.arange(1, T + 1, dtype=jnp.float64)
    fact = jnp.minimum(1.0, (i_plus_1 / float(T)) ** 6)  # shape (T,)

    # Q_t = alpha_1 · (fact_t * diag(ramped) + diag(constant))
    def Q_at_t(f):
        return alpha_1 * (jnp.diag(f * ramped_diag) + jnp.diag(constant_diag))

    Q = jax.vmap(Q_at_t)(fact)  # (T, state_dim, state_dim)

    # R = I_2 (unit control cost, matching |u_t|^2 in C&S Eq. 15)
    R_t = jnp.eye(2, dtype=jnp.float64)
    R = jnp.tile(R_t[None], (T, 1, 1))  # (T, 2, 2)

    # Terminal cost Q_f: ramp saturates to 1 → full diagonal active.
    Q_f = alpha_1 * (jnp.diag(ramped_diag) + jnp.diag(constant_diag))

    return CostSchedule(Q=Q, R=R, Q_f=Q_f)


def cost_schedule_from_loss_config(hps, plant: PlantLinearization) -> CostSchedule:
    """Build a Q,R schedule from a populated rlrmp ``hps`` namespace.

    This is the bridge between rlrmp's YAML loss config and the LQ-game
    representation. It handles the ``running_cost`` mode and the standard
    structured mode used in part2_5.

    Args:
        hps: A ``feedbax.types.TreeNamespace`` containing ``loss``, ``task``,
            and ``loss_update`` sub-namespaces (as produced by
            ``rlrmp.config``).
        plant: The plant linearisation.

    Returns:
        A ``CostSchedule``.
    """
    weights = getattr(hps.loss, "weights", {}) or {}

    def _w(name: str, default: float) -> float:
        val = getattr(weights, name, None)
        if val is None and isinstance(weights, dict):
            val = weights.get(name, None)
        return float(default if val is None else val)

    # Mid-period
    mid_pos_cfg = getattr(hps.loss, "effector_pos_mid", None)
    mid_pos_final = 0.1
    if mid_pos_cfg is not None:
        mid_pos_final = float(getattr(mid_pos_cfg, "ramp_final_weight", 0.1))
    mid_vel_cfg = getattr(hps.loss, "effector_vel_mid", None)
    mid_vel_final = 0.1
    if mid_vel_cfg is not None:
        mid_vel_final = float(getattr(mid_vel_cfg, "ramp_final_weight", 0.1))

    # Late-period
    late_pos_cfg = getattr(hps.loss, "effector_pos_late", None)
    late_pos_offset = 80
    late_pos_scale = 3.0
    if late_pos_cfg is not None:
        late_pos_offset = int(getattr(late_pos_cfg, "start_step_after_go", 80))
        late_pos_scale = float(getattr(late_pos_cfg, "final_scale_factor", 3.0))
    late_vel_cfg = getattr(hps.loss, "effector_vel_late", None)
    late_vel_scale = 3.0
    if late_vel_cfg is not None:
        late_vel_scale = float(getattr(late_vel_cfg, "final_scale_factor", 3.0))

    n_steps = int(hps.task.n_steps) - 1
    go_step = _resolve_go_step(hps)

    # Adaptive control cost: if loss_update is enabled, prefer the converged
    # value documented in synthesis_review section 2; else fall back to the
    # static weight in loss.weights.nn_output (typically 1e-6).
    R_weight = _w("nn_output", 1e-6)
    loss_update = getattr(hps, "loss_update", None)
    if loss_update is not None and bool(getattr(loss_update, "enabled", False)):
        target_ratio = float(getattr(loss_update, "target_ratio", 0.5))
        # Synthesis review section 2 reports the converged value as ~3e-5 for
        # rlrmp's standard config. Pass through unless caller overrides.
        R_weight = float(getattr(loss_update, "converged_R", 3.0e-5))
        # We don't know the converged ratio without running; expose the
        # target_ratio for documentation but keep R_weight as the override.
        del target_ratio

    spec = CostSpec(
        n_steps=n_steps,
        go_step=go_step,
        pos_mid_weight=_w("effector_pos_mid", 1.0) * mid_pos_final,
        vel_mid_weight=_w("effector_vel_mid", 0.0) * mid_vel_final,
        pos_late_weight=_w("effector_pos_late", 1.0),
        vel_late_weight=_w("effector_vel_late", 0.1),
        pos_late_scale_factor=late_pos_scale,
        vel_late_scale_factor=late_vel_scale,
        late_start_offset=late_pos_offset,
        R_weight=R_weight,
        terminal_pos_weight=_w("effector_pos_late", 1.0) * late_pos_scale,
        terminal_vel_weight=_w("effector_vel_late", 0.1) * late_vel_scale,
    )
    return cost_schedule_from_spec(spec, plant)


def _resolve_go_step(hps) -> int:
    """Resolve the go-cue step index from an hps namespace.

    rlrmp encodes epoch lengths as ranges in ``hps.task.epoch_len_ranges``.
    The go cue falls at the start of the last (movement) epoch, which is the
    sum of the means of all preceding epoch ranges. For a deterministic
    canonical schedule we use the upper bound of each range (matching the
    fixed-trial policy used for evaluation).
    """
    epoch_ranges = getattr(hps.task, "epoch_len_ranges", None)
    if epoch_ranges is None:
        # Conservative default: go cue ~1/4 into trial
        return int(hps.task.n_steps) // 4
    # Use upper bound of each pre-movement epoch for the canonical schedule
    return int(sum(r[1] - 1 for r in list(epoch_ranges)))


# =============================================================================
# Riccati recursion
# =============================================================================


# Numerical thresholds for admissibility. ``RHO_TOL`` is the margin below 1
# for the spectral radius gamma^{-2} B_w^T P B_w; ``COND_TOL`` is the maximum
# acceptable bracket condition number before we flag instability.
_RHO_TOL = 1e-6
_COND_TOL = 1e12


def solve_hinf_riccati(
    plant: PlantLinearization,
    schedule: CostSchedule,
    gamma: float,
) -> RiccatiSolution:
    """Solve the finite-horizon discrete-time H-infinity LQ-game Riccati.

    Implements the backward recursion described at the top of this module.
    The recursion is unrolled in Python (no ``jax.lax.scan``) so that we can
    short-circuit on inadmissibility and report per-step diagnostics. T is
    typically ~100 in the rlrmp setup, which is comfortably within the
    XLA-trace budget for single-call usage.

    Args:
        plant: Linearised plant (provides A, B, B_w).
        schedule: Time-varying Q, R, Q_f.
        gamma: H-infinity disturbance attenuation level. ``gamma -> infinity``
            recovers LQR; ``gamma`` near ``gamma_star`` is the most aggressive
            admissible H-infinity controller.

    Returns:
        A ``RiccatiSolution`` with ``P``, ``K``, admissibility flag, and
        per-step diagnostics. If the recursion crosses the boundary at some
        step ``t``, ``admissible = False`` and ``K[t:]`` carries the partial
        result up to that step (later entries are zero).

    Raises:
        ValueError: If ``gamma <= 0`` or shapes do not match.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    Bw = plant.Bw.astype(jnp.float64)
    n = A.shape[0]
    m_u = B.shape[1]
    m_w = Bw.shape[1]
    T = schedule.T

    if schedule.Q.shape[1:] != (n, n):
        raise ValueError(
            f"Q shape mismatch: got {schedule.Q.shape}, expected (T, {n}, {n})"
        )
    if schedule.R.shape[1:] != (m_u, m_u):
        raise ValueError(
            f"R shape mismatch: got {schedule.R.shape}, expected (T, {m_u}, {m_u})"
        )

    Q_f = schedule.Q_f.astype(jnp.float64)
    Q_seq = schedule.Q.astype(jnp.float64)
    R_seq = schedule.R.astype(jnp.float64)
    I_n = jnp.eye(n, dtype=jnp.float64)
    I_mw = jnp.eye(m_w, dtype=jnp.float64)
    inv_gamma2 = 1.0 / (gamma * gamma)

    P_list = [Q_f]
    K_list: list[jnp.ndarray] = []
    spectral_radii: list[float] = []
    bracket_conds: list[float] = []
    admissible = True

    for k in range(T - 1, -1, -1):
        P_next = P_list[-1]
        Q_t = Q_seq[k]
        R_t = R_seq[k]

        # Admissibility check on the m_w x m_w spectral radius
        # gamma^{-2} B_w^T P B_w. If max eigenvalue >= 1, gamma is too small.
        BwT_P_Bw = inv_gamma2 * (Bw.T @ P_next @ Bw)
        eigvals = jnp.linalg.eigvalsh(BwT_P_Bw)
        rho = float(jnp.max(jnp.real(eigvals)))
        spectral_radii.append(rho)

        if not jnp.isfinite(rho) or rho >= 1.0 - _RHO_TOL:
            admissible = False
            # Pad K with zeros for the remaining steps
            for _ in range(k + 1):
                K_list.append(jnp.zeros((m_u, n), dtype=jnp.float64))
                bracket_conds.append(float("inf"))
                if len(spectral_radii) <= k:
                    spectral_radii.append(float("inf"))
            # Keep P at last successful value (don't update further)
            for _ in range(k + 1):
                P_list.append(P_list[-1])
            break

        # M = B R^{-1} B^T - gamma^{-2} B_w B_w^T  (n x n)
        # We want (I_n + M @ P_next)^{-1} A. Compute via solve.
        # Since R is small (~3e-5), use solve(R, B^T) for stability.
        R_inv_BT = jnp.linalg.solve(R_t, B.T)  # (m_u, n)
        M = B @ R_inv_BT - inv_gamma2 * (Bw @ Bw.T)  # (n, n)
        bracket = I_n + M @ P_next  # (n, n)

        # Condition-number-based diagnostic
        try:
            sing_vals = jnp.linalg.svd(bracket, compute_uv=False)
            sv_min = float(jnp.min(sing_vals))
            sv_max = float(jnp.max(sing_vals))
            cond = sv_max / max(sv_min, 1e-30)
        except Exception:
            cond = float("inf")
        bracket_conds.append(cond)

        if not jnp.isfinite(cond) or cond > _COND_TOL:
            admissible = False
            for _ in range(k + 1):
                K_list.append(jnp.zeros((m_u, n), dtype=jnp.float64))
            for _ in range(k + 1):
                P_list.append(P_list[-1])
            break

        bracket_inv_A = jnp.linalg.solve(bracket, A)  # (n, n)
        P_t = Q_t + A.T @ P_next @ bracket_inv_A
        # Symmetrise to suppress drift
        P_t = 0.5 * (P_t + P_t.T)

        # Feedback gain K_t
        # Lambda_t = (I - gamma^{-2} P_next B_w B_w^T)^{-1} P_next  (n x n)
        # Equivalent: solve (I - gamma^{-2} P_next B_w B_w^T) X = P_next.
        Lambda_lhs = I_n - inv_gamma2 * (P_next @ Bw @ Bw.T)
        Lambda = jnp.linalg.solve(Lambda_lhs, P_next)
        # K_t = (R + B^T Lambda B)^{-1} B^T Lambda A  (m_u x n)
        K_lhs = R_t + B.T @ Lambda @ B
        K_rhs = B.T @ Lambda @ A
        K_t = jnp.linalg.solve(K_lhs, K_rhs)

        K_list.append(K_t)
        P_list.append(P_t)

    # P_list was built backwards from index T to 0. Reverse to get [P_0, P_1, ..., P_T].
    P_arr = jnp.stack(list(reversed(P_list)), axis=0)
    # K_list was built backwards from t=T-1 down to t=0. Reverse to t=0..T-1.
    K_arr = jnp.stack(list(reversed(K_list)), axis=0)
    spec_arr = jnp.array(list(reversed(spectral_radii)), dtype=jnp.float64)
    cond_arr = jnp.array(list(reversed(bracket_conds)), dtype=jnp.float64)

    # Pad spec_arr if recursion short-circuited (length should be T)
    if spec_arr.shape[0] < T:
        spec_arr = jnp.concatenate([
            jnp.full((T - spec_arr.shape[0],), float("inf"), dtype=jnp.float64),
            spec_arr,
        ])
    if cond_arr.shape[0] < T:
        cond_arr = jnp.concatenate([
            jnp.full((T - cond_arr.shape[0],), float("inf"), dtype=jnp.float64),
            cond_arr,
        ])

    # Compute max P condition number across t (only over admissible entries)
    if admissible:
        P_conds_per_t = jax.vmap(lambda P: jnp.linalg.cond(P + 1e-12 * I_n))(P_arr)
        max_P_cond = float(jnp.max(P_conds_per_t))
    else:
        max_P_cond = float("inf")

    return RiccatiSolution(
        P=P_arr,
        K=K_arr,
        admissible=admissible,
        spectral_radii=spec_arr,
        bracket_conditions=cond_arr,
        max_P_cond=max_P_cond,
        gamma=float(gamma),
    )


def solve_lqr(plant: PlantLinearization, schedule: CostSchedule) -> RiccatiSolution:
    """Solve the finite-horizon discrete-time LQR (gamma -> infinity limit).

    This is the standard discrete-time LQR backward recursion:

    .. math::

        P_t = Q_t + A^\\top P_{t+1} A
        - A^\\top P_{t+1} B (R_t + B^\\top P_{t+1} B)^{-1} B^\\top P_{t+1} A,

    initialised at :math:`P_T = Q_f`. Equivalent to ``solve_hinf_riccati`` in
    the limit ``gamma -> infinity``.

    Args:
        plant: Linearised plant.
        schedule: Cost schedule.

    Returns:
        A ``RiccatiSolution`` with ``admissible=True``, infinite ``gamma``,
        zero ``spectral_radii``.
    """
    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    n = A.shape[0]
    m_u = B.shape[1]
    T = schedule.T

    Q_f = schedule.Q_f.astype(jnp.float64)
    Q_seq = schedule.Q.astype(jnp.float64)
    R_seq = schedule.R.astype(jnp.float64)

    P_list = [Q_f]
    K_list: list[jnp.ndarray] = []
    for k in range(T - 1, -1, -1):
        P_next = P_list[-1]
        Q_t = Q_seq[k]
        R_t = R_seq[k]
        BPA = B.T @ P_next @ A
        K_lhs = R_t + B.T @ P_next @ B
        K_t = jnp.linalg.solve(K_lhs, BPA)
        P_t = Q_t + A.T @ P_next @ A - A.T @ P_next @ B @ K_t
        P_t = 0.5 * (P_t + P_t.T)
        K_list.append(K_t)
        P_list.append(P_t)

    P_arr = jnp.stack(list(reversed(P_list)), axis=0)
    K_arr = jnp.stack(list(reversed(K_list)), axis=0)
    return RiccatiSolution(
        P=P_arr,
        K=K_arr,
        admissible=True,
        spectral_radii=jnp.zeros((T,), dtype=jnp.float64),
        bracket_conditions=jnp.ones((T,), dtype=jnp.float64),
        max_P_cond=float(jnp.max(jax.vmap(jnp.linalg.cond)(P_arr))),
        gamma=float("inf"),
    )


def find_gamma_star(
    plant: PlantLinearization,
    schedule: CostSchedule,
    *,
    gamma_lo: float = 1e-4,
    gamma_hi: float = 1.0,
    max_iter: int = 60,
    tol: float = 1e-4,
) -> float:
    """Bisection for the smallest admissible gamma.

    Finds the boundary gamma_star such that ``gamma > gamma_star`` is
    admissible (the H-infinity Riccati has a finite, well-conditioned
    solution) and ``gamma < gamma_star`` is not.

    Args:
        plant: Linearised plant.
        schedule: Cost schedule.
        gamma_lo: Initial lower bracket. The function expands ``gamma_lo``
            downward by powers of 10 if it is admissible.
        gamma_hi: Initial upper bracket. The function expands ``gamma_hi``
            upward by powers of 10 if it is inadmissible.
        max_iter: Maximum bisection iterations.
        tol: Convergence tolerance on relative gamma width
            ``(gamma_hi - gamma_lo) / gamma_hi``.

    Returns:
        Estimated gamma_star (rounded to the inadmissible side; returned
        value is the smallest gamma still found admissible at the bisection
        precision).

    Raises:
        RuntimeError: If the bracket cannot be established within reasonable
            bounds.
    """
    # Expand gamma_hi upward until we get an admissible solution
    expand_iter = 0
    while expand_iter < 30:
        sol = solve_hinf_riccati(plant, schedule, gamma_hi)
        if sol.admissible:
            break
        gamma_hi *= 10.0
        expand_iter += 1
    else:
        raise RuntimeError(
            f"Could not find admissible gamma up to {gamma_hi}. "
            "Cost schedule may be pathological."
        )

    # Expand gamma_lo downward until we get an inadmissible solution
    expand_iter = 0
    while expand_iter < 30:
        sol = solve_hinf_riccati(plant, schedule, gamma_lo)
        if not sol.admissible:
            break
        gamma_lo *= 0.1
        expand_iter += 1
    else:
        raise RuntimeError(
            f"Could not find inadmissible gamma down to {gamma_lo}. "
            "The Riccati appears trivially solvable for all gamma."
        )

    # Bisection
    for _ in range(max_iter):
        gamma_mid = jnp.sqrt(gamma_lo * gamma_hi).item()  # geometric mean for log-scale
        sol = solve_hinf_riccati(plant, schedule, gamma_mid)
        if sol.admissible:
            gamma_hi = gamma_mid
        else:
            gamma_lo = gamma_mid
        if (gamma_hi - gamma_lo) / max(gamma_hi, 1e-30) < tol:
            break

    return float(gamma_hi)


# =============================================================================
# Flavor-(b) model-class disturbance: ΔA · [pos, vel]
# Bug: 97c227a — Riccati flavor-(b) extension via S-procedure / quadratic
# stability reduction. See module docstring for the mathematical derivation.
# =============================================================================


def _make_pos_vel_selector(plant: PlantLinearization) -> Float[Array, "n_q n"]:
    """Build the ``C_q`` selector that extracts ``[pos, vel]`` from the augmented state.

    For both the 4-state and 6-state plants, the ``[pos, vel]`` substate is
    a contiguous prefix of the state vector. ``C_q`` is shape ``(n_q, n)``
    with ``n_q = 4`` (2 position + 2 velocity), zeroing out any force-state
    coupling.

    Args:
        plant: Linearised plant.

    Returns:
        ``C_q`` of shape ``(4, n)``.
    """
    pos_lo, pos_hi = plant.pos_slice
    vel_lo, vel_hi = plant.vel_slice
    n_q = (pos_hi - pos_lo) + (vel_hi - vel_lo)
    n = plant.n
    C_q = jnp.zeros((n_q, n), dtype=jnp.float64)
    # Position block
    n_pos = pos_hi - pos_lo
    C_q = C_q.at[:n_pos, pos_lo:pos_hi].set(jnp.eye(n_pos, dtype=jnp.float64))
    # Velocity block
    n_vel = vel_hi - vel_lo
    C_q = C_q.at[n_pos:n_pos + n_vel, vel_lo:vel_hi].set(jnp.eye(n_vel, dtype=jnp.float64))
    return C_q


def _augment_schedule_for_modelclass(
    schedule: CostSchedule,
    *,
    eta: float,
    mass: float,
    C_q: Float[Array, "n_q n"],
) -> CostSchedule:
    """Augment ``Q_t`` and ``Q_f`` with the S-procedure ΔA penalty.

    Adds :math:`(m\\eta)^2 \\, C_q^\\top C_q` to ``Q_t`` for every ``t`` and
    to ``Q_f``. This is the minimum-conservatism quadratic-stability lift:
    it converts the flavor-(b) structured-uncertainty problem into a
    flavor-(a) Riccati on the augmented cost.

    Args:
        schedule: Original cost schedule (flavor-(a) or unperturbed LQ).
        eta: Frobenius-norm budget on ``ΔA``. Must be ≥ 0.
        mass: Effector mass. The factor ``mass`` arises because the force
            applied by ``DynamicsMatrixPerturb`` is ``f = mass · ΔA · [p,v]``;
            so the equivalent additive-force disturbance is bounded by
            ``mass · eta · ‖[p,v]‖``.
        C_q: ``[pos, vel]`` selector, shape ``(n_q, n)``.

    Returns:
        A new ``CostSchedule`` with ``Q_t`` and ``Q_f`` augmented.
    """
    if eta < 0:
        raise ValueError(f"eta must be non-negative, got {eta}")
    Q_aug_term = (mass * eta) ** 2 * (C_q.T @ C_q)
    Q_new = schedule.Q + Q_aug_term[None]
    Q_f_new = schedule.Q_f + Q_aug_term
    return CostSchedule(Q=Q_new, R=schedule.R, Q_f=Q_f_new)


def solve_hinf_riccati_modelclass(
    plant: PlantLinearization,
    schedule: CostSchedule,
    gamma: float,
    *,
    eta: float,
    C_q: Optional[Float[Array, "n_q n"]] = None,
    mass: Optional[float] = None,
) -> RiccatiSolution:
    """Flavor-(b) H-infinity Riccati: model-class ΔA disturbance via S-procedure.

    Solves the H-infinity LQ-game Riccati on the augmented cost
    :math:`Q_t \\to Q_t + (m\\eta)^2 \\, C_q^\\top C_q` while keeping the
    same force-channel ``B_w``. The result is a sufficient condition for
    robust LQ performance under any structured perturbation
    :math:`\\Delta A` with :math:`\\|\\Delta A\\|_F \\le \\eta` applied
    uniformly along the rollout (matching ``LinearDynamicsAdversary``).

    Mathematical derivation (S-procedure / quadratic-stability):
    With ``f = mass · ΔA · [p,v]`` (feedbax ``DynamicsMatrixPerturb``), the
    closed-loop disturbance entering the force channel is
    :math:`w_t = m\\,\\Delta A\\,C_q x_t`. Cauchy-Schwarz:
    :math:`\\|w_t\\|_2 \\le m\\eta\\,\\|C_q x_t\\|_2`. A sufficient
    LMI/Riccati condition for the LQ-game value
    :math:`x^\\top P x` to bound the worst-case cost-plus-perturbation
    energy is to penalise :math:`(m\\eta)^2\\,\\|C_q x\\|^2` in :math:`Q`
    while keeping :math:`B_w` unchanged. The resulting :math:`\\gamma_*^{(b)}`
    is monotone in :math:`\\eta` and recovers :math:`\\gamma_*^{(a)}` at
    :math:`\\eta = 0`.

    Args:
        plant: Linearised plant.
        schedule: Cost schedule (flavor-(a) form). Will be augmented
            internally; the input is not modified.
        gamma: H-infinity disturbance attenuation level.
        eta: Frobenius-norm budget on the structured ``ΔA``. Must be ≥ 0.
            ``eta = 0`` recovers ``solve_hinf_riccati`` exactly.
        C_q: State-selector matrix that extracts ``[pos, vel]`` from the
            augmented state, shape ``(n_q, n)``. If ``None``, built from
            ``plant.pos_slice`` and ``plant.vel_slice``.
        mass: Effector mass. If ``None``, recovered from the plant's
            continuous-time control matrix ``B_c`` (the velocity row scales
            as ``1/mass``).

    Returns:
        A ``RiccatiSolution`` from the augmented Riccati. Per-step
        diagnostics (``spectral_radii``, ``bracket_conditions``) are with
        respect to the augmented Riccati, not the flavor-(a) one.

    Raises:
        ValueError: If ``eta`` is negative or ``gamma`` non-positive.
    """
    if eta < 0:
        raise ValueError(f"eta must be non-negative, got {eta}")
    if C_q is None:
        C_q = _make_pos_vel_selector(plant)
    if mass is None:
        mass = _infer_mass_from_plant(plant)

    schedule_aug = _augment_schedule_for_modelclass(
        schedule, eta=float(eta), mass=float(mass), C_q=C_q.astype(jnp.float64)
    )
    return solve_hinf_riccati(plant, schedule_aug, gamma)


def find_gamma_star_modelclass(
    plant: PlantLinearization,
    schedule: CostSchedule,
    *,
    eta: float,
    C_q: Optional[Float[Array, "n_q n"]] = None,
    mass: Optional[float] = None,
    gamma_lo: float = 1e-4,
    gamma_hi: float = 1.0,
    max_iter: int = 60,
    tol: float = 1e-4,
) -> float:
    """Bisect for the smallest admissible gamma under the flavor-(b) Riccati.

    Same bisection scheme as ``find_gamma_star``, but on the augmented
    Riccati. Monotone in ``eta``: ``find_gamma_star_modelclass(plant,
    schedule, eta=0) == find_gamma_star(plant, schedule)`` (modulo
    bisection precision), and ``find_gamma_star_modelclass`` increases as
    ``eta`` increases.

    Args:
        plant: Linearised plant.
        schedule: Cost schedule (flavor-(a) form).
        eta: Frobenius-norm budget on ``ΔA``.
        C_q: Optional state selector. See ``solve_hinf_riccati_modelclass``.
        mass: Optional mass override. See ``solve_hinf_riccati_modelclass``.
        gamma_lo: Initial lower bracket.
        gamma_hi: Initial upper bracket.
        max_iter: Maximum bisection iterations.
        tol: Convergence tolerance on relative gamma width.

    Returns:
        Estimated :math:`\\gamma_*^{(b)}(\\eta)`.
    """
    if C_q is None:
        C_q = _make_pos_vel_selector(plant)
    if mass is None:
        mass = _infer_mass_from_plant(plant)
    schedule_aug = _augment_schedule_for_modelclass(
        schedule, eta=float(eta), mass=float(mass), C_q=C_q.astype(jnp.float64)
    )
    return find_gamma_star(
        plant,
        schedule_aug,
        gamma_lo=gamma_lo,
        gamma_hi=gamma_hi,
        max_iter=max_iter,
        tol=tol,
    )


def _infer_mass_from_plant(plant: PlantLinearization) -> float:
    """Recover effector mass from the continuous-time control matrix.

    For the rlrmp plant, the velocity row of ``B_c`` is ``I/mass`` (4-state)
    or the force row of ``B_c`` is ``I/tau`` and the velocity-to-force
    coupling in ``A_c`` is ``I/mass`` (6-state). Either way, the disturbance
    matrix ``Bw_c`` velocity row is ``I/mass``, so we read it off there.

    Returns:
        ``mass`` as a Python float.
    """
    vel_lo, vel_hi = plant.vel_slice
    Bw_vel_block = plant.Bw_c[vel_lo:vel_hi, :]
    # Bw_c velocity block is I/mass; pick the (0,0) entry (assumes isotropic).
    diag_val = float(Bw_vel_block[0, 0])
    if diag_val <= 0:
        raise ValueError(
            "Could not infer mass from plant: Bw_c velocity row has non-positive "
            f"(0,0) entry {diag_val}. Pass mass explicitly."
        )
    return 1.0 / diag_val


# =============================================================================
# Closed-loop simulation
# =============================================================================


def simulate_closed_loop(
    plant: PlantLinearization,
    K: Float[Array, "T m_u n"],
    x0: Float[Array, "n"],
    *,
    target_pos: Optional[Float[Array, "2"]] = None,
    w: Optional[Float[Array, "T m_w"]] = None,
) -> ClosedLoopRollout:
    """Simulate the discrete-time linear closed loop with feedback ``K``.

    The dynamics are

    .. math::

        x_{t+1} = A\\, x_t + B\\, u_t + B_w\\, w_t,
        \\qquad u_t = -K_t\\, x_t.

    The controller drives the state to zero. To represent a reach to a
    nonzero target, pass ``x0`` in **goal-centred coordinates** (i.e.
    ``init_pos - target_pos``) -- the simulator returns ``x`` in those same
    goal-centred coordinates. ``target_pos`` is used only by the
    lateral-deviation diagnostic to project absolute reach trajectories
    onto the init-to-target line; ``terminal_position_error`` is the norm
    of the goal-centred terminal position (i.e., distance from target).

    Args:
        plant: Linearised plant (must have the same ``n``, ``m_u``, ``m_w``
            as the gain ``K``).
        K: Time-varying feedback gain, shape (T, m_u, n).
        x0: Initial state in goal-centred coordinates. Build via
            ``make_reach_initial_state``.
        target_pos: Position of the target in workspace coordinates. Used
            for lateral-deviation diagnostic. If provided, lateral deviation
            is computed by mapping the goal-centred trajectory back to
            absolute coordinates (adding ``target_pos``) and projecting
            onto the line from ``init_pos`` to ``target_pos``. If ``None``,
            lateral deviation is computed in goal-centred coordinates with
            target = origin.
        w: Disturbance trajectory, shape (T, m_w). Default zero.

    Returns:
        A ``ClosedLoopRollout``. ``x`` is in goal-centred coordinates;
        ``terminal_position_error`` is the norm of the terminal goal-centred
        position (i.e., distance to target).
    """
    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    Bw = plant.Bw.astype(jnp.float64)
    T = K.shape[0]
    n = A.shape[0]
    m_u = B.shape[1]
    m_w = Bw.shape[1]
    if w is None:
        w_seq = jnp.zeros((T, m_w), dtype=jnp.float64)
    else:
        w_seq = w.astype(jnp.float64)

    x_seq = [x0.astype(jnp.float64)]
    u_seq: list[jnp.ndarray] = []
    for t in range(T):
        x_t = x_seq[-1]
        u_t = -K[t] @ x_t
        x_next = A @ x_t + B @ u_t + Bw @ w_seq[t]
        u_seq.append(u_t)
        x_seq.append(x_next)

    x_arr = jnp.stack(x_seq, axis=0)
    u_arr = jnp.stack(u_seq, axis=0)

    pos_lo, pos_hi = plant.pos_slice
    vel_lo, vel_hi = plant.vel_slice
    # Goal-centred trajectories
    pos_gc = x_arr[:, pos_lo:pos_hi]
    vel_t = x_arr[:, vel_lo:vel_hi]
    speed = jnp.linalg.norm(vel_t, axis=-1)
    peak_velocity_idx = int(jnp.argmax(speed))
    peak_velocity = float(jnp.max(speed))

    # Lateral deviation: convert goal-centred position to absolute (if
    # target_pos provided), then project onto the init-to-target line.
    if target_pos is None:
        target_abs = jnp.zeros((pos_hi - pos_lo,), dtype=jnp.float64)
    else:
        target_abs = target_pos.astype(jnp.float64)
    pos_abs = pos_gc + target_abs[None, :]  # absolute positions
    init_pos_abs = pos_abs[0]  # initial absolute position
    line_vec = target_abs - init_pos_abs  # vector from init to target
    line_len = jnp.linalg.norm(line_vec)
    if float(line_len) < 1e-12:
        peak_lat_dev = 0.0
        peak_forward_velocity = 0.0
        peak_forward_velocity_idx = 0
        peak_lateral_velocity = 0.0
    else:
        line_dir = line_vec / line_len
        rel = pos_abs - init_pos_abs
        proj = (rel @ line_dir)[:, None] * line_dir[None, :]
        perp = rel - proj
        peak_lat_dev = float(jnp.max(jnp.linalg.norm(perp, axis=-1)))
        # Forward velocity: signed projection of velocity onto reach axis.
        # Positive = motion toward target.
        v_forward_t = vel_t @ line_dir  # shape (T+1,)
        peak_forward_velocity = float(jnp.max(v_forward_t))
        peak_forward_velocity_idx = int(jnp.argmax(v_forward_t))
        # Lateral velocity: speed in the direction orthogonal to reach axis.
        v_lateral_t = jnp.linalg.norm(
            vel_t - v_forward_t[:, None] * line_dir[None, :], axis=-1
        )
        peak_lateral_velocity = float(jnp.max(v_lateral_t))

    control_effort = float(jnp.sum(jnp.linalg.norm(u_arr, axis=-1) ** 2) * plant.dt)
    # Terminal position error: distance from terminal absolute position to
    # target. In goal-centred coordinates, the terminal state is
    # ``pos_gc[-1]`` and the target is the origin, so the error is just
    # ``||pos_gc[-1]||``.
    terminal_pos_err = float(jnp.linalg.norm(pos_gc[-1]))

    return ClosedLoopRollout(
        x=x_arr,
        u=u_arr,
        peak_velocity=peak_velocity,
        peak_velocity_idx=peak_velocity_idx,
        peak_forward_velocity=peak_forward_velocity,
        peak_forward_velocity_idx=peak_forward_velocity_idx,
        peak_lateral_velocity=peak_lateral_velocity,
        peak_lateral_deviation=peak_lat_dev,
        control_effort=control_effort,
        terminal_position_error=terminal_pos_err,
    )


def make_reach_initial_state(
    plant: PlantLinearization,
    *,
    init_pos: Float[Array, "2"],
    target_pos: Float[Array, "2"],
) -> Float[Array, "n"]:
    """Build the goal-centred initial state vector for a reach.

    The Riccati / LQR controller is designed to drive the state to zero.
    For a reach from ``init_pos`` to ``target_pos``, the goal-centred
    state at t=0 is ``[init_pos - target_pos, 0_vel, 0_force]``.

    Args:
        plant: The plant linearisation.
        init_pos: Initial position in workspace coordinates, shape (2,).
        target_pos: Target position in workspace coordinates, shape (2,).

    Returns:
        Initial goal-centred state vector, shape (n,).
    """
    x0 = jnp.zeros((plant.n,), dtype=jnp.float64)
    pos_lo, pos_hi = plant.pos_slice
    x0 = x0.at[pos_lo:pos_hi].set((init_pos - target_pos).astype(jnp.float64))
    return x0


# =============================================================================
# High-level convenience: velocity inflation comparison
# =============================================================================


@dataclass(frozen=True)
class VelocityInflationResult:
    """Result of an LQR-vs-H-infinity peak-velocity comparison.

    The headline metric ``delta_v_percent`` is based on **peak forward
    velocity** — the signed projection of velocity onto the reach axis
    (initial→target, positive = toward target). This matches the metric
    reported by Crevecoeur & Scott (2019) Fig. 1e. C&S's robust-controller
    signature is opposite-signed across channels: higher peak forward velocity
    but lower peak lateral velocity relative to LQR.

    Attributes:
        gamma_star: Estimated boundary gamma*.
        gamma_evaluated: The gamma used for the H-infinity comparison.
        lqr_peak_velocity: Peak speed (2-norm) under the LQR controller.
            Retained for backward-compatibility; see ``lqr_peak_forward_velocity``
            for the headline metric.
        hinf_peak_velocity: Peak speed (2-norm) under the H-infinity controller.
        delta_v_percent: Percentage change based on peak *forward* velocity:
            ``(v_fwd_hinf - v_fwd_lqr) / max(v_fwd_lqr, 1e-12) * 100``.
            Positive = H∞ reaches higher forward velocity than LQR.
        lqr_peak_forward_velocity: Peak forward velocity under LQR.
        hinf_peak_forward_velocity: Peak forward velocity under H-infinity.
        lqr_peak_lateral_velocity: Peak lateral velocity under LQR.
        hinf_peak_lateral_velocity: Peak lateral velocity under H-infinity.
        delta_v_lateral_percent: Percentage change in peak lateral velocity:
            ``(v_lat_hinf - v_lat_lqr) / max(v_lat_lqr, 1e-12) * 100``.
            Negative = H∞ suppresses lateral velocity relative to LQR (C&S
            signature).
        lqr_rollout: Full rollout under LQR.
        hinf_rollout: Full rollout under H-infinity controller.
        riccati: The H-infinity Riccati solution at ``gamma_evaluated``.
    """

    gamma_star: float
    gamma_evaluated: float
    lqr_peak_velocity: float
    hinf_peak_velocity: float
    delta_v_percent: float
    lqr_peak_forward_velocity: float
    hinf_peak_forward_velocity: float
    lqr_peak_lateral_velocity: float
    hinf_peak_lateral_velocity: float
    delta_v_lateral_percent: float
    lqr_rollout: ClosedLoopRollout
    hinf_rollout: ClosedLoopRollout
    riccati: RiccatiSolution


def compute_velocity_inflation(
    plant: PlantLinearization,
    schedule: CostSchedule,
    *,
    init_pos: Float[Array, "2"],
    target_pos: Float[Array, "2"],
    gamma_factor: float = 1.5,
    gamma_star: Optional[float] = None,
) -> VelocityInflationResult:
    """LQR vs H-infinity peak-forward-velocity comparison on a single reach.

    This is the headline sanity check from synthesis_review section 11 step 1.
    Returns peak-forward-velocity inflation Delta v % at the requested gamma.

    The metric ``delta_v_percent`` is based on **peak forward velocity** —
    the projection of velocity onto the reach axis (initial→target, positive
    direction = toward target). This matches Crevecoeur & Scott (2019) Fig.
    1e, which explicitly compares peak velocity *toward the target* rather
    than peak speed (2-norm).

    C&S's signature for robust vs LQG control: higher peak forward velocity
    (+) and lower peak lateral velocity (−). Peak speed mixes both channels
    and the cancellation can suppress the inflation signal.

    ``delta_v_lateral_percent`` captures the lateral channel for diagnostics.

    Args:
        plant: Linearised plant.
        schedule: Cost schedule.
        init_pos: Reach starting position, shape (2,).
        target_pos: Reach target position, shape (2,).
        gamma_factor: Multiplier on gamma_star at which to evaluate the
            H-infinity controller. ``1.5`` matches synthesis_review's
            headline number.
        gamma_star: If provided, skip bisection and use this value. Useful
            for batched comparisons across configs.

    Returns:
        A ``VelocityInflationResult``.
    """
    if gamma_star is None:
        gamma_star = find_gamma_star(plant, schedule)
    gamma_eval = gamma_factor * gamma_star

    lqr_sol = solve_lqr(plant, schedule)
    hinf_sol = solve_hinf_riccati(plant, schedule, gamma_eval)
    if not hinf_sol.admissible:
        raise RuntimeError(
            f"H-infinity Riccati at gamma={gamma_eval} (factor={gamma_factor}) "
            f"was inadmissible despite gamma_star={gamma_star}. "
            "This usually indicates the bisection landed too close to the boundary."
        )

    x0 = make_reach_initial_state(plant, init_pos=init_pos, target_pos=target_pos)

    lqr_rollout = simulate_closed_loop(plant, lqr_sol.K, x0, target_pos=target_pos)
    # Note: simulate_closed_loop expects goal-centred x0; the controller drives
    # state to zero, so target_pos is passed only for diagnostics.
    hinf_rollout = simulate_closed_loop(plant, hinf_sol.K, x0, target_pos=target_pos)

    # Headline metric: peak forward velocity (signed, positive = toward target).
    # Bug: f90bf74 — switched from peak speed to peak forward velocity to match C&S 2019 Fig. 1e.
    delta = (
        100.0
        * (hinf_rollout.peak_forward_velocity - lqr_rollout.peak_forward_velocity)
        / max(lqr_rollout.peak_forward_velocity, 1e-12)
    )
    delta_lateral = (
        100.0
        * (hinf_rollout.peak_lateral_velocity - lqr_rollout.peak_lateral_velocity)
        / max(lqr_rollout.peak_lateral_velocity, 1e-12)
    )

    return VelocityInflationResult(
        gamma_star=float(gamma_star),
        gamma_evaluated=float(gamma_eval),
        lqr_peak_velocity=lqr_rollout.peak_velocity,
        hinf_peak_velocity=hinf_rollout.peak_velocity,
        delta_v_percent=float(delta),
        lqr_peak_forward_velocity=lqr_rollout.peak_forward_velocity,
        hinf_peak_forward_velocity=hinf_rollout.peak_forward_velocity,
        lqr_peak_lateral_velocity=lqr_rollout.peak_lateral_velocity,
        hinf_peak_lateral_velocity=hinf_rollout.peak_lateral_velocity,
        delta_v_lateral_percent=float(delta_lateral),
        lqr_rollout=lqr_rollout,
        hinf_rollout=hinf_rollout,
        riccati=hinf_sol,
    )


def compute_velocity_inflation_modelclass(
    plant: PlantLinearization,
    schedule: CostSchedule,
    *,
    init_pos: Float[Array, "2"],
    target_pos: Float[Array, "2"],
    eta: float,
    gamma_factor: float = 1.5,
    gamma_star: Optional[float] = None,
    C_q: Optional[Float[Array, "n_q n"]] = None,
    mass: Optional[float] = None,
) -> VelocityInflationResult:
    """LQR vs flavor-(b) H-infinity peak-forward-velocity comparison.

    Same headline metric as ``compute_velocity_inflation`` but using the
    model-class Riccati: the H-infinity controller is designed against a
    structured ``ΔA`` perturbation with Frobenius bound ``eta``.

    Bug: 97c227a — the flavor-(b) extension that the original
    ``compute_velocity_inflation`` (flavor-(a)) does not implement.

    Args:
        plant: Linearised plant.
        schedule: Cost schedule (flavor-(a) form; will be augmented).
        init_pos: Reach starting position.
        target_pos: Reach target position.
        eta: Frobenius-norm budget on ``ΔA``.
        gamma_factor: Multiplier on :math:`\\gamma_*^{(b)}` for the design
            level. ``1.5`` matches synthesis_review section 2.
        gamma_star: If provided, skip bisection and use this value as
            :math:`\\gamma_*^{(b)}`.
        C_q: Optional state selector.
        mass: Optional mass override.

    Returns:
        A ``VelocityInflationResult`` whose ``riccati`` field carries the
        flavor-(b) (augmented) Riccati solution at
        :math:`\\gamma_{factor} \\cdot \\gamma_*^{(b)}`.
    """
    if C_q is None:
        C_q = _make_pos_vel_selector(plant)
    if mass is None:
        mass = _infer_mass_from_plant(plant)

    if gamma_star is None:
        gamma_star = find_gamma_star_modelclass(
            plant, schedule, eta=eta, C_q=C_q, mass=mass
        )
    gamma_eval = gamma_factor * gamma_star

    lqr_sol = solve_lqr(plant, schedule)
    hinf_sol = solve_hinf_riccati_modelclass(
        plant, schedule, gamma_eval, eta=eta, C_q=C_q, mass=mass
    )
    if not hinf_sol.admissible:
        raise RuntimeError(
            f"Flavor-(b) H-infinity Riccati at gamma={gamma_eval} "
            f"(factor={gamma_factor}, eta={eta}) was inadmissible despite "
            f"gamma_star={gamma_star}. Check bisection precision."
        )

    x0 = make_reach_initial_state(plant, init_pos=init_pos, target_pos=target_pos)
    lqr_rollout = simulate_closed_loop(plant, lqr_sol.K, x0, target_pos=target_pos)
    hinf_rollout = simulate_closed_loop(plant, hinf_sol.K, x0, target_pos=target_pos)

    delta = (
        100.0
        * (hinf_rollout.peak_forward_velocity - lqr_rollout.peak_forward_velocity)
        / max(lqr_rollout.peak_forward_velocity, 1e-12)
    )
    delta_lateral = (
        100.0
        * (hinf_rollout.peak_lateral_velocity - lqr_rollout.peak_lateral_velocity)
        / max(lqr_rollout.peak_lateral_velocity, 1e-12)
    )

    return VelocityInflationResult(
        gamma_star=float(gamma_star),
        gamma_evaluated=float(gamma_eval),
        lqr_peak_velocity=lqr_rollout.peak_velocity,
        hinf_peak_velocity=hinf_rollout.peak_velocity,
        delta_v_percent=float(delta),
        lqr_peak_forward_velocity=lqr_rollout.peak_forward_velocity,
        hinf_peak_forward_velocity=hinf_rollout.peak_forward_velocity,
        lqr_peak_lateral_velocity=lqr_rollout.peak_lateral_velocity,
        hinf_peak_lateral_velocity=hinf_rollout.peak_lateral_velocity,
        delta_v_lateral_percent=float(delta_lateral),
        lqr_rollout=lqr_rollout,
        hinf_rollout=hinf_rollout,
        riccati=hinf_sol,
    )


# =============================================================================
# Linearisation fidelity check
# =============================================================================


def linearization_fidelity(
    plant: PlantLinearization,
    *,
    nonlinear_step_fn: Callable[[Float[Array, "n"], Float[Array, "m_u"]], Float[Array, "n"]],
    x0: Float[Array, "n"],
    u_seq: Float[Array, "T m_u"],
) -> dict:
    """Compare the linearised forward simulation to a nonlinear step function.

    Useful for verifying that the rlrmp plant truly is LTI (which it is by
    construction; this guards against parameter mismatches between
    ``linearize_pointmass`` and the model). Returns the per-timestep state
    error and a summary scalar.

    Args:
        plant: Linearised plant.
        nonlinear_step_fn: A function ``(x, u) -> x_next`` corresponding to
            one step of the nonlinear plant evolution. For an LTI rlrmp plant
            this should equal ``A x + B u`` exactly.
        x0: Initial state.
        u_seq: Control trajectory, shape (T, m_u).

    Returns:
        A dict with keys ``"x_lin"``, ``"x_nl"``, ``"per_step_err"``,
        ``"max_err"``, ``"final_err"``.
    """
    A = plant.A.astype(jnp.float64)
    B = plant.B.astype(jnp.float64)
    T = u_seq.shape[0]

    x_lin = [x0.astype(jnp.float64)]
    x_nl = [x0.astype(jnp.float64)]
    for t in range(T):
        x_lin.append(A @ x_lin[-1] + B @ u_seq[t])
        x_nl.append(nonlinear_step_fn(x_nl[-1], u_seq[t]))

    x_lin_arr = jnp.stack(x_lin, axis=0)
    x_nl_arr = jnp.stack(x_nl, axis=0)
    per_step_err = jnp.linalg.norm(x_lin_arr - x_nl_arr, axis=-1)
    return {
        "x_lin": x_lin_arr,
        "x_nl": x_nl_arr,
        "per_step_err": per_step_err,
        "max_err": float(jnp.max(per_step_err)),
        "final_err": float(per_step_err[-1]),
    }


__all__ = [
    "PlantLinearization",
    "Discretization",
    "CostSchedule",
    "CostSpec",
    "RiccatiSolution",
    "ClosedLoopRollout",
    "VelocityInflationResult",
    "linearize_pointmass",
    "cs_faithful_pointmass",
    "linearize_from_model",
    "cost_schedule_from_spec",
    "cs_eq15_cost_schedule",
    "apply_delay_distribution_to_schedule",
    "cost_schedule_from_loss_config",
    "solve_hinf_riccati",
    "solve_hinf_riccati_modelclass",
    "solve_lqr",
    "find_gamma_star",
    "find_gamma_star_modelclass",
    "simulate_closed_loop",
    "make_reach_initial_state",
    "compute_velocity_inflation",
    "compute_velocity_inflation_modelclass",
    "linearization_fidelity",
]
