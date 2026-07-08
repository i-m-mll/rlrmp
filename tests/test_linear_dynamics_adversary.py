"""Tests for ``LinearDynamicsAdversary`` (Bug: c723082).

Verifies the Frobenius-ball projection, gradient-ascent on a tiny synthetic
problem, and integration with the feedbax ``DynamicsMatrixPerturb``
intervenor (force-channel embedding of ``ΔA · x``).
"""

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from feedbax.intervene import DynamicsMatrixPerturb, DynamicsMatrixPerturbParams
from feedbax.runtime.graph import Component, Graph, Wire, init_state_from_component
from feedbax.runtime.iteration import run_component
from feedbax.runtime.state import CartesianState

from rlrmp.train.adversary import LinearDynamicsAdversary, _frobenius_project
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.intervention_compat import LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET
from rlrmp.train.minimax import build_hps
from rlrmp.train.minimax_native import _inject_adversary_delta_A
from rlrmp.train.task_model import setup_task_model_pair


class TestFrobeniusProjection:
    def test_inside_ball_unchanged(self):
        M = jnp.array([[0.1, 0.0], [0.0, 0.1]])
        radius = jnp.asarray(1.0)
        out = _frobenius_project(M, radius)
        assert jnp.allclose(out, M)

    def test_outside_ball_scaled_to_radius(self):
        M = jnp.array([[2.0, 0.0], [0.0, 2.0]])  # ||M||_F = 2*sqrt(2) ~ 2.83
        radius = jnp.asarray(1.0)
        out = _frobenius_project(M, radius)
        assert jnp.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)

    def test_zero_matrix_handled(self):
        M = jnp.zeros((2, 4))
        out = _frobenius_project(M, jnp.asarray(0.5))
        assert jnp.allclose(out, M)


class TestLinearDynamicsAdversary:
    def test_init_shape(self):
        adv = LinearDynamicsAdversary(
            n_state=4, n_dim=2, eta_max=0.1, key=jr.PRNGKey(0),
        )
        assert adv.delta_A.shape == (2, 4)
        assert adv.eta_max == 0.1
        assert adv.n_inner_steps == 5

    def test_project_inside_ball_idempotent(self):
        adv = LinearDynamicsAdversary(
            n_state=4, n_dim=2, eta_max=10.0, key=jr.PRNGKey(0),
        )
        # Initial delta_A is tiny (~1e-3); eta_max=10 ⇒ projection is no-op.
        adv_p = adv.project()
        assert jnp.allclose(adv_p.delta_A, adv.delta_A)

    def test_project_clamps_to_eta_max(self):
        adv = LinearDynamicsAdversary(
            n_state=4, n_dim=2, eta_max=0.05, key=jr.PRNGKey(0),
        )
        # Manually set a large delta_A
        big = jnp.ones((2, 4))  # norm = sqrt(8) ~ 2.83
        adv = eqx.tree_at(lambda a: a.delta_A, adv, big)
        adv_p = adv.project()
        assert float(adv_p.frobenius_norm()) == pytest.approx(0.05, abs=1e-6)

    def test_project_preserves_direction(self):
        adv = LinearDynamicsAdversary(
            n_state=4, n_dim=2, eta_max=0.1, key=jr.PRNGKey(0),
        )
        big = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        adv = eqx.tree_at(lambda a: a.delta_A, adv, big)
        adv_p = adv.project()
        # Direction preserved: same-sign first entry, others zero
        assert adv_p.delta_A[0, 0] > 0
        assert jnp.allclose(adv_p.delta_A[1], 0.0)
        assert jnp.allclose(adv_p.delta_A[0, 1:], 0.0)

    def test_call_returns_delta_A(self):
        adv = LinearDynamicsAdversary(
            n_state=4, n_dim=2, eta_max=0.1, key=jr.PRNGKey(0),
        )
        out = adv()
        assert out.shape == (2, 4)
        assert jnp.allclose(out, adv.delta_A)

    def test_pgd_increases_quadratic_loss(self):
        """Ascending a positive-semi-definite quadratic should grow loss
        toward the Frobenius-ball boundary."""
        adv = LinearDynamicsAdversary(
            n_state=4, n_dim=2, eta_max=0.5, learning_rate=0.05,
            n_inner_steps=20, key=jr.PRNGKey(1),
        )
        optimizer = optax.adam(adv.learning_rate)
        opt_state = optimizer.init(eqx.filter(adv, eqx.is_array))

        def loss_fn(a):
            # ||delta_A||^2 — a "trivial" quadratic the adversary maximises.
            return jnp.sum(a.delta_A ** 2)

        loss_history = [float(loss_fn(adv))]
        for _ in range(adv.n_inner_steps):
            grads = eqx.filter_grad(loss_fn)(adv)
            # Negate for ascent
            neg = jax.tree.map(lambda g: -g, grads)
            updates, opt_state = optimizer.update(
                eqx.filter(neg, eqx.is_array),
                opt_state,
                eqx.filter(adv, eqx.is_array),
            )
            adv = eqx.apply_updates(adv, updates)
            adv = adv.project()
            loss_history.append(float(loss_fn(adv)))

        # Loss should strictly increase, and final norm should be at the
        # boundary (eta_max=0.5 ⇒ loss = 0.25)
        assert loss_history[-1] > loss_history[0]
        assert float(adv.frobenius_norm()) == pytest.approx(0.5, abs=1e-3)

    def test_grad_through_call(self):
        """Verify autodiff works through ``__call__`` (smoke test)."""
        adv = LinearDynamicsAdversary(
            n_state=4, n_dim=2, eta_max=0.1, key=jr.PRNGKey(0),
        )
        x = jnp.array([1.0, 0.0, 0.5, -0.5])

        def loss(a):
            return jnp.sum((a() @ x) ** 2)

        g = eqx.filter_grad(loss)(adv)
        assert g.delta_A.shape == (2, 4)
        # Gradient should be non-trivial (delta_A is initialised nonzero)
        assert float(jnp.linalg.norm(g.delta_A)) > 0.0


class TestDynamicsMatrixPerturbIntegration:
    """Smoke tests verifying the rlrmp-side adversary connects to the
    feedbax-side ``DynamicsMatrixPerturb`` intervenor."""

    @staticmethod
    def _single_replicate_pair():
        args = SimpleNamespace(
            n_warmup_batches=1,
            n_adversary_batches=1,
            controller_lr=1e-4,
            loss_update_enabled=False,
            loss_update_ratio=0.5,
            hidden_type="gru",
            sisu_gating="additive",
            n_replicates=1,
            adversary_type="linear_dynamics",
        )
        return setup_task_model_pair(build_hps(args), key=jr.PRNGKey(0))

    def test_feedbax_intervenor_consumes_delta_A(self):
        from feedbax.intervene import (
            DynamicsMatrixPerturbParams,
        )
        from feedbax.runtime.state import CartesianState

        adv = LinearDynamicsAdversary(
            n_state=4, n_dim=2, eta_max=0.5, key=jr.PRNGKey(2),
        )
        # Set delta_A to a known matrix so the analytical answer is trivial.
        # Keep the component state and adversary parameter dtype explicit so
        # this test does not depend on suite-level JAX x64 state.
        comp = DynamicsMatrixPerturb(mass=1.0)
        default_dtype = jnp.dtype(jnp.float32)
        delta_A = jnp.array(
            [[0.0, 0.0, 0.3, 0.0], [0.0, 0.0, 0.0, 0.3]], dtype=default_dtype,
        )
        adv = eqx.tree_at(lambda a: a.delta_A, adv, delta_A)

        from equinox.nn import State
        state = State(comp).set(
            comp.params_index,
            DynamicsMatrixPerturbParams(active=True, delta_A=adv()),
        )
        eff = CartesianState(
            pos=jnp.zeros(2, dtype=default_dtype),
            vel=jnp.array([1.0, -2.0], dtype=default_dtype),
        )
        out, _ = comp(
            {"effector": eff, "force": jnp.zeros(2, dtype=default_dtype)},
            state,
            key=jr.PRNGKey(0),
        )
        # delta_A on velocity rows × vel = [0.3 * 1, 0.3 * -2] = [0.3, -0.6]
        # f = mass * Δ(dot v) = 1.0 * [0.3, -0.6]
        expected = jnp.array([0.3, -0.6], dtype=default_dtype)
        assert jnp.allclose(out["force"], expected)

    def test_linear_dynamics_setup_authors_dynamics_matrix_node(self):
        pair = self._single_replicate_pair()
        model = pair.model

        recurrent_wires = [
            wire
            for wire in model.wires
            if wire.source_node == "mechanics"
            and wire.source_port == "effector"
            and wire.target_node == PLANT_INTERVENOR_LABEL
            and wire.target_port == "effector"
        ]
        assert len(recurrent_wires) == 1
        assert recurrent_wires[0].temporality == "recurrent"
        assert model._needs_iteration
        assert isinstance(model.nodes[PLANT_INTERVENOR_LABEL], DynamicsMatrixPerturb)

    def test_delta_a_injection_activates_params_override_path(self):
        pair = self._single_replicate_pair()
        trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(0))
        trial = jax.tree.map(lambda x: x[None, ...] if eqx.is_array(x) else x, trial)
        delta_A = jnp.ones((2, 4), dtype=jnp.float32) * 0.01
        injected = _inject_adversary_delta_A(trial, delta_A, batch_size=1)

        trial_params = injected.intervene[PLANT_INTERVENOR_LABEL]
        assert trial_params.active.value.shape == (1, 139)
        assert trial_params.delta_A.value.shape == (1, 139, 2, 4)
        assert bool(trial_params.active.value[0, 0])
        assert jnp.allclose(trial_params.delta_A.value[0, 0], delta_A)

    def test_task_authors_inactive_dynamics_matrix_trial_params(self):
        pair = self._single_replicate_pair()
        trial = pair.task.get_train_trial_with_intervenor_params(jr.PRNGKey(0))
        trial_params = trial.intervene[PLANT_INTERVENOR_LABEL]
        assert trial_params.delta_A.shape == (2, 4)
        assert trial_params.scale.shape == ()
        assert trial_params.active.shape == ()
        assert not bool(trial_params.active)

    def test_warmup_active_zero_delta_a_is_controller_level_no_op(self):
        graph = _controller_dynamics_graph()
        inactive_loss, inactive_outputs, inactive_grads, inactive_updates = (
            _controller_graph_loss_grads_updates(graph, active=False)
        )
        active_loss, active_outputs, active_grads, active_updates = (
            _controller_graph_loss_grads_updates(graph, active=True)
        )
        assert jnp.allclose(inactive_loss, active_loss)
        _assert_array_trees_allclose(inactive_outputs, active_outputs)
        _assert_array_trees_allclose(inactive_grads, active_grads)
        _assert_array_trees_allclose(inactive_updates, active_updates)

        adversary = LinearDynamicsAdversary(
            n_state=4,
            n_dim=2,
            eta_max=0.1,
            key=jr.PRNGKey(3),
        )
        delta_before = adversary.delta_A
        _ = adversary()
        assert jnp.allclose(adversary.delta_A, delta_before)


class _LinearForceController(Component):
    input_ports = ("command",)
    output_ports = ("force",)

    gain: jnp.ndarray

    def __init__(self):
        self.gain = jnp.asarray([[0.5, -0.2], [0.1, 0.3]], dtype=jnp.float32)

    def __call__(self, inputs, state, *, key):
        return {"force": self.gain @ inputs["command"]}, state


def _controller_dynamics_graph() -> Graph:
    target = LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET
    params_key = (
        f"task:{target['source_data_id']}->"
        f"{target['target_node_id']}.{target['target_port']}"
    )
    return Graph(
        nodes={
            "controller": _LinearForceController(),
            PLANT_INTERVENOR_LABEL: DynamicsMatrixPerturb(mass=1.0),
        },
        wires=(
            Wire("controller", "force", PLANT_INTERVENOR_LABEL, "force"),
        ),
        input_ports=("command", "effector", params_key),
        output_ports=("force",),
        input_bindings={
            "command": ("controller", "command"),
            "effector": (PLANT_INTERVENOR_LABEL, "effector"),
            params_key: (PLANT_INTERVENOR_LABEL, "params_override"),
        },
        output_bindings={"force": (PLANT_INTERVENOR_LABEL, "force")},
    )


def _controller_graph_loss_grads_updates(graph: Graph, *, active: bool):
    target = LINEAR_DYNAMICS_ADVERSARY_COMPONENT_PARAMETER_TARGET
    params_key = (
        f"task:{target['source_data_id']}->"
        f"{target['target_node_id']}.{target['target_port']}"
    )
    n_steps = 5
    params = DynamicsMatrixPerturbParams(
        scale=jnp.ones((n_steps,), dtype=jnp.float32),
        active=jnp.full((n_steps,), active),
        delta_A=jnp.zeros((n_steps, 2, 4), dtype=jnp.float32),
    )
    inputs = {
        "command": jnp.stack(
            [
                jnp.linspace(-0.2, 0.2, n_steps),
                jnp.linspace(0.3, -0.1, n_steps),
            ],
            axis=-1,
        ),
        "effector": CartesianState(
            pos=jnp.zeros((n_steps, 2), dtype=jnp.float32),
            vel=jnp.ones((n_steps, 2), dtype=jnp.float32),
            force=jnp.zeros((n_steps, 2), dtype=jnp.float32),
        ),
        params_key: params,
    }

    def loss_fn(model):
        outputs, _final_state, _state_history = run_component(
            model,
            inputs,
            init_state_from_component(model),
            key=jr.PRNGKey(4),
            n_steps=n_steps,
        )
        return jnp.sum(outputs["force"] ** 2), outputs

    (loss, outputs), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(graph)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(graph, eqx.is_array))
    updates, _ = optimizer.update(grads, opt_state, graph)
    return loss, outputs, grads, updates


def _assert_array_trees_allclose(left, right) -> None:
    compared = 0
    for l_leaf, r_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if eqx.is_array(l_leaf):
            compared += 1
            assert jnp.allclose(l_leaf, r_leaf)
    assert compared > 0
