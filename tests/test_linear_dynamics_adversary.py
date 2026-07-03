"""Tests for ``LinearDynamicsAdversary`` (Bug: c723082).

Verifies the Frobenius-ball projection, gradient-ascent on a tiny synthetic
problem, and integration with the feedbax ``DynamicsMatrixPerturb``
intervenor (force-channel embedding of ``ΔA · x``).
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest
from feedbax.intervene import DynamicsMatrixPerturb

from rlrmp.adversary import LinearDynamicsAdversary, _frobenius_project
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.intervention_compat import (
    require_exactly_one_intervenor_for_dynamics_matrix_swap,
    swap_plant_intervenor_to_dynamics_matrix,
    swap_task_intervention_to_dynamics_matrix,
)
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.minimax import build_hps


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
        args = argparse.Namespace(
            n_warmup_batches=1,
            n_adversary_batches=1,
            controller_lr=1e-4,
            loss_update_enabled=False,
            loss_update_ratio=0.5,
            hidden_type="gru",
            sisu_gating="additive",
            n_replicates=1,
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
        # Use the state's default dtype so x64-mode (enabled elsewhere in the
        # suite, e.g. by test_hinf_riccati) doesn't cause a dtype mismatch
        # against the StateIndex's stored params.
        comp = DynamicsMatrixPerturb(mass=1.0)
        default_dtype = comp._initial_state.delta_A.dtype
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

    def test_intervenor_swap_requires_existing_matching_node(self):
        pair = self._single_replicate_pair()
        nodes = dict(pair.model.nodes)
        nodes.pop(PLANT_INTERVENOR_LABEL)
        model = eqx.tree_at(lambda g: g.nodes, pair.model, nodes)

        with pytest.raises(ValueError, match="exactly one intervenor node.*found 0"):
            require_exactly_one_intervenor_for_dynamics_matrix_swap(
                model,
                PLANT_INTERVENOR_LABEL,
            )

    def test_intervenor_swap_accepts_exactly_one_unswapped_node(self):
        pair = self._single_replicate_pair()

        swapped = swap_plant_intervenor_to_dynamics_matrix(
            pair.model,
            PLANT_INTERVENOR_LABEL,
        )

        assert isinstance(swapped.nodes[PLANT_INTERVENOR_LABEL], DynamicsMatrixPerturb)

    def test_intervenor_swap_rejects_duplicate_matching_nodes(self):
        pair = self._single_replicate_pair()
        nodes = dict(pair.model.nodes)
        nodes["duplicate_intervenor"] = nodes[PLANT_INTERVENOR_LABEL]
        model = eqx.tree_at(lambda g: g.nodes, pair.model, nodes)

        with pytest.raises(ValueError, match="exactly one intervenor node.*found 2"):
            require_exactly_one_intervenor_for_dynamics_matrix_swap(
                model,
                PLANT_INTERVENOR_LABEL,
            )

    def test_intervenor_swap_rejects_double_application(self):
        pair = self._single_replicate_pair()
        swapped = swap_plant_intervenor_to_dynamics_matrix(
            pair.model,
            PLANT_INTERVENOR_LABEL,
        )

        with pytest.raises(ValueError, match="applied twice"):
            swap_plant_intervenor_to_dynamics_matrix(
                swapped,
                PLANT_INTERVENOR_LABEL,
            )

    def test_task_swap_preserves_callable_pai_asf_schedules(self):
        """Linear-dynamics setup should keep PAI-ASF schedules trial-local."""
        pair = self._single_replicate_pair()
        task = pair.task
        model = swap_plant_intervenor_to_dynamics_matrix(pair.model, PLANT_INTERVENOR_LABEL)

        swapped = swap_task_intervention_to_dynamics_matrix(task, PLANT_INTERVENOR_LABEL)
        params = swapped.intervention_specs.training[PLANT_INTERVENOR_LABEL].params

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
        assert callable(params.scale)
        assert callable(params.active)
        trial = swapped.get_train_trial_with_intervenor_params(jr.PRNGKey(0))
        trial_params = trial.intervene[PLANT_INTERVENOR_LABEL]
        assert trial_params.delta_A.shape == (2, 4)
        assert trial_params.scale.shape == ()
        assert trial_params.active.shape == ()
