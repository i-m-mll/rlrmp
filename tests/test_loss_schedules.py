from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from rlrmp.loss import (
    CsAnalyticalQrfLoss,
    make_cs_eq15_stage_schedule,
    make_epoch_locked_ramp,
    make_movement_cs_eq15_stage_schedule,
)


def _trial_spec(*, n_steps: int = 140, epoch_bounds=(0, 0, 11, 139)):
    return SimpleNamespace(
        timeline=SimpleNamespace(
            n_steps=n_steps,
            epoch_bounds=jnp.asarray(epoch_bounds),
        )
    )


def test_epoch_locked_ramp_is_zero_before_movement_and_fixed_duration():
    weights = make_epoch_locked_ramp(duration_steps=60, start_epoch=-2)(_trial_spec())

    assert weights.shape == (139,)
    assert jnp.all(weights[:11] == 0.0)
    assert weights[11] == 0.0
    assert weights[41] == pytest.approx(0.5)
    assert weights[71] == pytest.approx(1.0)
    assert jnp.all(weights[71:] == 1.0)


@pytest.mark.parametrize(
    ("shape", "power", "expected_mid"),
    [
        ("linear", 2.0, 0.5),
        ("cosine", 2.0, 0.5),
        ("power", 2.0, 0.25),
        ("power", 4.0, 0.0625),
        ("power", 6.0, 0.015625),
    ],
)
def test_epoch_locked_ramp_shapes(shape, power, expected_mid):
    weights = make_epoch_locked_ramp(
        duration_steps=60,
        start_epoch=-2,
        shape=shape,
        power=power,
    )(_trial_spec())

    assert weights[41] == pytest.approx(expected_mid)
    assert weights[71] == pytest.approx(1.0)


def test_epoch_locked_ramp_handles_batched_epoch_bounds():
    weights = make_epoch_locked_ramp(duration_steps=60, start_epoch=-2)(
        _trial_spec(epoch_bounds=((0, 0, 11, 139), (0, 0, 21, 139)))
    )

    assert weights.shape == (2, 139)
    assert jnp.all(weights[0, :11] == 0.0)
    assert jnp.all(weights[1, :21] == 0.0)
    assert weights[0, 71] == pytest.approx(1.0)
    assert weights[1, 81] == pytest.approx(1.0)


def test_cs_eq15_stage_schedule_uses_one_indexed_sixty_stage_contract():
    weights = make_cs_eq15_stage_schedule(n_steps=61, power=6.0)

    assert weights.shape == (60,)
    assert weights[0] == pytest.approx((1.0 / 60.0) ** 6)
    assert weights[-1] == pytest.approx(1.0)


def test_movement_cs_eq15_stage_schedule_handles_batched_go_cues():
    weights = make_movement_cs_eq15_stage_schedule(horizon_steps=60)(
        _trial_spec(n_steps=90, epoch_bounds=((0, 10, 90), (0, 30, 90)))
    )

    assert weights.shape == (2, 90)
    assert jnp.all(weights[0, :10] == 0.0)
    assert jnp.all(weights[1, :30] == 0.0)
    assert weights[0, 10] == pytest.approx((1.0 / 60.0) ** 6)
    assert weights[1, 30] == pytest.approx((1.0 / 60.0) ** 6)
    assert weights[0, 69] == pytest.approx(1.0)
    assert weights[1, 89] == pytest.approx(1.0)


def _qrf_trial_spec(*, go_step: int = 10, n_steps: int = 90):
    return SimpleNamespace(
        inits={"mechanics.vector": jnp.zeros((8,), dtype=jnp.float32)},
        targets={},
        timeline=SimpleNamespace(
            n_steps=n_steps,
            epoch_bounds=jnp.asarray((0, go_step, n_steps)),
        ),
    )


def _qrf_states(*, n_steps: int = 90):
    return SimpleNamespace(
        mechanics=SimpleNamespace(vector=jnp.ones((n_steps, 8), dtype=jnp.float32)),
        net=SimpleNamespace(output=jnp.zeros((n_steps, 2), dtype=jnp.float32)),
    )


def test_delayed_full_qrf_default_stops_after_canonical_window():
    q = jnp.broadcast_to(
        jnp.diag(jnp.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=jnp.float32)), (60, 8, 8)
    )
    r = jnp.zeros((60, 2, 2), dtype=jnp.float32)
    q_f = q[-1]
    term = CsAnalyticalQrfLoss(Q=q, R=r, Q_f=q_f, target_pos=jnp.zeros((2,)))

    value = term.term(_qrf_states(), _qrf_trial_spec(), None)

    assert value == pytest.approx(60 * 4 + 4)


def test_delayed_full_qrf_flat_tail_keeps_terminal_running_weights_active():
    q = jnp.broadcast_to(
        jnp.diag(jnp.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=jnp.float32)), (60, 8, 8)
    )
    r = jnp.zeros((60, 2, 2), dtype=jnp.float32)
    q_f = q[-1]
    term = CsAnalyticalQrfLoss(
        Q=q,
        R=r,
        Q_f=q_f,
        target_pos=jnp.zeros((2,)),
        delayed_movement_cost_tail_mode="flat_after_canonical_horizon",
    )

    value = term.term(_qrf_states(), _qrf_trial_spec(), None)

    assert value == pytest.approx(80 * 4 + 4)
