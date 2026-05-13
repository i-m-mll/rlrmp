from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from rlrmp.loss import make_epoch_locked_ramp


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
