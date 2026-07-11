from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from feedbax.models.networks import MaskedLinear

from rlrmp.eval.legacy_checkpoints import (
    force_legacy_masked_readout,
    restore_legacy_static_numeric_metadata,
)


class _Leaf(eqx.Module):
    std: Any = None
    mean: Any = None
    noise_func: Any = None


class _Noise(eqx.Module):
    terms: tuple[Any, ...]


class _Port(eqx.Module):
    delay: Any
    noise_func: Any
    add_noise: Any
    init_value: Any


class _Population(eqx.Module):
    n_input_only: Any
    n_readout_only: Any
    n_recurrent_only: Any
    n_input_readout: Any


class _Network(eqx.Module):
    readout: Any
    input_size: Any
    hidden_size: Any
    out_size: Any
    population_structure: _Population
    dtype: Any = eqx.field(static=True)


class _NetworkNode(eqx.Module):
    net: _Network


class _Mechanics(eqx.Module):
    dt: Any


class _Model(eqx.Module):
    nodes: dict[str, Any]


def _replicated(value: Any) -> jnp.ndarray:
    return jnp.asarray([value, value])


def _model() -> _Model:
    readout = eqx.nn.Linear(3, 2, key=jr.PRNGKey(0))
    population = _Population(*(_replicated(value) for value in (1, 2, 3, 4)))
    net = _Network(
        readout,
        _replicated(3),
        _replicated(4),
        _replicated(2),
        population,
        jnp.float32,
    )
    efferent_noise = _Noise(
        (
            _Leaf(noise_func=_Leaf(std=_replicated(0.1), mean=_replicated(0.2))),
            _Leaf(std=_replicated(0.3), mean=_replicated(0.4)),
        )
    )
    sensory_noise = _Leaf(std=_replicated(0.5), mean=_replicated(0.6))
    return _Model(
        {
            "efferent": _Port(
                _replicated(1), efferent_noise, _replicated(True), _replicated(0.0)
            ),
            "mechanics": _Mechanics(_replicated(0.01)),
            "net": _NetworkNode(net),
            "sensory": _Port(
                _replicated(2), sensory_noise, _replicated(False), _replicated(0.0)
            ),
        }
    )


def test_force_legacy_masked_readout_preserves_linear_parameters() -> None:
    model = _model()

    restored = force_legacy_masked_readout(model)

    assert isinstance(restored.nodes["net"].net.readout, MaskedLinear)
    assert jnp.array_equal(
        restored.nodes["net"].net.readout.linear.weight,
        model.nodes["net"].net.readout.weight,
    )
    assert restored.nodes["net"].net.dtype is jnp.float64


def test_restore_legacy_static_numeric_metadata_returns_scalars() -> None:
    restored = restore_legacy_static_numeric_metadata(_model())

    assert restored.nodes["efferent"].delay == 1
    assert isinstance(restored.nodes["efferent"].delay, int)
    assert restored.nodes["efferent"].noise_func.terms[0].noise_func.std == pytest.approx(0.1)
    assert isinstance(restored.nodes["efferent"].add_noise, bool)
    assert restored.nodes["mechanics"].dt == pytest.approx(0.01)
    assert restored.nodes["net"].net.input_size == 3
    assert restored.nodes["net"].net.population_structure.n_input_readout == 4
    assert restored.nodes["sensory"].delay == 2
    assert isinstance(restored.nodes["sensory"].add_noise, bool)
