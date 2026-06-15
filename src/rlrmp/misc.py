import jax.numpy as jnp
import jax.random as jr
from jax_cookbook.misc import split_by
from jaxtyping import Array

from rlrmp.constants import RNN_INPUT_CHANNEL_SIZES
from rlrmp.types import RNNInputChannels


def split_rnn_input_channels(arr: Array, *, axis=-1):
    return RNNInputChannels(*split_by(arr, RNN_INPUT_CHANNEL_SIZES, axis=axis))


def get_field_amplitude(intervenor_params):
    from feedbax.intervene import CurlFieldParams, FixedFieldParams

    if isinstance(intervenor_params, FixedFieldParams):
        return jnp.linalg.norm(intervenor_params.field, axis=-1)
    if isinstance(intervenor_params, CurlFieldParams):
        return jnp.abs(intervenor_params.amplitude)
    raise ValueError(f"Unknown intervenor parameters type: {type(intervenor_params)}")


def vector_with_gaussian_length(key, shape=()):
    key_angle, key_length = jr.split(key)
    angle = jr.uniform(key_angle, shape, minval=-jnp.pi, maxval=jnp.pi)
    length = jr.normal(key_length, shape)
    vector = length * jnp.array([jnp.cos(angle), jnp.sin(angle)])
    return vector.T


def unit_circle_points(n):
    """Generate N evenly spaced points on a unit circle."""
    angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
    z = jnp.exp(1j * angles)
    return jnp.column_stack([z.real, z.imag])
