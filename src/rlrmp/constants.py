import jax.numpy as jnp

from rlrmp.types import RNNInputChannels

RNN_INPUT_CHANNEL_SIZES = RNNInputChannels(1, 2, 2, 2, 2)
RNN_INPUT_CHANNEL_IDXS = RNNInputChannels(
    jnp.arange(0, 1), jnp.arange(1, 3), jnp.arange(3, 5), jnp.arange(5, 7), jnp.arange(7, 9)
)
