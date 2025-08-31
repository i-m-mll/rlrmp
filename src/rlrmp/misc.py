from jax_cookbook.misc import split_by
from jaxtyping import Array

from rlrmp.constants import RNN_INPUT_CHANNEL_SIZES
from rlrmp.types import RNNInputChannels


def split_rnn_input_channels(arr: Array, *, axis=-1):
    return RNNInputChannels(*split_by(arr, RNN_INPUT_CHANNEL_SIZES, axis=axis))
