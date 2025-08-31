from collections import namedtuple
from typing import NamedTuple


class RNNInputChannels[T](NamedTuple):
    sisu: T
    goal_pos: T
    goal_vel: T
    fb_pos: T
    fb_vel: T


class RNNCellArgs[T](NamedTuple):
    input: T
    state: T
