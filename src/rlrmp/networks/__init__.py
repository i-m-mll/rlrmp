"""rlrmp.networks — controller architectures for rlrmp experiments.

Currently contains the linear-controller MVP for the decoupling acid test
(Bug: 410d7ac).
"""

from rlrmp.networks.linear_controllers import (
    LinearController,
    LinearTrackerController,
    point_mass_linear_controller,
)

__all__ = [
    "LinearController",
    "LinearTrackerController",
    "point_mass_linear_controller",
]
