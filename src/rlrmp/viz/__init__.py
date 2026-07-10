"""Visualization utilities for rlrmp."""

from rlrmp.viz.colors import hex_to_rgba
from rlrmp.viz.profile_grids import profile_comparison_grid
from rlrmp.viz.traces import (
    add_band_trace,
    add_line,
    add_profile_line,
    add_reference_trace,
    add_reduced_sample_trace,
    add_sample_band,
)

__all__ = [
    "add_band_trace",
    "add_line",
    "add_profile_line",
    "add_reference_trace",
    "add_reduced_sample_trace",
    "add_sample_band",
    "hex_to_rgba",
    "profile_comparison_grid",
]
