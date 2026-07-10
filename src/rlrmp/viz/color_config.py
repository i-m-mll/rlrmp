"""RLRMP-specific color configuration for Feedbax analysis plots."""

from __future__ import annotations

from types import MappingProxyType
from typing import Sequence

import plotly.colors as plc

from feedbax.config.namespace import TreeNamespace
from feedbax.plot.color_setup import ColorConfig, ColorscaleSpec, register_color_config

RLRMP_COLOR_CONFIG_SCHEMA_ID = "rlrmp.plot.color_config"
RLRMP_COLOR_CONFIG_SCHEMA_VERSION = "rlrmp.plot.color_config.v1"


def _perturbation_amplitudes(hps: TreeNamespace) -> Sequence:
    return hps.pert.amp


def _training_perturbation_stds(hps: TreeNamespace) -> Sequence:
    return hps.train.pert.std


def _sisu_levels(hps: TreeNamespace) -> Sequence:
    return hps.sisu


def _trial_indices(hps: TreeNamespace) -> Sequence[int]:
    return range(hps.eval_n)


RLRMP_COLOR_CONFIG = ColorConfig(
    schema_id=RLRMP_COLOR_CONFIG_SCHEMA_ID,
    schema_version=RLRMP_COLOR_CONFIG_SCHEMA_VERSION,
    colorscales=MappingProxyType(
        {
            "train__pert__std": "viridis",
            "pert__amp": "plotly3",
            "sisu": "thermal",
            "reach_condition": "phase",
            "replicate": "twilight",
            "trial": "Tealgrn",
            "pert_var": tuple(plc.qualitative.D3),
        }
    ),
    color_specs=MappingProxyType(
        {
            "pert__amp": ColorscaleSpec(_perturbation_amplitudes),
            "train__pert__std": ColorscaleSpec(_training_perturbation_stds),
            "sisu": ColorscaleSpec(_sisu_levels),
            "trial": ColorscaleSpec(_trial_indices),
        }
    ),
)


def register_rlrmp_color_config() -> ColorConfig:
    """Register RLRMP's project vocabulary with Feedbax's color registry."""

    return register_color_config(RLRMP_COLOR_CONFIG, replace=True)


__all__ = [
    "RLRMP_COLOR_CONFIG",
    "RLRMP_COLOR_CONFIG_SCHEMA_ID",
    "RLRMP_COLOR_CONFIG_SCHEMA_VERSION",
    "register_rlrmp_color_config",
]
