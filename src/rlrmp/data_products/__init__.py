"""Governed RLRMP data products loaded by typed identity, not baked as constants.

Generated or adopted empirical data (perturbation calibration tables, budget
anchors) is persisted as a tracked :class:`~feedbax.contracts.manifest.AnalysisDataProduct`
document and loaded here with fail-closed identity validation. Runtime code
defines schemas, loaders, and builders; it does not embed generated datasets as
source-code constants.
"""

from __future__ import annotations

from rlrmp.data_products.broad_epsilon import (
    BroadEpsilonAnchors,
    consumed_broad_epsilon_identity,
    load_broad_epsilon_anchors,
)
from rlrmp.data_products.calibration import (
    PerturbationCalibrationDefaults,
    OpenLoopCalibration,
    consumed_calibration_identity,
    consumed_perturbation_calibration_defaults_identity,
    load_open_loop_calibration,
    load_perturbation_calibration_defaults,
)
from rlrmp.data_products.envelope import DataProductError, load_data_product

__all__ = [
    "BroadEpsilonAnchors",
    "DataProductError",
    "OpenLoopCalibration",
    "PerturbationCalibrationDefaults",
    "consumed_broad_epsilon_identity",
    "consumed_calibration_identity",
    "consumed_perturbation_calibration_defaults_identity",
    "load_broad_epsilon_anchors",
    "load_data_product",
    "load_open_loop_calibration",
    "load_perturbation_calibration_defaults",
]
