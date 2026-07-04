"""Public-surface guard for governed data products."""

from __future__ import annotations

import inspect
from types import ModuleType
from typing import Any

import rlrmp.data_products as data_products
from rlrmp.data_products import broad_epsilon, calibration, envelope, registry


EXPECTED_PACKAGE_ALL = {
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
}

EXPECTED_MODULE_FUNCTIONS = {
    "rlrmp.data_products.envelope": {
        "load_data_product",
        "read_data_product",
        "validate_data_product",
        "write_data_product",
    },
    "rlrmp.data_products.calibration": {
        "calibration_data_product_requirement",
        "calibration_defaults_data_product_requirement",
        "consumed_calibration_identity",
        "consumed_perturbation_calibration_defaults_identity",
        "load_open_loop_calibration",
        "load_perturbation_calibration_defaults",
    },
    "rlrmp.data_products.broad_epsilon": {
        "broad_epsilon_data_product_requirement",
        "consumed_broad_epsilon_identity",
        "load_broad_epsilon_anchors",
    },
    "rlrmp.data_products.registry": {
        "register_data_product_identity",
        "registered_data_product_identities",
    },
}


def test_data_products_package_public_surface_is_pinned() -> None:
    # Issue 96ac0e5: bespoke build/write path bypassed by the real materializer
    # must not silently reaccrete.
    assert set(data_products.__all__) == EXPECTED_PACKAGE_ALL


def test_data_products_module_public_functions_are_pinned() -> None:
    modules = (envelope, calibration, broad_epsilon, registry)
    assert {
        module.__name__: _public_module_callables(module)
        for module in modules
    } == EXPECTED_MODULE_FUNCTIONS


def _public_module_callables(module: ModuleType) -> set[str]:
    return {
        name
        for name, value in vars(module).items()
        if _is_public_module_callable(module, name, value)
    }


def _is_public_module_callable(module: ModuleType, name: str, value: Any) -> bool:
    return (
        not name.startswith("_")
        and callable(value)
        and not inspect.isclass(value)
        and getattr(value, "__module__", None) == module.__name__
    )
