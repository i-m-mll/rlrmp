"""Open-loop perturbation calibration as a governed data product.

The open-loop unit-sensitivity table (``peak delta x`` per family and timing
bin) and the controller-visible velocity scale are *generated* by
``rlrmp.analysis.pipelines.gru_perturbation_calibration`` (extLQG nominal-command
open-loop replay). They used to live as source-code constants
(``DEFAULT_OPEN_LOOP_PEAK_DELTA_X_PER_UNIT`` /
``DEFAULT_CONTROLLER_VISIBLE_VELOCITY_SCALE_M_S``). They now live in a tracked
:class:`AnalysisDataProduct` document under ``results/ea6ccb4/data_products/`` and
are loaded here with fail-closed identity validation.

The controller-visible velocity scale is the C&S faithful-plant LQR peak forward
velocity, adopted from ``results/a7dad8a/notes/adversary_equivalence_manifest.json``
(``game_card_summary.lqr.peak_forward_velocity``); the persisted product records
that adoption. The force/filter native scale (``1.0 N``) is a unit convention,
carried alongside for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from feedbax.contracts.graph import AnalysisDataProductRequirement
from feedbax.contracts.manifest import AnalysisDataProduct

from rlrmp.data_products.envelope import load_data_product
from rlrmp.paths import REPO_ROOT, mkdir_p

__all__ = [
    "CALIBRATION_PRODUCT_IDENTITY_HASH",
    "CALIBRATION_PRODUCT_LOGICAL_NAME",
    "CALIBRATION_PRODUCT_PATH",
    "CALIBRATION_PRODUCT_ROLE",
    "CALIBRATION_PRODUCT_SCHEMA_ID",
    "CALIBRATION_PRODUCT_SCHEMA_VERSION",
    "OpenLoopCalibration",
    "build_open_loop_calibration_product",
    "calibration_data_product_requirement",
    "consumed_calibration_identity",
    "controller_visible_velocity_scale_m_s",
    "load_open_loop_calibration",
    "open_loop_peak_delta_x_per_unit",
    "write_open_loop_calibration_product",
]

CALIBRATION_PRODUCT_SCHEMA_ID = "rlrmp.perturbation_open_loop_calibration"
CALIBRATION_PRODUCT_SCHEMA_VERSION = "rlrmp.perturbation_open_loop_calibration.v2"
CALIBRATION_PRODUCT_ROLE = "perturbation_open_loop_calibration"
CALIBRATION_PRODUCT_LOGICAL_NAME = "cs_perturbation_open_loop_calibration"
CALIBRATION_PRODUCT_PRODUCER = (
    "rlrmp.analysis.pipelines.gru_perturbation_calibration."
    "materialize_perturbation_open_loop_calibration"
)
CALIBRATION_PRODUCT_RELPATH = (
    "results/ea6ccb4/data_products/perturbation_open_loop_calibration.json"
)
CALIBRATION_PRODUCT_PATH = REPO_ROOT / CALIBRATION_PRODUCT_RELPATH

# Pinned identity of the adopted calibration product. The loader fails closed if
# the persisted product's semantic identity ever diverges from this value.
CALIBRATION_PRODUCT_IDENTITY_HASH = (
    "03edd3141b62d1b1cf045097114caac7bc96f1236a433875976aec974d9bb97a"
)

CONTROLLER_VISIBLE_VELOCITY_SCALE_ADOPTION = {
    "source_issue": "a7dad8a",
    "source_note": "results/a7dad8a/notes/adversary_equivalence_manifest.json",
    "source_field": "game_card_summary.lqr.peak_forward_velocity",
    "description": "C&S faithful-plant LQR peak forward velocity used as the "
    "controller-visible native velocity scale",
}
CONTROLLER_VISIBLE_FORCE_FILTER_SCALE_CONVENTION = {
    "value_n": 1.0,
    "description": "native 1 N reference offset for force/filter controller-visible "
    "components; unit convention, not generated data",
}


@dataclass(frozen=True)
class OpenLoopCalibration:
    """Loaded open-loop perturbation calibration values with product identity."""

    peak_delta_x_per_unit: dict[str, dict[str, float]]
    controller_visible_velocity_scale_m_s: float
    controller_visible_force_filter_scale_n: float
    reference_reach_m: float
    product_identity_hash: str

    def __getitem__(self, key: str) -> dict[str, float]:
        """Index the unit-sensitivity table by perturbation family."""

        return self.peak_delta_x_per_unit[key]


def _calibration_parameters(
    peak_delta_x_per_unit: dict[str, dict[str, float]],
    controller_visible_velocity_scale_m_s: float,
    controller_visible_force_filter_scale_n: float,
    reference_reach_m: float,
) -> dict[str, Any]:
    return {
        "open_loop_peak_delta_x_per_unit": {
            str(family): {str(bin_): float(value) for bin_, value in bins.items()}
            for family, bins in peak_delta_x_per_unit.items()
        },
        "controller_visible_velocity_scale_m_s": float(controller_visible_velocity_scale_m_s),
        "controller_visible_force_filter_scale_n": float(controller_visible_force_filter_scale_n),
        "reference_reach_m": float(reference_reach_m),
        "scale_provenance": {
            "controller_visible_velocity_scale_m_s": CONTROLLER_VISIBLE_VELOCITY_SCALE_ADOPTION,
            "controller_visible_force_filter_scale_n": (
                CONTROLLER_VISIBLE_FORCE_FILTER_SCALE_CONVENTION
            ),
        },
    }


def build_open_loop_calibration_product(
    *,
    peak_delta_x_per_unit: dict[str, dict[str, float]],
    controller_visible_velocity_scale_m_s: float,
    controller_visible_force_filter_scale_n: float = 1.0,
    reference_reach_m: float = 0.15,
) -> AnalysisDataProduct:
    """Build the calibration :class:`AnalysisDataProduct` envelope."""

    return AnalysisDataProduct(
        product_schema_id=CALIBRATION_PRODUCT_SCHEMA_ID,
        product_schema_version=CALIBRATION_PRODUCT_SCHEMA_VERSION,
        role=CALIBRATION_PRODUCT_ROLE,
        logical_name=CALIBRATION_PRODUCT_LOGICAL_NAME,
        producer_manifest_id=CALIBRATION_PRODUCT_PRODUCER,
        parameters=_calibration_parameters(
            peak_delta_x_per_unit,
            controller_visible_velocity_scale_m_s,
            controller_visible_force_filter_scale_n,
            reference_reach_m,
        ),
        materialization={
            "materializer": CALIBRATION_PRODUCT_PRODUCER,
            "rerun_command": (
                "uv run python scripts/materialize_perturbation_open_loop_calibration.py"
            ),
            "open_loop_reference": "extLQG nominal command replay (no feedback correction)",
            "bulk_manifest_root": "_artifacts/1ad3c16/perturbation_open_loop_calibration",
        },
        metadata={
            "issue": "ea6ccb4",
            "provenance_issue": "1ad3c16",
            "note": "Distilled open-loop unit-sensitivity table; bulk per-row manifest "
            "lives under _artifacts and is not required at runtime.",
        },
    )


def write_open_loop_calibration_product(
    product: AnalysisDataProduct,
    *,
    path: Path | None = None,
) -> Path:
    """Persist the calibration product as tracked JSON and return its path."""

    path = path or CALIBRATION_PRODUCT_PATH
    mkdir_p(path.parent)
    path.write_text(
        product.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    return path


def calibration_data_product_requirement() -> AnalysisDataProductRequirement:
    """Return the fail-closed requirement for the open-loop calibration product."""

    return AnalysisDataProductRequirement(
        role=CALIBRATION_PRODUCT_ROLE,
        product_schema_id=CALIBRATION_PRODUCT_SCHEMA_ID,
        exact_product_schema_version=CALIBRATION_PRODUCT_SCHEMA_VERSION,
        logical_name=CALIBRATION_PRODUCT_LOGICAL_NAME,
        product_identity_hash=CALIBRATION_PRODUCT_IDENTITY_HASH,
    )


@lru_cache(maxsize=1)
def load_open_loop_calibration() -> OpenLoopCalibration:
    """Load and validate the open-loop calibration product (fail-closed, cached)."""

    product = load_data_product(
        CALIBRATION_PRODUCT_PATH,
        calibration_data_product_requirement(),
    )
    params = product.parameters
    table = {
        str(family): {str(bin_): float(value) for bin_, value in bins.items()}
        for family, bins in params["open_loop_peak_delta_x_per_unit"].items()
    }
    return OpenLoopCalibration(
        peak_delta_x_per_unit=table,
        controller_visible_velocity_scale_m_s=float(
            params["controller_visible_velocity_scale_m_s"]
        ),
        controller_visible_force_filter_scale_n=float(
            params["controller_visible_force_filter_scale_n"]
        ),
        reference_reach_m=float(params["reference_reach_m"]),
        product_identity_hash=str(product.product_identity_hash),
    )


def open_loop_peak_delta_x_per_unit() -> dict[str, dict[str, float]]:
    """Return the loaded open-loop unit-sensitivity table."""

    return load_open_loop_calibration().peak_delta_x_per_unit


def controller_visible_velocity_scale_m_s() -> float:
    """Return the loaded controller-visible native velocity scale."""

    return load_open_loop_calibration().controller_visible_velocity_scale_m_s


def consumed_calibration_identity() -> dict[str, str]:
    """Return the ``{role, schema, hash}`` consumed-identity record for run specs."""

    calibration = load_open_loop_calibration()
    return {
        "role": CALIBRATION_PRODUCT_ROLE,
        "schema": CALIBRATION_PRODUCT_SCHEMA_VERSION,
        "hash": calibration.product_identity_hash,
    }
