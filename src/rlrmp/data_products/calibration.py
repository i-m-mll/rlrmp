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

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from feedbax.contracts.graph import AnalysisDataProductRequirement
from feedbax.contracts.manifest import AnalysisDataProduct, ArtifactRef

from rlrmp.data_products.envelope import DataProductError, load_data_product
from rlrmp.data_products.registry import register_data_product_identity
from rlrmp.paths import REPO_ROOT

__all__ = [
    "CALIBRATION_PRODUCT_IDENTITY_HASH",
    "CALIBRATION_PRODUCT_LOGICAL_NAME",
    "CALIBRATION_PRODUCT_PATH",
    "CALIBRATION_PRODUCT_ROLE",
    "CALIBRATION_PRODUCT_SCHEMA_ID",
    "CALIBRATION_PRODUCT_SCHEMA_VERSION",
    "CALIBRATION_DEFAULTS_PAYLOAD_PATH",
    "CALIBRATION_DEFAULTS_PAYLOAD_RELPATH",
    "CALIBRATION_DEFAULTS_PAYLOAD_SCHEMA_ID",
    "CALIBRATION_DEFAULTS_PAYLOAD_SCHEMA_VERSION",
    "CALIBRATION_DEFAULTS_PAYLOAD_SHA256",
    "CALIBRATION_DEFAULTS_PRODUCT_IDENTITY_HASH",
    "CALIBRATION_DEFAULTS_PRODUCT_LOGICAL_NAME",
    "CALIBRATION_DEFAULTS_PRODUCT_PATH",
    "CALIBRATION_DEFAULTS_PRODUCT_RELPATH",
    "CALIBRATION_DEFAULTS_PRODUCT_ROLE",
    "CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_ID",
    "CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_VERSION",
    "NativeConvention",
    "OpenLoopCalibration",
    "PerturbationCalibrationDefaults",
    "ReachCalibrationPoint",
    "ReachRelativeLevel",
    "TimingCalibrationBin",
    "calibration_data_product_requirement",
    "calibration_defaults_data_product_requirement",
    "load_perturbation_calibration_defaults",
    "load_open_loop_calibration",
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

CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_ID = "rlrmp.perturbation_calibration_defaults"
CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_VERSION = "rlrmp.perturbation_calibration_defaults.v1"
CALIBRATION_DEFAULTS_PRODUCT_ROLE = "perturbation_calibration_defaults"
CALIBRATION_DEFAULTS_PRODUCT_LOGICAL_NAME = "cs_perturbation_calibration_defaults"
CALIBRATION_DEFAULTS_PRODUCT_PRODUCER = "tracked_calibration_defaults_document"
CALIBRATION_DEFAULTS_PRODUCT_RELPATH = (
    "results/ea6ccb4/data_products/perturbation_calibration_defaults.json"
)
CALIBRATION_DEFAULTS_PRODUCT_PATH = REPO_ROOT / CALIBRATION_DEFAULTS_PRODUCT_RELPATH
CALIBRATION_DEFAULTS_PAYLOAD_SCHEMA_ID = "rlrmp.perturbation_calibration_defaults.payload"
CALIBRATION_DEFAULTS_PAYLOAD_SCHEMA_VERSION = (
    "rlrmp.perturbation_calibration_defaults.payload.v1"
)
CALIBRATION_DEFAULTS_PAYLOAD_RELPATH = (
    "results/ea6ccb4/data_products/perturbation_calibration_defaults.payload.json"
)
CALIBRATION_DEFAULTS_PAYLOAD_PATH = REPO_ROOT / CALIBRATION_DEFAULTS_PAYLOAD_RELPATH
CALIBRATION_DEFAULTS_PAYLOAD_LOGICAL_NAME = "cs_perturbation_calibration_defaults_payload"

# Pinned identity and payload byte hash of the adopted runtime-default tables.
CALIBRATION_DEFAULTS_PRODUCT_IDENTITY_HASH = (
    "56a58d4ca3f5ff143e6acab62dee9bf25256cec9a8d540804736e684770b63e5"
)
CALIBRATION_DEFAULTS_PAYLOAD_SHA256 = (
    "2947ce1bbc3511d80c10d7c2428320de2d59fc74af7f416e9107a87c5e3072cc"
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


@dataclass(frozen=True)
class ReachCalibrationPoint:
    """A reach length whose relative perturbation levels should be calibrated."""

    label: str
    split: str
    reach_length_m: float
    role: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable calibration point."""

        return {
            "label": self.label,
            "split": self.split,
            "reach_length_m": float(self.reach_length_m),
            "role": self.role,
        }


@dataclass(frozen=True)
class ReachRelativeLevel:
    """A relative open-loop effect-size level."""

    name: str
    fraction_of_reach: float
    role: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable level definition."""

        return {
            "name": self.name,
            "fraction_of_reach": float(self.fraction_of_reach),
            "role": self.role,
        }


@dataclass(frozen=True)
class TimingCalibrationBin:
    """A deterministic timing bin used by perturbation calibration."""

    label: str
    start_time_index: int
    duration_steps: int
    role: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable timing-bin definition."""

        return {
            "label": self.label,
            "start_time_index": int(self.start_time_index),
            "duration_steps": int(self.duration_steps),
            "role": self.role,
        }


@dataclass(frozen=True)
class NativeConvention:
    """Native-unit convention for rows that are not open-loop plant calibrations."""

    family: str
    channel: str
    native_unit_rule: str
    timing_rule: str
    report_metric: str
    role: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable native convention."""

        return {
            "family": self.family,
            "channel": self.channel,
            "native_unit_rule": self.native_unit_rule,
            "timing_rule": self.timing_rule,
            "report_metric": self.report_metric,
            "role": self.role,
        }


@dataclass(frozen=True)
class PerturbationCalibrationDefaults:
    """Loaded adopted runtime defaults for perturbation calibration."""

    amplitude_factors: tuple[float, ...]
    reach_calibration_points: tuple[ReachCalibrationPoint, ...]
    reach_relative_levels: tuple[ReachRelativeLevel, ...]
    plant_timing_bins: tuple[TimingCalibrationBin, ...]
    controller_visible_timing_bins: tuple[TimingCalibrationBin, ...]
    native_conventions: tuple[NativeConvention, ...]
    product_identity_hash: str
    payload_sha256: str


def calibration_data_product_requirement() -> AnalysisDataProductRequirement:
    """Return the fail-closed requirement for the open-loop calibration product."""

    return AnalysisDataProductRequirement(
        role=CALIBRATION_PRODUCT_ROLE,
        product_schema_id=CALIBRATION_PRODUCT_SCHEMA_ID,
        exact_product_schema_version=CALIBRATION_PRODUCT_SCHEMA_VERSION,
        logical_name=CALIBRATION_PRODUCT_LOGICAL_NAME,
        product_identity_hash=CALIBRATION_PRODUCT_IDENTITY_HASH,
    )


def calibration_defaults_data_product_requirement() -> AnalysisDataProductRequirement:
    """Return the fail-closed requirement for adopted calibration defaults."""

    return AnalysisDataProductRequirement(
        role=CALIBRATION_DEFAULTS_PRODUCT_ROLE,
        product_schema_id=CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_ID,
        exact_product_schema_version=CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_VERSION,
        logical_name=CALIBRATION_DEFAULTS_PRODUCT_LOGICAL_NAME,
        product_identity_hash=CALIBRATION_DEFAULTS_PRODUCT_IDENTITY_HASH,
        artifact_sha256=CALIBRATION_DEFAULTS_PAYLOAD_SHA256,
    )


register_data_product_identity(
    role=CALIBRATION_PRODUCT_ROLE,
    product_schema_id=CALIBRATION_PRODUCT_SCHEMA_ID,
    product_schema_version=CALIBRATION_PRODUCT_SCHEMA_VERSION,
    logical_name=CALIBRATION_PRODUCT_LOGICAL_NAME,
    requirement_factory=calibration_data_product_requirement,
    document_relpath=CALIBRATION_PRODUCT_RELPATH,
)

register_data_product_identity(
    role=CALIBRATION_DEFAULTS_PRODUCT_ROLE,
    product_schema_id=CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_ID,
    product_schema_version=CALIBRATION_DEFAULTS_PRODUCT_SCHEMA_VERSION,
    logical_name=CALIBRATION_DEFAULTS_PRODUCT_LOGICAL_NAME,
    requirement_factory=calibration_defaults_data_product_requirement,
    document_relpath=CALIBRATION_DEFAULTS_PRODUCT_RELPATH,
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


@lru_cache(maxsize=1)
def load_perturbation_calibration_defaults() -> PerturbationCalibrationDefaults:
    """Load and validate adopted perturbation-calibration runtime defaults."""

    product = load_data_product(
        CALIBRATION_DEFAULTS_PRODUCT_PATH,
        calibration_defaults_data_product_requirement(),
    )
    artifact = _calibration_defaults_payload_artifact(product)
    if artifact.uri != product.parameters.get("payload_artifact_uri"):
        raise DataProductError(
            "calibration-defaults product payload artifact uri does not match parameters",
            kind="Mismatch",
            mismatch_class="artifact-uri",
        )
    payload_path = REPO_ROOT / str(artifact.uri)
    actual_sha256 = _sha256_file(payload_path)
    if actual_sha256 != artifact.sha256:
        raise DataProductError(
            f"calibration-defaults payload hash mismatch: required={artifact.sha256!r}, "
            f"actual={actual_sha256!r}",
            kind="Mismatch",
            mismatch_class="artifact-byte-hash",
        )
    payload = _read_calibration_defaults_payload(payload_path)
    _validate_calibration_defaults_payload(payload)
    return PerturbationCalibrationDefaults(
        amplitude_factors=tuple(float(value) for value in payload["amplitude_factors"]),
        reach_calibration_points=tuple(
            ReachCalibrationPoint(
                label=str(point["label"]),
                split=str(point["split"]),
                reach_length_m=float(point["reach_length_m"]),
                role=str(point["role"]),
            )
            for point in payload["reach_calibration_points"]
        ),
        reach_relative_levels=tuple(
            ReachRelativeLevel(
                name=str(level["name"]),
                fraction_of_reach=float(level["fraction_of_reach"]),
                role=str(level["role"]),
            )
            for level in payload["reach_relative_levels"]
        ),
        plant_timing_bins=tuple(
            TimingCalibrationBin(
                label=str(timing_bin["label"]),
                start_time_index=int(timing_bin["start_time_index"]),
                duration_steps=int(timing_bin["duration_steps"]),
                role=str(timing_bin["role"]),
            )
            for timing_bin in payload["plant_timing_bins"]
        ),
        controller_visible_timing_bins=tuple(
            TimingCalibrationBin(
                label=str(timing_bin["label"]),
                start_time_index=int(timing_bin["start_time_index"]),
                duration_steps=int(timing_bin["duration_steps"]),
                role=str(timing_bin["role"]),
            )
            for timing_bin in payload["controller_visible_timing_bins"]
        ),
        native_conventions=tuple(
            NativeConvention(
                family=str(convention["family"]),
                channel=str(convention["channel"]),
                native_unit_rule=str(convention["native_unit_rule"]),
                timing_rule=str(convention["timing_rule"]),
                report_metric=str(convention["report_metric"]),
                role=str(convention["role"]),
            )
            for convention in payload["native_conventions"]
        ),
        product_identity_hash=str(product.product_identity_hash),
        payload_sha256=str(artifact.sha256),
    )


def _calibration_defaults_payload_artifact(product: AnalysisDataProduct) -> ArtifactRef:
    matches = [
        artifact
        for artifact in product.artifacts
        if artifact.role == "calibration_defaults_payload"
    ]
    if len(matches) != 1:
        raise DataProductError(
            "calibration-defaults product must carry exactly one payload artifact",
            kind="Missing",
            mismatch_class="missing-payload-artifact",
        )
    artifact = matches[0]
    if artifact.sha256 != CALIBRATION_DEFAULTS_PAYLOAD_SHA256:
        raise DataProductError(
            f"calibration-defaults product has wrong payload artifact hash: "
            f"required={CALIBRATION_DEFAULTS_PAYLOAD_SHA256!r}, actual={artifact.sha256!r}",
            kind="Mismatch",
            mismatch_class="artifact-byte-hash",
        )
    if not artifact.uri:
        raise DataProductError(
            "calibration-defaults payload artifact is missing a repo-relative uri",
            kind="Missing",
            mismatch_class="missing-payload-artifact",
        )
    return artifact


def _read_calibration_defaults_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataProductError(
            f"calibration-defaults payload is missing: {path}",
            kind="Missing",
            mismatch_class="missing-payload-artifact",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataProductError(
            f"calibration-defaults payload could not be read: {path}: {exc}",
            kind="Missing",
            mismatch_class="unreadable-payload-artifact",
        ) from exc
    if not isinstance(payload, dict):
        raise DataProductError(
            "calibration-defaults payload must be a JSON object",
            kind="Mismatch",
            mismatch_class="payload-schema",
        )
    return payload


def _validate_calibration_defaults_payload(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_id": CALIBRATION_DEFAULTS_PAYLOAD_SCHEMA_ID,
        "schema_version": CALIBRATION_DEFAULTS_PAYLOAD_SCHEMA_VERSION,
        "role": CALIBRATION_DEFAULTS_PRODUCT_ROLE,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise DataProductError(
                f"calibration-defaults payload has wrong {key}: "
                f"required={value!r}, actual={payload.get(key)!r}",
                kind="Mismatch",
                mismatch_class="payload-schema",
            )
    for key in (
        "amplitude_factors",
        "reach_calibration_points",
        "reach_relative_levels",
        "plant_timing_bins",
        "controller_visible_timing_bins",
        "native_conventions",
    ):
        if not isinstance(payload.get(key), list) or not payload[key]:
            raise DataProductError(
                f"calibration-defaults payload missing non-empty list {key!r}",
                kind="Missing",
                mismatch_class="payload-schema",
            )


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DataProductError(
            f"could not hash calibration-defaults payload: {path}: {exc}",
            kind="Missing",
            mismatch_class="unreadable-payload-artifact",
        ) from exc
