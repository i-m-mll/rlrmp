"""Broad full-state epsilon budget anchors as a governed extraction product.

The broad-epsilon training lane draws its per-level closed-loop epsilon budget
from analytical game-card / adversary-equivalence manifests, not from a baked
Python table. The two levels adopt directly from their analytical sources:

* ``moderate`` (gamma factor 1.4) from
  ``results/cb98e58/notes/analytical_game_card_manifest.json``
  (``frontier`` entry with ``factor == 1.4``);
* ``strong`` (gamma factor 1.05) from
  ``results/a7dad8a/notes/adversary_equivalence_manifest.json``
  (``game_card_summary.frontier`` entry with ``factor == 1.05``).

The persisted product records an explicit adoption record per level (source
manifest, pointer, and adopted values). The extraction spec under
``results/ea6ccb4/data_products/`` owns the source selection and field mapping.
The loader fails closed if the persisted product's identity drifts, and
additionally re-runs the feedbax extraction engine so historical runs remain
reproducible through the adoption records rather than through silent constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from feedbax.contracts.graph import AnalysisDataProductRequirement
from feedbax.contracts.extraction import (
    DataProductDrift,
    ExtractionProductSpec,
    verify_extraction_product,
)

from rlrmp.data_products.envelope import DataProductError, load_data_product
from rlrmp.data_products.registry import register_data_product_identity
from rlrmp.paths import REPO_ROOT

__all__ = [
    "BROAD_EPSILON_PRODUCT_IDENTITY_HASH",
    "BROAD_EPSILON_PRODUCT_LOGICAL_NAME",
    "BROAD_EPSILON_PRODUCT_PATH",
    "BROAD_EPSILON_PRODUCT_ROLE",
    "BROAD_EPSILON_PRODUCT_SCHEMA_ID",
    "BROAD_EPSILON_PRODUCT_SCHEMA_VERSION",
    "BROAD_EPSILON_REFERENCE_REACH_M",
    "BROAD_EPSILON_EXTRACTION_SPEC_PATH",
    "BroadEpsilonAnchors",
    "broad_epsilon_data_product_requirement",
    "load_broad_epsilon_anchors",
]

BROAD_EPSILON_PRODUCT_SCHEMA_ID = "rlrmp.broad_epsilon_budget_anchors"
BROAD_EPSILON_PRODUCT_SCHEMA_VERSION = "rlrmp.broad_epsilon_budget_anchors.v1"
BROAD_EPSILON_PRODUCT_ROLE = "broad_epsilon_budget_anchors"
BROAD_EPSILON_PRODUCT_LOGICAL_NAME = "cs_broad_epsilon_budget_anchors"
BROAD_EPSILON_PRODUCT_PRODUCER = "scripts.materialize_broad_epsilon_budget_anchors"
BROAD_EPSILON_PRODUCT_RELPATH = "results/ea6ccb4/data_products/broad_epsilon_budget_anchors.json"
BROAD_EPSILON_PRODUCT_PATH = REPO_ROOT / BROAD_EPSILON_PRODUCT_RELPATH
BROAD_EPSILON_EXTRACTION_SPEC_RELPATH = (
    "results/ea6ccb4/data_products/broad_epsilon_budget_anchors.extraction.json"
)
BROAD_EPSILON_EXTRACTION_SPEC_PATH = REPO_ROOT / BROAD_EPSILON_EXTRACTION_SPEC_RELPATH
BROAD_EPSILON_REFERENCE_REACH_M: float

# Pinned identity of the adopted budget-anchor product.
BROAD_EPSILON_PRODUCT_IDENTITY_HASH = (
    "4e5d319c4848ef19d25ddf9dc8d21a6230cc0d336c5f565fe1a0b63516332542"
)

_CONTRACT_KEYS = (
    "gamma_factor",
    "closed_loop_epsilon_energy_15cm",
    "closed_loop_epsilon_l2_15cm",
    "delta_v_percent",
    "source_issue",
    "source_note",
)

_EXPECTED_LEVELS = ("moderate", "strong")


@dataclass(frozen=True)
class BroadEpsilonAnchors:
    """Loaded broad-epsilon budget anchors with product identity."""

    levels: dict[str, dict[str, Any]]
    reference_reach_m: float
    product_identity_hash: str

    def __contains__(self, level: str) -> bool:
        return level in self.levels

    def __getitem__(self, level: str) -> dict[str, Any]:
        return self.levels[level]

    def __iter__(self):
        return iter(self.levels)

    def keys(self):
        return self.levels.keys()


def _contract(anchor: dict[str, Any]) -> dict[str, Any]:
    return {key: anchor[key] for key in _CONTRACT_KEYS}


@lru_cache(maxsize=1)
def _broad_epsilon_extraction_spec() -> ExtractionProductSpec:
    """Read the tracked feedbax extraction spec for broad-epsilon anchors."""

    if not BROAD_EPSILON_EXTRACTION_SPEC_PATH.is_file():
        raise DataProductError(
            f"broad-epsilon extraction spec is missing: {BROAD_EPSILON_EXTRACTION_SPEC_PATH}",
            kind="Missing",
            mismatch_class="missing-extraction-spec",
        )
    try:
        return ExtractionProductSpec.model_validate_json(
            BROAD_EPSILON_EXTRACTION_SPEC_PATH.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise DataProductError(
            f"broad-epsilon extraction spec failed validation: {exc}",
            kind="Mismatch",
            mismatch_class="extraction-spec",
        ) from exc


def broad_epsilon_data_product_requirement() -> AnalysisDataProductRequirement:
    """Return the fail-closed requirement for the broad-epsilon anchors product."""

    return AnalysisDataProductRequirement(
        role=BROAD_EPSILON_PRODUCT_ROLE,
        product_schema_id=BROAD_EPSILON_PRODUCT_SCHEMA_ID,
        exact_product_schema_version=BROAD_EPSILON_PRODUCT_SCHEMA_VERSION,
        logical_name=BROAD_EPSILON_PRODUCT_LOGICAL_NAME,
        product_identity_hash=BROAD_EPSILON_PRODUCT_IDENTITY_HASH,
    )


register_data_product_identity(
    role=BROAD_EPSILON_PRODUCT_ROLE,
    product_schema_id=BROAD_EPSILON_PRODUCT_SCHEMA_ID,
    product_schema_version=BROAD_EPSILON_PRODUCT_SCHEMA_VERSION,
    logical_name=BROAD_EPSILON_PRODUCT_LOGICAL_NAME,
    requirement_factory=broad_epsilon_data_product_requirement,
    document_relpath=BROAD_EPSILON_PRODUCT_RELPATH,
)


@lru_cache(maxsize=1)
def load_broad_epsilon_anchors() -> BroadEpsilonAnchors:
    """Load, validate, and re-verify broad-epsilon anchors (fail-closed, cached).

    Returns anchors keyed by level in the legacy ``BROAD_EPSILON_LEVELS`` shape
    (``gamma_factor``, ``closed_loop_epsilon_energy_15cm``,
    ``closed_loop_epsilon_l2_15cm``, ``delta_v_percent``, ``source_issue``,
    ``source_note``). Fails closed if the persisted product identity is stale or
    if the persisted values no longer match the analytical source manifests.
    """

    product = load_data_product(
        BROAD_EPSILON_PRODUCT_PATH,
        broad_epsilon_data_product_requirement(),
    )
    try:
        product = verify_extraction_product(
            _broad_epsilon_extraction_spec(),
            product,
            REPO_ROOT,
        )
    except DataProductDrift as exc:
        raise DataProductError(
            f"broad-epsilon product no longer matches analytical sources: {exc}",
            kind="Mismatch",
            mismatch_class="analytical-source-drift",
        ) from exc
    persisted_levels = product.parameters["levels"]
    levels: dict[str, dict[str, Any]] = {}
    for level in _EXPECTED_LEVELS:
        if level not in persisted_levels:
            raise DataProductError(
                f"broad-epsilon product is missing level {level!r}",
                kind="Missing",
                mismatch_class="missing-level",
            )
        levels[level] = _contract(persisted_levels[level])
    return BroadEpsilonAnchors(
        levels=levels,
        reference_reach_m=float(product.parameters["reference_reach_m"]),
        product_identity_hash=str(product.product_identity_hash),
    )


def __getattr__(name: str) -> Any:
    """Resolve the legacy reference-reach export from the governed product."""

    if name == "BROAD_EPSILON_REFERENCE_REACH_M":
        return load_broad_epsilon_anchors().reference_reach_m
    raise AttributeError(name)
