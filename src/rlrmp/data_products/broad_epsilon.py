"""Broad full-state epsilon budget anchors as a governed, adopted data product.

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
manifest, pointer, and adopted values). The loader fails closed if the persisted
product's identity drifts, and additionally re-reads the analytical sources and
fails closed if the adopted values no longer match, so historical runs remain
reproducible through the adoption records rather than through silent constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from feedbax.contracts.graph import AnalysisDataProductRequirement
from feedbax.contracts.manifest import AnalysisDataProduct

from rlrmp.data_products.envelope import DataProductError, load_data_product
from rlrmp.paths import REPO_ROOT, mkdir_p

__all__ = [
    "BROAD_EPSILON_PRODUCT_IDENTITY_HASH",
    "BROAD_EPSILON_PRODUCT_LOGICAL_NAME",
    "BROAD_EPSILON_PRODUCT_PATH",
    "BROAD_EPSILON_PRODUCT_ROLE",
    "BROAD_EPSILON_PRODUCT_SCHEMA_ID",
    "BROAD_EPSILON_PRODUCT_SCHEMA_VERSION",
    "BROAD_EPSILON_REFERENCE_REACH_M",
    "BroadEpsilonAnchors",
    "broad_epsilon_data_product_requirement",
    "build_broad_epsilon_budget_anchors_product",
    "consumed_broad_epsilon_identity",
    "load_broad_epsilon_anchors",
    "write_broad_epsilon_budget_anchors_product",
]

BROAD_EPSILON_PRODUCT_SCHEMA_ID = "rlrmp.broad_epsilon_budget_anchors"
BROAD_EPSILON_PRODUCT_SCHEMA_VERSION = "rlrmp.broad_epsilon_budget_anchors.v1"
BROAD_EPSILON_PRODUCT_ROLE = "broad_epsilon_budget_anchors"
BROAD_EPSILON_PRODUCT_LOGICAL_NAME = "cs_broad_epsilon_budget_anchors"
BROAD_EPSILON_PRODUCT_PRODUCER = (
    "rlrmp.data_products.broad_epsilon.build_broad_epsilon_budget_anchors_product"
)
BROAD_EPSILON_PRODUCT_RELPATH = "results/ea6ccb4/data_products/broad_epsilon_budget_anchors.json"
BROAD_EPSILON_PRODUCT_PATH = REPO_ROOT / BROAD_EPSILON_PRODUCT_RELPATH
BROAD_EPSILON_REFERENCE_REACH_M = 0.15

# Pinned identity of the adopted budget-anchor product.
BROAD_EPSILON_PRODUCT_IDENTITY_HASH = (
    "4e5d319c4848ef19d25ddf9dc8d21a6230cc0d336c5f565fe1a0b63516332542"
)

# Contract keys returned to downstream consumers (exact legacy shape).
_CONTRACT_KEYS = (
    "gamma_factor",
    "closed_loop_epsilon_energy_15cm",
    "closed_loop_epsilon_l2_15cm",
    "delta_v_percent",
    "source_issue",
    "source_note",
)

# Per-level analytical adoption source. ``factor`` selects the frontier entry.
_LEVEL_SOURCES: dict[str, dict[str, Any]] = {
    "moderate": {
        "gamma_factor": 1.4,
        "source_issue": "cb98e58",
        "source_note": "results/cb98e58/notes/analytical_game_card_manifest.json",
        "source_manifest": "results/cb98e58/notes/analytical_game_card_manifest.json",
        "frontier_pointer": ("frontier",),
        "source_factor_key": "1p4",
    },
    "strong": {
        "gamma_factor": 1.05,
        "source_issue": "a7dad8a",
        "source_note": "results/a7dad8a/notes/adversary_equivalence_manifest.json",
        "source_manifest": "results/a7dad8a/notes/adversary_equivalence_manifest.json",
        "frontier_pointer": ("game_card_summary", "frontier"),
        "source_factor_key": "1p05",
    },
}


@dataclass(frozen=True)
class BroadEpsilonAnchors:
    """Loaded broad-epsilon budget anchors with product identity."""

    levels: dict[str, dict[str, Any]]
    product_identity_hash: str

    def __contains__(self, level: str) -> bool:
        return level in self.levels

    def __getitem__(self, level: str) -> dict[str, Any]:
        return self.levels[level]

    def __iter__(self):
        return iter(self.levels)

    def keys(self):
        return self.levels.keys()


def _read_frontier_entry(source: dict[str, Any]) -> dict[str, Any]:
    """Read the analytical frontier entry named by ``source`` (fail-closed)."""

    path = REPO_ROOT / source["source_manifest"]
    if not path.is_file():
        raise DataProductError(
            f"broad-epsilon analytical source is missing: {path}",
            kind="Missing",
            mismatch_class="missing-source",
        )
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    for key in source["frontier_pointer"]:
        payload = payload[key]
    factor = float(source["gamma_factor"])
    for entry in payload:
        if abs(float(entry.get("factor", "nan")) - factor) <= 1e-12:
            return entry
    raise DataProductError(
        f"broad-epsilon analytical source {source['source_manifest']} has no frontier "
        f"entry for factor {factor}",
        kind="Mismatch",
        mismatch_class="missing-source-entry",
    )


def _anchor_from_source(level: str, source: dict[str, Any]) -> dict[str, Any]:
    entry = _read_frontier_entry(source)
    return {
        "gamma_factor": float(source["gamma_factor"]),
        "closed_loop_epsilon_energy_15cm": float(entry["closed_loop_epsilon_energy"]),
        "closed_loop_epsilon_l2_15cm": float(entry["closed_loop_epsilon_l2"]),
        "delta_v_percent": float(entry["delta_v_percent"]),
        "source_issue": str(source["source_issue"]),
        "source_note": str(source["source_note"]),
    }


def _contract(anchor: dict[str, Any]) -> dict[str, Any]:
    return {key: anchor[key] for key in _CONTRACT_KEYS}


def _adoption_record(level: str, source: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_manifest": source["source_manifest"],
        "source_pointer": (
            f"{'.'.join(source['frontier_pointer'])}[factor={source['gamma_factor']}]"
        ),
        "source_factor_key": source["source_factor_key"],
        "adopted_fields": {
            "closed_loop_epsilon_energy_15cm": "closed_loop_epsilon_energy",
            "closed_loop_epsilon_l2_15cm": "closed_loop_epsilon_l2",
            "delta_v_percent": "delta_v_percent",
        },
        "adopted_values": _contract(anchor),
        "note": "Historical broad-epsilon runs used the identical baked values; the "
        "adoption record makes that provenance explicit and reproducible.",
    }


def build_broad_epsilon_budget_anchors_product() -> AnalysisDataProduct:
    """Build the broad-epsilon anchors :class:`AnalysisDataProduct` from sources."""

    levels_payload: dict[str, Any] = {}
    for level, source in _LEVEL_SOURCES.items():
        anchor = _anchor_from_source(level, source)
        levels_payload[level] = {
            **_contract(anchor),
            "adoption": _adoption_record(level, source, anchor),
        }
    return AnalysisDataProduct(
        product_schema_id=BROAD_EPSILON_PRODUCT_SCHEMA_ID,
        product_schema_version=BROAD_EPSILON_PRODUCT_SCHEMA_VERSION,
        role=BROAD_EPSILON_PRODUCT_ROLE,
        logical_name=BROAD_EPSILON_PRODUCT_LOGICAL_NAME,
        producer_manifest_id=BROAD_EPSILON_PRODUCT_PRODUCER,
        parameters={
            "reference_reach_m": BROAD_EPSILON_REFERENCE_REACH_M,
            "levels": levels_payload,
        },
        materialization={
            "materializer": BROAD_EPSILON_PRODUCT_PRODUCER,
            "adoption_mode": "read_analytical_frontier_entries",
        },
        metadata={
            "issue": "ea6ccb4",
            "note": "Broad-epsilon per-level closed-loop epsilon budgets adopted from "
            "analytical game-card / adversary-equivalence manifests.",
        },
    )


def write_broad_epsilon_budget_anchors_product(
    product: AnalysisDataProduct,
    *,
    path: Path | None = None,
) -> Path:
    """Persist the broad-epsilon anchors product as tracked JSON."""

    path = path or BROAD_EPSILON_PRODUCT_PATH
    mkdir_p(path.parent)
    path.write_text(
        product.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    return path


def broad_epsilon_data_product_requirement() -> AnalysisDataProductRequirement:
    """Return the fail-closed requirement for the broad-epsilon anchors product."""

    return AnalysisDataProductRequirement(
        role=BROAD_EPSILON_PRODUCT_ROLE,
        product_schema_id=BROAD_EPSILON_PRODUCT_SCHEMA_ID,
        exact_product_schema_version=BROAD_EPSILON_PRODUCT_SCHEMA_VERSION,
        logical_name=BROAD_EPSILON_PRODUCT_LOGICAL_NAME,
        product_identity_hash=BROAD_EPSILON_PRODUCT_IDENTITY_HASH,
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
    persisted_levels = product.parameters["levels"]
    levels: dict[str, dict[str, Any]] = {}
    for level, source in _LEVEL_SOURCES.items():
        if level not in persisted_levels:
            raise DataProductError(
                f"broad-epsilon product is missing level {level!r}",
                kind="Missing",
                mismatch_class="missing-level",
            )
        contract = _contract(persisted_levels[level])
        source_anchor = _contract(_anchor_from_source(level, source))
        if contract != source_anchor:
            raise DataProductError(
                f"broad-epsilon level {level!r} no longer matches its analytical "
                f"source {source['source_note']}: persisted={contract!r}, "
                f"source={source_anchor!r}",
                kind="Mismatch",
                mismatch_class="analytical-source-drift",
            )
        levels[level] = contract
    return BroadEpsilonAnchors(
        levels=levels,
        product_identity_hash=str(product.product_identity_hash),
    )


def consumed_broad_epsilon_identity() -> dict[str, str]:
    """Return the ``{role, schema, hash}`` consumed-identity record for run specs."""

    anchors = load_broad_epsilon_anchors()
    return {
        "role": BROAD_EPSILON_PRODUCT_ROLE,
        "schema": BROAD_EPSILON_PRODUCT_SCHEMA_VERSION,
        "hash": anchors.product_identity_hash,
    }
