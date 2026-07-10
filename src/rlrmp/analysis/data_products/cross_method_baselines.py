"""Governed empirical baseline rows for the cross-method comparison."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

from feedbax.contracts.graph import AnalysisDataProductRequirement

from rlrmp.data_products.envelope import DataProductError, load_data_product
from rlrmp.data_products.registry import register_data_product_identity
from rlrmp.paths import REPO_ROOT

PRODUCT_SCHEMA_ID = "rlrmp.cross_method_first_run_baselines"
PRODUCT_SCHEMA_VERSION = "rlrmp.cross_method_first_run_baselines.v1"
PRODUCT_ROLE = "cross_method_first_run_baselines"
PRODUCT_LOGICAL_NAME = "c723082_first_run_baselines"
PRODUCT_IDENTITY_HASH = "c881652ee47d4261ea13f90167fb3ea03a4a662a0d1c418bc81442251a4f4664"
PRODUCT_RELPATH = "results/c723082/data_products/first_run_baselines.json"
PAYLOAD_RELPATH = "results/c723082/data_products/first_run_baselines.payload.json"
PAYLOAD_SCHEMA_ID = "rlrmp.cross_method_first_run_baselines.payload"
PAYLOAD_SCHEMA_VERSION = "rlrmp.cross_method_first_run_baselines.payload.v1"
PAYLOAD_SHA256 = "ccc74ac8620aa2d655bbac4e1544e6d21a076a38d03b508721809ae88ce518ba"


@dataclass(frozen=True)
class CrossMethodFirstRunBaselines:
    """Validated empirical rows and their pinned product identity."""

    rows: tuple[Mapping[str, Any], ...]
    product_identity_hash: str


def first_run_baselines_requirement() -> AnalysisDataProductRequirement:
    """Return the exact requirement for the adopted empirical baseline product."""

    return AnalysisDataProductRequirement(
        role=PRODUCT_ROLE,
        product_schema_id=PRODUCT_SCHEMA_ID,
        logical_name=PRODUCT_LOGICAL_NAME,
        exact_product_schema_version=PRODUCT_SCHEMA_VERSION,
        product_identity_hash=PRODUCT_IDENTITY_HASH,
        artifact_sha256=PAYLOAD_SHA256,
    )


@lru_cache(maxsize=1)
def load_first_run_baselines() -> CrossMethodFirstRunBaselines:
    """Load baseline rows with fail-closed envelope and payload validation."""

    product = load_data_product(REPO_ROOT / PRODUCT_RELPATH, first_run_baselines_requirement())
    payload_path = REPO_ROOT / PAYLOAD_RELPATH
    payload_bytes = payload_path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != PAYLOAD_SHA256:
        raise DataProductError(
            "cross-method first-run baseline payload hash mismatch",
            kind="Mismatch",
            mismatch_class="artifact-byte-hash",
        )
    payload = json.loads(payload_bytes)
    if payload.get("schema_id") != PAYLOAD_SCHEMA_ID:
        raise DataProductError(
            "cross-method first-run baseline payload schema mismatch",
            kind="Mismatch",
            mismatch_class="schema",
        )
    if payload.get("schema_version") != PAYLOAD_SCHEMA_VERSION:
        raise DataProductError(
            "cross-method first-run baseline payload version mismatch",
            kind="Mismatch",
            mismatch_class="schema-version",
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise DataProductError(
            "cross-method first-run baseline payload has no rows",
            kind="Mismatch",
            mismatch_class="schema",
        )
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise DataProductError(
                "cross-method first-run baseline row is not a mapping",
                kind="Mismatch",
                mismatch_class="schema",
            )
        normalized.append(
            MappingProxyType(
                {
                    **row,
                    "g_sd": float("nan") if row.get("g_sd") is None else float(row["g_sd"]),
                    "g_sp": float("nan") if row.get("g_sp") is None else float(row["g_sp"]),
                }
            )
        )
    return CrossMethodFirstRunBaselines(
        rows=tuple(normalized),
        product_identity_hash=str(product.product_identity_hash),
    )


register_data_product_identity(
    role=PRODUCT_ROLE,
    product_schema_id=PRODUCT_SCHEMA_ID,
    product_schema_version=PRODUCT_SCHEMA_VERSION,
    logical_name=PRODUCT_LOGICAL_NAME,
    requirement_factory=first_run_baselines_requirement,
    document_relpath=PRODUCT_RELPATH,
)
