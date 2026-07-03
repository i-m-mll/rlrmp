"""Fail-closed loader for RLRMP data products carried on the Feedbax envelope.

Generated or adopted empirical data (perturbation calibration tables, budget
anchors) is persisted as a tracked :class:`AnalysisDataProduct` document rather
than as a source-code constant. Runtime code loads it here and validates the
typed product identity before use.

Validation is fail-closed. Loading raises :class:`DataProductError` on:

* a missing product document,
* a document whose stored ``product_identity_hash`` disagrees with its semantic
  envelope (tamper / value drift),
* a product/schema id mismatch against the caller's requirement,
* a product schema-version mismatch,
* a role mismatch,
* a stale ``descriptor_basis_hash``,
* a pinned ``product_identity_hash`` mismatch, or
* a pinned artifact byte-hash mismatch.

The requirement is expressed with Feedbax's
:class:`AnalysisDataProductRequirement`, and the mismatch semantics mirror the
Feedbax provider's data-product resolution so consumers fail closed the same way
whether the product is resolved at run-production time or loaded here.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from feedbax.contracts.graph import AnalysisDataProductRequirement
from feedbax.contracts.manifest import (
    AnalysisDataProduct,
    analysis_data_product_identity_hash,
)

__all__ = [
    "DataProductError",
    "load_data_product",
    "read_data_product",
    "validate_data_product",
]


class DataProductError(RuntimeError):
    """Raised when a required data product is missing or fails identity validation."""

    def __init__(self, message: str, *, kind: str, mismatch_class: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.mismatch_class = mismatch_class


def read_data_product(path: Path | str) -> AnalysisDataProduct:
    """Read and structurally validate a persisted analysis data product.

    The Feedbax model validator recomputes ``product_identity_hash`` from the
    semantic envelope, so a document whose stored hash no longer matches its
    values fails here (value drift / tamper detection).
    """

    path = Path(path)
    if not path.is_file():
        raise DataProductError(
            f"required analysis data product is missing: no file at {path}",
            kind="Missing",
            mismatch_class="missing-product",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataProductError(
            f"required analysis data product could not be read: {path}: {exc}",
            kind="Missing",
            mismatch_class="unreadable-product",
        ) from exc
    try:
        product = AnalysisDataProduct.model_validate(payload)
    except PydanticValidationError as exc:
        raise DataProductError(
            f"analysis data product at {path} failed schema/identity validation: {exc}",
            kind="Mismatch",
            mismatch_class="product-identity",
        ) from exc
    return product


def validate_data_product(
    product: AnalysisDataProduct,
    requirement: AnalysisDataProductRequirement,
    *,
    source: str = "<product>",
) -> None:
    """Raise :class:`DataProductError` unless ``product`` satisfies ``requirement``.

    Covers missing/mismatch failure modes: product schema id, product schema
    version, role, descriptor basis, pinned product-identity hash, and pinned
    artifact byte hash.
    """

    # Re-derive the identity hash to catch any post-construction mutation.
    recomputed = analysis_data_product_identity_hash(product)
    if product.product_identity_hash != recomputed:
        raise DataProductError(
            f"analysis data product {source} has a stale product_identity_hash: "
            f"stored={product.product_identity_hash!r}, computed={recomputed!r}",
            kind="Mismatch",
            mismatch_class="product-identity",
        )

    if requirement.logical_name is not None and product.logical_name != requirement.logical_name:
        raise DataProductError(
            f"analysis data product {source} has the wrong logical_name: "
            f"required={requirement.logical_name!r}, actual={product.logical_name!r}",
            kind="Missing",
            mismatch_class="missing-product",
        )
    if product.product_schema_id != requirement.product_schema_id:
        raise DataProductError(
            f"analysis data product {source} has the wrong product schema: "
            f"required={requirement.product_schema_id!r}, "
            f"actual={product.product_schema_id!r}",
            kind="Mismatch",
            mismatch_class="schema",
        )
    if not _schema_version_satisfies(
        product.product_schema_version,
        exact=requirement.exact_product_schema_version,
        minimum=requirement.min_product_schema_version,
        maximum=requirement.max_product_schema_version,
    ):
        raise DataProductError(
            f"analysis data product {source} has an incompatible product schema "
            f"version: required exact={requirement.exact_product_schema_version!r}, "
            f"actual={product.product_schema_version!r}",
            kind="Mismatch",
            mismatch_class="schema-version",
        )
    if product.role != requirement.role:
        raise DataProductError(
            f"analysis data product {source} has the wrong role: "
            f"required={requirement.role!r}, actual={product.role!r}",
            kind="Mismatch",
            mismatch_class="wrong-role",
        )
    if (
        requirement.descriptor_basis_hash is not None
        and product.descriptor_basis_hash != requirement.descriptor_basis_hash
    ):
        raise DataProductError(
            f"analysis data product {source} has the wrong descriptor_basis_hash: "
            f"required={requirement.descriptor_basis_hash!r}, "
            f"actual={product.descriptor_basis_hash!r}",
            kind="Mismatch",
            mismatch_class="wrong-basis",
        )
    if (
        requirement.product_identity_hash is not None
        and product.product_identity_hash != requirement.product_identity_hash
    ):
        raise DataProductError(
            f"analysis data product {source} has the wrong product_identity_hash: "
            f"required={requirement.product_identity_hash!r}, "
            f"actual={product.product_identity_hash!r}",
            kind="Mismatch",
            mismatch_class="product-identity",
        )
    if requirement.artifact_sha256 is not None:
        artifact_hashes = {artifact.sha256 for artifact in product.artifacts}
        if requirement.artifact_sha256 not in artifact_hashes:
            raise DataProductError(
                f"analysis data product {source} does not carry the pinned artifact "
                f"hash {requirement.artifact_sha256!r}",
                kind="Mismatch",
                mismatch_class="artifact-byte-hash",
            )


def load_data_product(
    path: Path | str,
    requirement: AnalysisDataProductRequirement,
) -> AnalysisDataProduct:
    """Read and validate a persisted analysis data product, failing closed."""

    product = read_data_product(path)
    validate_data_product(product, requirement, source=str(path))
    return product


def _schema_version_satisfies(
    version: str,
    *,
    exact: str | None,
    minimum: str | None,
    maximum: str | None,
) -> bool:
    if exact is not None:
        return version == exact
    if minimum is not None and _schema_version_compare(version, minimum) < 0:
        return False
    if maximum is not None and _schema_version_compare(version, maximum) > 0:
        return False
    return True


def _schema_version_compare(left: str, right: str) -> int:
    left_prefix, left_number = _schema_version_parts(left)
    right_prefix, right_number = _schema_version_parts(right)
    if left_prefix != right_prefix:
        return -1 if left < right else 1
    return (left_number > right_number) - (left_number < right_number)


def _schema_version_parts(version: str) -> tuple[str, int]:
    prefix, separator, suffix = version.rpartition(".v")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    return version, 0
