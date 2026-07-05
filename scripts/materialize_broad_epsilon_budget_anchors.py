#!/usr/bin/env python
"""Check or regenerate the broad-epsilon budget-anchors data product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedbax.contracts.extraction import (
    DataProductDrift,
    ExtractionProductIdentityMismatch,
    materialize_extraction_product,
    verify_extraction_product,
)
from feedbax.contracts.manifest import (
    AnalysisDataProduct,
    analysis_data_product_identity_hash,
)

from rlrmp.data_products.broad_epsilon import (
    BROAD_EPSILON_PRODUCT_IDENTITY_HASH,
    BROAD_EPSILON_PRODUCT_PATH,
    _broad_epsilon_extraction_spec,
    broad_epsilon_data_product_requirement,
)
from rlrmp.data_products.envelope import DataProductError, load_data_product, write_data_product
from rlrmp.paths import REPO_ROOT


class MaterializationMismatch(RuntimeError):
    """Raised when regenerated broad-epsilon product state fails a check."""

    def __init__(self, message: str, *, mismatch_class: str) -> None:
        super().__init__(message)
        self.mismatch_class = mismatch_class


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the regenerated product; default is check-only",
    )
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    if args.output_path is not None and not args.write:
        parser.error("--output-path is only valid with --write")

    mode = "write" if args.write else "check"
    path = args.output_path or BROAD_EPSILON_PRODUCT_PATH
    try:
        spec = _broad_epsilon_extraction_spec()
        product = _materialize_product()
        identity_hash = _verify_identity_pin(product)
        if args.write:
            write_data_product(product, path)
        _verify_serialized_bytes(product, path)
        persisted = load_data_product(path, broad_epsilon_data_product_requirement())
        try:
            verify_extraction_product(spec, persisted, REPO_ROOT)
        except DataProductDrift as exc:
            raise DataProductError(
                f"broad-epsilon product no longer matches analytical sources: {exc}",
                kind="Mismatch",
                mismatch_class="analytical-source-drift",
            ) from exc
    except DataProductError as exc:
        _print_summary(
            mode=mode,
            path=path,
            result="mismatch",
            mismatch_class=exc.mismatch_class or exc.kind,
            message=str(exc),
        )
        return 1
    except MaterializationMismatch as exc:
        _print_summary(
            mode=mode,
            path=path,
            result="mismatch",
            mismatch_class=exc.mismatch_class,
            message=str(exc),
        )
        return 1

    _print_summary(
        mode=mode,
        path=path,
        result="ok",
        identity_hash=identity_hash,
    )
    return 0


def _materialize_product() -> AnalysisDataProduct:
    try:
        return materialize_extraction_product(_broad_epsilon_extraction_spec(), REPO_ROOT)
    except ExtractionProductIdentityMismatch as exc:
        raise DataProductError(
            f"broad-epsilon extraction product identity mismatch: {exc}",
            kind="Mismatch",
            mismatch_class="product-identity",
        ) from exc


def _verify_identity_pin(product: AnalysisDataProduct) -> str:
    identity_hash = analysis_data_product_identity_hash(product)
    if identity_hash != BROAD_EPSILON_PRODUCT_IDENTITY_HASH:
        raise MaterializationMismatch(
            "regenerated broad-epsilon product_identity_hash does not match the "
            f"pinned value: regenerated={identity_hash!r}, "
            f"pinned={BROAD_EPSILON_PRODUCT_IDENTITY_HASH!r}",
            mismatch_class="product-identity",
        )
    return str(identity_hash)


def _verify_serialized_bytes(product: AnalysisDataProduct, path: Path) -> None:
    expected = _serialized_product_bytes(product)
    if not path.is_file():
        raise MaterializationMismatch(
            f"tracked broad-epsilon product is missing: {path}",
            mismatch_class="missing-product",
        )
    actual = path.read_bytes()
    if actual != expected:
        raise MaterializationMismatch(
            f"regenerated broad-epsilon product is not byte-identical to {path}",
            mismatch_class="product-bytes",
        )


def _serialized_product_bytes(product: AnalysisDataProduct) -> bytes:
    return (product.model_dump_json(indent=2, exclude_none=True) + "\n").encode("utf-8")


def _print_summary(
    *,
    mode: str,
    path: Path,
    result: str,
    identity_hash: str | None = None,
    mismatch_class: str | None = None,
    message: str | None = None,
) -> None:
    summary = {
        "mode": mode,
        "path": str(path),
        "result": result,
    }
    if identity_hash is not None:
        summary["identity_hash"] = identity_hash
    if mismatch_class is not None:
        summary["mismatch_class"] = mismatch_class
    if message is not None:
        summary["message"] = message
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
