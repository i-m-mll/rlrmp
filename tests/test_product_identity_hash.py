"""Live contract tests for governed data-product identity hashes (issue c223bb8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rlrmp.data_products.broad_epsilon import (
    BROAD_EPSILON_PRODUCT_IDENTITY_HASH,
    BROAD_EPSILON_PRODUCT_PATH,
    BROAD_EPSILON_PRODUCT_ROLE,
    broad_epsilon_data_product_requirement,
    load_broad_epsilon_anchors,
)
from rlrmp.data_products.calibration import (
    CALIBRATION_PRODUCT_IDENTITY_HASH,
    CALIBRATION_PRODUCT_LOGICAL_NAME,
    CALIBRATION_PRODUCT_PATH,
    CALIBRATION_PRODUCT_RELPATH,
    CALIBRATION_PRODUCT_ROLE,
    CALIBRATION_PRODUCT_SCHEMA_ID,
    CALIBRATION_PRODUCT_SCHEMA_VERSION,
    calibration_data_product_requirement,
    load_open_loop_calibration,
)
from rlrmp.data_products.envelope import DataProductError, load_data_product
from rlrmp.data_products.registry import (
    register_data_product_identity,
    registered_data_product_identities,
)
from rlrmp.paths import REPO_ROOT


pytestmark = pytest.mark.feedbax_contract


def test_calibration_product_identity_hash_loader_matches_pin() -> None:
    load_open_loop_calibration.cache_clear()
    calibration = load_open_loop_calibration()
    assert calibration.product_identity_hash == CALIBRATION_PRODUCT_IDENTITY_HASH
    assert calibration_data_product_requirement().product_identity_hash == (
        CALIBRATION_PRODUCT_IDENTITY_HASH
    )


def test_broad_epsilon_product_identity_hash_loader_matches_pin() -> None:
    load_broad_epsilon_anchors.cache_clear()
    anchors = load_broad_epsilon_anchors()
    assert anchors.product_identity_hash == BROAD_EPSILON_PRODUCT_IDENTITY_HASH
    assert broad_epsilon_data_product_requirement().product_identity_hash == (
        BROAD_EPSILON_PRODUCT_IDENTITY_HASH
    )


def test_calibration_product_identity_hash_tamper_fails_closed(tmp_path: Path) -> None:
    tampered = _tampered_copy(
        CALIBRATION_PRODUCT_PATH,
        tmp_path,
        lambda payload: payload["parameters"].__setitem__("reference_reach_m", 0.16),
    )

    with pytest.raises(DataProductError, match="product_identity|identity validation"):
        load_data_product(tampered, calibration_data_product_requirement())


def test_broad_epsilon_product_identity_hash_tamper_fails_closed(tmp_path: Path) -> None:
    tampered = _tampered_copy(
        BROAD_EPSILON_PRODUCT_PATH,
        tmp_path,
        lambda payload: payload["parameters"]["levels"]["moderate"].__setitem__(
            "delta_v_percent",
            4.25,
        ),
    )

    with pytest.raises(DataProductError, match="product_identity|identity validation"):
        load_data_product(tampered, broad_epsilon_data_product_requirement())


@pytest.mark.parametrize("colliding_key", ["role", "product_schema_id", "logical_name"])
def test_data_product_registry_duplicate_role_fails_closed(colliding_key: str) -> None:
    register_data_product_identity(
        role=CALIBRATION_PRODUCT_ROLE,
        product_schema_id=CALIBRATION_PRODUCT_SCHEMA_ID,
        product_schema_version=CALIBRATION_PRODUCT_SCHEMA_VERSION,
        logical_name=CALIBRATION_PRODUCT_LOGICAL_NAME,
        requirement_factory=calibration_data_product_requirement,
        document_relpath=CALIBRATION_PRODUCT_RELPATH,
    )

    candidate = {
        "role": f"{CALIBRATION_PRODUCT_ROLE}_copy",
        "product_schema_id": f"{CALIBRATION_PRODUCT_SCHEMA_ID}.copy",
        "product_schema_version": f"{CALIBRATION_PRODUCT_SCHEMA_VERSION}.copy",
        "logical_name": f"{CALIBRATION_PRODUCT_LOGICAL_NAME}_copy",
        "requirement_factory": calibration_data_product_requirement,
        "document_relpath": "results/ea6ccb4/data_products/copy.json",
    }
    candidate[colliding_key] = {
        "role": CALIBRATION_PRODUCT_ROLE,
        "product_schema_id": CALIBRATION_PRODUCT_SCHEMA_ID,
        "logical_name": CALIBRATION_PRODUCT_LOGICAL_NAME,
    }[colliding_key]

    with pytest.raises(ValueError) as exc_info:
        register_data_product_identity(**candidate)

    message = str(exc_info.value)
    assert colliding_key in message
    assert CALIBRATION_PRODUCT_ROLE in message
    assert candidate["role"] in message


def test_registered_data_products_resolve_fail_closed() -> None:
    identities = registered_data_product_identities()
    assert {CALIBRATION_PRODUCT_ROLE, BROAD_EPSILON_PRODUCT_ROLE} <= set(identities)

    for identity in identities.values():
        path = REPO_ROOT / identity.document_relpath
        assert path.is_file()
        product = load_data_product(path, identity.requirement_factory())
        assert product.role == identity.role
        assert product.product_schema_id == identity.product_schema_id
        assert product.product_schema_version == identity.product_schema_version
        assert product.logical_name == identity.logical_name


def _tampered_copy(
    source: Path,
    tmp_path: Path,
    mutate: Any,
) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutate(payload)
    destination = tmp_path / source.name
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
