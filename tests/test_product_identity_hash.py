"""Live contract tests for governed data-product identity hashes (issue c223bb8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rlrmp.data_products.broad_epsilon import (
    BROAD_EPSILON_PRODUCT_IDENTITY_HASH,
    BROAD_EPSILON_PRODUCT_PATH,
    broad_epsilon_data_product_requirement,
    load_broad_epsilon_anchors,
)
from rlrmp.data_products.calibration import (
    CALIBRATION_PRODUCT_IDENTITY_HASH,
    CALIBRATION_PRODUCT_PATH,
    calibration_data_product_requirement,
    load_open_loop_calibration,
)
from rlrmp.data_products.envelope import DataProductError, load_data_product


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
