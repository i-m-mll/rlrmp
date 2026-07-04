"""Live contract tests for governed data-product identity hashes (issue c223bb8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from feedbax.contracts.expressions import expression_hash
from feedbax.contracts.extraction import (
    DataProductDrift,
    ExtractionProductSpec,
    materialize_extraction_product,
    verify_extraction_product,
)
from feedbax.contracts.manifest import AnalysisDataProduct

from rlrmp.data_products.broad_epsilon import (
    BROAD_EPSILON_EXTRACTION_SPEC_PATH,
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


def test_broad_epsilon_extraction_round_trip_preserves_tracked_bytes(
    tmp_path: Path,
) -> None:
    spec = _broad_epsilon_extraction_spec()
    product = materialize_extraction_product(spec, REPO_ROOT)
    destination = tmp_path / BROAD_EPSILON_PRODUCT_PATH.name
    destination.write_bytes(_serialized_product_bytes(product))

    assert destination.read_bytes() == BROAD_EPSILON_PRODUCT_PATH.read_bytes()
    assert product.product_identity_hash == BROAD_EPSILON_PRODUCT_IDENTITY_HASH


def test_broad_epsilon_extraction_verify_passes_and_drift_is_typed() -> None:
    spec = _broad_epsilon_extraction_spec()
    tracked_payload = json.loads(BROAD_EPSILON_PRODUCT_PATH.read_text(encoding="utf-8"))

    verified = verify_extraction_product(spec, tracked_payload, REPO_ROOT)
    assert verified.product_identity_hash == BROAD_EPSILON_PRODUCT_IDENTITY_HASH

    mutated_payload = json.loads(BROAD_EPSILON_PRODUCT_PATH.read_text(encoding="utf-8"))
    mutated_payload["parameters"]["levels"]["moderate"]["delta_v_percent"] = 0.0
    mutated_payload.pop("product_identity_hash")
    mutated = AnalysisDataProduct.model_validate(mutated_payload)
    with pytest.raises(DataProductDrift) as excinfo:
        verify_extraction_product(spec, mutated, REPO_ROOT)

    drift = excinfo.value
    assert drift.output_path == "levels.moderate.delta_v_percent"
    assert drift.source_uri == "results/cb98e58/notes/analytical_game_card_manifest.json"
    assert drift.persisted_value == 0.0
    assert drift.source_value == 4.041729916548296


def test_broad_epsilon_extraction_spec_document_identity_is_stable() -> None:
    spec = _broad_epsilon_extraction_spec()

    assert spec.expected_identity_hash == BROAD_EPSILON_PRODUCT_IDENTITY_HASH
    assert [source.uri for source in spec.sources] == [
        "results/cb98e58/notes/analytical_game_card_manifest.json",
        "results/a7dad8a/notes/adversary_equivalence_manifest.json",
    ]
    assert {field.output_path: expression_hash(field.query) for field in spec.fields} == {
        "levels.moderate.gamma_factor": (
            "45713b8134841a432ae1b2ff75ec1da62825479bbd3602316c57e546838ffca6"
        ),
        "levels.moderate.closed_loop_epsilon_energy_15cm": (
            "87eff684f370e22c656d1acabf91e2601b3a7891db293b2dfaffd655d995bb90"
        ),
        "levels.moderate.closed_loop_epsilon_l2_15cm": (
            "a8c5fda106b696e3fe4be0ac8ded02f9ec7ec13b7f4315d7b5778bf1eab5b255"
        ),
        "levels.moderate.delta_v_percent": (
            "19e58577aa252faed6478c62c0a3fef6f5a26460d7e9d3fedb104560720e1ff2"
        ),
        "levels.strong.gamma_factor": (
            "2908747c5991a7352b43dfe75a3d6928d3a30a39c4cbe40f7f6b27774a41d14e"
        ),
        "levels.strong.closed_loop_epsilon_energy_15cm": (
            "8e44c1ef42da5dec61f687e44df2531c76c0fed678192958c59a7e48fd9319d2"
        ),
        "levels.strong.closed_loop_epsilon_l2_15cm": (
            "c277b578e53a3ed1fcdb37ff1dddcdaebad5053c562823dac15786671a4d030f"
        ),
        "levels.strong.delta_v_percent": (
            "d93e299bdf1b8d51cbf82e2abe98dfac6e986bb3a8fda77fc4f38de6a7c9b142"
        ),
    }


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


def _broad_epsilon_extraction_spec() -> ExtractionProductSpec:
    return ExtractionProductSpec.model_validate_json(
        BROAD_EPSILON_EXTRACTION_SPEC_PATH.read_text(encoding="utf-8")
    )


def _serialized_product_bytes(product: AnalysisDataProduct) -> bytes:
    return (product.model_dump_json(indent=2, exclude_none=True) + "\n").encode("utf-8")


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
