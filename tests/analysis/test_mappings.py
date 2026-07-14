"""Semantic tests for shared analysis mapping coercion."""

from __future__ import annotations

from rlrmp.mappings import as_mapping


def test_as_mapping_preserves_mapping_identity() -> None:
    payload = {"value": 1}

    assert as_mapping(payload) is payload


def test_as_mapping_coerces_non_mappings_to_empty() -> None:
    assert as_mapping(None) == {}
    assert as_mapping([("value", 1)]) == {}
