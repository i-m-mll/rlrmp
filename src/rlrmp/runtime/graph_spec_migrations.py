"""Compatibility helpers for historical Feedbax graph payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION,
    LEGACY_GRAPH_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.migrations import migrate_graph_spec


_RLRMP_LEGACY_GRAPH_SCHEMA_VERSION_ALIAS = "feedbax.spec.graph.v1"


def migrate_feedbax_graph_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a Feedbax GraphSpec payload migrated to the current schema."""

    graph_payload = _normalize_legacy_graph_schema_aliases(payload)
    migrated = migrate_graph_spec(graph_payload).payload
    migrated.setdefault("schema_id", GRAPH_SPEC_SCHEMA_ID)
    migrated["schema_version"] = GRAPH_SPEC_SCHEMA_VERSION
    return migrated


def _normalize_legacy_graph_schema_aliases(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "schema_version" and value == _RLRMP_LEGACY_GRAPH_SCHEMA_VERSION_ALIAS:
            normalized[key] = LEGACY_GRAPH_SPEC_SCHEMA_VERSION
        elif isinstance(value, Mapping):
            normalized[key] = _normalize_legacy_graph_schema_aliases(value)
        elif isinstance(value, list):
            normalized[key] = [
                _normalize_legacy_graph_schema_aliases(item)
                if isinstance(item, Mapping)
                else item
                for item in value
            ]
        else:
            normalized[key] = value
    return normalized
