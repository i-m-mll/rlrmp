"""Compatibility helpers for historical Feedbax graph payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from feedbax.contracts.graph import (
    GRAPH_SPEC_SCHEMA_ID,
    GRAPH_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.migrations import migrate_graph_spec


def migrate_feedbax_graph_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a Feedbax GraphSpec payload migrated to the current schema."""

    migrated = migrate_graph_spec(payload).payload
    migrated.setdefault("schema_id", GRAPH_SPEC_SCHEMA_ID)
    migrated["schema_version"] = GRAPH_SPEC_SCHEMA_VERSION
    return migrated
