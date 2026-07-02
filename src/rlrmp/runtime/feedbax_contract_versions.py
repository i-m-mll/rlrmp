"""Version pins for RLRMP contracts consumed by the Feedbax drift gate."""

from __future__ import annotations

from feedbax.contracts.manifest import SCHEMA_VERSION as FEEDBAX_MANIFEST_SCHEMA_VERSION

from rlrmp.model.feedbax_graph import SUPPORTED_GRAPH_SPEC_VERSIONS
from rlrmp.runtime.spec_migrations import RUN_SPEC_SCHEMA_VERSION


SUPPORTED_FEEDBAX_MANIFEST_SCHEMA_VERSIONS = (FEEDBAX_MANIFEST_SCHEMA_VERSION,)
SUPPORTED_TRAINING_RUN_SPEC_VERSIONS = (RUN_SPEC_SCHEMA_VERSION,)
SUPPORTED_RUN_STATUS_CHECKPOINT_SCHEMA_VERSIONS = (1,)

PENDING_VERSION_PINS = {
    "descriptor_basis_hash": "owned by issue 844acc6",
    "data_product_payload": "owned by issue 108b4d3/product identity and follow-on data-product work",
}


def assert_supported_graph_spec_version(version: str) -> None:
    """Raise when a Feedbax GraphSpec version is outside the pinned gate set."""

    if version not in SUPPORTED_GRAPH_SPEC_VERSIONS:
        raise ValueError(
            f"Unsupported Feedbax GraphSpec version {version!r}; "
            f"supported versions: {SUPPORTED_GRAPH_SPEC_VERSIONS!r}"
        )


__all__ = [
    "PENDING_VERSION_PINS",
    "SUPPORTED_FEEDBAX_MANIFEST_SCHEMA_VERSIONS",
    "SUPPORTED_GRAPH_SPEC_VERSIONS",
    "SUPPORTED_RUN_STATUS_CHECKPOINT_SCHEMA_VERSIONS",
    "SUPPORTED_TRAINING_RUN_SPEC_VERSIONS",
    "assert_supported_graph_spec_version",
]
