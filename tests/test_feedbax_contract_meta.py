"""Meta-tests for the Feedbax contract gate manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
from feedbax.analysis.validation import RecipeValidationError, validate_analysis_recipe

from feedbax.contracts.manifest import load_manifest
from feedbax.testing import (
    ContractFamily,
    ContractSuiteManifest,
    assert_live_family_counts,
    assert_negative_canaries_collected,
    collect_contract_nodeids,
    load_suite_manifest,
)
from rlrmp.runtime.feedbax_contract_versions import (
    PENDING_VERSION_PINS,
    SUPPORTED_FEEDBAX_MANIFEST_SCHEMA_VERSIONS,
    SUPPORTED_GRAPH_SPEC_VERSIONS,
    SUPPORTED_RUN_STATUS_CHECKPOINT_SCHEMA_VERSIONS,
    SUPPORTED_TRAINING_RUN_SPEC_VERSIONS,
    assert_supported_graph_spec_version,
)
from rlrmp.runtime.spec_migrations import RUN_SPEC_SCHEMA_VERSION
from rlrmp.runtime.spec_migrations import ensure_rlrmp_spec_families


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_MANIFEST_PATH = REPO_ROOT / "ci" / "feedbax-contract-suite.toml"
FIXTURE_MANIFEST = (
    REPO_ROOT
    / "results"
    / "9455785"
    / "manifests"
    / "training_runs"
    / "feedbax_training_run_rlrmp_artifact_normalization_fixture.json"
)


def test_feedbax_contract_suite_manifest_collects_live_families() -> None:
    manifest = load_suite_manifest(SUITE_MANIFEST_PATH, marker="feedbax_contract")
    collection = collect_contract_nodeids(rootdir=REPO_ROOT, marker=manifest.marker)
    assert_live_family_counts(manifest, collection.nodeids)
    assert_negative_canaries_collected(manifest, collection.nodeids)


def test_feedbax_contract_suite_has_no_skip_or_non_strict_xfail_marks() -> None:
    collect_contract_nodeids(rootdir=REPO_ROOT, marker="feedbax_contract", extra_args=("-ra",))


def test_suite_manifest_negative_canary_rejects_zero_live_family() -> None:
    manifest = ContractSuiteManifest(
        marker="feedbax_contract",
        families=(ContractFamily("empty", "live", "tests/does_not_exist.py::", 1),),
    )

    with pytest.raises(AssertionError, match="empty"):
        assert_live_family_counts(manifest, ["tests/other.py::test_example"])


def test_analysis_recipe_negative_canary_rejects_bad_signature() -> None:
    def bad_recipe(run_spec, root):  # noqa: ANN001, ANN202
        return None

    with pytest.raises(RecipeValidationError):
        validate_analysis_recipe("rlrmp.bad", bad_recipe)


def test_version_pin_negative_canary_rejects_unsupported_graph_version() -> None:
    assert "0.0.0" not in SUPPORTED_GRAPH_SPEC_VERSIONS


def test_manifest_fixture_negative_canary_rejects_wrong_run_spec_version() -> None:
    ensure_rlrmp_spec_families()
    manifest = load_manifest(FIXTURE_MANIFEST)
    assert manifest.training_spec is not None
    assert manifest.training_spec.schema_version == RUN_SPEC_SCHEMA_VERSION
    assert manifest.training_spec.schema_version != "rlrmp.run_spec.v0"


def test_feedbax_contract_version_pins_cover_live_contracts() -> None:
    assert SUPPORTED_GRAPH_SPEC_VERSIONS == ("1.0.0",)
    assert SUPPORTED_FEEDBAX_MANIFEST_SCHEMA_VERSIONS == ("feedbax.manifest.v1",)
    assert SUPPORTED_TRAINING_RUN_SPEC_VERSIONS == (RUN_SPEC_SCHEMA_VERSION,)
    assert SUPPORTED_RUN_STATUS_CHECKPOINT_SCHEMA_VERSIONS == (1,)
    assert PENDING_VERSION_PINS == {
        "data_product_payload": (
            "owned by issue 108b4d3/product identity and follow-on data-product work"
        ),
    }
    assert_supported_graph_spec_version("1.0.0")
    with pytest.raises(ValueError, match="Unsupported Feedbax GraphSpec version"):
        assert_supported_graph_spec_version("0.0.0")
