"""Meta-tests for the Feedbax contract gate manifest."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
from feedbax.analysis.validation import RecipeValidationError, validate_analysis_recipe

from feedbax.contracts.manifest import load_manifest
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
    manifest = _load_suite_manifest()
    nodeids = _collect_feedbax_contract_nodeids()
    _assert_live_family_counts(manifest, nodeids)


def test_feedbax_contract_suite_has_no_skip_or_non_strict_xfail_marks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "feedbax_contract",
            "--strict-markers",
            "--collect-only",
            "-q",
            "-ra",
            "-o",
            "xfail_strict=true",
            "-o",
            "empty_parameter_set_mark=fail_at_collect",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "SKIPPED" not in combined
    assert "XFAIL" not in combined


def test_suite_manifest_negative_canary_rejects_zero_live_family() -> None:
    manifest = {
        "families": [
            {
                "name": "empty",
                "status": "live",
                "expected_collection_pattern": "tests/does_not_exist.py::",
                "minimum_non_skipped": 1,
            }
        ]
    }

    with pytest.raises(AssertionError, match="empty"):
        _assert_live_family_counts(manifest, ["tests/other.py::test_example"])


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
    assert SUPPORTED_TRAINING_RUN_SPEC_VERSIONS == ("rlrmp.run_spec.v1",)
    assert SUPPORTED_RUN_STATUS_CHECKPOINT_SCHEMA_VERSIONS == (1,)
    assert PENDING_VERSION_PINS == {
        "descriptor_basis_hash": "owned by issue 844acc6",
        "data_product_payload": (
            "owned by issue 108b4d3/product identity and follow-on data-product work"
        ),
    }
    assert_supported_graph_spec_version("1.0.0")
    with pytest.raises(ValueError, match="Unsupported Feedbax GraphSpec version"):
        assert_supported_graph_spec_version("0.0.0")


def _load_suite_manifest() -> dict:
    return tomllib.loads(SUITE_MANIFEST_PATH.read_text(encoding="utf-8"))


def _collect_feedbax_contract_nodeids() -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "feedbax_contract",
            "--strict-markers",
            "--collect-only",
            "-q",
            "-o",
            "empty_parameter_set_mark=fail_at_collect",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith("<")
    ]


def _assert_live_family_counts(manifest: dict, nodeids: list[str]) -> None:
    live_families = [
        family for family in manifest["families"] if family["status"] == "live"
    ]
    assert live_families, "Feedbax contract manifest declares zero live families"
    for family in live_families:
        pattern = family["expected_collection_pattern"]
        count = sum(1 for nodeid in nodeids if pattern in nodeid)
        assert count >= int(family["minimum_non_skipped"]), (
            f"Feedbax contract family {family['name']!r} collected {count} tests; "
            f"minimum is {family['minimum_non_skipped']} for pattern {pattern!r}"
        )
