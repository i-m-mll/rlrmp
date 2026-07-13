"""Meta-tests for the Feedbax contract gate manifest."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib
from typing import Any

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
PINNED_FAMILY_HISTORY = frozenset(
    {
        "analysis_recipe_contract",
        "graph_spec_contract",
        "artifact_manifest_normalization",
        "import_boundary",
        "suite_manifest_meta",
        "graph_sidecar_audit",
        "converted_sidecar_loadability",
        "tracked_results_reference_integrity",
        "retired_id_scan",
        "retired_results_script_imports",
        "descriptor_conformance_canary",
        "model_export_parity",
        "descriptor_basis_hash",
        "feedback_descriptor_scan",
        "product_identity_hash",
        "analysis_write_custody",
        "analysis_eval_dependency",
        "write_surface",
        "reaccretion_ratchet",
        "experiment_kpi_gates",
        "tracked_materialization_gate",
        "native_executor_deletion_gate",
        "generated_data_constant_scan",
        "data_in_code_scan",
        "import_time_jax_config_scan",
        "defaults_schema_ownership_scan",
        "lane_b_terminal_gate",
        "lane_c_terminal_gate",
        "pipeline_contract_native",
        "perturbation_bank_strangler",
        "training_config_flat_key_scan",
        "declarative_figure_stock",
        "six_cell_figure_migration",
        "declarative_profile_stock",
        "delayed_profile_figure_contract",
        "prego_figure_migration",
        "response_norm_figure_contract",
        "steady_state_figure_contract",
        "literature_replication_figure_contract",
        "scalar_diagnostic_figure_contract",
        "movement_ramp_figure_contract",
        "sisu_spectrum_figure_contract",
        "gru_postrun_figure_contract",
        "nominal_profile_figure_contract",
        "moderate_calibrated_figure_contract",
        "stabilization_response_figure_contract",
    }
)


def test_feedbax_contract_suite_manifest_collects_live_families() -> None:
    manifest = load_suite_manifest(SUITE_MANIFEST_PATH, marker="feedbax_contract")
    collection = collect_contract_nodeids(rootdir=REPO_ROOT, marker=manifest.marker)
    assert_live_family_counts(manifest, collection.nodeids)
    assert_negative_canaries_collected(manifest, collection.nodeids)


def test_contract_family_lifecycle_metadata_requires_due_migrations_to_retire() -> None:
    payload = tomllib.loads(SUITE_MANIFEST_PATH.read_text(encoding="utf-8"))

    _assert_contract_family_lifecycle(payload, repo_root=REPO_ROOT)


def test_feedbax_contract_suite_has_no_skip_or_non_strict_xfail_marks() -> None:
    collect_contract_nodeids(rootdir=REPO_ROOT, marker="feedbax_contract", extra_args=("-ra",))


def test_suite_manifest_negative_canary_rejects_zero_live_family() -> None:
    manifest = ContractSuiteManifest(
        marker="feedbax_contract",
        families=(ContractFamily("empty", "live", "tests/does_not_exist.py::", 1),),
    )

    with pytest.raises(AssertionError, match="empty"):
        assert_live_family_counts(manifest, ["tests/other.py::test_example"])


def test_lifecycle_negative_canary_rejects_satisfied_registered_migration(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "inventory.toml"
    inventory.write_text("legacy = []\n", encoding="utf-8")
    payload = {
        "families": [
            {
                "name": "successor",
                "owner": "1234abc",
                "status": "live",
                "lifecycle": "permanent",
            },
            {
                "name": "due_migration",
                "owner": "7654abc",
                "status": "live",
                "lifecycle": "migration",
                "successor": "successor",
                "exit_predicate": {
                    "kind": "toml_arrays_empty",
                    "path": "inventory.toml",
                    "keys": ["legacy"],
                },
            },
        ],
        "retired_families": [],
        "family_history": ["successor", "due_migration"],
    }

    with pytest.raises(AssertionError, match="must retire.*due_migration"):
        _assert_contract_family_lifecycle(
            payload,
            repo_root=tmp_path,
            pinned_history=frozenset({"successor", "due_migration"}),
        )


def test_lifecycle_negative_canary_rejects_silent_history_deletion(tmp_path: Path) -> None:
    payload = {
        "families": [
            {
                "name": "survivor",
                "owner": "1234abc",
                "status": "live",
                "lifecycle": "permanent",
            }
        ],
        "retired_families": [],
        "family_history": ["survivor"],
    }

    with pytest.raises(AssertionError, match="deleted_family"):
        _assert_contract_family_lifecycle(
            payload,
            repo_root=tmp_path,
            pinned_history=frozenset({"survivor", "deleted_family"}),
        )


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


def _assert_contract_family_lifecycle(
    payload: dict[str, Any],
    *,
    repo_root: Path,
    pinned_history: frozenset[str] = PINNED_FAMILY_HISTORY,
) -> None:
    active = payload.get("families")
    retired = payload.get("retired_families", [])
    assert isinstance(active, list) and active, "manifest must declare active families"
    assert isinstance(retired, list), "retired_families must be an array of tables"

    all_families = [*active, *retired]
    names = [family.get("name") for family in all_families]
    assert all(isinstance(name, str) and name for name in names)
    assert len(names) == len(set(names)), "active and retired family names must be unique"
    history = payload.get("family_history")
    assert isinstance(history, list) and len(history) == len(set(history))
    missing_history = sorted(pinned_history - set(history))
    assert not missing_history, (
        f"pinned historical contract families cannot disappear: {missing_history}"
    )
    assert set(names) == set(history), (
        "family_history is append-only; move retired registrations to retired_families"
    )

    active_by_name = {family["name"]: family for family in active}
    owner_pattern = re.compile(r"^[0-9a-f]{7}$")
    due: list[str] = []
    for family in all_families:
        name = family["name"]
        assert owner_pattern.fullmatch(str(family.get("owner", ""))), (
            f"contract family {name!r} needs a 7-character owner issue"
        )
        lifecycle = family.get("lifecycle")
        assert lifecycle in {"permanent", "migration"}, (
            f"contract family {name!r} has invalid lifecycle {lifecycle!r}"
        )
        if lifecycle == "permanent":
            assert family in active, f"permanent contract family {name!r} cannot be retired"
            assert family.get("status") == "live", (
                f"active permanent family {name!r} must declare status = 'live'"
            )
            assert "successor" not in family and "exit_predicate" not in family
            continue

        successor = family.get("successor")
        assert isinstance(successor, str) and successor in active_by_name, (
            f"migration family {name!r} needs a live successor family"
        )
        assert active_by_name[successor].get("status") == "live", (
            f"migration family {name!r} successor {successor!r} is not live"
        )
        predicate = family.get("exit_predicate")
        assert isinstance(predicate, dict), (
            f"migration family {name!r} needs a machine-checkable exit_predicate"
        )
        satisfied = _exit_predicate_satisfied(predicate, repo_root=repo_root)
        if family in active:
            assert family.get("status") == "live", (
                f"active migration family {name!r} must declare status = 'live'"
            )
            if satisfied:
                due.append(name)
        if family in retired:
            assert family.get("status") == "retired", (
                f"retired family {name!r} must declare status = 'retired'"
            )
            assert owner_pattern.fullmatch(str(family.get("retired_by", ""))), (
                f"retired family {name!r} needs a 7-character retiring issue"
            )
            assert satisfied, f"retired family {name!r} has an unsatisfied exit predicate"

    assert not due, f"migration families with satisfied exit predicates must retire: {due}"


def _exit_predicate_satisfied(predicate: dict[str, Any], *, repo_root: Path) -> bool:
    kind = predicate.get("kind")
    assert kind == "toml_arrays_empty", f"unsupported exit predicate kind {kind!r}"

    relative_path = predicate.get("path")
    keys = predicate.get("keys")
    assert isinstance(relative_path, str) and relative_path
    assert isinstance(keys, list) and keys and all(isinstance(key, str) for key in keys)
    inventory_path = repo_root / relative_path
    assert inventory_path.is_file(), f"exit-predicate inventory does not exist: {relative_path}"
    inventory = tomllib.loads(inventory_path.read_text(encoding="utf-8"))
    return all(inventory.get(key, []) == [] for key in keys)
