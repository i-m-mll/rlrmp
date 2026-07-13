"""Shrink-only ratchet for out-of-schema default fallback sites (issue 5b3aabe).

This freezes the current inventory of ``.get("key", literal_default)`` and
``getattr(obj, "key", literal_default)`` sites in schema-owning runtime,
evaluation, analysis, model, training, and benchmark modules. Retiring a site is
ceremony-free: stale allowlist entries do not fail this test. Adding a new site
fails unless ``ci/defaults-ratchet-allowlist.toml`` is deliberately updated to
name the owning ledger issue.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re
import tomllib

import pytest
import rlrmp
from feedbax.analysis import evaluation as feedbax_evaluation
from feedbax.analysis.reports import registered_report_types
from feedbax.plugins.registry import ExperimentRegistry
from pydantic import BaseModel

from rlrmp.runtime.defaults_scan import (
    SCAN_TARGETS,
    DefaultFallbackSite,
    DefaultValueDriftException,
    count_default_fallback_sites,
    find_value_drifts,
    scan_default_fallback_site_instances,
    scan_default_fallback_sites,
    scan_default_fallback_sites_in_paths,
    scan_authored_identity_defaults,
)
from rlrmp.runtime.params_models import params_model_for, registered_params_models


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "ci" / "defaults-ratchet-allowlist.toml"
CANARY_PATH = REPO_ROOT / "tests" / "fixtures" / "defaults_scan_canary.py"
REQUIRED_AUTHORING_SCAN_TARGETS = (
    "src/rlrmp/train/config_materialization.py",
    "src/rlrmp/train/run_spec_authoring.py",
    "src/rlrmp/train/training_configs.py",
)
LEGACY_OUTPUT_SURFACES = (
    "src/rlrmp/model/feedbax_graph.py",
    "src/rlrmp/train/config_materialization.py",
    "src/rlrmp/train/run_spec_authoring.py",
)

VALUE_DRIFT_EXCEPTIONS: tuple[DefaultValueDriftException, ...] = (
    DefaultValueDriftException(
        key="status",
        reason=(
            "Generic analysis/report result-state metadata; values are local state labels, "
            "not one shared schema default."
        ),
    ),
    DefaultValueDriftException(
        key="reason",
        reason=(
            "Generic missingness/failure explanation metadata; values describe local "
            "diagnostic states, not one shared parameter."
        ),
    ),
    DefaultValueDriftException(
        key="type",
        reason=(
            "Generic payload discriminator used by unrelated training-task and perturbation "
            "records."
        ),
    ),
    DefaultValueDriftException(
        key="role",
        reason=(
            "Generic artifact/input role metadata; report rendering and diagnostic bundles use "
            "different local absence labels."
        ),
    ),
    DefaultValueDriftException(
        key="hidden_size",
        path="src/rlrmp/model/feedbax_graph.py",
        literal_repr="1",
        reason=(
            "Feedbax graph hidden-size fallback is a deserialization shape placeholder, not the "
            "training model width default."
        ),
    ),
    DefaultValueDriftException(
        key="level",
        path="src/rlrmp/benchmarks/packing.py",
        literal_repr="'not_applicable'",
        reason=(
            "Benchmark packing rows use not_applicable as result metadata; perturbation training "
            "uses level as a calibration strength."
        ),
    ),
    DefaultValueDriftException(
        key="n_replicates",
        path="src/rlrmp/train/cs_nominal_gru.py",
        literal_repr="5",
        reason=(
            "The nominal training CLI's legacy run default differs from scalar materialization "
            "fallbacks that collapse to one replicate."
        ),
    ),
    DefaultValueDriftException(
        key="n_steps",
        path="src/rlrmp/benchmarks/packing.py",
        literal_repr="0",
        reason=(
            "Benchmark packing uses zero as an unavailable step-count marker; perturbation "
            "training uses n_steps as a real perturbation horizon."
        ),
    ),
    DefaultValueDriftException(
        key="sign",
        path="src/rlrmp/eval/feedback_ablation.py",
        literal_repr="0",
        reason=(
            "Feedback-ablation summaries use neutral sign metadata; perturbation calibration "
            "uses sign as a directional pulse parameter."
        ),
    ),
    DefaultValueDriftException(
        key="sign",
        path="src/rlrmp/eval/perturbation_bank.py",
        literal_repr="0",
        reason=(
            "Perturbation-bank filtering treats absent sign metadata as neutral; perturbation "
            "calibration uses sign as a directional pulse parameter."
        ),
    ),
)


def test_default_fallback_sites_match_allowlist() -> None:
    allowlist = _load_allowlist()
    allowed = _allowlisted_sites(allowlist)

    found = scan_default_fallback_sites(REPO_ROOT)

    new_instances = _new_or_grown_instances(found, allowed)
    assert not new_instances, (
        "New out-of-schema default fallback site(s) found without an allowlist "
        f"entry: {new_instances}. Add entries to "
        f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)} naming the owning ledger issue, "
        "or route the default through a schema-owned params model. Stale "
        "allowlist entries are permitted, so retiring sites does not require a "
        "same-commit allowlist edit."
    )
    assert found, "Default-fallback scan found zero sites; scan scope may be broken"


def test_training_authoring_files_stay_in_scanner_scope_and_have_no_fallbacks() -> None:
    assert all(target in SCAN_TARGETS for target in REQUIRED_AUTHORING_SCAN_TARGETS)
    paths = [REPO_ROOT / target for target in REQUIRED_AUTHORING_SCAN_TARGETS]

    assert scan_default_fallback_sites_in_paths(paths, repo_root=REPO_ROOT) == []


def test_current_authoring_surfaces_do_not_emit_legacy_labeled_fields() -> None:
    found: list[str] = []
    for relpath in LEGACY_OUTPUT_SURFACES:
        tree = ast.parse((REPO_ROOT / relpath).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key in node.keys:
                if (
                    not isinstance(key, ast.Constant)
                    or not isinstance(key.value, str)
                    or not key.value.startswith("legacy_")
                ):
                    continue
                found.append(f"{relpath}:{key.lineno}:{key.value}")

    assert found == []


def test_default_fallback_allowlist_entries_carry_owner_and_count() -> None:
    allowlist = _load_allowlist()

    issue_pattern = re.compile(r"^[0-9a-f]{7}$")
    for entry in allowlist["default_fallback_sites"]:
        site = DefaultFallbackSite(
            path=entry.get("path", ""),
            key=entry.get("key", ""),
            literal_repr=entry.get("literal_repr", ""),
        )
        count = entry.get("count")
        owner = entry.get("owner", "")

        assert site.path and site.path.endswith(".py"), f"Invalid allowlist path: {entry}"
        assert site.key, f"Invalid allowlist key: {entry}"
        assert site.literal_repr, f"Invalid allowlist literal_repr: {entry}"
        assert isinstance(count, int) and count > 0, (
            f"Allowlist entry {entry} must carry a positive occurrence count."
        )
        assert issue_pattern.match(owner), (
            f"Allowlist entry {entry} is missing a valid 7-character owning issue."
        )


def test_default_fallback_values_do_not_drift_across_files() -> None:
    for exception in VALUE_DRIFT_EXCEPTIONS:
        assert exception.reason, f"Value-drift exception needs a justification: {exception}"

    sites = scan_default_fallback_site_instances(REPO_ROOT)
    drifts = find_value_drifts(sites, exceptions=VALUE_DRIFT_EXCEPTIONS)

    assert not drifts, (
        "Same-key default fallback values drift across files: "
        f"{[drift.display() for drift in drifts]}. Route the value through a "
        "schema-owned params model, or add a narrowly justified exception for "
        "genuinely different concepts that share a key name."
    )


def test_defaults_scanner_negative_canary_detects_out_of_schema_default() -> None:
    found = count_default_fallback_sites(
        scan_default_fallback_sites_in_paths([CANARY_PATH], repo_root=REPO_ROOT)
    )
    assert (
        found[
            DefaultFallbackSite(
                path="tests/fixtures/defaults_scan_canary.py",
                key="default_scan_canary",
                literal_repr="12345",
            )
        ]
        == 1
    )


def test_training_defaults_do_not_embed_authored_issue_identities() -> None:
    assert scan_authored_identity_defaults(REPO_ROOT) == []


def test_authored_identity_default_scanner_detects_assignment_and_argument_paths(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "src/rlrmp/train"
    train_dir.mkdir(parents=True)
    (train_dir / "example.py").write_text(
        "DEFAULT_OUTPUT = '_artifacts/a1b2c3d/runs/default'\n"
        "class Config:\n"
        "    output = 'results/1a2b3c4/runs/default.json'\n"
        "def build(path='results/abc1234/runs/default.json'):\n"
        "    return path\n",
        encoding="utf-8",
    )

    assert {site.identity for site in scan_authored_identity_defaults(tmp_path)} == {
        "a1b2c3d",
        "1a2b3c4",
        "abc1234",
    }


def test_registered_eval_and_report_recipes_have_params_models() -> None:
    rlrmp.register_experiment_package(ExperimentRegistry())

    registered_eval_types = sorted(
        evaluation_type
        for evaluation_type in feedbax_evaluation._EVALUATION_RECIPES
        if evaluation_type.startswith("rlrmp.")
    )
    registered_reports = sorted(
        report_type for report_type in registered_report_types() if report_type.startswith("rlrmp.")
    )
    registered_recipe_names = registered_eval_types + registered_reports

    assert registered_recipe_names
    for recipe_name in registered_recipe_names:
        model_class = params_model_for(recipe_name)
        assert issubclass(model_class, BaseModel), recipe_name
        assert registered_params_models()[recipe_name] is model_class
    with pytest.raises(KeyError):
        params_model_for("rlrmp.eval.unregistered")


def _load_allowlist() -> dict:
    return tomllib.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))


def _allowlisted_sites(allowlist: dict) -> Counter[DefaultFallbackSite]:
    counter: Counter[DefaultFallbackSite] = Counter()
    for entry in allowlist["default_fallback_sites"]:
        site = DefaultFallbackSite(entry["path"], entry["key"], entry["literal_repr"])
        counter[site] += int(entry["count"])
    return counter


def _new_or_grown_instances(
    found: Counter[DefaultFallbackSite],
    allowed: Counter[DefaultFallbackSite],
) -> list[dict[str, str | int]]:
    instances: list[dict[str, str | int]] = []
    for site, found_count in sorted(found.items()):
        allowed_count = allowed[site]
        if found_count <= allowed_count:
            continue
        instances.append(
            {
                "path": site.path,
                "key": site.key,
                "literal_repr": site.literal_repr,
                "found_count": found_count,
                "allowed_count": allowed_count,
            }
        )
    return instances
