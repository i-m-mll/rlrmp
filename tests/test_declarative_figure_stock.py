"""Shrink-only gate for the pre-declarative figure stock (issue f6f38b6).

Tracked figure intent must either validate as a native Feedbax ``FigureSpec`` or
be an explicitly inventoried legacy record. Frozen structured outputs are kept
only as reasoned archival parity oracles. Living result-family producers may
not assemble Plotly figures or legacy spec dictionaries unless they are in the
same shrink-only inventory.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tomllib
from typing import Any

import pytest
from feedbax.contracts.figures import FigureSpec


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "ci" / "legacy-figure-stock-allowlist.toml"
ISSUE_RE = re.compile(r"^[0-9a-f]{7}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
ARCHIVE_STATUS = "archived_structured_output"
ARCHIVE_REASON_ISSUE = "7ae2916"
PROVENANCE_HEAVY_KEYS = {
    "checkpoint_policy",
    "data",
    "panels",
    "replicate_summaries",
    "replicate_summary",
    "runtime_provenance",
}


@dataclass(frozen=True)
class FigureStockFact:
    """Classification of one tracked figure spec."""

    path: str
    role: str
    provenance_heavy_keys: tuple[str, ...]


def test_figure_stock_inventory_is_non_vacuous_and_classified() -> None:
    facts = _scan_figure_stock()

    assert len(facts) >= 70, "figure-stock scan found too few tracked specs"
    assert any(fact.role == "archival" for fact in facts)
    assert any(fact.role == "legacy_living" for fact in facts)
    assert any("perturbation_response_norms" in fact.path for fact in facts), (
        "response-norm figure capability fell out of the governed stock"
    )
    assert {fact.role for fact in facts} <= {"native", "legacy_living", "archival"}


def test_non_native_living_figure_specs_match_shrink_only_allowlist() -> None:
    observed = {fact.path for fact in _scan_figure_stock() if fact.role == "legacy_living"}
    allowed = {entry["path"] for entry in _allowlist_entries("legacy_specs")}

    assert observed == allowed, _inventory_message(
        "non-native living figure spec", observed=observed, allowed=allowed
    )


def test_living_direct_figure_assembly_matches_shrink_only_allowlist() -> None:
    observed = set(_living_direct_assembly_files())
    allowed = {entry["path"] for entry in _allowlist_entries("living_materializers")}

    assert observed == allowed, _inventory_message(
        "living direct figure/spec assembly", observed=observed, allowed=allowed
    )


def test_archival_figure_specs_are_reasoned_parity_oracles() -> None:
    archival = [fact for fact in _scan_figure_stock() if fact.role == "archival"]
    assert archival, "figure-stock scan found no archival parity oracles"

    for fact in archival:
        payload = _load_json(REPO_ROOT / fact.path)
        archive = payload["archive"]
        assert archive["status"] == ARCHIVE_STATUS, fact.path
        assert SHA_RE.fullmatch(str(archive["last_runnable_revision"])), fact.path
        assert archive["rationale_issue"] == ARCHIVE_REASON_ISSUE, fact.path
        assert archive["parity_oracle"] == "archived_structured_outputs", fact.path
        assert len(str(archive["reason"]).strip()) >= 20, fact.path


def test_allowlist_entries_are_reasoned_and_point_to_living_files() -> None:
    for family in ("legacy_specs", "living_materializers"):
        entries = _allowlist_entries(family)
        assert entries, f"{family} inventory must remain explicit while legacy stock exists"
        paths = [entry["path"] for entry in entries]
        assert len(paths) == len(set(paths)), f"duplicate {family} allowlist entries"
        for entry in entries:
            assert ISSUE_RE.fullmatch(str(entry.get("owner", ""))), entry
            assert len(str(entry.get("reason", "")).strip()) >= 20, entry
            assert (REPO_ROOT / entry["path"]).is_file(), entry


def test_authored_intent_rejects_provenance_heavy_legacy_payload() -> None:
    native_payload = FigureSpec(
        name="intrinsic-panels-data-bound-facets",
        template="rlrmp.profile_comparison",
        slot_bindings={},
        panels=[{"name": "intrinsic-profile-panel"}],
        facet_bindings={"condition": {"item": "manifest", "path": "metadata.facets"}},
    ).model_dump(mode="json")
    native_fact = _classify_payload(
        "results/fffffff/figures/native/spec.json", native_payload
    )
    assert native_fact.role == "native"
    assert "panels" in native_fact.provenance_heavy_keys
    archival_fact = FigureStockFact(
        path="results/eeeeeee/figures/archive/spec.json",
        role="archival",
        provenance_heavy_keys=(),
    )
    assert _living_figure_issue_ids([native_fact, archival_fact]) == {"fffffff"}

    payload = {
        "schema_version": "rlrmp.figure_spec.v1",
        "topic": "new_legacy_figure",
        "checkpoint_policy": "best checkpoint after inspecting training history",
        "panels": [{"replicate_summary": {"peak": 1.0}}],
        "data": {"time": [0.0, 0.1], "mean": [0.0, 1.0]},
    }

    fact = _classify_payload("results/fffffff/figures/new/spec.json", payload)

    assert fact.role == "legacy_living"
    assert set(fact.provenance_heavy_keys) >= {
        "checkpoint_policy",
        "data",
        "panels",
        "replicate_summary",
    }
    with pytest.raises(AssertionError, match="non-native living figure spec"):
        _assert_no_unlisted_legacy([fact], allowed=set())


def test_direct_assembly_negative_canary_rejects_plotly_materializer() -> None:
    source = """
import plotly.graph_objects as go

def materialize_new_figure():
    spec = {"schema_version": "legacy.v1", "panels": [{"data": [1, 2]}]}
    return go.Figure(), spec
"""
    assert _has_direct_figure_assembly(source)
    with pytest.raises(AssertionError, match="direct figure/spec assembly"):
        _assert_no_unlisted_materializers(
            ["results/fffffff/scripts/materialize_new_figure.py"], allowed=set()
        )


def _figure_spec_files() -> list[Path]:
    roots = (REPO_ROOT / "results", REPO_ROOT / "manuscript")
    return sorted(
        path
        for root in roots
        if root.exists()
        for path in root.glob("**/figures/**/spec.json")
        if path.is_file()
    )


def _scan_figure_stock() -> list[FigureStockFact]:
    return [
        _classify_payload(path.relative_to(REPO_ROOT).as_posix(), _load_json(path))
        for path in _figure_spec_files()
    ]


def _classify_payload(path: str, payload: dict[str, Any]) -> FigureStockFact:
    heavy = tuple(sorted(_find_provenance_heavy_keys(payload)))
    archive = payload.get("archive")
    if isinstance(archive, dict) and archive.get("status") == ARCHIVE_STATUS:
        role = "archival"
    else:
        try:
            FigureSpec.model_validate(payload)
        except Exception:  # Pydantic emits several validation-error subclasses.
            role = "legacy_living"
        else:
            role = "native"
    return FigureStockFact(path=path, role=role, provenance_heavy_keys=heavy)


def _find_provenance_heavy_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PROVENANCE_HEAVY_KEYS:
                found.add(key)
            found.update(_find_provenance_heavy_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_provenance_heavy_keys(item))
    return found


def _living_direct_assembly_files() -> list[str]:
    living_issues = _living_figure_issue_ids(_scan_figure_stock())
    paths = []
    for issue in sorted(living_issues):
        scripts = REPO_ROOT / "results" / issue / "scripts"
        for path in sorted(scripts.glob("*.py")):
            if _has_direct_figure_assembly(path.read_text(encoding="utf-8")):
                paths.append(path.relative_to(REPO_ROOT).as_posix())
    return paths


def _living_figure_issue_ids(facts: list[FigureStockFact]) -> set[str]:
    return {
        Path(fact.path).parts[1]
        for fact in facts
        if fact.role != "archival" and Path(fact.path).parts[0] == "results"
    }


def _has_direct_figure_assembly(source: str) -> bool:
    tree = ast.parse(source)
    imports_plotly = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name.startswith("plotly") for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("plotly"))
        for node in ast.walk(tree)
    )
    legacy_spec_dict = any(
        isinstance(node, ast.Dict)
        and {key.value for key in node.keys if isinstance(key, ast.Constant)}
        & {"schema_version", "checkpoint_policy", "panels", "figure_link"}
        for node in ast.walk(tree)
    )
    return imports_plotly or legacy_spec_dict


def _allowlist_entries(family: str) -> list[dict[str, Any]]:
    payload = tomllib.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    return list(payload.get(family, []))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), path
    return payload


def _assert_no_unlisted_legacy(facts: list[FigureStockFact], *, allowed: set[str]) -> None:
    unlisted = {fact.path for fact in facts if fact.role == "legacy_living"} - allowed
    assert not unlisted, f"non-native living figure spec(s): {sorted(unlisted)}"


def _assert_no_unlisted_materializers(paths: list[str], *, allowed: set[str]) -> None:
    unlisted = set(paths) - allowed
    assert not unlisted, (
        f"direct figure/spec assembly in living materializer(s): {sorted(unlisted)}"
    )


def _inventory_message(kind: str, *, observed: set[str], allowed: set[str]) -> str:
    return (
        f"{kind} inventory drifted. Unlisted={sorted(observed - allowed)}; "
        f"stale={sorted(allowed - observed)}. Migrate/remove the old surface or "
        f"make a deliberate, reasoned shrink-only allowlist update."
    )
