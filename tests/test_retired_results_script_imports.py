"""Deny retired namespace imports in tracked results scripts (issue 81c150f)."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import tomllib

import pytest


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "ci" / "retired-results-script-import-allowlist.toml"
RETIRED_NAMESPACES = frozenset(
    {
        "rlrmp.analysis.multi_cell_driver",
        "rlrmp.analysis.pipelines",
        "rlrmp.viz.figures",
        "rlrmp.viz.profile_grids",
    }
)


def _tracked_results_scripts() -> tuple[str, ...]:
    output = subprocess.check_output(
        ["git", "--no-optional-locks", "ls-files", "results/**/scripts/*.py"],
        cwd=REPO_ROOT,
        text=True,
    )
    return tuple(path for path in output.splitlines() if path and (REPO_ROOT / path).is_file())


def _retired_root(module: str) -> str | None:
    for namespace in RETIRED_NAMESPACES:
        if module == namespace or module.startswith(f"{namespace}."):
            return namespace
    return None


def _scan_source(source: str, *, filename: str) -> frozenset[str]:
    tree = ast.parse(source, filename=filename)
    hits: set[str] = set()
    for node in ast.walk(tree):
        modules: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            modules = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules = (node.module,)
        for module in modules:
            retired = _retired_root(module)
            if retired is not None:
                hits.add(retired)
    return frozenset(hits)


def _observed_imports() -> dict[str, frozenset[str]]:
    observed: dict[str, frozenset[str]] = {}
    for relpath in _tracked_results_scripts():
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        hits = _scan_source(source, filename=relpath)
        if hits:
            observed[relpath] = hits
    return observed


def _allowed_imports() -> dict[str, frozenset[str]]:
    payload = tomllib.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    allowed: dict[str, frozenset[str]] = {}
    for entry in payload.get("exceptions", []):
        path = str(entry["path"])
        assert path not in allowed, f"duplicate retired-import exception: {path}"
        assert entry.get("owner") == "34699b2", entry
        assert len(str(entry.get("reason", "")).strip()) >= 40, entry
        namespaces = frozenset(str(value) for value in entry["namespaces"])
        assert namespaces and namespaces <= RETIRED_NAMESPACES, entry
        allowed[path] = namespaces
    return allowed


def test_tracked_results_scripts_match_retired_import_allowlist() -> None:
    observed = _observed_imports()
    allowed = _allowed_imports()

    assert observed == allowed, (
        "Retired results-script imports differ from the shrink-only inventory. "
        f"Unlisted or changed: {observed.keys() - allowed.keys()}; "
        f"dead entries: {allowed.keys() - observed.keys()}; "
        f"namespace mismatches: "
        f"{sorted(path for path in observed.keys() & allowed.keys() if observed[path] != allowed[path])}"
    )


def test_retired_import_negative_canary_flags_direct_and_submodule_imports() -> None:
    source = """
import rlrmp.viz.figures
from rlrmp.analysis.pipelines.retired_materializer import materialize
from rlrmp.analysis.multi_cell_driver import run
from rlrmp.viz.profile_grids import build
"""
    assert _scan_source(source, filename="synthetic.py") == RETIRED_NAMESPACES
