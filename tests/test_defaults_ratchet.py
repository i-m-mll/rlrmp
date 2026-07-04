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
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import tomllib

import pytest


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "ci" / "defaults-ratchet-allowlist.toml"

SCAN_TARGETS = (
    "src/rlrmp/eval/",
    "src/rlrmp/analysis/pipelines/",
    "src/rlrmp/analysis/reports.py",
    "src/rlrmp/runtime/training_run_specs.py",
    "src/rlrmp/train/cs_nominal_gru.py",
    "src/rlrmp/train/cs_perturbation_training.py",
    "src/rlrmp/benchmarks/packing.py",
    "src/rlrmp/model/",
    "src/rlrmp/train/closed_loop_distillation.py",
    "src/rlrmp/eval/minimax_io.py",
)


@dataclass(frozen=True, order=True)
class DefaultFallbackSite:
    path: str
    key: str
    literal_repr: str


def test_default_fallback_sites_match_allowlist() -> None:
    allowlist = _load_allowlist()
    allowed = _allowlisted_sites(allowlist)

    found = _scan_default_fallback_sites()

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


def _load_allowlist() -> dict:
    return tomllib.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))


def _allowlisted_sites(allowlist: dict) -> Counter[DefaultFallbackSite]:
    counter: Counter[DefaultFallbackSite] = Counter()
    for entry in allowlist["default_fallback_sites"]:
        site = DefaultFallbackSite(
            path=entry["path"],
            key=entry["key"],
            literal_repr=entry["literal_repr"],
        )
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


def _scan_default_fallback_sites() -> Counter[DefaultFallbackSite]:
    found: Counter[DefaultFallbackSite] = Counter()
    for path in _scan_files():
        relpath = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            site = _default_fallback_site(node, relpath)
            if site is not None:
                found[site] += 1
    return found


def _scan_files() -> list[Path]:
    files: set[Path] = set()
    for target in SCAN_TARGETS:
        path = REPO_ROOT / target
        if path.is_dir():
            files.update(path.rglob("*.py"))
        elif path.exists():
            files.add(path)
        else:
            raise AssertionError(f"Configured default-fallback scan target does not exist: {target}")
    return sorted(files)


def _default_fallback_site(call: ast.Call, relpath: str) -> DefaultFallbackSite | None:
    key_node: ast.AST | None = None
    default_node: ast.AST | None = None

    if isinstance(call.func, ast.Attribute) and call.func.attr == "get":
        if len(call.args) >= 2:
            key_node = call.args[0]
            default_node = call.args[1]
    elif isinstance(call.func, ast.Name) and call.func.id == "getattr":
        if len(call.args) >= 3:
            key_node = call.args[1]
            default_node = call.args[2]

    if key_node is None or default_node is None:
        return None

    key = _literal_value(key_node)
    default = _literal_value(default_node)
    if not isinstance(key, str) or not _is_meaningful_literal(default):
        return None
    return DefaultFallbackSite(relpath, key, repr(default))


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _is_meaningful_literal(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        return True
    if isinstance(value, str):
        return value != ""
    if isinstance(value, tuple):
        return bool(value) and all(_is_tuple_scalar(item) for item in value)
    return False


def _is_tuple_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int | float):
        return True
    if isinstance(value, str):
        return value != ""
    return False
