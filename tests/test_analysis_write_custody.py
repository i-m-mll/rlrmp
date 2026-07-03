"""Analysis-domain durable-write custody ratchet (issue c223bb8).

This freezes the current inventory of raw durable writes under
``src/rlrmp/analysis/**`` while the analysis materializers are migrated onto
governed Feedbax custody surfaces. New write sites fail by default; retired
sites must be removed from the allowlist.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re
import tomllib

import pytest


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / "src" / "rlrmp" / "analysis"
ALLOWLIST_PATH = REPO_ROOT / "ci" / "analysis-write-custody-allowlist.toml"

_DUMP_MODULES = {"json", "pickle", "toml", "yaml"}
_PATH_WRITE_METHODS = {"write_text", "write_bytes"}
_FIGURE_WRITE_METHODS = {"write_html", "write_image", "write_json"}
_ARRAY_WRITE_METHODS = {"save", "savez", "savez_compressed"}


class WriteSite:
    """One structurally keyed raw write site."""

    __slots__ = ("path", "symbol", "kind", "target")

    def __init__(self, *, path: str, symbol: str, kind: str, target: str) -> None:
        self.path = path
        self.symbol = symbol
        self.kind = kind
        self.target = target

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.path, self.symbol, self.kind, self.target)


def test_analysis_write_scan_is_non_vacuous() -> None:
    sites = _scan_analysis_tree()
    assert sites, "analysis durable-write scan found zero sites; scan scope is broken"
    assert any(site.path.startswith("src/rlrmp/analysis/math/") for site in sites)
    assert any(site.path.startswith("src/rlrmp/analysis/pipelines/") for site in sites)
    assert any(site.kind == "write_html" for site in sites), (
        "analysis figure HTML writers are not covered by the scan"
    )


def test_analysis_durable_write_sites_match_allowlist() -> None:
    found = _site_counts(_scan_analysis_tree())
    allowed = _allowlist_counts(_load_allowlist())

    new_sites = sorted(set(found) - set(allowed))
    assert not new_sites, (
        "New analysis durable-write site(s) found without an allowlist entry: "
        f"{new_sites}. Route the write through governed custody, or add a "
        f"deliberate entry to {ALLOWLIST_PATH.relative_to(REPO_ROOT)} with owner "
        "and reason."
    )

    count_mismatches = sorted(
        (key, f"found={found[key]} allowed={allowed[key]}")
        for key in set(found) & set(allowed)
        if found[key] != allowed[key]
    )
    assert not count_mismatches, (
        "Analysis durable-write allowlist count mismatch: "
        f"{count_mismatches}. Counts catch duplicate writes under one structural key; "
        "update the inventory deliberately."
    )


def test_analysis_write_allowlist_has_no_dead_entries() -> None:
    found = _site_counts(_scan_analysis_tree())
    allowed = _allowlist_counts(_load_allowlist())
    dead = sorted(set(allowed) - set(found))
    assert not dead, (
        "Analysis durable-write allowlist names site(s) that no longer exist: "
        f"{dead}. Remove stale entries; shrinking the inventory is required."
    )


def test_analysis_write_allowlist_entries_carry_owner_and_reason() -> None:
    issue_re = re.compile(r"^[0-9a-f]{7}$")
    entries = _load_allowlist().get("durable_write_sites", [])
    assert entries, "analysis write allowlist declares zero durable sites"
    for entry in entries:
        assert issue_re.match(entry.get("owner", "")), (
            f"Allowlist entry {entry} is missing a 7-character owning issue"
        )
        assert isinstance(entry.get("reason"), str) and len(entry["reason"].strip()) >= 20, (
            f"Allowlist entry {entry} needs a brief reason"
        )
        assert isinstance(entry.get("count"), int) and entry["count"] >= 1, (
            f"Allowlist entry {entry} needs a positive integer count"
        )


def test_analysis_write_negative_canary_flags_unlisted_durable_write() -> None:
    found = _site_counts(
        _scan_source(
            """
from pathlib import Path

def materialize_new_report(output_path: Path) -> None:
    output_path.write_text("new durable report", encoding="utf-8")
""",
            relpath="src/rlrmp/analysis/pipelines/new_report.py",
        )
    )
    allowed: dict[tuple[str, str, str, str], int] = {}

    with pytest.raises(AssertionError, match="New analysis durable-write"):
        _assert_no_new_sites(found, allowed)


def _scan_analysis_tree() -> list[WriteSite]:
    sites: list[WriteSite] = []
    for path in sorted(ANALYSIS_ROOT.rglob("*.py")):
        sites.extend(_scan_file(path))
    return sites


def _scan_file(path: Path) -> list[WriteSite]:
    return _scan_source(
        path.read_text(encoding="utf-8"),
        relpath=path.relative_to(REPO_ROOT).as_posix(),
    )


def _scan_source(source: str, *, relpath: str) -> list[WriteSite]:
    tree = ast.parse(source, filename=relpath)
    scanner = _AnalysisWriteScanner(relpath)
    scanner.visit(tree)
    return scanner.sites


class _AnalysisWriteScanner(ast.NodeVisitor):
    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.stack: list[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.sites: list[WriteSite] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.stack.append(node)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        detected = _detect_write_call(node)
        if detected is not None:
            kind, target = detected
            self.sites.append(
                WriteSite(
                    path=self.relpath,
                    symbol=_symbol(self.stack),
                    kind=kind,
                    target=_target_label(target),
                )
            )
        self.generic_visit(node)


def _detect_write_call(call: ast.Call) -> tuple[str, ast.expr | None] | None:
    func = call.func
    name = _function_name(func)
    if name == "open" and _is_write_mode(_open_mode(call)):
        return ("open_w", call.args[0] if call.args else None)

    if not isinstance(func, ast.Attribute):
        return None

    attr = func.attr
    if attr == "open" and _is_write_mode(_open_mode(call)):
        return ("path.open_w", func.value)
    if attr in _PATH_WRITE_METHODS:
        return (attr, func.value)
    if attr in _FIGURE_WRITE_METHODS:
        return (attr, call.args[0] if call.args else None)
    if attr in _ARRAY_WRITE_METHODS:
        return (attr, call.args[0] if call.args else None)
    if attr == "dump" and isinstance(func.value, ast.Name) and func.value.id in _DUMP_MODULES:
        return (f"{func.value.id}.dump", call.args[1] if len(call.args) > 1 else None)
    return None


def _function_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _function_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return ""


def _open_mode(call: ast.Call) -> str | None:
    mode = call.args[1] if len(call.args) > 1 else None
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return mode.value
    return None


def _is_write_mode(mode: str | None) -> bool:
    return bool(mode) and any(flag in mode for flag in "wax+")


def _symbol(stack: list[ast.AST]) -> str:
    names = []
    for node in stack:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return ".".join(names) or "<module>"


def _target_label(expr: ast.expr | None) -> str:
    if expr is None:
        return "<unknown>"
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Constant):
        return repr(expr.value)
    if isinstance(expr, ast.Attribute):
        return f"{_target_label(expr.value)}.{expr.attr}"
    if isinstance(expr, ast.Subscript):
        return f"{_target_label(expr.value)}[]"
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        return f"{_target_label(expr.left)}/{_target_label(expr.right)}"
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        return f"{_target_label(expr.func.value)}.{expr.func.attr}()"
    if isinstance(expr, ast.JoinedStr):
        return "<f-string>"
    return "<expr>"


def _load_allowlist() -> dict:
    return tomllib.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))


def _site_counts(sites: list[WriteSite]) -> dict[tuple[str, str, str, str], int]:
    return Counter(site.key for site in sites)


def _allowlist_counts(allowlist: dict) -> dict[tuple[str, str, str, str], int]:
    counts: dict[tuple[str, str, str, str], int] = {}
    for entry in allowlist.get("durable_write_sites", []):
        key = (entry["path"], entry["symbol"], entry["kind"], entry["target"])
        counts[key] = int(entry["count"])
    return counts


def _assert_no_new_sites(
    found: dict[tuple[str, str, str, str], int],
    allowed: dict[tuple[str, str, str, str], int],
) -> None:
    new_sites = sorted(set(found) - set(allowed))
    assert not new_sites, f"New analysis durable-write site(s): {new_sites}"
