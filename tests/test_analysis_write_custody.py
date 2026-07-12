"""Analysis-domain durable-write custody ratchet (issue c223bb8).

This freezes the current inventory of raw durable writes under
``src/rlrmp/analysis/**`` while the analysis materializers are migrated onto
governed Feedbax custody surfaces. New write sites fail by default; retired
sites must be removed from the allowlist.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
import re
import tomllib

import pytest
from feedbax.testing import (
    AllowlistEntry,
    Scope,
    SiteVisitor,
    diff_allowlist,
    scan_domain,
    target_label as kit_target_label,
)


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
    assert not list((REPO_ROOT / "src/rlrmp/analysis/pipelines").glob("*.py"))
    # Figure renders now route through Feedbax custody. The final direct
    # ``write_html`` analysis site was the retired response-norm producer, so
    # requiring one here would force legacy durable writers to remain alive.


def test_analysis_durable_write_sites_match_allowlist() -> None:
    diff = _allowlist_diff(_scan_analysis_tree())
    new_sites = sorted(site.key for site, _occurrence in diff.unlisted)
    assert not new_sites, (
        "New analysis durable-write site(s) found without an allowlist entry: "
        f"{new_sites}. Route the write through governed custody, or add a "
        f"deliberate entry to {ALLOWLIST_PATH.relative_to(REPO_ROOT)} with owner "
        "and reason."
    )


def test_analysis_write_allowlist_has_no_dead_entries() -> None:
    diff = _allowlist_diff(_scan_analysis_tree())
    dead = sorted(entry.key for entry in diff.dead_entries)
    assert not dead, (
        "Analysis durable-write allowlist names site(s) that no longer exist: "
        f"{dead}. Remove stale entries; shrinking the inventory is required."
    )


def test_analysis_write_allowlist_entries_carry_owner_and_reason() -> None:
    issue_re = re.compile(r"^[0-9a-f]{7}$")
    entries = _raw_allowlist_entries()
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


@pytest.mark.parametrize(
    ("relative_path", "function_name"),
    (
        ("src/rlrmp/analysis/soft_lambda.py", "run_soft_lambda_materializer"),
    ),
)
def test_analysis_json_sidecars_use_governed_writer(
    relative_path: str,
    function_name: str,
) -> None:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    calls = {_function_name(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}

    assert "write_json" in calls
    assert not any(name.endswith(".write_text") for name in calls)


def test_analysis_write_negative_canary_flags_unlisted_durable_write() -> None:
    sites = _scan_source(
        """
from pathlib import Path

def materialize_new_report(output_path: Path) -> None:
    output_path.write_text("new durable report", encoding="utf-8")
""",
        relpath="src/rlrmp/analysis/pipelines/new_report.py",
    )

    with pytest.raises(AssertionError, match="New analysis durable-write"):
        _assert_no_new_sites(sites, [])


def _scan_analysis_tree() -> list[WriteSite]:
    return scan_domain(
        ANALYSIS_ROOT.rglob("*.py"),
        root=REPO_ROOT,
        visitor_factory=_AnalysisWriteScanner,
    )


def _scan_source(source: str, *, relpath: str) -> list[WriteSite]:
    tree = ast.parse(source, filename=relpath)
    scanner = _AnalysisWriteScanner(relpath=relpath)
    scanner.visit(tree)
    return scanner.sites


class _AnalysisWriteScanner(SiteVisitor[WriteSite]):
    def visit_Call(self, node: ast.Call) -> None:
        detected = _detect_write_call(node)
        if detected is not None:
            kind, target = detected
            self.sites.append(
                WriteSite(
                    path=self.relpath,
                    symbol=self.qualname,
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


def _target_label(expr: ast.expr | None) -> str:
    if expr is None:
        return "<unknown>"
    if isinstance(expr, (ast.Name, ast.Constant)):
        return kit_target_label(expr)
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


def _raw_allowlist_entries() -> list[dict]:
    data = tomllib.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    entries = data.get("durable_write_sites", [])
    assert isinstance(entries, list), "durable_write_sites must be a TOML list"
    return entries


def _numbered_sites(sites: Iterable[WriteSite]) -> list[tuple[WriteSite, int]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    numbered = []
    for site in sites:
        counts[site.key] += 1
        numbered.append((site, counts[site.key]))
    return numbered


def _site_key_from_entry(entry: dict) -> tuple[str, str, str, str]:
    return (entry["path"], entry["symbol"], entry["kind"], entry["target"])


def _kit_allowlist_entries(
    raw_entries: Iterable[dict],
) -> list[AllowlistEntry[tuple[str, str, str, str, int]]]:
    entries = []
    for raw in raw_entries:
        key = _site_key_from_entry(raw)
        for occurrence in range(1, int(raw["count"]) + 1):
            entries.append(
                AllowlistEntry(
                    scope=Scope("python_scope", raw["path"], raw["symbol"]),
                    owner=raw["owner"],
                    reason=raw["reason"],
                    key=(*key, occurrence),
                )
            )
    return entries


def _allowlist_diff(sites: Iterable[WriteSite]):
    return diff_allowlist(
        _numbered_sites(sites),
        _kit_allowlist_entries(_raw_allowlist_entries()),
        site_key=lambda item: (*item[0].key, item[1]),
        site_location=lambda item: (item[0].path, item[0].symbol),
    )


def _assert_no_new_sites(
    sites: Iterable[WriteSite],
    entries: list[AllowlistEntry[tuple[str, str, str, str, int]]],
) -> None:
    diff = diff_allowlist(
        _numbered_sites(sites),
        entries,
        site_key=lambda item: (*item[0].key, item[1]),
        site_location=lambda item: (item[0].path, item[0].symbol),
    )
    new_sites = sorted(site.key for site, _occurrence in diff.unlisted)
    assert not new_sites, f"New analysis durable-write site(s): {new_sites}"
