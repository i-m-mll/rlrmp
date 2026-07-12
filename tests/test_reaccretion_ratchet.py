"""Re-accretion ratchet: allowlist-pinned legacy-pattern inventories (issue 9728133).

Freezes the inventories of three legacy-pattern classes the 64a04e0 umbrella
exists to retire, so the migration target cannot silently grow while the
migration is pending:

1. RLRMP-branded GraphSpec component-ID strings defined under ``src/``.
2. Argparse-first training entry points.
3. Hand-rolled run-spec/run.json writer sites outside sanctioned emitters.

Shrinking an inventory (retiring a pattern) never fails these tests --
removing an allowlist entry after retiring the underlying code is
ceremony-free. Growing an inventory (a new branded component ID, a new
argparse-first launcher, or a new hand-rolled run-spec writer) fails unless
``ci/legacy-pattern-allowlist.toml`` is deliberately edited to name the new
instance and its owning ledger issue. See that file's header comment for the
exact scan-scope rationale for each inventory.
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import TypeVar

import pytest
from feedbax.testing import SiteVisitor, TomlListBaseline, compare_ratchet, scan_domain


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "ci" / "legacy-pattern-allowlist.toml"

# --- Inventory 1: branded GraphSpec component-ID strings -------------------

BRANDED_COMPONENT_ID_PATTERN = re.compile(r"^RLRMP[A-Z][A-Za-z0-9]*$")

BRANDED_COMPONENT_ID_SCAN_FILES = (
    "src/rlrmp/model/feedbax_graph.py",
    "src/rlrmp/model/cs_lss_gru.py",
    "src/rlrmp/runtime/run_specs.py",
    "src/rlrmp/train/cs_perturbation_training.py",
)

# --- Inventories 2 and 3: training-entry-point domain -----------------------

SANCTIONED_RUN_SPEC_EMITTER_NAMES = frozenset(
    {
        "validate_nominal_gru_run_spec",
        "validate_nominal_gru_run_spec_file",
        "validate_minimax_run_spec",
        "validate_minimax_run_spec_file",
        "stamp_current_schema",
        "accept_rlrmp_spec_payload",
    }
)
RUN_SPEC_NAME_PATTERN = re.compile(r"(run_spec|spec_path|run_path)", re.IGNORECASE)
SCHEMA_GENERATED_PARSER_MODULES: frozenset[str] = frozenset()

KeyT = TypeVar("KeyT", bound=Hashable)


BRANDED_ID_BASELINE = TomlListBaseline(
    ALLOWLIST_PATH,
    "branded_component_ids",
    key_from_entry=lambda entry: entry["value"],
)
ARGPARSE_BASELINE = TomlListBaseline(
    ALLOWLIST_PATH,
    "argparse_training_entry_points",
    key_from_entry=lambda entry: entry["path"],
)
RUN_SPEC_WRITER_BASELINE = TomlListBaseline(
    ALLOWLIST_PATH,
    "run_spec_writer_sites",
    key_from_entry=lambda entry: (entry["path"], entry["function"]),
)


def test_branded_graph_spec_component_ids_match_allowlist() -> None:
    found = _scan_branded_component_ids()

    _assert_no_growth(
        found,
        BRANDED_ID_BASELINE.load(),
        "New RLRMP-branded GraphSpec component-ID string(s) found without an "
        "allowlist entry: {added}. Add an entry to "
        f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)} naming the owning ledger issue, "
        "or confirm this is not meant to be a new active-spec default.",
    )
    # Shrink-only: retired IDs may remain listed without failing this test.
    assert found, "Branded-component-ID scan found zero strings; scan scope may be broken"


def test_argparse_training_entry_points_match_allowlist() -> None:
    found = _scan_argparse_entry_points()

    _assert_no_growth(
        found,
        ARGPARSE_BASELINE.load(),
        "New argparse-first training entry point(s) found without an allowlist "
        "entry: {added}. Add an entry to "
        f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)} naming the owning ledger issue.",
    )
    # This retired-pattern inventory may reach zero now that canonical training
    # CLIs are generated from registered config schemas. The negative canary
    # below still proves a new argparse-first launcher is rejected.


def test_hand_rolled_run_spec_writer_sites_match_allowlist() -> None:
    found = _scan_run_spec_writer_sites()

    _assert_no_growth(
        found,
        RUN_SPEC_WRITER_BASELINE.load(),
        "New hand-rolled run-spec/run.json writer site(s) found without an "
        "allowlist entry: {added}. Add an entry to "
        f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)} naming the owning ledger issue, "
        "or route the write through a sanctioned emitter "
        "(validate_nominal_gru_run_spec / stamp_current_schema / "
        "accept_rlrmp_spec_payload).",
    )
    # This retired-pattern inventory is allowed to reach zero when the last
    # hand-rolled run-spec writer has been removed. The negative canary below
    # still proves new unallowlisted writers are rejected.


def test_reaccretion_ratchet_negative_canary_rejects_unlisted_instance() -> None:
    with pytest.raises(AssertionError, match="New RLRMP-branded"):
        _assert_no_growth(
            {"RLRMPSomeExistingComponent", "RLRMPBrandNewComponent"},
            {"RLRMPSomeExistingComponent"},
            "New RLRMP-branded component ID(s): {added}",
        )
    with pytest.raises(AssertionError, match="New argparse-first"):
        _assert_no_growth(
            {"scripts/train_existing.py", "scripts/train_new_launcher.py"},
            {"scripts/train_existing.py"},
            "New argparse-first entry point(s): {added}",
        )
    with pytest.raises(AssertionError, match="New hand-rolled"):
        _assert_no_growth(
            {
                ("scripts/train_existing.py", "existing"),
                ("scripts/train_new.py", "run_training"),
            },
            {("scripts/train_existing.py", "existing")},
            "New hand-rolled run-spec writer site(s): {added}",
        )


def test_reaccretion_ratchet_allowlist_entries_carry_owning_issue() -> None:
    allowlist = _load_allowlist()
    issue_pattern = re.compile(r"^[0-9a-f]{7}$")
    for family in (
        "branded_component_ids",
        "argparse_training_entry_points",
        "run_spec_writer_sites",
    ):
        for entry in allowlist[family]:
            owner = entry.get("owner", "")
            assert issue_pattern.match(owner), (
                f"Allowlist entry {entry} in [[{family}]] is missing a valid "
                "7-character owning-issue annotation."
            )


def _load_allowlist() -> dict:
    return tomllib.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))


def _assert_no_growth(current: Iterable[KeyT], baseline: Iterable[KeyT], message: str) -> None:
    diff = compare_ratchet(current, baseline)
    assert not diff.added, message.format(added=sorted(diff.added))


# --- Inventory 1 scan --------------------------------------------------------


def _scan_branded_component_ids() -> set[str]:
    return set(
        scan_domain(
            (REPO_ROOT / relpath for relpath in BRANDED_COMPONENT_ID_SCAN_FILES),
            root=REPO_ROOT,
            visitor_factory=_BrandedIdVisitor,
        )
    )


class _BrandedIdVisitor(SiteVisitor[str]):
    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and BRANDED_COMPONENT_ID_PATTERN.fullmatch(node.value):
            self.sites.append(node.value)


# --- Inventories 2/3 shared domain -------------------------------------------


def _training_entry_point_domain_files() -> list[Path]:
    files = sorted((REPO_ROOT / "src" / "rlrmp" / "train").glob("*.py"))
    files += sorted((REPO_ROOT / "scripts").glob("train_*.py"))
    return files


# --- Inventory 2 scan --------------------------------------------------------


def _scan_argparse_entry_points() -> set[str]:
    return set(
        scan_domain(
            _training_entry_point_domain_files(),
            root=REPO_ROOT,
            visitor_factory=_ArgparseEntrypointVisitor,
        )
    )


class _ArgparseEntrypointVisitor(SiteVisitor[str]):
    def visit_Module(self, node: ast.Module) -> None:
        if (
            _constructs_argument_parser(node)
            and self.relpath not in SCHEMA_GENERATED_PARSER_MODULES
        ):
            self.sites.append(self.relpath)


def _constructs_argument_parser(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
            if isinstance(func.value, ast.Name) and func.value.id == "argparse":
                return True
        if isinstance(func, ast.Name) and func.id == "ArgumentParser":
            return True
    return False


# --- Inventory 3 scan --------------------------------------------------------


def _scan_run_spec_writer_sites() -> set[tuple[str, str]]:
    return set(
        scan_domain(
            _training_entry_point_domain_files(),
            root=REPO_ROOT,
            visitor_factory=_RunSpecWriterVisitor,
        )
    )


class _RunSpecWriterVisitor(SiteVisitor[tuple[str, str]]):
    def visit_Module(self, node: ast.Module) -> None:
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and _is_hand_rolled_run_spec_writer(child):
                self.sites.append((self.relpath, child.name))


def _is_hand_rolled_run_spec_writer(func: ast.FunctionDef) -> bool:
    calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)]

    writes_json = any(
        _is_json_dump_call(call) or _is_write_text_of_json_dumps_call(call) for call in calls
    )
    if not writes_json:
        return False

    if any(_call_func_name(call) in SANCTIONED_RUN_SPEC_EMITTER_NAMES for call in calls):
        return False

    target_names = _open_write_target_names(func) | _write_text_receiver_names(func)
    name_looks_like_spec = any(RUN_SPEC_NAME_PATTERN.search(name) for name in target_names)
    func_name_looks_like_spec = bool(RUN_SPEC_NAME_PATTERN.search(func.name))
    return name_looks_like_spec or func_name_looks_like_spec


def _call_func_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_json_dump_call(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "dump"
        and isinstance(func.value, ast.Name)
        and func.value.id == "json"
    )


def _is_json_dumps_expr(node: ast.AST) -> bool:
    """True if ``node`` is (recursively, through string concatenation) a json.dumps(...) call."""

    if isinstance(node, ast.BinOp):
        return _is_json_dumps_expr(node.left) or _is_json_dumps_expr(node.right)
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "dumps":
        return isinstance(func.value, ast.Name) and func.value.id == "json"
    if isinstance(func, ast.Name):
        # Local wrappers around json.dumps (e.g. compact_json_dumps, _json_dumps).
        return "json_dumps" in func.id
    return False


def _is_write_text_of_json_dumps_call(call: ast.Call) -> bool:
    func = call.func
    if not (isinstance(func, ast.Attribute) and func.attr == "write_text"):
        return False
    args = list(call.args) + [kw.value for kw in call.keywords]
    return any(_is_json_dumps_expr(arg) for arg in args)


def _open_write_target_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            names.add(node.args[0].id)
    return names


def _write_text_receiver_names(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and _is_write_text_of_json_dumps_call(node):
            receiver = node.func.value  # type: ignore[union-attr]
            if isinstance(receiver, ast.Name):
                names.add(receiver.id)
    return names
