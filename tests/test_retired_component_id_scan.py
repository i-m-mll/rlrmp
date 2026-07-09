"""Terminal-gate retired GraphSpec component-ID confinement scan (issue 7811e47).

Gate check 1 of the 419eed1 semantic-safety gate. Asserts inventory-complete
CONFINEMENT of the retired RLRMP-branded GraphSpec component-ID strings: every
occurrence -- across ``src/``, ``tests/``, production ``scripts/``, and the
manifest / fixture JSON under ``results/`` -- must fall inside a bounded,
annotated allowlist entry in ``ci/retired-component-id-confinement.toml``.

This is distinct from the 9728133 re-accretion ratchet
(``tests/test_reaccretion_ratchet.py``): the ratchet freezes the *inventory* of
branded component-ID VALUES against growth in four src modules; this scan pins
the *locations* every retired ID may legally appear, so a retired ID reappearing
on an active clean-target path (a new builder, script, or emission site) fails
even though its value is already in the ratchet inventory.

Scan mechanism:
  * ``.py`` under src/tests/scripts/results -- AST string-literal scan; each
    occurrence is bound to its enclosing-function qualname so the two mixed
    model modules (native active builders + legacy loader/serialization code)
    get function-region confinement, not whole-file.
  * ``.json`` under tests/results -- recursive JSON-literal scan (keys + values).

Skips count as failures: an in-scope ``.py`` that fails to parse or ``.json``
that fails to load is a hard error, and a stale allowlist entry that matches no
occurrence fails the dead-entry check.

The retired-ID inventory is not re-listed here; it is read from the 9728133
ratchet's ``[[branded_component_ids]]`` table so the two gates share one source
of truth.
"""

from __future__ import annotations

import ast
import json
import tomllib
import warnings
from pathlib import Path

import pytest


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
RATCHET_ALLOWLIST_PATH = REPO_ROOT / "ci" / "legacy-pattern-allowlist.toml"
CONFINEMENT_ALLOWLIST_PATH = REPO_ROOT / "ci" / "retired-component-id-confinement.toml"

# Files that legitimately carry retired-ID strings as data/inventory and are not
# emission surfaces: the two allowlist TOMLs and this scan module itself.
SELF_EXCLUDED_RELPATHS = frozenset(
    {
        "ci/legacy-pattern-allowlist.toml",
        "ci/retired-component-id-confinement.toml",
        "tests/test_retired_component_id_scan.py",
    }
)

MODULE_SCOPE = "<module>"

RETIRED_STANDALONE_MATERIALIZER_MODULES = frozenset(
    {
        "materialize_adversary_equivalence",
        "materialize_analytical_game_card",
        "materialize_cs_stochastic_phase1",
        "materialize_cs_stochastic_phase3",
        "materialize_linear_equivalence_certificate",
        "materialize_linear_round_trip",
    }
)

RETIRED_STANDALONE_MATERIALIZER_PATHS = frozenset(
    f"scripts/{module}.py" for module in RETIRED_STANDALONE_MATERIALIZER_MODULES
)


# --------------------------------------------------------------------------- #
# Inventory + allowlist loading
# --------------------------------------------------------------------------- #


def _retired_id_inventory() -> frozenset[str]:
    ratchet = tomllib.loads(RATCHET_ALLOWLIST_PATH.read_text(encoding="utf-8"))
    return frozenset(entry["value"] for entry in ratchet["branded_component_ids"])


def _load_confinement_allowlist() -> dict:
    return tomllib.loads(CONFINEMENT_ALLOWLIST_PATH.read_text(encoding="utf-8"))


class _AllowlistIndex:
    """Indexed, order-independent view of the confinement allowlist."""

    def __init__(self, allowlist: dict) -> None:
        self.python_scopes: set[tuple[str, str]] = {
            (entry["path"], entry["qualname"]) for entry in allowlist.get("python_scope", [])
        }
        self.files: set[str] = {entry["path"] for entry in allowlist.get("file", [])}
        self.globs: dict[str, frozenset[str]] = {
            entry["pattern"]: frozenset(
                p.relative_to(REPO_ROOT).as_posix() for p in REPO_ROOT.glob(entry["pattern"])
            )
            for entry in allowlist.get("glob", [])
        }

    def covering_entries(self, occ: "_Occurrence") -> list[tuple[str, object]]:
        """Return every allowlist entry that covers ``occ`` (empty == uncovered)."""

        covering: list[tuple[str, object]] = []
        for scope in self.python_scopes:
            path, qualname = scope
            if path == occ.path and qualname == occ.qualname:
                covering.append(("python_scope", scope))
        if occ.path in self.files:
            covering.append(("file", occ.path))
        for pattern, members in self.globs.items():
            if occ.path in members:
                covering.append(("glob", pattern))
        return covering

    def all_entry_keys(self) -> set[tuple[str, object]]:
        keys: set[tuple[str, object]] = set()
        keys.update(("python_scope", s) for s in self.python_scopes)
        keys.update(("file", f) for f in self.files)
        keys.update(("glob", g) for g in self.globs)
        return keys


class _Occurrence:
    __slots__ = ("path", "component_id", "qualname", "lineno")

    def __init__(self, path: str, component_id: str, qualname: str | None, lineno: int) -> None:
        self.path = path
        self.component_id = component_id
        self.qualname = qualname  # enclosing-function qualname for .py; None for .json
        self.lineno = lineno

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        loc = f"{self.qualname}" if self.qualname is not None else "<json>"
        return f"{self.path}:{self.lineno} [{loc}] {self.component_id}"


# --------------------------------------------------------------------------- #
# Occurrence scanning
# --------------------------------------------------------------------------- #


def _in_scope_relpaths() -> tuple[list[str], list[str]]:
    """Return (python_relpaths, json_relpaths) in scan scope, self-exclusions removed."""

    py: set[str] = set()
    for root in ("src", "tests", "scripts", "results"):
        py.update(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / root).rglob("*.py"))
    js: set[str] = set()
    for root in ("tests", "results"):
        js.update(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / root).rglob("*.json"))
    py -= SELF_EXCLUDED_RELPATHS
    js -= SELF_EXCLUDED_RELPATHS
    return sorted(py), sorted(js)


class _ConstantScopeVisitor(ast.NodeVisitor):
    def __init__(self, inventory: frozenset[str]) -> None:
        self._inventory = inventory
        self._stack: list[str] = []
        self.hits: list[tuple[str, str, int]] = []  # (component_id, qualname, lineno)

    def _visit_scoped(self, node: ast.AST) -> None:
        self._stack.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _visit_scoped
    visit_AsyncFunctionDef = _visit_scoped

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value in self._inventory:
            qualname = ".".join(self._stack) if self._stack else MODULE_SCOPE
            self.hits.append((node.value, qualname, getattr(node, "lineno", 0)))
        self.generic_visit(node)


def _scan_python(relpath: str, inventory: frozenset[str], errors: list[str]) -> list[_Occurrence]:
    path = REPO_ROOT / relpath
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError) as exc:  # skips count as failures
        errors.append(f"{relpath}: unparseable/unreadable in-scope python file: {exc}")
        return []
    visitor = _ConstantScopeVisitor(inventory)
    visitor.visit(tree)
    return [_Occurrence(relpath, cid, qual, lineno) for cid, qual, lineno in visitor.hits]


class _RetiredEntrypointImportVisitor(ast.NodeVisitor):
    def __init__(self, inventory: frozenset[str]) -> None:
        self._inventory = inventory
        self.hits: list[tuple[str, int]] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in self._inventory:
                self.hits.append((alias.name, node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in self._inventory:
            self.hits.append((node.module, node.lineno))


def _iter_json_strings(node: object):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str):
                yield key
            yield from _iter_json_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_json_strings(value)


def _scan_json(relpath: str, inventory: frozenset[str], errors: list[str]) -> list[_Occurrence]:
    path = REPO_ROOT / relpath
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:  # skips count as failures
        errors.append(f"{relpath}: unloadable/unreadable in-scope json file: {exc}")
        return []
    seen: set[str] = set()
    for string in _iter_json_strings(data):
        if string in inventory:
            seen.add(string)
    return [_Occurrence(relpath, cid, None, 0) for cid in sorted(seen)]


def _collect_occurrences(
    inventory: frozenset[str],
) -> tuple[list[_Occurrence], list[str]]:
    errors: list[str] = []
    occurrences: list[_Occurrence] = []
    py_files, json_files = _in_scope_relpaths()
    for relpath in py_files:
        occurrences.extend(_scan_python(relpath, inventory, errors))
    for relpath in json_files:
        occurrences.extend(_scan_json(relpath, inventory, errors))
    return occurrences, errors


def _retired_entrypoint_import_hits(
    source: str,
    *,
    relpath: str,
    inventory: frozenset[str] = RETIRED_STANDALONE_MATERIALIZER_MODULES,
) -> list[str]:
    tree = ast.parse(source, filename=relpath)
    visitor = _RetiredEntrypointImportVisitor(inventory)
    visitor.visit(tree)
    return [f"{relpath}:{lineno}: {module}" for module, lineno in visitor.hits]


def _scan_retired_entrypoint_imports() -> tuple[list[str], list[str]]:
    hits: list[str] = []
    errors: list[str] = []
    for root in ("src", "tests", "scripts"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            relpath = path.relative_to(REPO_ROOT).as_posix()
            if relpath in RETIRED_STANDALONE_MATERIALIZER_PATHS:
                continue
            if relpath in SELF_EXCLUDED_RELPATHS:
                continue
            try:
                hits.extend(
                    _retired_entrypoint_import_hits(
                        path.read_text(encoding="utf-8"),
                        relpath=relpath,
                    )
                )
            except (SyntaxError, UnicodeDecodeError, OSError) as exc:
                errors.append(f"{relpath}: unparseable/unreadable in-scope python file: {exc}")
    return hits, errors


# --------------------------------------------------------------------------- #
# Pure evaluators (shared by the live scan and the negative canaries)
# --------------------------------------------------------------------------- #


def _uncovered_occurrences(
    occurrences: list[_Occurrence], index: _AllowlistIndex
) -> list[_Occurrence]:
    return [occ for occ in occurrences if not index.covering_entries(occ)]


def _matched_entry_keys(
    occurrences: list[_Occurrence], index: _AllowlistIndex
) -> set[tuple[str, object]]:
    matched: set[tuple[str, object]] = set()
    for occ in occurrences:
        matched.update(index.covering_entries(occ))
    return matched


def _dead_entry_keys(
    occurrences: list[_Occurrence], index: _AllowlistIndex
) -> set[tuple[str, object]]:
    return index.all_entry_keys() - _matched_entry_keys(occurrences, index)


# --------------------------------------------------------------------------- #
# Live gate tests
# --------------------------------------------------------------------------- #


def test_retired_id_inventory_is_nonempty_and_sourced_from_ratchet() -> None:
    inventory = _retired_id_inventory()
    assert inventory, "retired-ID inventory read from the 9728133 ratchet is empty"
    # Sanity: the named terminal-gate IDs must be present in the shared inventory.
    for required in (
        "RLRMPSimpleStagedNetwork",
        "RLRMPLinearController",
        "RLRMPLinearTrackerController",
        "RLRMPCsLssInitialHiddenStagedNetwork",
        "RLRMPCsLssFiniteEpsilonPolicy",
    ):
        assert required in inventory, f"{required} missing from ratchet inventory source"


def test_no_retired_component_id_outside_confinement_allowlist() -> None:
    inventory = _retired_id_inventory()
    index = _AllowlistIndex(_load_confinement_allowlist())
    occurrences, errors = _collect_occurrences(inventory)

    assert not errors, (
        "in-scope files could not be scanned (skips count as failures):\n" + "\n".join(errors)
    )
    # Scope-not-broken guard: the scan must actually see the retired IDs.
    assert occurrences, "retired-ID scan found zero occurrences; scan scope may be broken"

    uncovered = _uncovered_occurrences(occurrences, index)
    assert not uncovered, (
        "Retired GraphSpec component-ID(s) found on a path with no confinement "
        f"allowlist entry in {CONFINEMENT_ALLOWLIST_PATH.relative_to(REPO_ROOT)}:\n"
        + "\n".join(f"  {occ}" for occ in sorted(uncovered, key=str))
        + "\n\nEither route the emission through a Feedbax-native component ID, or "
        "(if this is legacy/conversion/fixture code) add an annotated allowlist "
        "entry naming the owning ledger issue."
    )


def test_confinement_allowlist_has_no_dead_entries() -> None:
    inventory = _retired_id_inventory()
    index = _AllowlistIndex(_load_confinement_allowlist())
    occurrences, errors = _collect_occurrences(inventory)
    assert not errors, "in-scope files could not be scanned:\n" + "\n".join(errors)

    dead = _dead_entry_keys(occurrences, index)
    assert not dead, (
        "Stale confinement allowlist entry/entries match no retired-ID occurrence "
        "(dead entries count as skips, and skips count as failures). Remove them "
        f"from {CONFINEMENT_ALLOWLIST_PATH.relative_to(REPO_ROOT)}:\n"
        + "\n".join(f"  {kind}: {key}" for kind, key in sorted(dead, key=str))
    )


def test_confinement_allowlist_entries_carry_owner_category_and_note() -> None:
    import re

    allowlist = _load_confinement_allowlist()
    issue_pattern = re.compile(r"^[0-9a-f]{7}$")
    for family in ("python_scope", "file", "glob"):
        for entry in allowlist.get(family, []):
            owner = entry.get("owner", "")
            assert issue_pattern.match(owner), (
                f"[[{family}]] entry {entry} is missing a valid 7-char owner annotation"
            )
            assert entry.get("category"), f"[[{family}]] entry {entry} is missing a category"
            assert entry.get("note"), f"[[{family}]] entry {entry} is missing a note"


def test_retired_standalone_materializer_entrypoints_stay_deleted() -> None:
    remaining = sorted(
        relpath
        for relpath in RETIRED_STANDALONE_MATERIALIZER_PATHS
        if (REPO_ROOT / relpath).exists()
    )
    assert not remaining, (
        "Retired standalone legacy materializer entrypoint(s) reappeared. "
        "Restore from git history only for archaeology, or file a new native "
        f"materialization issue before reviving: {remaining}"
    )


def test_retired_standalone_materializers_are_not_imported_by_active_code() -> None:
    hits, errors = _scan_retired_entrypoint_imports()
    assert not errors, (
        "in-scope files could not be scanned for retired materializer imports:\n"
        + "\n".join(errors)
    )
    assert not hits, (
        "Active src/tests/scripts code imports retired standalone materializer "
        "entrypoint(s):\n"
        + "\n".join(hits)
    )


# --------------------------------------------------------------------------- #
# Negative canaries (teeth)
# --------------------------------------------------------------------------- #


def test_scan_negative_canary_flags_unconfined_active_emission() -> None:
    index = _AllowlistIndex(_load_confinement_allowlist())
    # A retired ID stamped in a brand-new active builder function that is not in
    # any allowlist scope must be flagged.
    unconfined = _Occurrence(
        "src/rlrmp/model/new_active_builder.py",
        "RLRMPSimpleStagedNetwork",
        "build_active_controller_spec",
        42,
    )
    confined = _Occurrence(
        "src/rlrmp/model/feedbax_graph.py",
        "RLRMPSimpleStagedNetwork",
        "register_rlrmp_graph_components",
        148,
    )
    uncovered = _uncovered_occurrences([unconfined, confined], index)
    assert uncovered == [unconfined]


def test_scan_negative_canary_flags_retired_id_in_new_function_of_allowlisted_file() -> None:
    index = _AllowlistIndex(_load_confinement_allowlist())
    # Same allowlisted file, but a function that is NOT an allowlisted scope:
    # region-level confinement must still flag it.
    occ = _Occurrence(
        "src/rlrmp/model/feedbax_graph.py",
        "RLRMPLinearController",
        "_build_native_point_mass_active",
        999,
    )
    assert _uncovered_occurrences([occ], index) == [occ]


def test_scan_negative_canary_treats_unreadable_file_as_error_not_skip() -> None:
    errors: list[str] = []
    sentinel = frozenset({"RLRMPNoMatchInventorySentinel"})
    # Happy path: a real, clean module scanned with a non-matching inventory
    # returns no occurrences and records no error.
    result = _scan_python("src/rlrmp/model/feedback_descriptors.py", sentinel, errors)
    assert result == [] and not errors
    # Error path: a missing in-scope file must be recorded as an error, not
    # silently skipped (skips count as failures).
    missing = _scan_python("results/does_not_exist_synthetic.py", sentinel, errors)
    assert missing == []
    assert errors


def test_entrypoint_negative_canary_flags_import_of_retired_materializer() -> None:
    hits = _retired_entrypoint_import_hits(
        "import materialize_linear_round_trip\n",
        relpath="tests/test_new_legacy_call.py",
    )
    assert hits == ["tests/test_new_legacy_call.py:1: materialize_linear_round_trip"]
