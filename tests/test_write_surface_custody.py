"""Deny-by-default durable-output custody + provenance-lineage audit (issue f5d9695).

This is the terminal lineage-audit gate for the 64a04e0 feedbax-native umbrella.
It enforces two things as executable, gate-registered checks (family
``write_surface`` in ``ci/feedbax-contract-suite.toml``):

1.  **Write-surface custody guard (the genuinely new machinery).** A static AST
    scan over the training run-production/emission domain
    (``scripts/train_*.py`` + ``src/rlrmp/train/*.py`` -- the same
    training-entry-point domain the re-accretion ratchet pins) enumerates every
    *raw* durable-output write call (``fbx_save`` / ``eqx.tree_serialise_leaves``
    / ``np.save*`` / ``open(..., "w")`` / ``Path.write_text`` / ``Path.write_bytes``).
    Each site is classified by its raw write target:

    - **ephemeral / atomic-staging** -- the raw write targets a ``tmp``-rooted
      path (staged then ``os.replace``-d / renamed into place). These are safe:
      the durable materialization is the atomic rename, not a raw serialization
      to a declared durable root.
    - **durable** -- everything else (a write whose raw target is an
      ``output_dir`` / ``spec_path`` / ``paths[...]`` / parameter-rooted path).

    Every *durable* raw-write site must be named in
    ``ci/write-surface-allowlist.toml`` with an owning issue and a role.
    A new, unlisted durable raw-write fails the gate: deny-by-default custody by
    construction, not by spot-check. The ``conditional`` metadata is generated
    from AST control flow (enclosing ``if`` guards and ternary dispatch over
    writer functions), not hand-curated. See the allowlist header for the
    documented escape modes (subprocess / native-library /
    remote-object-store writes, symlink / hardlink traversal) that this static
    guard does not by itself close.

2.  **Six provenance-lineage invariants (assert-integrate).** Invariants 1-6 from
    the issue body are enforced here as marked checks that reference and
    assert-integrate the existing (mostly unmarked) production anchors and unit
    tests, pulling them into the required CI standing gate. Where an invariant is
    already structurally guaranteed (e.g. no ``comparable_training_spec`` parity
    reconstruction exists anywhere; exactly one ``TrainingRunManifest`` emitter),
    the check asserts that structural fact directly.

``skips count as failures`` in this gate: the family is enrolled ``live`` with a
``minimum_non_skipped`` floor, and ``tests/conftest.py`` +
``test_feedbax_contract_meta.py`` forbid SKIP / non-strict XFAIL under the
``feedbax_contract`` marker.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
import tomllib

import pytest


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "ci" / "write-surface-allowlist.toml"

# ---------------------------------------------------------------------------
# Domain: the training run-production / emission path.
#
# This is the same training-entry-point domain the re-accretion ratchet
# (tests/test_reaccretion_ratchet.py, issue 9728133) pins: scripts/train_*.py
# plus src/rlrmp/train/*.py. Analysis pipelines (src/rlrmp/analysis/**) and eval
# scripts write DataProduct / ReportManifest governed outputs on a separate
# custody substrate (issues 108b4d3 / ea6ccb4 and the ReportManifest contract)
# and are out of scope for the *training-manifest* lineage guarded here.
# ---------------------------------------------------------------------------


def _domain_files() -> list[Path]:
    files = sorted((REPO_ROOT / "src" / "rlrmp" / "train").glob("*.py"))
    files += sorted((REPO_ROOT / "scripts").glob("train_*.py"))
    return [p for p in files if p.is_file()]


_EPHEMERAL_ROOT_RE = re.compile(r"^_?tmp", re.IGNORECASE)


class WriteSite:
    """One raw durable-or-ephemeral write call site, keyed structurally."""

    __slots__ = ("relpath", "function", "kind", "target", "ephemeral", "conditional")

    def __init__(
        self,
        *,
        relpath: str,
        function: str,
        kind: str,
        target: str,
        ephemeral: bool,
        conditional: bool,
    ) -> None:
        self.relpath = relpath
        self.function = function
        self.kind = kind
        self.target = target
        self.ephemeral = ephemeral
        self.conditional = conditional

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.relpath, self.function, self.kind, self.target)


# ---------------------------------------------------------------------------
# Raw-write call detection
# ---------------------------------------------------------------------------


def _write_target_expr(call: ast.Call) -> ast.expr | None:
    """Return the AST expression naming what a raw-write call writes to."""

    func = call.func
    if isinstance(func, ast.Name) and func.id in {"fbx_save", "open"}:
        return call.args[0] if call.args else None
    if isinstance(func, ast.Attribute):
        # eqx.tree_serialise_leaves(target, value); np.save*(target, ...)
        if func.attr in {"tree_serialise_leaves", "save", "savez", "savez_compressed"}:
            return call.args[0] if call.args else None
        # <receiver>.write_text(...) / <receiver>.write_bytes(...)
        if func.attr in {"write_text", "write_bytes"}:
            return func.value
    return None


def _is_raw_write_call(call: ast.Call) -> bool:
    if _write_target_expr(call) is None:
        return False
    func = call.func
    if isinstance(func, ast.Name):
        if func.id == "fbx_save":
            return True
        if func.id == "open":
            # Only writing opens are durable-output writes.
            mode = call.args[1] if len(call.args) > 1 else None
            for kw in call.keywords:
                if kw.arg == "mode":
                    mode = kw.value
            return isinstance(mode, ast.Constant) and "w" in str(mode.value)
    if isinstance(func, ast.Attribute):
        return func.attr in {
            "tree_serialise_leaves",
            "save",
            "savez",
            "savez_compressed",
            "write_text",
            "write_bytes",
        }
    return False


def _call_kind(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return "open_w" if func.id == "open" else func.id
    assert isinstance(func, ast.Attribute)
    return func.attr


def _target_root_name(expr: ast.expr | None) -> str | None:
    """Return the root variable name of a write-target path expression."""

    node = expr
    while node is not None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            node = node.left
            continue
        if isinstance(node, ast.Subscript):
            node = node.value
            continue
        if isinstance(node, ast.Attribute):
            node = node.value
            continue
        if isinstance(node, ast.Call):
            # e.g. path.with_name(...), Path(...); recurse into the receiver.
            if isinstance(node.func, ast.Attribute):
                node = node.func.value
                continue
            if node.args:
                node = node.args[0]
                continue
        return None
    return None


def _target_label(expr: ast.expr | None) -> str:
    """Stable, edit-resilient label for a write target (root + terminal)."""

    if expr is None:
        return "<unknown>"
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
        return f"{_target_label(expr.left)}/{_terminal_label(expr.right)}"
    if isinstance(expr, ast.Subscript):
        return f"{_target_label(expr.value)}[{_terminal_label(expr.slice)}]"
    if isinstance(expr, ast.Attribute):
        return f"{_target_label(expr.value)}.{expr.attr}"
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        return f"{_target_label(expr.func.value)}.{expr.func.attr}()"
    return "<expr>"


def _terminal_label(expr: ast.expr) -> str:
    if isinstance(expr, ast.Constant):
        return str(expr.value)
    if isinstance(expr, ast.Name):
        return f"<{expr.id}>"
    if isinstance(expr, ast.JoinedStr):
        parts = []
        for value in expr.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            else:
                parts.append("{}")
        return "".join(parts)
    return "<term>"


# ---------------------------------------------------------------------------
# Scan: durable + ephemeral sites
# ---------------------------------------------------------------------------


def _enclosing_function(stack: list[ast.AST]) -> str:
    for node in reversed(stack):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return "<module>"


def _module_conditional_dispatch(tree: ast.Module) -> dict[str, set[str]]:
    """Map a function name to the set of sibling functions it is mutually
    exclusive with via a ternary dispatch ``f = A if cond else B``."""

    pairs: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        body, orelse = node.body, node.orelse
        if isinstance(body, ast.Name) and isinstance(orelse, ast.Name):
            a, b = body.id, orelse.id
            pairs.setdefault(a, set()).add(b)
            pairs.setdefault(b, set()).add(a)
    return pairs


def _scan_file(path: Path) -> list[WriteSite]:
    """Return write sites for one domain file."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    relpath = path.relative_to(REPO_ROOT).as_posix()
    dispatch = _module_conditional_dispatch(tree)

    # Attach parent + guard context by a manual DFS with a stack.
    sites: list[WriteSite] = []
    def visit(node: ast.AST, stack: list[ast.AST]) -> None:
        function = _enclosing_function(stack)
        if isinstance(node, ast.Call) and _is_raw_write_call(node):
            target_expr = _write_target_expr(node)
            root = _target_root_name(target_expr)
            ephemeral = bool(root and _EPHEMERAL_ROOT_RE.match(root))
            has_if_guard = any(isinstance(anc, ast.If) for anc in stack)
            in_dispatch = function in dispatch
            sites.append(
                WriteSite(
                    relpath=relpath,
                    function=function,
                    kind=_call_kind(node),
                    target=_target_label(target_expr),
                    ephemeral=ephemeral,
                    conditional=has_if_guard or in_dispatch,
                )
            )
        for child in ast.iter_child_nodes(node):
            visit(child, stack + [node])

    visit(tree, [])
    return sites


def _scan_domain() -> list[WriteSite]:
    all_sites: list[WriteSite] = []
    for path in _domain_files():
        all_sites.extend(_scan_file(path))
    return all_sites


def _durable_sites(sites: list[WriteSite]) -> list[WriteSite]:
    return [s for s in sites if not s.ephemeral]


def _load_allowlist() -> dict:
    return tomllib.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))


def _allowlist_index(allowlist: dict) -> dict[tuple[str, str, str, str], dict]:
    index: dict[tuple[str, str, str, str], dict] = {}
    for entry in allowlist.get("durable_write_sites", []):
        key = (entry["path"], entry["function"], entry["kind"], entry["target"])
        index[key] = entry
    return index


# ===========================================================================
# Write-surface custody guard
# ===========================================================================


def test_domain_scan_is_non_vacuous() -> None:
    """The scan must actually see the training-emission write surface."""

    sites = _scan_domain()
    durable = _durable_sites(sites)
    assert durable, "Durable write-site scan found zero sites; scan scope is broken"
    scanned_files = {s.relpath for s in durable}
    assert "src/rlrmp/train/run_spec_authoring.py" in scanned_files
    assert "scripts/train_minimax.py" not in scanned_files


def test_durable_write_sites_match_allowlist() -> None:
    """Deny-by-default: every durable raw-write must be allowlisted."""

    sites = _scan_domain()
    allowlist = _load_allowlist()
    allowed = _allowlist_index(allowlist)

    found_keys = {s.key for s in _durable_sites(sites)}
    unlisted = sorted(found_keys - set(allowed))
    assert not unlisted, (
        "New durable raw-write site(s) to a declared durable root without an "
        f"allowlist entry: {unlisted}. Route the write through the sanctioned "
        "feedbax custody writer / a run-spec emitter, or add an entry to "
        f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)} naming the owning ledger issue "
        "and role. (Local scratch that is atomically renamed into place must "
        "target a `tmp`-rooted path so it classifies as ephemeral.)"
    )


def test_allowlist_has_no_dead_entries() -> None:
    """Every allowlisted durable site must still exist in the tree."""

    sites = _scan_domain()
    allowlist = _load_allowlist()
    found_keys = {s.key for s in _durable_sites(sites)}
    dead = sorted(set(_allowlist_index(allowlist)) - found_keys)
    assert not dead, (
        "Allowlist names durable write site(s) that no longer exist in the "
        f"scanned domain: {dead}. Remove the stale entries (shrinking the "
        "custody inventory is ceremony-free)."
    )


def test_allowlist_entries_carry_owner_and_role() -> None:
    allowlist = _load_allowlist()
    issue_re = re.compile(r"^[0-9a-f]{7}$")
    allowed_roles = {
        "run_spec",
        "model_artifact",
        "history_artifact",
        "adversary_artifact",
        "log_data_product",
        "write_helper",
        "materialized_checkpoint",
    }
    entries = allowlist.get("durable_write_sites", [])
    assert entries, "write-surface allowlist declares zero durable sites"
    for entry in entries:
        assert issue_re.match(entry.get("owner", "")), (
            f"Durable-write allowlist entry {entry} missing a valid 7-char owner"
        )
        assert entry.get("role") in allowed_roles, (
            f"Durable-write allowlist entry {entry} has unknown role "
            f"{entry.get('role')!r}; allowed: {sorted(allowed_roles)}"
        )
        assert isinstance(entry.get("conditional"), bool), (
            f"Durable-write allowlist entry {entry} missing bool `conditional`"
        )


def test_conditional_flag_matches_generated_control_flow() -> None:
    """Allowlisted `conditional` must equal the AST-generated conditionality.

    The declared conditionality cannot drift from the writer control flow
    actually present in the code.
    """

    sites = _scan_domain()
    allowlist = _load_allowlist()
    allowed = _allowlist_index(allowlist)

    mismatches = []
    for site in _durable_sites(sites):
        entry = allowed.get(site.key)
        if entry is None:
            continue  # covered by test_durable_write_sites_match_allowlist
        if bool(entry["conditional"]) != site.conditional:
            mismatches.append(
                (site.key, f"declared={entry['conditional']} generated={site.conditional}")
            )
    assert not mismatches, (
        f"Allowlisted conditional flag disagrees with generated control flow: {mismatches}"
    )


def test_conditional_durable_write_surface_is_absent() -> None:
    """The old conditional-emitter branch matrix is gone, not silently empty."""

    sites = _scan_domain()
    conditional_durable = sorted(s.key for s in _durable_sites(sites) if s.conditional)
    assert not conditional_durable, (
        "Conditional durable raw-write site(s) reappeared. Restore an explicit "
        "non-vacuous guard for that surface instead of relying on the deleted "
        f"branch-matrix machinery: {conditional_durable}"
    )
    conditional_entries = [
        entry for entry in _load_allowlist().get("durable_write_sites", []) if entry["conditional"]
    ]
    assert not conditional_entries, (
        "Allowlist declares conditional durable entries even though the live "
        f"durable conditional surface is absent: {conditional_entries}"
    )


def test_ephemeral_writes_are_tmp_staged() -> None:
    """Ephemeral classification is sound: every ephemeral site is tmp-rooted."""

    sites = _scan_domain()
    ephemeral = [s for s in sites if s.ephemeral]
    for site in ephemeral:
        assert _EPHEMERAL_ROOT_RE.match(site.target.split("/")[0].split("[")[0]), (
            f"Ephemeral site {site.key} is not tmp-rooted; classification is unsound"
        )


def test_write_surface_negative_canary_flags_new_durable_write() -> None:
    """A new raw durable write to output_dir is flagged as an unlisted site."""

    src = (
        "import numpy as np\n"
        "def emit(output_dir, model):\n"
        "    fbx_save(output_dir / 'sneaky.eqx', model)\n"
        "    np.savez(output_dir / 'sneaky.npz', x=1)\n"
    )
    tree = ast.parse(src)
    found: set[tuple] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_raw_write_call(node):
            expr = _write_target_expr(node)
            root = _target_root_name(expr)
            if not (root and _EPHEMERAL_ROOT_RE.match(root)):
                found.add((_call_kind(node), _target_label(expr)))
    assert ("fbx_save", "output_dir/sneaky.eqx") in found
    assert ("savez", "output_dir/sneaky.npz") in found


def test_write_surface_negative_canary_ignores_tmp_staged_write() -> None:
    """A raw write staged under tmp/ is not flagged as a durable-root write."""

    src = (
        "def stage(tmp, model):\n"
        "    import equinox as eqx\n"
        "    eqx.tree_serialise_leaves(tmp / 'model.eqx', model)\n"
    )
    tree = ast.parse(src)
    classified = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_raw_write_call(node):
            expr = _write_target_expr(node)
            root = _target_root_name(expr)
            classified.add(
                (
                    _call_kind(node),
                    _target_label(expr),
                    bool(root and _EPHEMERAL_ROOT_RE.match(root)),
                )
            )
    assert classified == {("tree_serialise_leaves", "tmp/model.eqx", True)}


def test_single_custody_pytree_writer() -> None:
    """rlrmp routes checkpoint PyTree custody through the one feedbax writer.

    The single feedbax-owned durable transaction writer is
    ``feedbax.training.checkpoint_custody.write_checkpoint_transaction``; rlrmp
    reaches it only through the two adapters in
    ``src/rlrmp/runtime/checkpoint_custody.py``. No other rlrmp module may call
    ``write_checkpoint_transaction`` directly.
    """

    adapter = REPO_ROOT / "src" / "rlrmp" / "runtime" / "checkpoint_custody.py"
    adapter_src = adapter.read_text(encoding="utf-8")
    assert "write_checkpoint_transaction" in adapter_src
    assert "from feedbax.training.checkpoint_custody import" in adapter_src

    callers: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "rlrmp").rglob("*.py")):
        if path == adapter:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "write_checkpoint_transaction":
                    callers.append(path.relative_to(REPO_ROOT).as_posix())
    assert not callers, (
        "feedbax write_checkpoint_transaction is called outside the sanctioned "
        f"rlrmp custody adapter: {sorted(set(callers))}"
    )


# ===========================================================================
# Provenance-lineage invariants 1-6 (assert-integrate)
# ===========================================================================


def test_invariant1_no_comparable_spec_parity_reconstruction() -> None:
    """Inv 1: no ``comparable_training_spec()``-style parity reconstruction.

    The native ``TrainingRunManifest`` is the sole source of truth; there is no
    derivation path that reconstructs parity from a legacy run.json. Enforced by
    absence across the whole source + script tree.
    """

    offenders: list[str] = []
    for base in ("src", "scripts"):
        for path in sorted((REPO_ROOT / base).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and "comparable_training_spec" in node.name:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")
    assert not offenders, (
        f"comparable_training_spec-style parity reconstruction present: {offenders}"
    )
    resolver = (REPO_ROOT / "src" / "rlrmp" / "runtime" / "run_specs.py").read_text(
        encoding="utf-8"
    )
    assert "resolve_run_record" in resolver and "TrainingRunManifest" in resolver


def test_invariant2_post_run_provenance_field_set_complete(tmp_path) -> None:
    """Inv 2: every native manifest carries a full post_run_provenance stamp."""

    from rlrmp.runtime.training_run_specs import (
        POST_RUN_SCHEMA_VERSION,
        attach_post_run_provenance,
    )

    run_spec = {"issue": "f5d9695", "feedbax_graph": {}}
    stamped = attach_post_run_provenance(
        run_spec,
        run_spec_path=tmp_path / "run.json",
        artifact_dir=tmp_path,
        manifest_root=tmp_path / "feedbax_runs",
    )
    prov = stamped["post_run_provenance"]
    assert prov["schema_version"] == POST_RUN_SCHEMA_VERSION
    assert set(prov["rlrmp"]) >= {"commit", "branch", "dirty", "remote"}
    assert set(prov["feedbax"]) >= {"commit", "branch", "dirty", "remote"}
    assert set(prov["schemas"]) >= {
        "post_run_provenance",
        "feedbax_manifest",
        "feedbax_provider",
    }
    root = prov["feedbax_manifest_root"]
    assert set(root) >= {"path", "absolute_path_sha256", "env"}
    assert re.fullmatch(r"[0-9a-f]{64}", root["absolute_path_sha256"])
    assert "feedbax_graph" in prov and "graph_spec_version" in prov["feedbax_graph"]


def test_invariant3_consumed_data_identity_shape() -> None:
    """Inv 3: consumed-data identities carry role / schema / hash."""

    from rlrmp.runtime.training_run_specs import (
        CONSUMED_DATA_IDENTITIES_KEY,
        add_consumed_data_identity,
    )

    updated = add_consumed_data_identity(
        {}, role="calibration_table", schema="rlrmp.data_product.v1", hash="abc123"
    )
    entries = updated[CONSUMED_DATA_IDENTITIES_KEY]
    assert entries == [
        {"role": "calibration_table", "schema": "rlrmp.data_product.v1", "hash": "abc123"}
    ]
    for bad in (
        {"role": "", "schema": "s", "hash": "h"},
        {"role": "r", "schema": "", "hash": "h"},
        {"role": "r", "schema": "s", "hash": ""},
    ):
        with pytest.raises(ValueError):
            add_consumed_data_identity({}, **bad)


def test_invariant4_legacy_discriminators_are_explicit() -> None:
    """Inv 4: legacy runs surface explicit not_found / archive-only discriminators."""

    resolver = (REPO_ROOT / "src" / "rlrmp" / "runtime" / "run_specs.py").read_text(
        encoding="utf-8"
    )
    assert "not_found" in resolver and "archive-only" in resolver, (
        "run-record resolver must raise an explicit not_found / archive-only "
        "discriminator for legacy runs, never silently miss or misclassify"
    )
    migrations = (REPO_ROOT / "src" / "rlrmp" / "runtime" / "spec_migrations.py").read_text(
        encoding="utf-8"
    )
    assert "ArchiveOnlySpecError" in migrations and "archive-only" in migrations


def test_invariant5_single_native_manifest_emitter() -> None:
    """Inv 5: exactly one TrainingRunManifest construction site (no parallel emitter)."""

    construction_sites: list[str] = []
    for base in ("src", "scripts"):
        for path in sorted((REPO_ROOT / base).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "TrainingRunManifest":
                        construction_sites.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert construction_sites == ["src/rlrmp/runtime/training_run_specs.py:1021"] or (
        len(construction_sites) == 1
        and construction_sites[0].startswith("src/rlrmp/runtime/training_run_specs.py:")
    ), f"expected exactly one native emitter, found: {construction_sites}"


def test_invariant6_single_substrate_no_parallel_versioning_registry() -> None:
    """Inv 6: provenance-versioning schema machinery lives only in feedbax migrations."""

    substrate_classes = {"SpecSchemaRegistry", "SpecSchemaFamily", "SchemaMigration"}
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "rlrmp").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in substrate_classes:
                offenders.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")
    assert not offenders, (
        "rlrmp defines a parallel provenance-versioning substrate class: "
        f"{offenders}. Schema families/migrations must come from "
        "feedbax.contracts.migrations."
    )
    migrations = (REPO_ROOT / "src" / "rlrmp" / "runtime" / "spec_migrations.py").read_text(
        encoding="utf-8"
    )
    assert "from feedbax.contracts.migrations import" in migrations
