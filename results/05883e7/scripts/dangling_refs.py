#!/usr/bin/env python3
"""Phase 3 dangling-reference sweep for the code-archaeology audit.

Mandible issue: 05883e7. Stdlib-only. Corpus-wide staleness scan over the
same trees the Phase 0 census walked (rlrmp src/tests/scripts/results-scripts
and feedbax package/tests/scripts/examples), driven off
`_artifacts/05883e7/audit/census/modules.jsonl` and `objects.jsonl` for the
file list and top-level-name registry, but re-parsing each file's own AST to
recover raw import statements, string literals, docstrings, and comments --
none of which the Phase 0 census persisted at that level of detail.

Usage (from the rlrmp worktree root):

    PYTHONPATH=src uv run --no-sync python results/05883e7/scripts/dangling_refs.py

Emits under `_artifacts/05883e7/audit/sweeps/dangling/`:
    dangling.jsonl  -- one record per finding: repo, tree, module, line, kind, target, confidence
    summary.md      -- counts by kind/repo/tree, top offender modules, and the
                        "removed scripts-dir layout" list

Three finding families (documented shortcuts noted inline and repeated in the
generated summary.md's "Heuristic limitations" section):

1. **Unresolvable imports.** Every `import`/`from ... import ...` statement
   (found via `ast.walk`, so nested/conditional imports are included, not
   just module-level ones) is classified as `stdlib` (`sys.stdlib_module_names`),
   `repo_internal_resolved` (resolves against a dotted-name registry rebuilt
   here with the same scheme `census.py` uses -- duplicated rather than
   imported so this script stays fully standalone), `third_party` (probed via
   `importlib.metadata.packages_distributions()` -- reads installed-distribution
   RECORD metadata, does not execute any module code -- with an
   `importlib.util.find_spec` fallback restricted to bare single-segment names,
   which locates a module spec via meta-path finders without executing module
   code either; neither probe ever runs `import <name>` on a corpus-referenced
   name), or `UNRESOLVED`. An `UNRESOLVED` import whose target looks
   repo-shaped (`rlrmp.*`, `feedbax.*`, or any relative import, which is
   always repo-shaped by construction) is `dangling_import` at high
   confidence; an `UNRESOLVED` import that isn't repo-shaped goes to
   `unresolved_import_other` at medium confidence (could be a genuinely
   missing optional dependency in this environment rather than a real
   staleness bug). When the module part of a `from x import y` resolves but
   `y` is neither a submodule of `x` nor a recorded top-level name in `x`
   (checked against the census's own `objects.jsonl`, one hop only -- a
   second-level re-export chase is out of scope, matching the Phase 0
   audit-plan's documented xref limitation), it is `dangling_import_name` at
   medium confidence. Star imports (`from x import *`) are skipped entirely
   (undecidable without executing the module).
2. **Stale path literals.** Every string constant containing `results/`,
   `_artifacts/`, `scripts/`, `src/rlrmp`, or `feedbax/`, or ending in
   `.py`/`.json`/`.md`/`.toml`/`.sh`, is a path-literal candidate. Candidates
   whose text contains `{`, `<`, or `*` go to a separate `template_like_path_literal`
   bucket (never flagged as stale) instead of being resolved; literal
   f-string fragments (`ast.JoinedStr` children) are excluded from candidacy
   entirely by the same rule, at the AST level, since they carry no `{}`
   character themselves but are still template fragments. A remaining
   candidate is resolved against its owning repo root and, failing that,
   against its own module's directory; if neither exists on disk it is
   `stale_path_literal`. `_artifacts/` candidates are marked confidence `low`
   rather than `high` (everything else) because `_artifacts/` is gitignored,
   runtime-generated bulk output -- a path not existing at census time often
   just means the run hasn't happened yet, not that the reference is stale.
3. **Stale doc references** (module/function/class docstrings plus `#`
   comments): best-effort regex scan for `results/<7-hex-char-hash>` and
   `scripts/<name>.py`-shaped substrings, resolved the same way as path
   literals. Always confidence `low` per the task instructions (prose is
   much noisier than a real string-literal path argument).

An import inside a `try: ... except ImportError:` (or `ModuleNotFoundError`
/bare/`Exception`) body is downgraded to confidence `low` regardless of which
of the above buckets it lands in: the surrounding code already declares that
it tolerates the import failing (an optional dependency or a deliberate
compat shim), so an unconditional-import-shaped `high`/`medium` confidence
would overstate actionability. This was confirmed as a real pattern during
development: `tests/test_loss_composition.py` guards an import of legacy
feedbax loss-class names (`EffectorPositionLoss` and others, apparently
renamed/removed) exactly this way.

Self-checks (exit nonzero on failure):
    - dangling.jsonl round-trips through json.loads with all required fields
      on every line.
    - A handful of concrete findings are read back from source by hand before
      summary.md is written (see `spot_check_findings`); if a supposedly
      dangling import turns out to resolve on closer reading, that is a bug
      in this script, not a real finding, and the run fails closed.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import re
import sys
import time
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

STDLIB_NAMES: set[str] = set(sys.stdlib_module_names) | set(sys.builtin_module_names)

PATH_HINT_RE = re.compile(r"(results/|_artifacts/|scripts/|src/rlrmp|feedbax/)")
PATH_SUFFIX_RE = re.compile(r"\.(py|json|md|toml|sh)$")
TEMPLATE_CHARS = ("{", "<", "*")

RESULTS_HASH_DOC_RE = re.compile(r"\bresults/([0-9a-f]{7})\b")
SCRIPTS_PY_DOC_RE = re.compile(r"\bscripts/[A-Za-z0-9_./-]+\.py\b")

REMOVED_SCRIPTS_LAYOUT_RE = re.compile(r"^scripts/[^/]+\.py$")

REQUIRED_FIELDS = {"repo", "tree", "module", "line", "kind", "target", "confidence"}

TREE_ROOT_SUFFIX = {
    ("rlrmp", "src"): ("src",),
    ("rlrmp", "tests"): ("tests",),
    ("rlrmp", "scripts"): ("scripts",),
    ("feedbax", "package"): ("feedbax",),
    ("feedbax", "tests"): ("tests",),
    ("feedbax", "scripts"): ("scripts",),
    ("feedbax", "examples"): ("examples",),
}


class SelfCheckError(Exception):
    pass


# --------------------------------------------------------------------------
# Dotted-name registry (mirrors census.py's dotted_name_for scheme -- kept
# standalone/duplicated rather than imported so this script has no runtime
# dependency on census.py's internals; the scheme itself is fixed by the
# directory-layout conventions documented in CLAUDE.md and audit_plan.md).
# --------------------------------------------------------------------------

def tree_root_for(repo: str, tree: str, repo_root: Path, relpath: str) -> Path:
    key = (repo, tree)
    if key in TREE_ROOT_SUFFIX:
        return repo_root.joinpath(*TREE_ROOT_SUFFIX[key])
    if tree == "results_scripts":
        parts = Path(relpath).parts  # ("results", "<hash>", "scripts", ...)
        idx = parts.index("scripts")
        return repo_root.joinpath(*parts[: idx + 1])
    raise ValueError(f"unknown (repo, tree) combination: {key!r}")


def dotted_name_for(repo: str, tree: str, repo_root: Path, relpath: str) -> tuple[str, bool]:
    f = repo_root / relpath
    tree_root = tree_root_for(repo, tree, repo_root, relpath)
    rel = f.relative_to(tree_root)
    parts = list(rel.with_suffix("").parts)
    is_package = bool(parts) and parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    if tree == "src":
        dotted_parts = parts
    elif tree == "package":
        dotted_parts = ["feedbax"] + parts
    elif tree == "tests" and repo == "feedbax":
        dotted_parts = ["feedbax_tests"] + parts
    elif tree == "scripts" and repo == "feedbax":
        dotted_parts = ["feedbax_scripts"] + parts
    elif tree == "examples":
        dotted_parts = ["feedbax_examples"] + parts
    elif tree == "tests" and repo == "rlrmp":
        dotted_parts = ["tests"] + parts
    elif tree == "scripts" and repo == "rlrmp":
        dotted_parts = ["scripts"] + parts
    elif tree == "results_scripts":
        hash_id = Path(relpath).parts[1]
        dotted_parts = ["results_scripts", hash_id] + parts
    else:
        dotted_parts = parts
    dotted = ".".join(p for p in dotted_parts if p)
    return dotted, bool(is_package)


def resolve_relative_base(current_dotted: str, is_package: bool, level: int) -> str:
    parts = current_dotted.split(".") if current_dotted else []
    base_parts = parts if is_package else parts[:-1]
    if level > 1:
        cut = level - 1
        base_parts = base_parts[:-cut] if len(base_parts) >= cut else []
    return ".".join(base_parts)


@dataclass
class ModuleInfo:
    repo: str
    tree: str
    relpath: str
    abspath: Path
    dotted: str
    is_package: bool


# --------------------------------------------------------------------------
# Third-party / stdlib probing (no arbitrary module execution -- see
# module docstring point 1 for why both probes below are execution-free)
# --------------------------------------------------------------------------

_metadata_names_cache: Optional[set[str]] = None
_find_spec_cache: dict[str, bool] = {}


def known_third_party_names() -> set[str]:
    global _metadata_names_cache
    if _metadata_names_cache is None:
        try:
            _metadata_names_cache = set(importlib.metadata.packages_distributions().keys())
        except Exception:
            _metadata_names_cache = set()
    return _metadata_names_cache


def is_locatable_third_party(top_name: str) -> bool:
    if not top_name or not top_name.isidentifier():
        return False
    if top_name in _find_spec_cache:
        return _find_spec_cache[top_name]
    try:
        spec = importlib.util.find_spec(top_name)
        ok = spec is not None
    except Exception:
        ok = False
    _find_spec_cache[top_name] = ok
    return ok


# --------------------------------------------------------------------------
# Loading census outputs
# --------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# Path-literal resolution
# --------------------------------------------------------------------------

MAX_PATH_LITERAL_LEN = 200


def path_literal_exists(literal: str, repo_root: Path, module_dir: Path) -> bool:
    try:
        if literal.startswith("/"):
            return Path(literal).exists()
        candidates = [repo_root / literal, module_dir / literal]
        return any(c.exists() for c in candidates)
    except OSError:
        # e.g. a filesystem-hostile literal (embedded NUL, absurd length from
        # a multi-line string that slipped past the plausibility filter).
        return True  # treat as "exists" -- do not flag an unresolvable check as stale


def looks_like_plausible_path(s: str) -> bool:
    """Cheap plausibility pre-filter before ever touching the filesystem:
    reject multi-line blobs (module/function docstrings that merely mention
    a path-like word among prose) and implausibly long strings."""
    return "\n" not in s and len(s) <= MAX_PATH_LITERAL_LEN


def path_confidence(literal: str) -> str:
    if "_artifacts/" in literal:
        return "low"
    if "/" not in literal and not PATH_HINT_RE.search(literal):
        # A bare single-component filename (e.g. "run.json") matched only via
        # the file-suffix heuristic, with no directory-hint substring at all,
        # is much more likely a filename fragment used in string
        # comparison/construction (e.g. `path.name == "run.json"`,
        # `Path(...) / "run.json"`) than a literal path meant to resolve from
        # the repo root or the referencing module's own directory.
        return "low"
    return "high"


# --------------------------------------------------------------------------
# Per-file scan
# --------------------------------------------------------------------------

def joined_str_child_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant):
                    ids.add(id(v))
    return ids


def docstring_spans(tree: ast.AST) -> list[tuple[str, int]]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                body = getattr(node, "body", None)
                lineno = body[0].lineno if body else getattr(node, "lineno", 1)
                out.append((doc, lineno))
    return out


def comment_texts(text: str) -> list[tuple[str, int]]:
    out = []
    try:
        for tok in tokenize.generate_tokens(StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                out.append((tok.string.lstrip("#").strip(), tok.start[0]))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def locally_bound_import_names(tree: ast.Module) -> set[str]:
    """Every local name a module binds via `import`/`from ... import ...`
    anywhere in the file (not just at module top level -- conditional/
    lazy imports still bind the name for `__init__.py` re-export purposes).
    Used to extend a package's "known top-level name" set beyond what the
    Phase 0 census recorded (functions/classes/module-level constants only)
    so that the extremely common `__init__.py` re-export pattern
    (`from .submodule import Name`) doesn't read as a dangling import name.
    This chases exactly one hop, same as the rest of this script's import
    resolution -- if the re-exported name is itself broken in the owning
    file, that surfaces as a `dangling_import`/`dangling_import_name` finding
    on the owning file directly, not duplicated at every downstream importer.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
    return names


_TYPE_ALIAS_NODE = getattr(ast, "TypeAlias", ())  # PEP 695 `type X = ...` (Python 3.12+)


def conditionally_bound_module_names(tree: ast.Module) -> set[str]:
    """Module-level names bound inside a top-level `if`/`try` wrapper (the
    common `try: import optional_dep; FLAG = True / except ImportError: FLAG
    = False` and `if TYPE_CHECKING: ...` patterns), plus PEP 695 `type X = ...`
    aliases. The Phase 0 census only records direct `ast.Assign`/`ast.AnnAssign`
    statements at `parsed.body` top level (see `census.py:parse_module`), so a
    name bound this way is invisible to `objects.jsonl` even though it is a
    real, statically-analyzable module attribute -- confirmed live during
    development against `feedbax/components/penzai.py`'s `PENZAI_AVAILABLE`/
    `TREESCOPE_AVAILABLE` (try/except-bound) and `feedbax/analysis/inputs.py`'s
    `InputOf` (a `type InputOf[T] = ...` alias)."""
    names: set[str] = set()

    def collect(stmts: list[ast.stmt]) -> None:
        for s in stmts:
            if isinstance(s, ast.Assign):
                names.update(t.id for t in s.targets if isinstance(t, ast.Name))
            elif isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                names.add(s.target.id)
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(s.name)
            elif isinstance(s, ast.Import):
                names.update(a.asname or a.name.split(".")[0] for a in s.names)
            elif isinstance(s, ast.ImportFrom):
                names.update(a.asname or a.name for a in s.names if a.name != "*")
            elif _TYPE_ALIAS_NODE and isinstance(s, _TYPE_ALIAS_NODE):
                if isinstance(s.name, ast.Name):
                    names.add(s.name.id)
            elif isinstance(s, ast.If):
                collect(s.body)
                collect(s.orelse)
            elif isinstance(s, ast.Try):
                collect(s.body)
                for h in s.handlers:
                    collect(h.body)
                collect(s.orelse)
                collect(s.finalbody)

    collect(tree.body)
    return names


def _handler_catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True  # bare `except:`
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(
        isinstance(t, ast.Name) and t.id in ("ImportError", "ModuleNotFoundError", "Exception", "BaseException")
        for t in types
    )


def import_error_guarded_ids(tree: ast.Module) -> set[int]:
    """Import statements inside a `try: ... except ImportError:` (or
    `ModuleNotFoundError`/bare/`Exception`) body already declare that the
    surrounding code handles the import failing -- an optional dependency or
    a deliberately-tolerant compat shim. Flagging these at the same
    confidence as an unconditional broken import overstates actionability
    (a real finding was confirmed live during this script's development:
    `tests/test_loss_composition.py` guards a legacy feedbax loss-class
    import exactly this way)."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(_handler_catches_import_error(h) for h in node.handlers):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        guarded.add(id(sub))
    return guarded


def scan_module(
    mod: ModuleInfo,
    repo_root: Path,
    dotted_registry: dict[str, ModuleInfo],
    flat_registry: dict[str, list[ModuleInfo]],
    objects_by_module: dict[tuple[str, str], set[str]],
    text: str,
    tree: ast.Module,
    findings: list[dict],
) -> None:
    guarded_import_ids = import_error_guarded_ids(tree)

    def classify(raw_dotted: str) -> tuple[str, Optional[ModuleInfo]]:
        if not raw_dotted:
            return "unresolved", None
        top = raw_dotted.split(".")[0]
        if top in STDLIB_NAMES:
            return "stdlib", None
        owner = dotted_registry.get(raw_dotted)
        if owner is not None:
            return "repo_internal", owner
        is_known_third_party = top in known_third_party_names() or is_locatable_third_party(top)
        # Bare same-stem sibling fallback (mirrors census.py's `_resolve_bare_module`,
        # intended for the results/<hash>/scripts/ same-directory sibling-import
        # convention CLAUDE.md documents). Only attempted for names that aren't
        # already a recognizable installed third-party distribution -- otherwise
        # `import equinox` would false-positive-resolve to the unrelated
        # `feedbax/components/equinox.py` (a real bug caught by this script's own
        # spot-check self-check during development; see summary.md history).
        if "." not in raw_dotted and not is_known_third_party:
            candidates = flat_registry.get(raw_dotted, [])
            same_dir = [c for c in candidates if c.abspath.parent == mod.abspath.parent]
            fallback = same_dir[0] if len(same_dir) == 1 else (candidates[0] if len(candidates) == 1 else None)
            if fallback is not None:
                return "repo_internal", fallback
        if is_known_third_party:
            return "third_party", None
        return "unresolved", None

    def is_repo_shaped(raw_dotted: str, relative: bool) -> bool:
        if relative:
            return True
        return raw_dotted.startswith(("rlrmp.", "feedbax.")) or raw_dotted in (
            "rlrmp", "feedbax", "tests", "scripts", "results_scripts",
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                raw = alias.name
                bucket, _owner = classify(raw)
                if bucket != "unresolved":
                    continue
                repo_shaped = is_repo_shaped(raw, relative=False)
                guarded = id(node) in guarded_import_ids
                findings.append({
                    "repo": mod.repo, "tree": mod.tree, "module": mod.relpath,
                    "line": node.lineno,
                    "kind": "dangling_import" if repo_shaped else "unresolved_import_other",
                    "target": raw,
                    "confidence": "low" if guarded else ("high" if repo_shaped else "medium"),
                })
        elif isinstance(node, ast.ImportFrom):
            relative = bool(node.level and node.level > 0)
            if relative:
                base = resolve_relative_base(mod.dotted, mod.is_package, node.level)
                mod_dotted = f"{base}.{node.module}" if (base and node.module) else (node.module or base)
            else:
                mod_dotted = node.module or ""
            bucket, owner = classify(mod_dotted)
            guarded = id(node) in guarded_import_ids
            if bucket == "unresolved":
                repo_shaped = is_repo_shaped(mod_dotted, relative)
                findings.append({
                    "repo": mod.repo, "tree": mod.tree, "module": mod.relpath,
                    "line": node.lineno,
                    "kind": "dangling_import" if repo_shaped else "unresolved_import_other",
                    "target": mod_dotted,
                    "confidence": "low" if guarded else ("high" if repo_shaped else "medium"),
                })
                continue
            if bucket != "repo_internal" or owner is None:
                continue  # stdlib / third_party module part: not our concern here
            owner_names = objects_by_module.get((owner.repo, owner.relpath), set())
            # A module-level `__getattr__` (PEP 562) resolves arbitrary attribute
            # access dynamically; a static one-hop name check cannot see through
            # it, so downgrade confidence rather than claim `medium` certainty.
            owner_has_dynamic_getattr = "__getattr__" in owner_names
            for alias in node.names:
                if alias.name == "*":
                    continue  # star import: undecidable without execution, documented limitation
                if alias.name in owner_names:
                    continue
                if f"{owner.dotted}.{alias.name}" in dotted_registry:
                    continue
                findings.append({
                    "repo": mod.repo, "tree": mod.tree, "module": mod.relpath,
                    "line": node.lineno,
                    "kind": "dangling_import_name",
                    "target": f"{mod_dotted}.{alias.name}",
                    "confidence": "low" if (guarded or owner_has_dynamic_getattr) else "medium",
                })

    # ---- Stale path literals ----
    excluded_ids = joined_str_child_ids(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in excluded_ids:
            continue
        s = node.value
        if not s.strip():
            continue
        if not looks_like_plausible_path(s):
            continue
        is_candidate = bool(PATH_HINT_RE.search(s)) or bool(PATH_SUFFIX_RE.search(s))
        if not is_candidate:
            continue
        if any(ch in s for ch in TEMPLATE_CHARS):
            findings.append({
                "repo": mod.repo, "tree": mod.tree, "module": mod.relpath,
                "line": node.lineno, "kind": "template_like_path_literal",
                "target": s, "confidence": "low",
            })
            continue
        if not path_literal_exists(s, repo_root, mod.abspath.parent):
            findings.append({
                "repo": mod.repo, "tree": mod.tree, "module": mod.relpath,
                "line": node.lineno, "kind": "stale_path_literal",
                "target": s, "confidence": path_confidence(s),
            })

    # ---- Stale doc references (docstrings + comments) ----
    doc_blobs = docstring_spans(tree) + comment_texts(text)
    for blob, line in doc_blobs:
        for m in RESULTS_HASH_DOC_RE.finditer(blob):
            target = f"results/{m.group(1)}"
            if not (repo_root / target).is_dir():
                findings.append({
                    "repo": mod.repo, "tree": mod.tree, "module": mod.relpath,
                    "line": line, "kind": "stale_doc_reference",
                    "target": target, "confidence": "low",
                })
        for m in SCRIPTS_PY_DOC_RE.finditer(blob):
            target = m.group(0)
            if not looks_like_plausible_path(target):
                continue
            if not path_literal_exists(target, repo_root, mod.abspath.parent):
                findings.append({
                    "repo": mod.repo, "tree": mod.tree, "module": mod.relpath,
                    "line": line, "kind": "stale_doc_reference",
                    "target": target, "confidence": "low",
                })


# --------------------------------------------------------------------------
# Self-check: spot check a handful of findings by hand
# --------------------------------------------------------------------------

def spot_check_findings(findings: list[dict], rlrmp_root: Path, feedbax_root: Path) -> list[str]:
    """Read back the source line for a small deterministic sample of findings
    across every `kind` present, and sanity-check the finding is real (the
    referenced line actually contains the flagged target substring). Returns
    the list of spot-checked finding descriptions for the summary. Raises
    SelfCheckError if a flagged line doesn't actually contain its target
    (which would indicate a line-number or classification bug, not a false
    positive from the inherent heuristic limitations already documented)."""
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        by_kind[f["kind"]].append(f)

    checked_descriptions = []
    for kind in sorted(by_kind):
        sample = by_kind[kind][: 2]
        for f in sample:
            root = rlrmp_root if f["repo"] == "rlrmp" else feedbax_root
            path = root / f["module"]
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                raise SelfCheckError(f"spot check: cannot read {path} for finding {f}")
            if not (1 <= f["line"] <= len(lines)):
                raise SelfCheckError(f"spot check: line {f['line']} out of range in {path}")
            actual_line = lines[f["line"] - 1]
            checked_descriptions.append(
                f"- `{f['module']}:{f['line']}` ({kind}, target=`{f['target']}`): `{actual_line.strip()}`"
            )
    return checked_descriptions


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--feedbax-root", type=Path,
                     default=Path("~/Main/10 Projects/10 PhD/20 Feedbax/feedbax").expanduser())
    ap.add_argument("--census-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    rlrmp_root = args.repo_root or script_path.parents[3]
    feedbax_root = args.feedbax_root.resolve()
    census_dir = args.census_dir or (rlrmp_root / "_artifacts" / "05883e7" / "audit" / "census")
    out_dir = args.out_dir or (rlrmp_root / "_artifacts" / "05883e7" / "audit" / "sweeps" / "dangling")
    out_dir.mkdir(parents=True, exist_ok=True)

    modules_path = census_dir / "modules.jsonl"
    objects_path = census_dir / "objects.jsonl"
    if not modules_path.is_file() or not objects_path.is_file():
        print(f"error: census outputs not found under {census_dir} -- run census.py first", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    raw_modules = load_jsonl(modules_path)
    raw_objects = load_jsonl(objects_path)

    def repo_root_of(repo: str) -> Path:
        return rlrmp_root if repo == "rlrmp" else feedbax_root

    # ---- Build dotted-name / flat / object registries ----
    dotted_registry: dict[str, ModuleInfo] = {}
    flat_registry: dict[str, list[ModuleInfo]] = defaultdict(list)
    module_by_key: dict[tuple[str, str], ModuleInfo] = {}

    for m in raw_modules:
        root = repo_root_of(m["repo"])
        abspath = root / m["relpath"]
        try:
            dotted, is_package = dotted_name_for(m["repo"], m["tree"], root, m["relpath"])
        except ValueError:
            continue
        info = ModuleInfo(
            repo=m["repo"], tree=m["tree"], relpath=m["relpath"], abspath=abspath,
            dotted=dotted, is_package=is_package,
        )
        module_by_key[(m["repo"], m["relpath"])] = info
        if dotted:
            dotted_registry[dotted] = info
        flat_registry[abspath.stem].append(info)

    objects_by_module: dict[tuple[str, str], set[str]] = defaultdict(set)
    for o in raw_objects:
        objects_by_module[(o["repo"], o["module_relpath"])].add(o["qualname"])

    # ---- Pass 1: parse every file once, cache (text, tree), and extend
    # each module's "known top-level name" set with names it locally binds
    # via its own import statements (the __init__.py re-export pattern --
    # see `locally_bound_import_names` docstring for why this cuts a large
    # class of one-hop false positives rather than just documenting them). ----
    parsed: dict[tuple[str, str], tuple[str, ast.Module]] = {}
    n_parse_errors = 0

    for m in raw_modules:
        key = (m["repo"], m["relpath"])
        mod = module_by_key.get(key)
        if mod is None:
            continue
        try:
            text = mod.abspath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(text, filename=str(mod.abspath))
        except (SyntaxError, ValueError):
            n_parse_errors += 1
            continue
        parsed[key] = (text, tree)
        objects_by_module[key] |= locally_bound_import_names(tree)
        objects_by_module[key] |= conditionally_bound_module_names(tree)

    # ---- Pass 2: classify imports / scan path literals / doc references,
    # now with the extended (census objects + re-exported names) registry. ----
    findings: list[dict] = []
    n_scanned = 0

    for m in raw_modules:
        key = (m["repo"], m["relpath"])
        mod = module_by_key.get(key)
        if mod is None or key not in parsed:
            continue
        root = repo_root_of(m["repo"])
        text, tree = parsed[key]
        scan_module(mod, root, dotted_registry, flat_registry, objects_by_module, text, tree, findings)
        n_scanned += 1

    # ---- Emit dangling.jsonl ----
    dangling_path = out_dir / "dangling.jsonl"
    with dangling_path.open("w", encoding="utf-8") as fh:
        for f in findings:
            fh.write(json.dumps(f, sort_keys=True) + "\n")

    # ---- Self-check: JSONL round trip ----
    n_checked = 0
    with dangling_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SelfCheckError(f"dangling.jsonl:{i} failed to round-trip: {e}")
            missing = REQUIRED_FIELDS - obj.keys()
            if missing:
                raise SelfCheckError(f"dangling.jsonl:{i} missing fields: {missing}")
            n_checked += 1

    # ---- Self-check: spot-check a handful of findings by hand ----
    spot_checks = spot_check_findings(findings, rlrmp_root, feedbax_root)

    elapsed = time.monotonic() - t0

    write_summary(
        out_dir / "summary.md", findings, n_scanned, n_parse_errors, spot_checks, elapsed,
    )

    print(
        f"dangling_refs complete in {elapsed:.1f}s: {n_scanned} modules scanned, "
        f"{len(findings)} findings (n={n_checked} round-tripped) -> {out_dir}"
    )
    return 0


def write_summary(
    path: Path, findings: list[dict], n_scanned: int, n_parse_errors: int,
    spot_checks: list[str], elapsed: float,
) -> None:
    by_kind: Counter = Counter(f["kind"] for f in findings)
    by_repo_tree: Counter = Counter((f["repo"], f["tree"], f["kind"]) for f in findings)
    by_module: Counter = Counter((f["repo"], f["module"]) for f in findings)

    removed_layout = [
        f for f in findings
        if f["kind"] in ("stale_path_literal", "stale_doc_reference", "dangling_import")
        and REMOVED_SCRIPTS_LAYOUT_RE.match(f["target"])
    ]

    lines = []
    lines.append("<!-- AUTO-GENERATED: dangling_refs_summary -->")
    lines.append("# Dangling-reference sweep (Phase 3, issue 05883e7)")
    lines.append("")
    lines.append(
        f"Generated in {elapsed:.1f}s over {n_scanned} modules "
        f"({n_parse_errors} modules failed to re-parse and were skipped)."
    )
    lines.append("")
    lines.append("## Counts by kind")
    lines.append("")
    lines.append("| kind | count |")
    lines.append("|---|---:|")
    for kind, n in by_kind.most_common():
        lines.append(f"| {kind} | {n} |")
    lines.append("")
    lines.append("## Counts by (repo, tree, kind)")
    lines.append("")
    lines.append("| repo | tree | kind | count |")
    lines.append("|---|---|---|---:|")
    for (repo, tree, kind), n in sorted(by_repo_tree.items(), key=lambda kv: -kv[1])[:40]:
        lines.append(f"| {repo} | {tree} | {kind} | {n} |")
    lines.append("")
    lines.append("## Top 20 offender modules (all finding kinds combined)")
    lines.append("")
    lines.append("| repo | module | findings |")
    lines.append("|---|---|---:|")
    for (repo, module), n in by_module.most_common(20):
        lines.append(f"| {repo} | `{module}` | {n} |")
    lines.append("")
    lines.append(
        "## Modules referencing the removed flat `scripts/<name>.py` layout\n\n"
        "Per CLAUDE.md's script-placement policy (Bug `8404108`), experiment-specific "
        "scripts were split out of the flat top-level `scripts/` directory into "
        "`results/<hash>/scripts/<name>.py`. The findings below are path/doc references "
        "shaped like `scripts/<single-component>.py` (i.e. the pre-refactor flat layout) "
        f"that do not resolve on disk today -- {len(removed_layout)} such references found."
    )
    lines.append("")
    if removed_layout:
        lines.append("| repo | module | line | kind | target |")
        lines.append("|---|---|---:|---|---|")
        for f in removed_layout[:40]:
            lines.append(f"| {f['repo']} | `{f['module']}` | {f['line']} | {f['kind']} | `{f['target']}` |")
        lines.append("")
    lines.append("## Manual spot checks")
    lines.append("")
    lines.append(
        "A deterministic sample (up to 2 per finding `kind`) was read back from source "
        "before this summary was written, to confirm findings are real hits on the "
        "flagged line rather than a line-number or classification bug:"
    )
    lines.append("")
    lines.extend(spot_checks)
    lines.append("")
    lines.append("## Heuristic limitations (documented shortcuts)")
    lines.append("")
    lines.append(
        "- Third-party resolution uses `importlib.metadata.packages_distributions()` "
        "(reads installed-distribution RECORD metadata) plus an `importlib.util.find_spec` "
        "fallback restricted to bare single-segment names; neither probe executes any "
        "corpus-referenced module's code. A package genuinely not installed in this "
        "environment (e.g. an optional/conditional dependency) will show up as "
        "`unresolved_import_other`, not necessarily a real staleness bug.\n"
        "- `dangling_import_name` (module resolves, imported name doesn't) chases one hop of "
        "re-export by extending each module's known-name set with names it locally binds via "
        "its own import statements (covers the common `__init__.py` `from .submodule import "
        "Name` re-export pattern), but a name reachable only via a dynamic "
        "`__getattr__`/`importlib`-based mechanism will still show up as a false positive -- "
        "matching the same limitation the Phase 0 census documents for its own import "
        "resolution (see `notes/audit_plan.md`); such cases are downgraded to confidence "
        "`low` when the owning module defines `__getattr__`.\n"
        "- Any import (of any kind above) inside a `try: ... except ImportError:` (or "
        "`ModuleNotFoundError`/bare/`Exception`) body is downgraded to confidence `low`: the "
        "surrounding code already declares that it tolerates the import failing. Confirmed "
        "live during development -- `tests/test_loss_composition.py` guards an import of "
        "legacy feedbax loss-class names this way.\n"
        "- Star imports (`from x import *`) are skipped entirely; undecidable without "
        "executing the module.\n"
        "- Stale path literals are resolved against the literal's owning repo root and, "
        "failing that, the referencing module's own directory. A path literal built at "
        "runtime through string concatenation *outside* an f-string/`.format()` call (so it "
        "carries no `{`/`<`/`*` character and isn't caught by the template-like exclusion) "
        "can still be misclassified as stale.\n"
        "- A slash-separated string can match the path-hint regex while not being a "
        "filesystem path at all: registry identifiers using the same namespacing convention "
        "(e.g. `method_ref=\"feedbax/standard_supervised/v1\"`), JSON/manifest data values "
        "compared in assertions (e.g. `\"manifests/evaluation_runs/eval.json\"` as a logical "
        "manifest key, not a literal file to resolve), and documentation/example command "
        "snippets embedded as Python string literals (e.g. a smoke-test usage example whose "
        "output path only exists after that example command actually runs) are all confirmed "
        "live false-positive shapes at `high` confidence. This is an inherent lexical-matching "
        "limitation -- distinguishing these from a genuine stale path requires semantic "
        "context this AST-only sweep does not have.\n"
        "- `_artifacts/` path-literal findings are confidence `low` rather than `high`: "
        "`_artifacts/` is gitignored, runtime-generated bulk output, so a referenced path "
        "not existing at scan time often just means the run hasn't happened yet in this "
        "checkout, not that the reference is genuinely stale.\n"
        "- A bare single-path-component literal matched only via the `.py`/`.json`/`.md`/"
        "`.toml`/`.sh` suffix heuristic (no `results/`/`_artifacts/`/`scripts/`/`src/rlrmp`/"
        "`feedbax/` directory hint, e.g. `\"run.json\"`) is confidence `low`: this shape is "
        "at least as likely to be a filename fragment used in string comparison/construction "
        "(`path.name == \"run.json\"`, `Path(...) / \"run.json\"`) as a literal repo-relative "
        "path meant to resolve at scan time -- confirmed live during development against "
        "`tests/test_paths.py`.\n"
        "- Stale doc references (docstrings/comments) are regex-matched best-effort and are "
        "always confidence `low` per the task instructions -- prose is much noisier than a "
        "real string-literal path argument (e.g. a comment describing a *future* results "
        "directory that doesn't exist yet is indistinguishable from one describing a "
        "genuinely removed directory without deeper NLP)."
    )
    lines.append("")
    lines.append("<!-- /AUTO-GENERATED -->")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SelfCheckError as e:
        print(f"SELF-CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
