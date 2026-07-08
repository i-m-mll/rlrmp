#!/usr/bin/env python3
"""Phase 0 deterministic census for the rlrmp+feedbax code archaeology audit.

Mandible issue: 05883e7. Stdlib-only (ast, json, subprocess, pathlib, re,
tokenize is not actually needed -- docstrings/comments come from ast). Censuses
rlrmp's `src/`, `tests/`, `scripts/`, `results/*/scripts/` trees plus feedbax's
`feedbax/` package, `tests/`, `scripts/`, and `examples/` trees, emitting a
per-module and per-top-level-object corpus plus a best-effort whole-corpus
cross-reference index under `_artifacts/05883e7/audit/census/`.

Usage (from the rlrmp worktree root):

    PYTHONPATH=src uv run --no-sync python results/05883e7/scripts/census.py

See results/05883e7/notes/audit_plan.md and notes/record_schema.md for the
phase this feeds and the classification schema later phases emit against.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

HP_LIKE_KEYS = {
    "lr", "learning_rate", "n_batches", "batch_size", "seed", "hidden_size",
    "n_replicates", "n_steps", "epsilon", "gamma", "sigma", "noise", "weight",
    "cost", "tau", "dt", "clip", "warmup", "schedule",
}

PATH_LITERAL_RE = re.compile(r"(results/|_artifacts/|\.json|\.eqx|\.npz|/)")
LEGACY_BANNER_RE = re.compile(r"LEGACY \(frozen")
TRAILER_RE = re.compile(r"^Mandible-Issue:\s*(\S+)", re.MULTILINE)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMERIC_TYPES = (int, float)  # bool is an int subclass; excluded explicitly below

SKIP_DIR_NAMES = {
    "__pycache__", ".venv", "venv", "_artifacts", "worktrees", ".git",
    "node_modules", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".idea",
}

MAX_CHUNK_OBJECT_LOC = 1500
MAX_CHUNK_OBJECTS = 40

GIT_LOG_FIELD_SEP = "\x1f"
GIT_LOG_RECORD_SEP = "\x1e"

REQUIRED_MODULE_FIELDS = {
    "id", "repo", "tree", "relpath", "loc", "n_objects", "imports",
    "docstring_first_line", "has_legacy_banner", "introducing_commit",
    "last_touch_commit", "mandible_issue_trailers", "parse_error",
}
REQUIRED_OBJECT_FIELDS = {
    "id", "repo", "tree", "module_relpath", "qualname", "kind", "line_start",
    "line_end", "loc", "decorators", "bases", "docstring_first_line",
    "is_private", "outbound_names", "n_numeric_literals",
    "max_float_container_size", "hp_like_keys", "n_string_path_literals",
    "registered", "in_all", "method_names", "n_methods", "last_touch_commit",
    "last_touch_date",
}
REQUIRED_XREF_FIELDS = {
    "id", "inbound_by_tree", "inbound_total", "unresolved_name_hits",
    "unresolved_by_tree", "string_hits", "in_all", "registered",
}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class ModuleRecord:
    id: str
    repo: str
    tree: str
    relpath: str
    abspath: Path
    dotted: str  # synthetic dotted name used for import resolution
    flat_name: str  # bare filename stem, for sibling-import fallback resolution
    is_package: bool  # True for __init__.py
    loc: int = 0
    n_objects: int = 0
    imports_resolved: set[str] = field(default_factory=set)
    docstring_first_line: Optional[str] = None
    has_legacy_banner: bool = False
    introducing_commit: Optional[dict[str, str]] = None
    last_touch_commit: Optional[dict[str, str]] = None
    mandible_issue_trailers: list[str] = field(default_factory=list)
    parse_error: bool = False
    object_ids_by_name: dict[str, str] = field(default_factory=dict)


@dataclass
class ObjectRecord:
    id: str
    repo: str
    tree: str
    module_relpath: str
    qualname: str
    kind: str
    line_start: int
    line_end: int
    loc: int
    decorators: list[str]
    bases: list[str]
    docstring_first_line: Optional[str]
    is_private: bool
    outbound_names: list[str]
    n_numeric_literals: int
    max_float_container_size: int
    hp_like_keys: list[str]
    n_string_path_literals: int
    registered: bool
    in_all: bool
    method_names: list[str]
    n_methods: int
    last_touch_commit: Optional[str] = None
    last_touch_date: Optional[str] = None


# --------------------------------------------------------------------------
# Tree discovery
# --------------------------------------------------------------------------

def iter_py_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_dir():
                if e.name in SKIP_DIR_NAMES:
                    continue
                stack.append(e)
            elif e.suffix == ".py":
                out.append(e)
    return sorted(out)


def discover_trees(rlrmp_root: Path, feedbax_root: Path) -> list[tuple[str, str, Path, Path]]:
    """Return (repo, tree, repo_root, file) tuples is too coarse; instead
    return (repo, tree, repo_root, tree_root) groups to walk."""
    groups: list[tuple[str, str, Path, Path]] = [
        ("rlrmp", "src", rlrmp_root, rlrmp_root / "src"),
        ("rlrmp", "tests", rlrmp_root, rlrmp_root / "tests"),
        ("rlrmp", "scripts", rlrmp_root, rlrmp_root / "scripts"),
        ("feedbax", "package", feedbax_root, feedbax_root / "feedbax"),
        ("feedbax", "tests", feedbax_root, feedbax_root / "tests"),
        ("feedbax", "scripts", feedbax_root, feedbax_root / "scripts"),
        ("feedbax", "examples", feedbax_root, feedbax_root / "examples"),
    ]
    # results/*/scripts/** (recursive, includes nested dirs like a vendored
    # compat shim package under results/183cba9/scripts/sqlalchemy/).
    results_dir = rlrmp_root / "results"
    if results_dir.is_dir():
        for hash_dir in sorted(results_dir.iterdir()):
            scripts_dir = hash_dir / "scripts"
            if scripts_dir.is_dir():
                groups.append(("rlrmp", "results_scripts", rlrmp_root, scripts_dir))
    return groups


def dotted_name_for(repo: str, tree: str, tree_root: Path, repo_root: Path, f: Path) -> tuple[str, bool]:
    """Compute a synthetic dotted name for import resolution, and whether the
    file is a package __init__."""
    rel = f.relative_to(tree_root)
    parts = list(rel.with_suffix("").parts)
    is_package = parts and parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    if tree == "src":
        # src/rlrmp/... -> rel is already "rlrmp/...", so parts is the full
        # dotted path (e.g. ["rlrmp", "train", "cs_nominal_gru"]).
        dotted_parts = parts
    elif tree == "package":
        dotted_parts = ["feedbax"] + parts
    elif tree == "tests" and repo == "feedbax":
        dotted_parts = ["feedbax_tests"] + parts
    elif tree == "scripts" and repo == "feedbax":
        dotted_parts = ["feedbax_scripts"] + parts
    elif tree == "examples" and repo == "feedbax":
        dotted_parts = ["feedbax_examples"] + parts
    elif tree == "tests" and repo == "rlrmp":
        dotted_parts = ["tests"] + parts
    elif tree == "scripts" and repo == "rlrmp":
        dotted_parts = ["scripts"] + parts
    elif tree == "results_scripts":
        # tree_root is results/<hash>/scripts; encode the hash in the dotted
        # path so ids stay unique across experiments.
        hash_id = tree_root.parent.name
        dotted_parts = ["results_scripts", hash_id] + parts
    else:
        dotted_parts = parts
    dotted = ".".join(p for p in dotted_parts if p)
    return dotted, bool(is_package)


# --------------------------------------------------------------------------
# Git provenance helpers
# --------------------------------------------------------------------------

def run_git(repo_root: Path, args: list[str], timeout: float = 30.0) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout


def file_git_history(repo_root: Path, relpath: str) -> tuple[Optional[dict], Optional[dict], list[str]]:
    """Return (introducing_commit, last_touch_commit, mandible_issue_trailers)."""
    fmt = f"%H{GIT_LOG_FIELD_SEP}%ad{GIT_LOG_FIELD_SEP}%B{GIT_LOG_RECORD_SEP}"
    out = run_git(
        repo_root,
        ["log", "--follow", f"--format={fmt}", "--date=short", "--", relpath],
    )
    if not out:
        return None, None, []
    records = [r for r in out.split(GIT_LOG_RECORD_SEP) if r.strip()]
    trailers: set[str] = set()
    parsed = []
    for r in records:
        parts = r.split(GIT_LOG_FIELD_SEP)
        if len(parts) < 3:
            continue
        h, d, body = parts[0].strip(), parts[1].strip(), parts[2]
        parsed.append({"hash": h, "date": d})
        for m in TRAILER_RE.finditer(body):
            trailers.add(m.group(1))
    if not parsed:
        return None, None, []
    last_touch = parsed[0]
    introducing = parsed[-1]
    return introducing, last_touch, sorted(trailers)


def file_blame_line_dates(repo_root: Path, relpath: str) -> tuple[dict[int, str], dict[int, str]]:
    """One `git blame --porcelain` pass -> ({final_line_no: 'YYYY-MM-DD'}, {final_line_no: commit_hash})."""
    out = run_git(repo_root, ["blame", "--porcelain", "--", relpath], timeout=60.0)
    if not out:
        return {}, {}
    header_re = re.compile(r"^([0-9a-f]{40}) (\d+) (\d+)")
    line_commit: dict[int, str] = {}
    commit_time: dict[str, int] = {}
    current_hash = None
    for line in out.splitlines():
        m = header_re.match(line)
        if m:
            current_hash = m.group(1)
            final_line = int(m.group(3))
            line_commit[final_line] = current_hash
            continue
        if current_hash is None:
            continue
        if line.startswith("committer-time "):
            try:
                commit_time[current_hash] = int(line.split(" ", 1)[1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("author-time "):
            try:
                commit_time.setdefault(current_hash, int(line.split(" ", 1)[1]))
            except (ValueError, IndexError):
                pass
    out_map: dict[int, str] = {}
    for ln, h in line_commit.items():
        ts = commit_time.get(h)
        if ts is not None:
            out_map[ln] = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return out_map, line_commit


# --------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------

def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def docstring_first_line(node) -> Optional[str]:
    doc = ast.get_docstring(node)
    if not doc:
        return None
    first = doc.strip().splitlines()[0].strip()
    return first or None


def is_numeric_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, NUMERIC_TYPES)
        and not isinstance(node.value, bool)
    )


def literal_stats(node: ast.AST) -> tuple[int, int, set[str], int]:
    """Return (n_numeric_literals, max_float_container_size, hp_like_keys,
    n_string_path_literals) over the given subtree."""
    n_numeric = 0
    max_container = 0
    hp_keys: set[str] = set()
    n_path_strings = 0

    for sub in ast.walk(node):
        if is_numeric_constant(sub):
            n_numeric += 1
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if PATH_LITERAL_RE.search(sub.value):
                n_path_strings += 1

        if isinstance(sub, (ast.List, ast.Tuple)):
            elts = sub.elts
            if elts and all(is_numeric_constant(e) for e in elts):
                max_container = max(max_container, len(elts))
        elif isinstance(sub, ast.Dict):
            keys, values = sub.keys, sub.values
            if values and all(v is not None and is_numeric_constant(v) for v in values):
                max_container = max(max_container, len(values))
            for k in keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    if k.value.lower() in HP_LIKE_KEYS:
                        hp_keys.add(k.value.lower())
        elif isinstance(sub, ast.Call):
            for kw in sub.keywords:
                if kw.arg and kw.arg.lower() in HP_LIKE_KEYS:
                    hp_keys.add(kw.arg.lower())

    return n_numeric, max_container, hp_keys, n_path_strings


def collect_outbound_names(node: ast.AST, own_name: str) -> list[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if sub.id != own_name:
                names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
    return sorted(names)


def has_register_signal(text: str) -> bool:
    return "register" in text.lower()


def decorator_registered(decorators: list[ast.expr]) -> bool:
    return any(has_register_signal(safe_unparse(d)) for d in decorators)


def collect_module_level_string_blob(tree: ast.Module) -> str:
    parts = []
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            parts.append(sub.value)
    return "\n".join(parts)


def find_all_list(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for stmt in tree.body:
        target_names = []
        if isinstance(stmt, ast.Assign):
            target_names = [t for t in stmt.targets if isinstance(t, ast.Name) and t.id == "__all__"]
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "__all__":
            target_names = [stmt.target]
        if target_names and isinstance(stmt.value, (ast.List, ast.Tuple, ast.Set)):
            for elt in stmt.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.add(elt.value)
    return names


# --------------------------------------------------------------------------
# Import table resolution
# --------------------------------------------------------------------------

@dataclass
class ImportBinding:
    kind: str  # "module" | "object" | "unresolved"
    target: Optional[str] = None  # dotted module name, when kind in {module, object-owning-module}
    object_name: Optional[str] = None  # only for kind == "object"


def resolve_relative_base(current_dotted: str, is_package: bool, level: int) -> str:
    parts = current_dotted.split(".") if current_dotted else []
    base_parts = parts if is_package else parts[:-1]
    if level > 1:
        cut = level - 1
        base_parts = base_parts[:-cut] if len(base_parts) >= cut else []
    return ".".join(base_parts)


def _resolve_bare_module(
    name: str,
    importing_module: ModuleRecord,
    dotted_registry: dict[str, ModuleRecord],
    flat_registry: dict[str, list[ModuleRecord]],
) -> Optional[str]:
    """Resolve a single-segment bare module reference (e.g. `import foo` or
    `from foo import Bar`) that doesn't match the dotted registry directly.

    This covers the sibling-import pattern CLAUDE.md explicitly permits within
    a single `results/<hash>/scripts/` directory (and, as a violation-detection
    side effect, the same pattern in flat `scripts/` if it occurs there): a
    same-directory sibling module with a matching bare filename stem.
    """
    if "." in name:
        return None
    candidates = flat_registry.get(name, [])
    if not candidates:
        return None
    same_dir = [c for c in candidates if c.abspath.parent == importing_module.abspath.parent]
    if len(same_dir) == 1:
        return same_dir[0].dotted
    if len(candidates) == 1:
        return candidates[0].dotted
    return None  # ambiguous across multiple same-named files; leave unresolved


def build_import_table(
    tree: ast.Module,
    module: ModuleRecord,
    dotted_registry: dict[str, ModuleRecord],
    flat_registry: dict[str, list[ModuleRecord]],
) -> dict[str, ImportBinding]:
    table: dict[str, ImportBinding] = {}
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                local = alias.asname or alias.name.split(".")[0]
                full = alias.name if alias.asname else alias.name.split(".")[0]
                target_dotted = full if full in dotted_registry else _resolve_bare_module(
                    full, module, dotted_registry, flat_registry
                )
                if target_dotted is not None:
                    table[local] = ImportBinding(kind="module", target=target_dotted)
                else:
                    table[local] = ImportBinding(kind="unresolved")
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.level and stmt.level > 0:
                base = resolve_relative_base(module.dotted, module.is_package, stmt.level)
                mod_dotted = f"{base}.{stmt.module}" if (base and stmt.module) else (stmt.module or base)
            else:
                mod_dotted = stmt.module or ""
                if mod_dotted not in dotted_registry:
                    resolved_bare = _resolve_bare_module(mod_dotted, module, dotted_registry, flat_registry)
                    if resolved_bare is not None:
                        mod_dotted = resolved_bare
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                owning_mod = dotted_registry.get(mod_dotted)
                if owning_mod and alias.name in owning_mod.object_ids_by_name:
                    table[local] = ImportBinding(kind="object", target=mod_dotted, object_name=alias.name)
                    continue
                submod_dotted = f"{mod_dotted}.{alias.name}" if mod_dotted else alias.name
                if submod_dotted in dotted_registry:
                    table[local] = ImportBinding(kind="module", target=submod_dotted)
                    continue
                table[local] = ImportBinding(kind="unresolved")
    return table


# --------------------------------------------------------------------------
# Per-file parse pass
# --------------------------------------------------------------------------

def parse_module(
    repo: str, tree_label: str, repo_root: Path, tree_root: Path, f: Path,
) -> tuple[Optional[ModuleRecord], Optional[ast.Module], list[ObjectRecord], str]:
    relpath = str(f.relative_to(repo_root))
    text = f.read_text(encoding="utf-8", errors="replace")
    loc = len(text.splitlines())
    dotted, is_package = dotted_name_for(repo, tree_label, tree_root, repo_root, f)
    flat_name = f.stem
    mod_id = f"{repo}:{relpath}:__module__"
    mod = ModuleRecord(
        id=mod_id, repo=repo, tree=tree_label, relpath=relpath, abspath=f,
        dotted=dotted, flat_name=flat_name, is_package=is_package, loc=loc,
    )
    try:
        parsed = ast.parse(text, filename=str(f))
    except (SyntaxError, ValueError) as e:
        mod.parse_error = True
        return mod, None, [], text

    mod.docstring_first_line = docstring_first_line(parsed)
    mod.has_legacy_banner = bool(LEGACY_BANNER_RE.search(text))

    all_names = find_all_list(parsed)
    objects: list[ObjectRecord] = []

    for stmt in parsed.body:
        qualname = None
        kind = None
        decorators: list[str] = []
        bases: list[str] = []
        method_names: list[str] = []
        node_for_stats: ast.AST = stmt

        if isinstance(stmt, ast.FunctionDef):
            qualname, kind = stmt.name, "function"
            decorators = [safe_unparse(d) for d in stmt.decorator_list]
        elif isinstance(stmt, ast.AsyncFunctionDef):
            qualname, kind = stmt.name, "async_function"
            decorators = [safe_unparse(d) for d in stmt.decorator_list]
        elif isinstance(stmt, ast.ClassDef):
            qualname, kind = stmt.name, "class"
            decorators = [safe_unparse(d) for d in stmt.decorator_list]
            bases = [safe_unparse(b) for b in stmt.bases]
            method_names = [
                b.name for b in stmt.body
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            qualname, kind = stmt.targets[0].id, "constant"
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            qualname, kind = stmt.target.id, "constant"

        if qualname is None:
            continue

        line_start = stmt.lineno
        line_end = getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno
        obj_loc = line_end - line_start + 1
        n_num, max_container, hp_keys, n_path = literal_stats(node_for_stats)
        registered = decorator_registered(getattr(stmt, "decorator_list", [])) or (
            kind == "constant" and isinstance(getattr(stmt, "value", None), ast.Call)
            and has_register_signal(safe_unparse(stmt.value.func))
        )
        obj_id = f"{repo}:{relpath}:{qualname}"
        objects.append(ObjectRecord(
            id=obj_id, repo=repo, tree=tree_label, module_relpath=relpath,
            qualname=qualname, kind=kind, line_start=line_start, line_end=line_end,
            loc=obj_loc, decorators=decorators, bases=bases,
            docstring_first_line=docstring_first_line(stmt) if kind in ("function", "async_function", "class") else None,
            is_private=qualname.startswith("_"),
            outbound_names=collect_outbound_names(node_for_stats, qualname),
            n_numeric_literals=n_num, max_float_container_size=max_container,
            hp_like_keys=sorted(hp_keys), n_string_path_literals=n_path,
            registered=bool(registered), in_all=qualname in all_names,
            method_names=method_names, n_methods=len(method_names),
        ))
        mod.object_ids_by_name[qualname] = obj_id

    mod.n_objects = len(objects)
    return mod, parsed, objects, text


# --------------------------------------------------------------------------
# Chunk planning
# --------------------------------------------------------------------------

def build_chunk_plan(
    modules: list[ModuleRecord], objects_by_module: dict[str, list[ObjectRecord]],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current = {"chunk_id": "", "modules": [], "objects": [], "object_loc_total": 0, "n_objects": 0}

    def flush():
        nonlocal current
        if current["objects"] or current["modules"]:
            current["chunk_id"] = f"chunk_{len(chunks) + 1:04d}"
            chunks.append(current)
        current = {"chunk_id": "", "modules": [], "objects": [], "object_loc_total": 0, "n_objects": 0}

    for mod in sorted(modules, key=lambda m: (m.repo, m.tree, m.relpath)):
        items = sorted(objects_by_module.get(mod.id, []), key=lambda o: o.line_start)
        if not items:
            if mod.id not in current["modules"]:
                current["modules"].append(mod.id)
            continue
        idx = 0
        module_attached = False
        while idx < len(items):
            item = items[idx]
            would_loc = current["object_loc_total"] + item.loc
            would_count = current["n_objects"] + 1
            fits = current["n_objects"] == 0 or (
                would_loc <= MAX_CHUNK_OBJECT_LOC and would_count <= MAX_CHUNK_OBJECTS
            )
            if not fits:
                flush()
                module_attached = False
                continue
            current["objects"].append({
                "id": item.id, "repo": item.repo, "module_relpath": item.module_relpath,
                "qualname": item.qualname, "line_start": item.line_start,
                "line_end": item.line_end, "loc": item.loc,
            })
            current["object_loc_total"] += item.loc
            current["n_objects"] += 1
            idx += 1
            if not module_attached:
                current["modules"].append(mod.id)
                module_attached = True
    flush()
    return chunks


# --------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------

class SelfCheckError(Exception):
    pass


def check_jsonl(path: Path, required_fields: set[str]) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SelfCheckError(f"{path}:{i} failed to round-trip json.loads: {e}")
            missing = required_fields - obj.keys()
            if missing:
                raise SelfCheckError(f"{path}:{i} missing required fields: {missing}")
            n += 1
    return n


# --------------------------------------------------------------------------
# Main census
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="rlrmp repo root (default: inferred from this script's location)")
    ap.add_argument("--feedbax-root", type=Path,
                    default=Path("~/Main/10 Projects/10 PhD/20 Feedbax/feedbax").expanduser(),
                    help="feedbax repo root (read-only)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: <repo-root>/_artifacts/05883e7/audit/census)")
    ap.add_argument("--skip-git", action="store_true",
                    help="skip git provenance/blame (fast dev iteration only)")
    args = ap.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = args.repo_root or script_path.parents[3]
    feedbax_root = args.feedbax_root.resolve()
    out_dir = args.out_dir or (repo_root / "_artifacts" / "05883e7" / "audit" / "census")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (repo_root / "src" / "rlrmp").is_dir():
        print(f"error: {repo_root} does not look like the rlrmp repo root (no src/rlrmp)", file=sys.stderr)
        return 2
    if not (feedbax_root / "feedbax").is_dir():
        print(f"error: {feedbax_root} does not look like the feedbax repo root (no feedbax/)", file=sys.stderr)
        return 2

    t0 = time.monotonic()

    groups = discover_trees(repo_root, feedbax_root)

    # ---- Pass 1: parse every file, build module registry ----
    all_modules: list[ModuleRecord] = []
    all_objects_by_module: dict[str, list[ObjectRecord]] = {}
    parsed_trees: dict[str, ast.Module] = {}
    module_texts: dict[str, str] = {}
    dotted_registry: dict[str, ModuleRecord] = {}
    flat_registry: dict[str, list[ModuleRecord]] = defaultdict(list)
    n_parse_errors = 0

    for repo, tree_label, group_repo_root, tree_root in groups:
        for f in iter_py_files(tree_root):
            mod, parsed, objects, text = parse_module(repo, tree_label, group_repo_root, tree_root, f)
            if mod is None:
                continue
            all_modules.append(mod)
            all_objects_by_module[mod.id] = objects
            module_texts[mod.id] = text
            if parsed is not None:
                parsed_trees[mod.id] = parsed
            else:
                n_parse_errors += 1
            if mod.dotted:
                dotted_registry[mod.dotted] = mod
            flat_registry[mod.flat_name].append(mod)

    modules_by_id: dict[str, ModuleRecord] = {m.id: m for m in all_modules}

    # ---- Pass 2: git provenance (module-level) + blame (object-level) ----
    if not args.skip_git:
        for mod in all_modules:
            repo_root_for = repo_root if mod.repo == "rlrmp" else feedbax_root
            intro, last, trailers = file_git_history(repo_root_for, mod.relpath)
            mod.introducing_commit = intro
            mod.last_touch_commit = last
            mod.mandible_issue_trailers = trailers
            line_dates, line_commit = file_blame_line_dates(repo_root_for, mod.relpath)
            objects = all_objects_by_module.get(mod.id, [])
            for obj in objects:
                best_date = None
                best_hash = None
                for ln in range(obj.line_start, obj.line_end + 1):
                    d = line_dates.get(ln)
                    if d is not None and (best_date is None or d > best_date):
                        best_date = d
                        best_hash = line_commit.get(ln)
                obj.last_touch_date = best_date
                obj.last_touch_commit = best_hash

    # ---- Pass 3: import tables + global name/string indices ----
    module_import_tables: dict[str, dict[str, ImportBinding]] = {}
    for mod_id, tree in parsed_trees.items():
        mod = modules_by_id[mod_id]
        module_import_tables[mod_id] = build_import_table(tree, mod, dotted_registry, flat_registry)

    objects_by_name: dict[str, list[str]] = defaultdict(list)
    for objs in all_objects_by_module.values():
        for o in objs:
            objects_by_name[o.qualname].append(o.id)

    global_token_counts: Counter[str] = Counter()
    for mod_id, tree in parsed_trees.items():
        blob = collect_module_level_string_blob(tree)
        if blob:
            global_token_counts.update(TOKEN_RE.findall(blob))

    # ---- Pass 4: cross-reference resolution ----
    inbound_by_tree: dict[str, Counter] = defaultdict(Counter)  # obj_id -> Counter[(repo,tree)]
    unresolved_by_tree: dict[str, Counter] = defaultdict(Counter)

    for mod_id, tree in parsed_trees.items():
        mod = modules_by_id[mod_id]
        consumer_key = f"{mod.repo}:{mod.tree}"
        import_table = module_import_tables[mod_id]

        name_load_counts: Counter[str] = Counter()
        attr_counts: Counter[tuple[str, str]] = Counter()
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                name_load_counts[sub.id] += 1
            elif isinstance(sub, ast.Attribute):
                root = sub.value
                if isinstance(root, ast.Name):
                    attr_counts[(root.id, sub.attr)] += 1

        accounted: set[str] = set()
        for alias, binding in import_table.items():
            if binding.kind == "object" and binding.target is not None and binding.object_name is not None:
                owning_mod = dotted_registry.get(binding.target)
                if owning_mod is None:
                    continue
                obj_id = owning_mod.object_ids_by_name.get(binding.object_name)
                if obj_id is None:
                    continue
                occ = name_load_counts.get(alias, 0)
                if occ > 0:
                    inbound_by_tree[obj_id][consumer_key] += occ
                    accounted.add(alias)
            elif binding.kind == "module" and binding.target is not None:
                owning_mod = dotted_registry.get(binding.target)
                if owning_mod is None:
                    continue
                accounted.add(alias)
                for (root, tail), occ in attr_counts.items():
                    if root != alias:
                        continue
                    obj_id = owning_mod.object_ids_by_name.get(tail)
                    if obj_id is not None:
                        inbound_by_tree[obj_id][consumer_key] += occ

        # bare-name fallback (best-effort, documented as noisy in audit_plan.md)
        for name, occ in name_load_counts.items():
            if name in accounted:
                continue
            for obj_id in objects_by_name.get(name, ()):
                unresolved_by_tree[obj_id][consumer_key] += occ

    # ---- Emit modules.jsonl ----
    modules_path = out_dir / "modules.jsonl"
    with modules_path.open("w", encoding="utf-8") as fh:
        for mod in sorted(all_modules, key=lambda m: (m.repo, m.tree, m.relpath)):
            rec = {
                "id": mod.id, "repo": mod.repo, "tree": mod.tree, "relpath": mod.relpath,
                "loc": mod.loc, "n_objects": mod.n_objects,
                "imports": sorted({
                    b.target for b in module_import_tables.get(mod.id, {}).values()
                    if b.target
                }),
                "docstring_first_line": mod.docstring_first_line,
                "has_legacy_banner": mod.has_legacy_banner,
                "introducing_commit": mod.introducing_commit,
                "last_touch_commit": mod.last_touch_commit,
                "mandible_issue_trailers": mod.mandible_issue_trailers,
                "parse_error": mod.parse_error,
            }
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # ---- Emit objects.jsonl ----
    objects_path = out_dir / "objects.jsonl"
    with objects_path.open("w", encoding="utf-8") as fh:
        for mod in sorted(all_modules, key=lambda m: (m.repo, m.tree, m.relpath)):
            for o in sorted(all_objects_by_module.get(mod.id, []), key=lambda x: x.line_start):
                rec = {
                    "id": o.id, "repo": o.repo, "tree": o.tree,
                    "module_relpath": o.module_relpath, "qualname": o.qualname,
                    "kind": o.kind, "line_start": o.line_start, "line_end": o.line_end,
                    "loc": o.loc, "decorators": o.decorators, "bases": o.bases,
                    "docstring_first_line": o.docstring_first_line,
                    "is_private": o.is_private, "outbound_names": o.outbound_names,
                    "n_numeric_literals": o.n_numeric_literals,
                    "max_float_container_size": o.max_float_container_size,
                    "hp_like_keys": o.hp_like_keys,
                    "n_string_path_literals": o.n_string_path_literals,
                    "registered": o.registered, "in_all": o.in_all,
                    "method_names": o.method_names, "n_methods": o.n_methods,
                    "last_touch_commit": o.last_touch_commit,
                    "last_touch_date": o.last_touch_date,
                }
                fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # ---- Emit xref.jsonl ----
    xref_path = out_dir / "xref.jsonl"
    with xref_path.open("w", encoding="utf-8") as fh:
        for mod in sorted(all_modules, key=lambda m: (m.repo, m.tree, m.relpath)):
            for o in sorted(all_objects_by_module.get(mod.id, []), key=lambda x: x.line_start):
                inbound = dict(inbound_by_tree.get(o.id, {}))
                unresolved = dict(unresolved_by_tree.get(o.id, {}))
                rec = {
                    "id": o.id,
                    "inbound_by_tree": inbound,
                    "inbound_total": sum(inbound.values()),
                    "unresolved_name_hits": sum(unresolved.values()),
                    "unresolved_by_tree": unresolved,
                    "string_hits": global_token_counts.get(o.qualname, 0),
                    "in_all": o.in_all,
                    "registered": o.registered,
                }
                fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # ---- Chunk plan ----
    chunk_plan = build_chunk_plan(all_modules, all_objects_by_module)
    chunk_plan_path = out_dir / "chunk_plan.json"
    chunk_plan_path.write_text(json.dumps(chunk_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- Self-checks ----
    n_modules = check_jsonl(modules_path, REQUIRED_MODULE_FIELDS)
    n_objects = check_jsonl(objects_path, REQUIRED_OBJECT_FIELDS)
    n_xref = check_jsonl(xref_path, REQUIRED_XREF_FIELDS)

    for mod in all_modules:
        obj_loc_total = sum(o.loc for o in all_objects_by_module.get(mod.id, []))
        if obj_loc_total > mod.loc:
            raise SelfCheckError(
                f"{mod.relpath}: object LOC total {obj_loc_total} exceeds file LOC {mod.loc}"
            )

    planned_ids = [o.id for objs in all_objects_by_module.values() for o in objs
                   if o.qualname == "planned_ef9c882_start_pos_hold_rows"]
    if not planned_ids:
        raise SelfCheckError("planned_ef9c882_start_pos_hold_rows not found in objects.jsonl")

    lc_ids = [o.id for objs in all_objects_by_module.values() for o in objs
              if o.qualname == "LaunchContinuation"]
    if not lc_ids:
        raise SelfCheckError("LaunchContinuation not found in objects.jsonl")
    lc_has_xref = any(
        inbound_by_tree.get(oid) or unresolved_by_tree.get(oid) for oid in lc_ids
    )
    if not lc_has_xref:
        raise SelfCheckError("LaunchContinuation has zero xref rows (expected self-file usage hits)")

    elapsed = time.monotonic() - t0

    # ---- summary.md ----
    write_summary(
        out_dir / "summary.md", all_modules, all_objects_by_module,
        inbound_by_tree, unresolved_by_tree, chunk_plan, n_parse_errors, elapsed,
    )

    print(f"census complete in {elapsed:.1f}s: {n_modules} modules, {n_objects} objects, "
          f"{n_xref} xref rows, {len(chunk_plan)} chunks -> {out_dir}")
    return 0


def write_summary(
    path: Path,
    all_modules: list[ModuleRecord],
    all_objects_by_module: dict[str, list[ObjectRecord]],
    inbound_by_tree: dict[str, Counter],
    unresolved_by_tree: dict[str, Counter],
    chunk_plan: list[dict[str, Any]],
    n_parse_errors: int,
    elapsed: float,
) -> None:
    all_objects = [o for objs in all_objects_by_module.values() for o in objs]

    loc_by_repo_tree: Counter = Counter()
    obj_by_repo_tree: Counter = Counter()
    for mod in all_modules:
        loc_by_repo_tree[(mod.repo, mod.tree)] += mod.loc
    for o in all_objects:
        obj_by_repo_tree[(o.repo, o.tree)] += 1

    top20 = sorted(all_modules, key=lambda m: m.loc, reverse=True)[:20]

    kind_counts = Counter(o.kind for o in all_objects)

    n_zero_inbound = 0
    n_eligible = 0
    for o in all_objects:
        if o.registered or o.in_all:
            continue
        if len(TOKEN_RE.findall(o.qualname)) == 0:
            continue
        n_eligible += 1
        if not inbound_by_tree.get(o.id) and not unresolved_by_tree.get(o.id):
            n_zero_inbound += 1
    pct_zero_inbound = (100.0 * n_zero_inbound / n_eligible) if n_eligible else 0.0

    n_legacy = sum(1 for m in all_modules if m.has_legacy_banner)
    n_hp_flagged = sum(1 for o in all_objects if o.hp_like_keys or o.max_float_container_size >= 4)

    lines = []
    lines.append("<!-- AUTO-GENERATED: census_summary -->")
    lines.append("# Census summary")
    lines.append("")
    lines.append(f"Generated in {elapsed:.1f}s. {n_parse_errors} files failed to parse and were skipped.")
    lines.append("")
    lines.append("## Objects / modules / LOC per repo x tree")
    lines.append("")
    lines.append("| repo | tree | modules | LOC | objects |")
    lines.append("|---|---|---:|---:|---:|")
    mod_counts = Counter((m.repo, m.tree) for m in all_modules)
    for key in sorted(loc_by_repo_tree, key=lambda k: -loc_by_repo_tree[k]):
        repo, tree_label = key
        lines.append(
            f"| {repo} | {tree_label} | {mod_counts[key]} | {loc_by_repo_tree[key]} | {obj_by_repo_tree[key]} |"
        )
    lines.append("")
    lines.append("## Top 20 modules by LOC")
    lines.append("")
    lines.append("| module | repo | tree | LOC |")
    lines.append("|---|---|---|---:|")
    for m in top20:
        lines.append(f"| `{m.relpath}` | {m.repo} | {m.tree} | {m.loc} |")
    lines.append("")
    lines.append("## Counts by object kind")
    lines.append("")
    lines.append("| kind | count |")
    lines.append("|---|---:|")
    for k, c in kind_counts.most_common():
        lines.append(f"| {k} | {c} |")
    lines.append("")
    lines.append(
        f"## Zero-inbound-reference rate\n\n"
        f"{n_zero_inbound} / {n_eligible} objects ({pct_zero_inbound:.1f}%) have zero inbound "
        f"references (routed or unresolved-bare-name) and are not registered or `__all__`-listed. "
        f"This is a *candidate-dead* signal only -- see the xref-resolution limitations in "
        f"notes/audit_plan.md before treating any of these as confirmed dead."
    )
    lines.append("")
    lines.append(f"## LEGACY-bannered modules\n\n{n_legacy} modules contain a `LEGACY (frozen` banner.")
    lines.append("")
    lines.append(
        f"## Objects with hyperparameter-literal signals\n\n"
        f"{n_hp_flagged} objects have at least one hp-like dict-key/kwarg-name hit or a numeric "
        f"container of size >= 4."
    )
    lines.append("")
    lines.append(f"## Chunk plan\n\n{len(chunk_plan)} classification chunks.")
    lines.append("")
    lines.append("<!-- /AUTO-GENERATED -->")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SelfCheckError as e:
        print(f"SELF-CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
