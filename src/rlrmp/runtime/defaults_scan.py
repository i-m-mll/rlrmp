"""AST scanner for out-of-schema literal default fallbacks."""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


SCAN_TARGETS = (
    "src/rlrmp/eval/",
    "src/rlrmp/analysis/reports.py",
    "src/rlrmp/runtime/training_run_specs.py",
    "src/rlrmp/train/cs_nominal_gru.py",
    "src/rlrmp/train/cs_perturbation_training.py",
    "src/rlrmp/benchmarks/packing.py",
    "src/rlrmp/model/",
    "src/rlrmp/train/config_materialization.py",
    "src/rlrmp/train/distillation_entry.py",
    "src/rlrmp/train/distillation_native/closed_loop_kernel.py",
    "src/rlrmp/train/run_spec_authoring.py",
    "src/rlrmp/train/training_configs.py",
    "src/rlrmp/eval/minimax_io.py",
)

ISSUE_ID_PATTERN = re.compile(
    r"(?<![0-9a-f])(?=[0-9a-f]{0,6}[0-9])[0-9a-f]{7}(?![0-9a-f])",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class AuthoredIdentityDefaultSite:
    """A seven-hex authored identity embedded in a Python default surface."""

    path: str
    qualname: str
    identity: str
    lineno: int


def scan_authored_identity_defaults(repo_root: Path) -> list[AuthoredIdentityDefaultSite]:
    """Reject issue/experiment identities authored as defaults under ``train``."""

    root = repo_root / "src/rlrmp/train"
    findings: list[AuthoredIdentityDefaultSite] = []
    for path in sorted(root.rglob("*.py")):
        relpath = path.relative_to(repo_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        findings.extend(_AuthoredIdentityDefaultVisitor(relpath).scan(tree))
    return sorted(findings)


class _AuthoredIdentityDefaultVisitor(ast.NodeVisitor):
    """Collect identity literals only from assignment and argument defaults."""

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.scope: list[str] = []
        self.scope_kinds: list[str] = []
        self.findings: list[AuthoredIdentityDefaultSite] = []

    def scan(self, tree: ast.AST) -> list[AuthoredIdentityDefaultSite]:
        self.visit(tree)
        return self.findings

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.scope_kinds.append("class")
        self.generic_visit(node)
        self.scope_kinds.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scan_function_defaults(node)
        self.scope.append(node.name)
        self.scope_kinds.append("function")
        self.generic_visit(node)
        self.scope_kinds.pop()
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        if "function" not in self.scope_kinds:
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            name = names[0] if len(names) == 1 else "<assignment>"
            if self.scope_kinds or _is_default_assignment_name(name) or isinstance(
                node.value, ast.Constant
            ):
                self._record(node.value, name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if "function" not in self.scope_kinds and node.value is not None:
            name = node.target.id if isinstance(node.target, ast.Name) else "<assignment>"
            if self.scope_kinds or _is_default_assignment_name(name) or isinstance(
                node.value, ast.Constant
            ):
                self._record(node.value, name)
        self.generic_visit(node)

    def _scan_function_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        positional = [*node.args.posonlyargs, *node.args.args]
        defaulted = positional[-len(node.args.defaults) :] if node.args.defaults else []
        for argument, default in zip(defaulted, node.args.defaults, strict=True):
            self._record(default, f"{node.name}.{argument.arg}")
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
            if default is not None:
                self._record(default, f"{node.name}.{argument.arg}")

    def _record(self, value: ast.AST, name: str) -> None:
        for child in ast.walk(value):
            if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
                continue
            for match in ISSUE_ID_PATTERN.finditer(child.value):
                qualname = ".".join((*self.scope, name))
                self.findings.append(
                    AuthoredIdentityDefaultSite(
                        path=self.relpath,
                        qualname=qualname,
                        identity=match.group(0),
                        lineno=child.lineno,
                    )
                )


def _is_default_assignment_name(name: str) -> bool:
    tokens = name.upper().split("_")
    return "DEFAULT" in tokens or "PROFILE" in tokens


@dataclass(frozen=True, order=True)
class DefaultFallbackSite:
    """One literal default fallback site, aggregated by path/key/default value."""

    path: str
    key: str
    literal_repr: str
    lineno: int | None = field(default=None, compare=False)

    def display(self) -> str:
        """Return a compact human-readable location."""

        suffix = f":{self.lineno}" if self.lineno is not None else ""
        return f"{self.path}{suffix} key={self.key!r} default={self.literal_repr}"


@dataclass(frozen=True)
class DefaultValueDriftException:
    """Allowed concept-specific value drift for one exact fallback site."""

    key: str
    reason: str
    path: str | None = None
    literal_repr: str | None = None

    def matches(self, site: DefaultFallbackSite) -> bool:
        """Return whether this exception covers ``site``."""

        if site.key != self.key:
            return False
        if self.path is not None and site.path != self.path:
            return False
        if self.literal_repr is not None and site.literal_repr != self.literal_repr:
            return False
        return True


@dataclass(frozen=True)
class DefaultValueDrift:
    """Same fallback key has multiple default values across files."""

    key: str
    sites_by_literal: Mapping[str, tuple[DefaultFallbackSite, ...]]

    def display(self) -> dict[str, object]:
        """Return assertion-friendly drift details."""

        return {
            "key": self.key,
            "values": {
                literal: [site.display() for site in sites]
                for literal, sites in sorted(self.sites_by_literal.items())
            },
        }


def scan_files(
    repo_root: Path,
    scan_targets: Sequence[str] = SCAN_TARGETS,
) -> list[Path]:
    """Return Python files from the configured production scan targets."""

    files: set[Path] = set()
    for target in scan_targets:
        path = repo_root / target
        if path.is_dir():
            files.update(path.rglob("*.py"))
        elif path.exists():
            files.add(path)
        else:
            raise FileNotFoundError(
                f"Configured default-fallback scan target does not exist: {target}"
            )
    return sorted(files)


def scan_default_fallback_sites(
    repo_root: Path,
    scan_targets: Sequence[str] = SCAN_TARGETS,
) -> Counter[DefaultFallbackSite]:
    """Count meaningful literal fallback sites in the production scan set."""

    return count_default_fallback_sites(
        scan_default_fallback_site_instances(repo_root, scan_targets)
    )


def count_default_fallback_sites(
    sites: Iterable[DefaultFallbackSite],
) -> Counter[DefaultFallbackSite]:
    """Aggregate fallback sites by ``(path, key, literal_repr)``."""

    counter: Counter[DefaultFallbackSite] = Counter()
    for site in sites:
        counter[site] += 1
    return counter


def scan_default_fallback_site_instances(
    repo_root: Path,
    scan_targets: Sequence[str] = SCAN_TARGETS,
) -> list[DefaultFallbackSite]:
    """Return individual fallback sites, preserving line numbers for diagnostics."""

    return scan_default_fallback_sites_in_paths(
        scan_files(repo_root, scan_targets),
        repo_root=repo_root,
    )


def scan_default_fallback_sites_in_paths(
    paths: Iterable[Path],
    *,
    repo_root: Path,
) -> list[DefaultFallbackSite]:
    """Return individual fallback sites from explicit files."""

    sites: list[DefaultFallbackSite] = []
    for path in sorted(paths):
        relpath = path.relative_to(repo_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            site = default_fallback_site(node, relpath)
            if site is not None:
                sites.append(site)
    return sorted(sites)


def default_fallback_site(call: ast.Call, relpath: str) -> DefaultFallbackSite | None:
    """Return a fallback site for meaningful ``.get``/``getattr`` literal defaults."""

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
    if not isinstance(key, str) or not is_meaningful_literal(default):
        return None
    return DefaultFallbackSite(
        path=relpath,
        key=key,
        literal_repr=repr(default),
        lineno=getattr(call, "lineno", None),
    )


def find_value_drifts(
    sites: Iterable[DefaultFallbackSite],
    *,
    exceptions: Iterable[DefaultValueDriftException] = (),
) -> list[DefaultValueDrift]:
    """Find same-key literal defaults that disagree across files."""

    exception_list = tuple(exceptions)
    by_key: dict[str, dict[str, list[DefaultFallbackSite]]] = defaultdict(lambda: defaultdict(list))
    for site in sites:
        if any(exception.matches(site) for exception in exception_list):
            continue
        by_key[site.key][site.literal_repr].append(site)

    drifts: list[DefaultValueDrift] = []
    for key, sites_by_literal in by_key.items():
        literals_by_path = {
            literal: {site.path for site in literal_sites}
            for literal, literal_sites in sites_by_literal.items()
        }
        if len(literals_by_path) < 2:
            continue
        all_paths = set().union(*literals_by_path.values())
        if len(all_paths) < 2:
            continue
        drifts.append(
            DefaultValueDrift(
                key=key,
                sites_by_literal={
                    literal: tuple(sorted(literal_sites))
                    for literal, literal_sites in sites_by_literal.items()
                },
            )
        )
    return sorted(drifts, key=lambda drift: drift.key)


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def is_meaningful_literal(value: Any) -> bool:
    """Return whether a literal default is a schema-ownership concern."""

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
