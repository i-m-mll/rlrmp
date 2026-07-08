"""Import-time side-effect linting."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ImportTimeJaxConfigFinding",
    "scan_source",
    "scan_tree",
    "violations",
]


@dataclass(frozen=True)
class ImportTimeJaxConfigFinding:
    """A module-level JAX config mutation found in source."""

    relpath: str
    lineno: int
    option: str


def scan_source(text: str, relpath: str) -> list[ImportTimeJaxConfigFinding]:
    """Return module-level ``jax_enable_x64`` config updates in ``text``."""

    tree = ast.parse(text)
    findings: list[ImportTimeJaxConfigFinding] = []
    for node in tree.body:
        call = _expr_call(node)
        if call is None:
            continue
        option = _jax_config_update_option(call)
        if option == "jax_enable_x64":
            findings.append(
                ImportTimeJaxConfigFinding(
                    relpath=relpath,
                    lineno=node.lineno,
                    option=option,
                )
            )
    return findings


def scan_tree(src_root: Path, *, repo_root: Path) -> list[ImportTimeJaxConfigFinding]:
    """Scan Python files under ``src_root`` for import-time x64 config flips."""

    findings: list[ImportTimeJaxConfigFinding] = []
    for path in sorted(src_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relpath = path.resolve().relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            findings.extend(scan_source(text, relpath))
        except SyntaxError:
            continue
    return findings


def violations(src_root: Path, *, repo_root: Path) -> list[ImportTimeJaxConfigFinding]:
    """Return disallowed import-time JAX x64 config updates."""

    return scan_tree(src_root, repo_root=repo_root)


def _expr_call(node: ast.stmt) -> ast.Call | None:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return node.value
    return None


def _jax_config_update_option(call: ast.Call) -> str | None:
    if not call.args:
        return None
    func = call.func
    is_jax_config_update = (
        isinstance(func, ast.Attribute)
        and func.attr == "update"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "config"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "jax"
    )
    is_imported_config_update = (
        isinstance(func, ast.Attribute)
        and func.attr == "update"
        and isinstance(func.value, ast.Name)
        and func.value.id == "config"
    )
    if not (is_jax_config_update or is_imported_config_update):
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None
