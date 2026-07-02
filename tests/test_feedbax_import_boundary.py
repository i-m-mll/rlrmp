"""AST import-boundary checks for Feedbax and jax-cookbook imports."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import tomllib

import pytest


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "ci" / "feedbax-public-imports.toml"
SCAN_ROOTS = ("src", "scripts", "results")
PRIVATE_STRING_TOKENS = ("feedbax._io",)


@dataclass(frozen=True)
class ImportUse:
    path: str
    lineno: int
    module: str
    symbol: str | None

    @property
    def package(self) -> str:
        return self.module.split(".", maxsplit=1)[0]


def test_feedbax_and_jax_cookbook_imports_use_canonical_public_homes() -> None:
    manifest = _load_manifest()
    violations = _find_import_violations(_iter_import_uses(), manifest)

    assert violations == []


def test_only_allowlisted_private_feedbax_strings_remain() -> None:
    manifest = _load_manifest()
    allowed = tuple(
        (entry["path"], entry["line"])
        for entry in manifest.get("allowed_private_strings", [])
    )
    violations: list[str] = []
    for path in _scan_paths():
        relpath = _relpath(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            for token in PRIVATE_STRING_TOKENS:
                if token in line and not any(
                    relpath == allowed_path and token in allowed_line
                    for allowed_path, allowed_line in allowed
                ):
                    violations.append(f"{relpath}: private Feedbax string {token!r}")

    assert violations == []


def test_import_boundary_negative_canaries_reject_private_and_uncanonical_imports() -> None:
    manifest = _load_manifest()
    private_use = ImportUse(
        path="src/example.py",
        lineno=1,
        module="feedbax.contracts._private",
        symbol="Thing",
    )
    uncanonical_use = ImportUse(
        path="src/example.py",
        lineno=2,
        module="feedbax.contracts.graph",
        symbol="NotDeclaredPublicHome",
    )

    violations = _find_import_violations([private_use, uncanonical_use], manifest)

    assert len(violations) == 2
    assert "private import path" in violations[0]
    assert "not declared in canonical-home manifest" in violations[1]


def _load_manifest() -> dict:
    return tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _scan_paths() -> list[Path]:
    paths: list[Path] = []
    paths.extend((REPO_ROOT / "src").rglob("*.py"))
    paths.extend((REPO_ROOT / "scripts").glob("*.py"))
    paths.extend((REPO_ROOT / "results").glob("*/scripts/*.py"))
    return sorted(paths)


def _relpath(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _iter_import_uses() -> list[ImportUse]:
    uses: list[ImportUse] = []
    for path in _scan_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relpath = _relpath(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    package = alias.name.split(".", maxsplit=1)[0]
                    if package in {"feedbax", "jax_cookbook"}:
                        uses.append(ImportUse(relpath, node.lineno, alias.name, None))
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
                package = module.split(".", maxsplit=1)[0]
                if package not in {"feedbax", "jax_cookbook"}:
                    continue
                for alias in node.names:
                    uses.append(ImportUse(relpath, node.lineno, module, alias.name))
    assert uses, "Feedbax import-boundary scan found zero imports"
    return uses


def _find_import_violations(uses: list[ImportUse], manifest: dict) -> list[str]:
    public_import_modules = set(manifest["public_import_modules"])
    canonical = {
        (entry["module"], symbol)
        for entry in manifest["canonical_homes"]
        for symbol in entry["symbols"]
    }
    known_in_flight = {
        (entry["path"], entry["module"], symbol)
        for entry in manifest.get("known_in_flight", [])
        for symbol in entry["symbols"]
    }
    violations: list[str] = []
    for use in uses:
        if use.symbol is None:
            if _has_private_segment(use.module):
                violations.append(f"{use.path}:{use.lineno}: private import path {use.module}")
            elif use.module not in public_import_modules:
                violations.append(
                    f"{use.path}:{use.lineno}: imported module {use.module!r} is not "
                    "declared as a public import module"
                )
            continue

        key = (use.path, use.module, use.symbol)
        if key in known_in_flight:
            continue
        if use.symbol == "*":
            violations.append(f"{use.path}:{use.lineno}: wildcard import from {use.module}")
        elif _has_private_segment(use.module) or use.symbol.startswith("_"):
            violations.append(f"{use.path}:{use.lineno}: private import path {use.module}")
        elif (use.module, use.symbol) not in canonical:
            violations.append(
                f"{use.path}:{use.lineno}: {use.module}.{use.symbol} is not declared "
                "in canonical-home manifest"
            )
    return violations


def _has_private_segment(module: str) -> bool:
    return any(part.startswith("_") for part in module.split(".")[1:])
