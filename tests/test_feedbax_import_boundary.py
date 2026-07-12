"""AST import-boundary checks for Feedbax and jax-cookbook imports."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import tomllib

import pytest
from feedbax.testing import (
    AllowlistDiff,
    AllowlistEntry,
    Scope,
    SiteVisitor,
    diff_allowlist,
    scan_domain,
)


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


@dataclass(frozen=True)
class PrivateStringUse:
    path: str
    token: str


def test_feedbax_and_jax_cookbook_imports_use_canonical_public_homes() -> None:
    manifest = _load_manifest()
    uses = _iter_import_uses()
    diff = _known_in_flight_diff(uses, manifest)
    violations = _format_import_violations(diff.unlisted, manifest)

    assert violations == []
    assert diff.dead_entries == ()


def test_only_allowlisted_private_feedbax_strings_remain() -> None:
    manifest = _load_manifest()
    diff = _private_string_diff(_iter_private_string_uses(), manifest)
    violations = [f"{use.path}: private Feedbax string {use.token!r}" for use in diff.unlisted]

    assert violations == []
    assert diff.dead_entries == ()


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
    uses = scan_domain(_scan_paths(), root=REPO_ROOT, visitor_factory=_ImportVisitor)
    assert uses, "Feedbax import-boundary scan found zero imports"
    return uses


class _ImportVisitor(SiteVisitor[ImportUse]):
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            package = alias.name.split(".", maxsplit=1)[0]
            if package in {"feedbax", "jax_cookbook"}:
                self.sites.append(ImportUse(self.relpath, node.lineno, alias.name, None))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        package = module.split(".", maxsplit=1)[0]
        if node.level or package not in {"feedbax", "jax_cookbook"}:
            return
        for alias in node.names:
            self.sites.append(ImportUse(self.relpath, node.lineno, module, alias.name))


def _find_import_violations(uses: list[ImportUse], manifest: dict) -> list[str]:
    return _format_import_violations(_known_in_flight_diff(uses, manifest).unlisted, manifest)


def _format_import_violations(uses: tuple[ImportUse, ...], manifest: dict) -> list[str]:
    public_import_modules = set(manifest["public_import_modules"])
    canonical = {
        (entry["module"], symbol)
        for entry in manifest["canonical_homes"]
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


def _known_in_flight_diff(
    uses: list[ImportUse], manifest: dict
) -> AllowlistDiff[tuple[str, str | None], ImportUse]:
    public_import_modules = set(manifest["public_import_modules"])
    canonical = {
        (entry["module"], symbol)
        for entry in manifest["canonical_homes"]
        for symbol in entry["symbols"]
    }
    findings = [
        use
        for use in uses
        if (
            use.symbol is None
            and (_has_private_segment(use.module) or use.module not in public_import_modules)
        )
        or (
            use.symbol is not None
            and (
                use.symbol == "*"
                or _has_private_segment(use.module)
                or use.symbol.startswith("_")
                or (use.module, use.symbol) not in canonical
            )
        )
    ]
    entries = [
        AllowlistEntry(
            scope=Scope("file", entry["path"]),
            owner="import_boundary",
            reason=str(entry.get("reason", "")),
            key=(entry["module"], symbol),
        )
        for entry in manifest.get("known_in_flight", [])
        for symbol in entry["symbols"]
    ]
    return diff_allowlist(
        findings,
        entries,
        site_key=lambda use: (use.module, use.symbol),
        site_location=lambda use: (use.path, None),
    )


def _iter_private_string_uses() -> list[PrivateStringUse]:
    return [
        PrivateStringUse(_relpath(path), token)
        for path in _scan_paths()
        for line in path.read_text(encoding="utf-8").splitlines()
        for token in PRIVATE_STRING_TOKENS
        if token in line
    ]


def _private_string_diff(
    uses: list[PrivateStringUse], manifest: dict
) -> AllowlistDiff[str, PrivateStringUse]:
    entries = [
        AllowlistEntry(
            scope=Scope("file", entry["path"]),
            owner="import_boundary",
            reason=str(entry.get("reason", "")),
            key=token,
        )
        for entry in manifest.get("allowed_private_strings", [])
        for token in PRIVATE_STRING_TOKENS
        if token in entry["line"]
    ]
    return diff_allowlist(
        uses,
        entries,
        site_key=lambda use: use.token,
        site_location=lambda use: (use.path, None),
    )


def _has_private_segment(module: str) -> bool:
    return any(part.startswith("_") for part in module.split(".")[1:])
