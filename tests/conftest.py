"""Shared pytest policy for RLRMP tests."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_MANIFEST_PATH = REPO_ROOT / "ci" / "feedbax-contract-suite.toml"


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """During the Feedbax gate, collect only manifest-enrolled live files."""

    markexpr = getattr(config.option, "markexpr", "") or ""
    if "feedbax_contract" not in markexpr:
        return False
    try:
        relpath = Path(collection_path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return False
    if not relpath.startswith("tests"):
        return True

    live_files = _feedbax_contract_live_files()
    path = Path(relpath)
    if Path(collection_path).is_dir():
        return not any(_path_is_under(Path(live_file), path) for live_file in live_files)
    return relpath not in live_files


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep the Feedbax contract gate non-skippable."""

    violations: list[str] = []
    for item in items:
        if item.get_closest_marker("feedbax_contract") is None:
            continue
        for marker in item.iter_markers():
            if marker.name in {"skip", "skipif"}:
                violations.append(f"{item.nodeid}: feedbax_contract tests may not be skipped")
            if marker.name == "xfail" and marker.kwargs.get("strict") is not True:
                violations.append(
                    f"{item.nodeid}: feedbax_contract xfail marks must set strict=True"
                )
    if violations:
        raise pytest.UsageError("\n".join(violations))


def _feedbax_contract_live_files() -> set[str]:
    manifest = tomllib.loads(SUITE_MANIFEST_PATH.read_text(encoding="utf-8"))
    files: set[str] = set()
    for family in manifest["families"]:
        if family["status"] != "live":
            continue
        pattern = family["expected_collection_pattern"]
        if "::" in pattern:
            files.add(pattern.split("::", maxsplit=1)[0])
    return files


def _path_is_under(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents
