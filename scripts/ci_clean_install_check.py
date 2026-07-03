#!/usr/bin/env python
"""Run the rlrmp clean non-editable install import gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile
import textwrap
import tomllib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "scripts" / "ci_clean_install_import_manifest.json"
FEEDBAX_REF_PATH = REPO_ROOT / "ci" / "feedbax-ref.toml"
FEEDBAX_REF_ENV = "RLRMP_CI_FEEDBAX_REF"
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Keep the temporary venv and print its path for debugging.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Use the interpreter from --python instead of creating/installing into a temp venv.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Python executable to use with --skip-install.",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    _scan_legacy_homes(manifest)

    if args.skip_install:
        if args.python is None:
            raise SystemExit("--skip-install requires --python")
        python = args.python
        _run_import_gate(python, manifest)
        return 0

    feedbax_spec = _resolve_feedbax_install_spec()
    temp_dir = tempfile.TemporaryDirectory(prefix="rlrmp-clean-install-")
    try:
        root = Path(temp_dir.name)
        venv = root / "venv"
        python = venv / "bin" / "python"
        _run(["uv", "venv", str(venv)], cwd=REPO_ROOT)
        _run(["uv", "pip", "install", "--python", str(python), feedbax_spec], cwd=REPO_ROOT)
        rlrmp_deps = _rlrmp_dependencies_without_feedbax()
        if rlrmp_deps:
            _run(["uv", "pip", "install", "--python", str(python), *rlrmp_deps], cwd=REPO_ROOT)
        _run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(REPO_ROOT)],
            cwd=REPO_ROOT,
        )
        _run_import_gate(python, manifest)
        print("clean-install gate passed")
        if args.keep_venv:
            print(f"kept venv: {venv}")
            temp_dir.cleanup = lambda: None  # type: ignore[method-assign]
        return 0
    finally:
        temp_dir.cleanup()


def _load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise SystemExit(f"Unsupported import manifest schema: {manifest.get('schema_version')!r}")
    return manifest


def _resolve_feedbax_install_spec() -> str:
    override = os.environ.get(FEEDBAX_REF_ENV)
    if override:
        spec, expected_sha = _parse_ref_value(override)
        _verify_checkout_sha(spec, expected_sha)
        return spec

    if not FEEDBAX_REF_PATH.exists():
        raise SystemExit(
            f"{FEEDBAX_REF_PATH.relative_to(REPO_ROOT)} is absent. "
            f"Set {FEEDBAX_REF_ENV} to a feedbax checkout path, path@sha, git URL, or 40-char SHA "
            "for local clean-install runs until the recorded feedbax ref file lands."
        )

    with FEEDBAX_REF_PATH.open("rb") as handle:
        data = tomllib.load(handle)
    ref = _first_ref_value(data)
    if not ref:
        raise SystemExit(
            f"{FEEDBAX_REF_PATH.relative_to(REPO_ROOT)} must define feedbax.rev, "
            "feedbax.sha, feedbax.ref, rev, sha, or ref."
        )
    spec, expected_sha = _parse_ref_value(ref)
    _verify_checkout_sha(spec, expected_sha)
    return spec


def _first_ref_value(data: dict[str, Any]) -> str | None:
    feedbax = data.get("feedbax")
    if isinstance(feedbax, dict):
        for key in ("rev", "sha", "ref"):
            value = feedbax.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("rev", "sha", "ref"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _rlrmp_dependencies_without_feedbax() -> list[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    dependencies = data.get("project", {}).get("dependencies", [])
    return [dep for dep in dependencies if _dependency_name(dep) != "feedbax"]


def _dependency_name(requirement: str) -> str:
    return re.split(r"[\[<>=!~;]", requirement, maxsplit=1)[0].strip().lower().replace("_", "-")


def _parse_ref_value(value: str) -> tuple[str, str | None]:
    if "@" in value:
        maybe_path, maybe_sha = value.rsplit("@", 1)
        if Path(maybe_path).exists():
            return str(Path(maybe_path).resolve()), maybe_sha
    if Path(value).exists():
        return str(Path(value).resolve()), None
    if GIT_SHA_RE.fullmatch(value):
        return f"git+https://github.com/i-m-mll/feedbax.git@{value}", value
    return value, None


def _verify_checkout_sha(spec: str, expected_sha: str | None) -> None:
    if expected_sha is None:
        return
    path = Path(spec)
    if not path.exists():
        return
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != expected_sha:
        raise SystemExit(
            f"{FEEDBAX_REF_ENV} requested {expected_sha}, but {path} is checked out at {actual}."
        )


def _scan_legacy_homes(manifest: dict[str, Any]) -> None:
    legacy_terms = [entry["legacy"] for entry in manifest["canonical_homes"]]
    allowlist = {
        (entry["path"], int(entry["line"]), entry["legacy"])
        for entry in manifest.get("allowlisted_references", [])
    }
    failures: list[str] = []
    scanned_files = 0
    for root_name in manifest["scan_roots"]:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if root_name == "results" and "scripts" not in path.parts:
                continue
            scanned_files += 1
            rel = path.relative_to(REPO_ROOT).as_posix()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for legacy in legacy_terms:
                    if legacy in line and (rel, lineno, legacy) not in allowlist:
                        failures.append(f"{rel}:{lineno}: stale legacy home {legacy!r}")
    if failures:
        raise SystemExit("stale facade scan failed:\n" + "\n".join(failures))
    print(f"stale facade scan passed ({scanned_files} Python files)")


def _run_import_gate(python: Path, manifest: dict[str, Any]) -> None:
    _run_python(
        python,
        _package_inventory_code(manifest["production_module_inventory"]["package"]),
        label="production module inventory",
    )
    for script_path in _script_probe_paths(manifest):
        if script_path.suffix == ".py":
            _run_python(python, _script_import_code(script_path), label=str(script_path))
        elif script_path.suffix == ".sh":
            _run(["bash", "-n", str(script_path)], cwd=REPO_ROOT)


def _script_probe_paths(manifest: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for group in manifest["script_probes"]:
        for pattern in group["paths"]:
            matches = sorted(REPO_ROOT.glob(pattern))
            if not matches:
                raise SystemExit(f"script probe pattern matched no files: {pattern}")
            paths.extend(path for path in matches if path.is_file())
    return sorted(set(paths))


def _package_inventory_code(package: str) -> str:
    return textwrap.dedent(
        f"""
        import importlib
        import json
        import pkgutil
        import sys

        package = importlib.import_module({package!r})
        failures = []
        imported = []
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                importlib.import_module(info.name)
                imported.append(info.name)
            except Exception as exc:
                failures.append({{"module": info.name, "error": repr(exc)}})
        if failures:
            print(json.dumps(failures, indent=2), file=sys.stderr)
            raise SystemExit(1)
        print(f"imported {{len(imported)}} {package!r} submodules")
        """
    )


def _script_import_code(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return textwrap.dedent(
        f"""
        import importlib.util
        import pathlib
        import sys

        path = pathlib.Path({str(path)!r})
        sys.argv = [{rel!r}]
        spec = importlib.util.spec_from_file_location("rlrmp_script_probe", path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load script spec for {{path}}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        sys.path.insert(0, str(path.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            try:
                sys.path.remove(str(path.parent))
            except ValueError:
                pass
            sys.modules.pop(spec.name, None)
        print("imported script {rel}")
        """
    )


def _run_python(python: Path, code: str, *, label: str) -> None:
    print(f"probe: {label}")
    _run([str(python), "-c", code], cwd=REPO_ROOT)


def _run(argv: list[str], *, cwd: Path) -> None:
    printable = " ".join(shlex.quote(part) for part in argv)
    print(f"+ {printable}")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(argv, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
