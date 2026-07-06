"""Memoized full-suite runner for rlrmp.

This wrapper records only clean, full-suite green runs. If any fingerprint
component cannot be resolved, it runs the suite instead of trusting the memo.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tomllib
from typing import Any, Callable, Sequence


SCHEMA_VERSION = 1
DEFAULT_MEMO_DIR = Path("_artifacts") / "test_cache" / "full_suite_memo"
DEFAULT_PYTEST_ARGS = ("tests/", "-q")
EXECUTION_RELEVANT_UNTRACKED_PATHS = (
    "feedbax/",
    "pyproject.toml",
    "scripts/",
    "src/",
    "tests/",
    "uv.lock",
)


@dataclasses.dataclass(frozen=True)
class CommandResult:
    stdout: str
    returncode: int


@dataclasses.dataclass(frozen=True)
class Fingerprint:
    ok: bool
    payload: dict[str, Any]
    digest: str | None
    reason: str | None = None


CommandRunner = Callable[[Sequence[str], Path], CommandResult]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memo-dir",
        type=Path,
        default=DEFAULT_MEMO_DIR,
        help="Directory for green-tree memo records, relative to the repo root by default.",
    )
    parser.add_argument("--no-memo", action="store_true", help="Run the suite without memo lookup.")
    parser.add_argument("--force", action="store_true", help="Run even when the memo is green.")
    parser.add_argument(
        "--print-fingerprint",
        action="store_true",
        help="Print the current fingerprint payload and exit without running pytest.",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="pytest-xdist worker count for the full suite. Default: auto.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    fingerprint = build_fingerprint(repo_root)
    if args.print_fingerprint:
        print(json.dumps(dataclasses.asdict(fingerprint), indent=2, sort_keys=True))
        return 0 if fingerprint.ok else 2

    if not fingerprint.ok:
        print(f"full-suite memo disabled: {fingerprint.reason}", file=sys.stderr)
    elif not args.no_memo and not args.force and memo_has_green(
        _memo_dir(repo_root, args.memo_dir), fingerprint
    ):
        print(f"full-suite memo hit: {fingerprint.digest}")
        return 0

    if importlib.util.find_spec("xdist") is None:
        print(
            "pytest-xdist is required for scripts/full_suite.sh; run uv sync in this worktree.",
            file=sys.stderr,
        )
        return 2

    pytest_args = [*DEFAULT_PYTEST_ARGS, "-n", args.workers]
    env = {
        **dict(os_environ()),
        "PYTHONPATH": _pythonpath(repo_root),
    }
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *pytest_args],
        cwd=repo_root,
        env=env,
        check=False,
    )
    if completed.returncode == 0 and fingerprint.ok:
        record_green(
            _memo_dir(repo_root, args.memo_dir),
            fingerprint,
            command=[sys.executable, "-m", "pytest", *pytest_args],
        )
    elif completed.returncode == 0:
        print("full suite passed, but no memo was recorded because the fingerprint was unresolved.")
    return completed.returncode


def os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def build_fingerprint(repo_root: Path, run: CommandRunner | None = None) -> Fingerprint:
    if run is None:
        run = _run

    components: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    failures: list[str] = []

    root_tree = _git_stdout(repo_root, ["rev-parse", "HEAD^{tree}"], run)
    root_dirty = _git_dirty(repo_root, run)
    if root_tree is None:
        failures.append("rlrmp git tree is unresolved")
    elif root_dirty is None:
        failures.append("rlrmp git dirty state is unresolved")
    elif root_dirty:
        failures.append("rlrmp worktree has tracked or untracked changes")
    else:
        components["rlrmp"] = {"tree": root_tree}

    lock_path = repo_root / "uv.lock"
    if lock_path.is_file():
        components["uv_lock_sha256"] = _sha256_file(lock_path)
    else:
        failures.append("uv.lock is missing")

    try:
        components["python"] = {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        }
        import jax
        import jaxlib

        components["jax"] = {"jax": jax.__version__, "jaxlib": jaxlib.__version__}
    except Exception as exc:  # pragma: no cover - exact import failure is environment-specific.
        failures.append(f"Python/JAX version probe failed: {exc}")

    feedbax_path = _feedbax_path(repo_root)
    if feedbax_path is None:
        failures.append("feedbax editable checkout path is unresolved")
    else:
        feedbax_head = _git_stdout(feedbax_path, ["rev-parse", "HEAD"], run)
        feedbax_dirty = _git_dirty(feedbax_path, run)
        if feedbax_head is None:
            failures.append("feedbax checkout HEAD is unresolved")
        elif feedbax_dirty is None:
            failures.append("feedbax dirty state is unresolved")
        elif feedbax_dirty:
            failures.append("feedbax checkout has tracked or untracked changes")
        else:
            components["feedbax"] = {
                "head": feedbax_head,
            }

    if failures:
        return Fingerprint(ok=False, payload=components, digest=None, reason="; ".join(failures))

    digest = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Fingerprint(ok=True, payload=components, digest=digest)


def memo_has_green(memo_dir: Path, fingerprint: Fingerprint) -> bool:
    if not fingerprint.ok or fingerprint.digest is None:
        return False
    entry = _read_memo(memo_dir, fingerprint)
    return isinstance(entry, dict) and entry.get("result") == "passed"


def record_green(memo_dir: Path, fingerprint: Fingerprint, command: Sequence[str]) -> None:
    if not fingerprint.ok or fingerprint.digest is None:
        raise ValueError("cannot record an unresolved fingerprint")
    entry = {
        "schema_version": SCHEMA_VERSION,
        "result": "passed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint.payload,
        "command": list(command),
    }
    memo_dir.mkdir(parents=True, exist_ok=True)
    path = _memo_file(memo_dir, fingerprint)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_memo(memo_dir: Path, fingerprint: Fingerprint) -> dict[str, Any]:
    path = _memo_file(memo_dir, fingerprint)
    if not path.exists():
        return {}
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if entry.get("schema_version") != SCHEMA_VERSION:
        return {}
    if entry.get("fingerprint") != fingerprint.payload:
        return {}
    return entry


def _memo_file(memo_dir: Path, fingerprint: Fingerprint) -> Path:
    if fingerprint.digest is None:
        raise ValueError("cannot address memo for unresolved fingerprint")
    return memo_dir / f"{fingerprint.digest}.json"


def _memo_dir(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _pythonpath(repo_root: Path) -> str:
    current = os_environ().get("PYTHONPATH")
    src = str(repo_root / "src")
    return src if not current else f"{src}:{current}"


def _feedbax_path(repo_root: Path) -> Path | None:
    try:
        pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        raw_path = pyproject["tool"]["uv"]["sources"]["feedbax"]["path"]
    except (KeyError, OSError, tomllib.TOMLDecodeError):
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _git_dirty(repo: Path, run: CommandRunner) -> bool | None:
    result = run(["git", "status", "--porcelain"], repo)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        if status != "??":
            return True
        path = line[3:]
        if _is_execution_relevant_untracked_path(path):
            return True
    return False


def _is_execution_relevant_untracked_path(path: str) -> bool:
    return any(
        path == relevant_path.rstrip("/") or path.startswith(relevant_path)
        for relevant_path in EXECUTION_RELEVANT_UNTRACKED_PATHS
    )


def _git_stdout(repo: Path, args: Sequence[str], run: CommandRunner) -> str | None:
    result = run(["git", *args], repo)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _run(cmd: Sequence[str], cwd: Path) -> CommandResult:
    completed = subprocess.run(
        list(cmd),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return CommandResult(stdout=completed.stdout, returncode=completed.returncode)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
