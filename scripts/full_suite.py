"""Memoized full-suite runner for rlrmp.

This wrapper records only clean, full-suite green runs. If any fingerprint
component cannot be resolved, it runs the suite instead of trusting the memo.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import dataclasses
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import tempfile
import tomllib
from types import TracebackType
from typing import Any, Callable, Self, Sequence


SCHEMA_VERSION = 1
LOCK_PROTOCOL_VERSION = 1
LOCK_BUSY_EXIT = 75
LOCK_ENV_VAR = "FULL_SUITE_LOCK_DIR"
LOCK_FILENAME = "full-suite.lock"
LOCK_REPOSITORY = "rlrmp"
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


class FullSuiteLockBusy(RuntimeError):
    """Raised when another participating repository owns the suite lock."""


class FullSuiteLock(AbstractContextManager["FullSuiteLock"]):
    """Hold the machine-wide advisory lock used by participating test suites."""

    def __init__(
        self,
        path: Path,
        *,
        repo_root: Path,
        repository: str = LOCK_REPOSITORY,
        command: Sequence[str] | None = None,
    ) -> None:
        self.path = path
        self.repo_root = repo_root
        self.repository = repository
        self.command = list(sys.argv if command is None else command)
        self._handle: Any | None = None

    def _read_holder(self) -> str:
        assert self._handle is not None
        self._handle.seek(0)
        raw = self._handle.read().strip()
        if not raw:
            return "holder metadata unavailable"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return f"unreadable holder metadata: {raw!r}"
        return ", ".join(
            f"{key}={data[key]}"
            for key in ("repository", "pid", "host", "started_at", "worktree", "command")
            if data.get(key) is not None
        )

    def _write_holder(self) -> None:
        assert self._handle is not None
        holder = {
            "schema_version": LOCK_PROTOCOL_VERSION,
            "protocol_version": LOCK_PROTOCOL_VERSION,
            "repository": self.repository,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "worktree": str(self.repo_root),
            "command": self.command,
        }
        self._handle.seek(0)
        self._handle.truncate()
        json.dump(holder, self._handle, sort_keys=True)
        self._handle.write("\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise FullSuiteLockBusy(
                    f"Full suite already running; active holder: {self._read_holder()}"
                ) from exc
            self._write_holder()
            print(f"Acquired full-suite lock: {self.path}", file=sys.stderr, flush=True)
            return self
        except BaseException:
            self._handle.close()
            self._handle = None
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None


def full_suite_lock_dir(environ: dict[str, str] | None = None) -> Path:
    """Return the shared, user-scoped full-suite lock directory."""
    env = os_environ() if environ is None else environ
    if override := env.get(LOCK_ENV_VAR):
        return Path(override).expanduser()
    try:
        user_token = str(os.getuid())
    except AttributeError:  # pragma: no cover - exercised only on platforms without getuid.
        user_token = re.sub(r"[^A-Za-z0-9_.-]", "_", getpass.getuser()) or "unknown"
    return Path(tempfile.gettempdir()) / f"full-suite-lock-{user_token}"


def full_suite_lock_path(environ: dict[str, str] | None = None) -> Path:
    """Return the shared lock-file path used across participating repositories."""
    return full_suite_lock_dir(environ) / LOCK_FILENAME


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
    if args.print_fingerprint:
        fingerprint = build_fingerprint(repo_root)
        print(json.dumps(dataclasses.asdict(fingerprint), indent=2, sort_keys=True))
        return 0 if fingerprint.ok else 2

    try:
        with FullSuiteLock(full_suite_lock_path(), repo_root=repo_root):
            return _run_full_suite(args, repo_root)
    except FullSuiteLockBusy as exc:
        print(f"ERROR: scripts/full_suite.sh refused to run: {exc}", file=sys.stderr)
        return LOCK_BUSY_EXIT


def _run_full_suite(args: argparse.Namespace, repo_root: Path) -> int:
    fingerprint = build_fingerprint(repo_root)

    if not fingerprint.ok:
        print(f"full-suite memo disabled: {fingerprint.reason}", file=sys.stderr)
    elif (
        not args.no_memo
        and not args.force
        and memo_has_green(_memo_dir(repo_root, args.memo_dir), fingerprint)
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
