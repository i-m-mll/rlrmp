"""Selective pytest runner for rlrmp development loops.

This runner is an inner-loop convenience only. It never records full-suite memo
entries and is not an integration or auth gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


DEFAULT_TEST_TARGET = "tests/"
DEFAULT_TESTMON_BASE = "main"
DEFAULT_TESTMON_DATA = ".testmondata"
TESTMON_STATE_SUFFIX = ".rlrmp-state.json"
PYTEST_OPTIONS_WITH_VALUES = {
    "--confcutdir",
    "--cov",
    "--cov-append",
    "--cov-config",
    "--cov-fail-under",
    "--cov-report",
    "--deselect",
    "--doctest-glob",
    "--durations",
    "--ignore",
    "--ignore-glob",
    "--import-mode",
    "--junit-prefix",
    "--junit-xml",
    "--lfnf",
    "--log-cli-date-format",
    "--log-cli-format",
    "--log-cli-level",
    "--log-date-format",
    "--log-file",
    "--log-file-date-format",
    "--log-file-format",
    "--log-file-level",
    "--log-format",
    "--log-level",
    "--maxfail",
    "--override-ini",
    "--rootdir",
    "--tb",
    "--verbosity",
    "-k",
    "-m",
    "-n",
    "-o",
    "-p",
    "-W",
}


@dataclasses.dataclass(frozen=True)
class TestmonState:
    branch: str
    head: str
    base_ref: str
    merge_base: str


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args, pytest_args = parse_args(argv)
    pytest_args = normalize_pytest_args(pytest_args)

    if args.testmon and "--testmon" not in pytest_args:
        pytest_args = ["--testmon", *pytest_args]

    if args.workers is not None and "-n" not in pytest_args:
        pytest_args = ["-n", args.workers, *pytest_args]

    command = [sys.executable, "-m", "pytest", *pytest_args]
    if args.dry_run:
        print(" ".join(command))
        return 0

    if args.testmon:
        state = build_testmon_state(repo_root, args.testmon_base)
        testmon_data = resolve_repo_path(repo_root, Path(DEFAULT_TESTMON_DATA))
        prepare_testmon_data(testmon_data, state)

    completed = subprocess.run(
        command,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": pythonpath(repo_root, os.environ)},
        check=False,
    )

    if args.testmon:
        write_testmon_state(testmon_data, state)
    return completed.returncode


def parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run targeted pytest checks for development iteration.",
        epilog=(
            "Examples: scripts/dev_tests.sh tests/test_x.py::test_y; "
            "scripts/dev_tests.sh -k descriptor; scripts/dev_tests.sh --lf; "
            "scripts/dev_tests.sh --testmon"
        ),
    )
    parser.add_argument(
        "--testmon",
        action="store_true",
        help="Run pytest-testmon after invalidating stale .testmondata.",
    )
    parser.add_argument(
        "--testmon-base",
        default=DEFAULT_TESTMON_BASE,
        help="Git ref used to record the testmon merge-base. Default: main.",
    )
    parser.add_argument(
        "--workers",
        help="Optional pytest-xdist worker count for local targeted runs, e.g. auto or 4.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pytest command without running it.",
    )
    return parser.parse_known_args(argv)


def normalize_pytest_args(pytest_args: Sequence[str]) -> list[str]:
    args = list(pytest_args)
    if not has_explicit_test_target(args):
        return [DEFAULT_TEST_TARGET, *args]
    return args


def has_explicit_test_target(pytest_args: Sequence[str]) -> bool:
    skip_next = False
    for arg in pytest_args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            continue
        if arg in PYTEST_OPTIONS_WITH_VALUES:
            skip_next = True
            continue
        if arg.startswith("--") and "=" in arg:
            continue
        if arg.startswith("-"):
            continue
        return True
    return False


def build_testmon_state(repo_root: Path, base_ref: str) -> TestmonState:
    branch = git_stdout(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    head = git_stdout(repo_root, ["rev-parse", "HEAD"])
    merge_base = git_stdout(repo_root, ["merge-base", "HEAD", base_ref])
    return TestmonState(branch=branch, head=head, base_ref=base_ref, merge_base=merge_base)


def prepare_testmon_data(testmon_data: Path, state: TestmonState) -> None:
    metadata_path = testmon_state_path(testmon_data)
    prior_state = read_testmon_state(metadata_path)
    if prior_state == state:
        return
    remove_path(testmon_data)
    remove_path(metadata_path)


def write_testmon_state(testmon_data: Path, state: TestmonState) -> None:
    metadata_path = testmon_state_path(testmon_data)
    metadata_path.write_text(
        json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_testmon_state(metadata_path: Path) -> TestmonState | None:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    try:
        return TestmonState(
            branch=payload["branch"],
            head=payload["head"],
            base_ref=payload["base_ref"],
            merge_base=payload["merge_base"],
        )
    except KeyError:
        return None


def testmon_state_path(testmon_data: Path) -> Path:
    return testmon_data.with_name(f"{testmon_data.name}{TESTMON_STATE_SUFFIX}")


def resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def git_stdout(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown git error"
        raise SystemExit(f"dev-tests testmon state failed: git {' '.join(args)}: {detail}")
    return completed.stdout.strip()


def pythonpath(repo_root: Path, environ: Mapping[str, str]) -> str:
    src = str(repo_root / "src")
    current = environ.get("PYTHONPATH")
    return src if not current else f"{src}:{current}"


if __name__ == "__main__":
    raise SystemExit(main())
