"""Focused checks for the selective development test runner."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dev_tests.py"
SPEC = importlib.util.spec_from_file_location("dev_tests", SCRIPT_PATH)
assert SPEC is not None
dev_tests = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = dev_tests
SPEC.loader.exec_module(dev_tests)


def test_normalize_defaults_to_tests_for_keyword_selection() -> None:
    assert dev_tests.normalize_pytest_args(["-k", "descriptor"]) == [
        "tests/",
        "-k",
        "descriptor",
    ]


def test_normalize_defaults_to_tests_for_last_failed_selection() -> None:
    assert dev_tests.normalize_pytest_args(["--lf"]) == ["tests/", "--lf"]


def test_normalize_preserves_explicit_node_id() -> None:
    node = "tests/test_dev_tests_runner.py::test_normalize_preserves_explicit_node_id"
    assert dev_tests.normalize_pytest_args([node, "-q"]) == [node, "-q"]


def test_testmon_preparation_removes_stale_data_on_branch_switch(tmp_path: Path) -> None:
    testmon_data = tmp_path / ".testmondata"
    metadata_path = dev_tests.testmon_state_path(testmon_data)
    testmon_data.write_text("sqlite bytes would live here\n", encoding="utf-8")
    stale_state = dev_tests.TestmonState(
        branch="feature/old",
        head="1111111",
        base_ref="main",
        merge_base="2222222",
    )
    fresh_state = dataclasses.replace(stale_state, branch="feature/new")
    dev_tests.write_testmon_state(testmon_data, stale_state)

    dev_tests.prepare_testmon_data(testmon_data, fresh_state)

    assert not testmon_data.exists()
    assert not metadata_path.exists()


def test_testmon_preparation_removes_stale_data_on_rebase(tmp_path: Path) -> None:
    testmon_data = tmp_path / ".testmondata"
    metadata_path = dev_tests.testmon_state_path(testmon_data)
    testmon_data.write_text("sqlite bytes would live here\n", encoding="utf-8")
    stale_state = dev_tests.TestmonState(
        branch="feature/current",
        head="1111111",
        base_ref="main",
        merge_base="2222222",
    )
    rebased_state = dataclasses.replace(stale_state, head="3333333", merge_base="4444444")
    dev_tests.write_testmon_state(testmon_data, stale_state)

    dev_tests.prepare_testmon_data(testmon_data, rebased_state)

    assert not testmon_data.exists()
    assert not metadata_path.exists()


def test_testmon_preparation_keeps_matching_data(tmp_path: Path) -> None:
    testmon_data = tmp_path / ".testmondata"
    testmon_data.write_text("sqlite bytes would live here\n", encoding="utf-8")
    state = dev_tests.TestmonState(
        branch="feature/current",
        head="1111111",
        base_ref="main",
        merge_base="2222222",
    )
    dev_tests.write_testmon_state(testmon_data, state)

    dev_tests.prepare_testmon_data(testmon_data, state)

    assert testmon_data.exists()
    assert dev_tests.read_testmon_state(dev_tests.testmon_state_path(testmon_data)) == state


def test_testmon_preparation_removes_data_without_metadata(tmp_path: Path) -> None:
    testmon_data = tmp_path / ".testmondata"
    testmon_data.write_text("sqlite bytes would live here\n", encoding="utf-8")
    state = dev_tests.TestmonState(
        branch="feature/current",
        head="1111111",
        base_ref="main",
        merge_base="2222222",
    )

    dev_tests.prepare_testmon_data(testmon_data, state)

    assert not testmon_data.exists()
