"""Regression tests for import-time JAX x64 side effects."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from rlrmp.runtime.import_lint import scan_source, violations


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
NO_IMPORT_X64_MODULES = (
    "rlrmp.analysis",
    "rlrmp.analysis.math.hinf_riccati",
    "rlrmp.analysis.math.output_feedback",
    "rlrmp.analysis.math.robust_bellman",
    "rlrmp.analysis.math.linear_round_trip",
    "rlrmp.analysis.math.linear_equivalence_certificate",
    "rlrmp.analysis.math.cs_released_simulation",
    "rlrmp.analysis.math.induced_gain",
)

_IMPORT_TIME_X64_SNIPPET = """
import jax

jax.config.update("jax_enable_x64", True)
"""

_FUNCTION_SCOPE_X64_SNIPPET = """
import jax

def main():
    jax.config.update("jax_enable_x64", True)
"""


def test_import_lint_negative_canary_flags_module_level_x64_update() -> None:
    findings = scan_source(_IMPORT_TIME_X64_SNIPPET, "src/rlrmp/example.py")
    assert [(finding.option, finding.lineno) for finding in findings] == [
        ("jax_enable_x64", 4)
    ]


def test_import_lint_ignores_function_scope_x64_update() -> None:
    assert scan_source(_FUNCTION_SCOPE_X64_SNIPPET, "src/rlrmp/example.py") == []


def test_src_tree_has_no_import_time_jax_x64_updates() -> None:
    found = violations(SRC_ROOT, repo_root=REPO_ROOT)
    assert found == [], [
        f"{finding.relpath}:{finding.lineno} updates {finding.option!r} at import time"
        for finding in found
    ]


def test_analysis_imports_leave_jax_x64_disabled_in_fresh_process() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["JAX_ENABLE_X64"] = "False"
    code = """
import importlib
import jax

modules = {modules!r}
if jax.config.jax_enable_x64:
    raise SystemExit("x64 was enabled before imports")
for name in modules:
    importlib.import_module(name)
    if jax.config.jax_enable_x64:
        raise SystemExit(f"{{name}} enabled x64 at import time")
""".format(modules=NO_IMPORT_X64_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
