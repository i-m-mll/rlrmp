"""Structural guard for experiment-script relocations owned by b632f57."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DESTINATIONS = {
    "eval_part2_5.py": "results/2ef67ca/scripts/eval_part2_5.py",
    "eval_part2_5_figures.py": "results/2ef67ca/scripts/eval_part2_5_figures.py",
    "probe_round_trip_ratio.py": "results/72fb8d9/scripts/probe_round_trip_ratio.py",
    "diag_probe_anomalies.py": "results/72fb8d9/scripts/diag_probe_anomalies.py",
    "diag_cs_baseline.py": "results/72fb8d9/scripts/diag_cs_baseline.py",
    "diag_cs_bw_full_state.py": "results/72fb8d9/scripts/diag_cs_bw_full_state.py",
    "diag_cs_bw_full_state_sweeps.py": (
        "results/72fb8d9/scripts/diag_cs_bw_full_state_sweeps.py"
    ),
    "materialize_output_feedback_optimizer_basin_diagnostic.py": (
        "results/1c014e5/scripts/materialize_output_feedback_optimizer_basin_diagnostic.py"
    ),
    "materialize_output_feedback_observer_error_coverage.py": (
        "results/3becdec/scripts/materialize_output_feedback_observer_error_coverage.py"
    ),
}


def test_relocated_experiment_scripts_are_absent_from_top_level_scripts() -> None:
    assert not [
        name for name in DESTINATIONS if (REPO_ROOT / "scripts" / name).exists()
    ]


def test_relocated_scripts_exist_without_runpy_or_sys_path_hacks() -> None:
    for destination in DESTINATIONS.values():
        path = REPO_ROOT / destination
        assert path.is_file(), destination
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=destination)
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name == "runpy" for alias in node.names)
            for node in ast.walk(tree)
        ), destination
        assert "sys.path" not in path.read_text(encoding="utf-8"), destination


def test_relocated_destinations_import_in_isolated_processes() -> None:
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    for destination in DESTINATIONS.values():
        code = (
            "import importlib.util; "
            f"p={str(REPO_ROOT / destination)!r}; "
            "s=importlib.util.spec_from_file_location('relocated_script', p); "
            "m=importlib.util.module_from_spec(s); "
            "s.loader.exec_module(m)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{destination}\n{result.stdout}\n{result.stderr}"
