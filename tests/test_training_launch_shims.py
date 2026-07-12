"""Tests for error-only legacy training-script shims."""

from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLACEMENT = (
    "PYTHONPATH=src uv run --no-sync python scripts/launch_training.py "
    "execute <authored-matrix.json>"
)


@pytest.mark.parametrize(
    "script_name",
    [
        "train_cs_nominal_gru.py",
        "train_minimax.py",
        "train_guided_distillation.py",
        "train_closed_loop_distillation.py",
    ],
)
def test_legacy_training_script_is_error_only_shim(script_name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--scientific-flag"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == REPLACEMENT
