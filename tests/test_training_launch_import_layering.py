"""Import-layering guard for authored training-matrix validation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_authored_launch_validate_does_not_import_jax_or_initialize_backend(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_id": "feedbax.spec.training_run_matrix",
                "schema_version": "feedbax.spec.training_run_matrix.v3",
                "name": "import-layering probe",
                "base": {
                    "kind": "resolved_output",
                    "ref": "resolved.json",
                    "resolved_root_hash": "0" * 64,
                },
                "rows": [{"row_id": "only"}],
            }
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT / "src"), environment.get("PYTHONPATH", ""))
    )
    probe = """
import importlib.abc
import runpy
import sys

attempted = []

class JaxImportProbe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "jax" or fullname.startswith(("jax.", "jaxlib")):
            attempted.append(fullname)
        return None

sys.meta_path.insert(0, JaxImportProbe())
sys.argv = [sys.argv[1], "validate", sys.argv[2], "--repo-root", sys.argv[3]]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as error:
    assert error.code == 0, error.code

assert "jax" not in sys.modules
assert not any(name == "jax" or name.startswith(("jax.", "jaxlib")) for name in sys.modules)
assert attempted == [], f"JAX import attempted during validate: {attempted}"
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
            str(REPO_ROOT / "scripts" / "launch_training.py"),
            str(matrix_path),
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "valid TrainingRunMatrixSpec" in completed.stdout
