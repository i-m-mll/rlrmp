from __future__ import annotations

import json
from pathlib import Path

from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.training.run_matrix import materialize_run_matrix, render_spec_lock_table

from rlrmp.runtime.training_run_specs import (
    register_rlrmp_cs_supervised_method,
    register_rlrmp_distillation_methods,
)
from rlrmp.train.minimax import ensure_minimax_training_method_registered


def _register_methods() -> None:
    register_rlrmp_cs_supervised_method()
    register_rlrmp_distillation_methods()
    ensure_minimax_training_method_registered()


def test_ef9c882_matrix_materializes_planned_run_id_goldens() -> None:
    _register_methods()
    repo_root = Path.cwd()
    matrix = TrainingRunMatrixSpec.model_validate_json(
        (repo_root / "results" / "ef9c882" / "runs" / "matrix.json").read_text(
            encoding="utf-8"
        )
    )

    materialized = materialize_run_matrix(matrix, repo_root=repo_root)
    run_ids = {row.row_id: row.planned_run_id for row in materialized.rows}
    expected = json.loads(
        (repo_root / "tests" / "fixtures" / "ef9c882_matrix_planned_run_ids.json").read_text(
            encoding="utf-8"
        )
    )

    assert run_ids == expected
    assert len(materialized.rows) == 8
    assert "Row count: 8" in render_spec_lock_table(matrix, materialized)
