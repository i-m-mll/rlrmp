from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.training.run_matrix import materialize_run_matrix, render_spec_lock_table

from rlrmp.runtime.training_run_specs import (
    register_rlrmp_cs_supervised_method,
    register_rlrmp_distillation_methods,
)
from rlrmp.train.minimax_native import ensure_minimax_training_method_registered


LEGACY_SPEC_STOCK_COMMIT = "ab380a0fde3428ae2385bce50fff230a836d0bac"


def _legacy_flat_run(repo_root: Path, row_id: str) -> dict[str, object]:
    result = subprocess.run(
        [
            "git",
            "show",
            f"{LEGACY_SPEC_STOCK_COMMIT}:results/ef9c882/runs/{row_id}.json",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _delayed_pre_go_semantics(loss_summary: dict[str, object]) -> dict[str, object]:
    auxiliary = deepcopy(loss_summary["delayed_pre_go_auxiliary_terms"])
    assert isinstance(auxiliary, dict)
    for group_name in ("active_terms", "terms"):
        group = auxiliary[group_name]
        assert isinstance(group, dict)
        start_position = group.get("delayed_pre_go_start_pos_hold")
        if isinstance(start_position, dict) and "norm" not in start_position:
            start_position["norm"] = "l2"
    weights = loss_summary["weights"]
    assert isinstance(weights, dict)
    return {
        "auxiliary": auxiliary,
        "weights": {
            key: value for key, value in weights.items() if key.startswith("delayed_pre_go_")
        },
    }


def _register_methods() -> None:
    register_rlrmp_cs_supervised_method()
    register_rlrmp_distillation_methods()
    ensure_minimax_training_method_registered()


def test_ef9c882_matrix_materializes_all_compact_historical_rows() -> None:
    _register_methods()
    repo_root = Path.cwd()
    matrix = TrainingRunMatrixSpec.model_validate_json(
        (repo_root / "results" / "ef9c882" / "runs" / "matrix.json").read_text(
            encoding="utf-8"
        )
    )

    materialized = materialize_run_matrix(matrix, repo_root=repo_root)
    expected_rows = [
        "hold__force_filter",
        "hold__start_pos_zero_vel",
        "hold__start_pos_zero_vel_lr1e-2",
        "hold__start_pos_zero_vel_lr3e-2",
        "hold_start_pos_l1__w1e4",
        "hold_start_pos_l1__w1e5",
        "hold_start_pos_l1__w1e6",
        "hold_start_pos_l1__w1e8",
        "hold_start_pos_l1_ffpert__w1e5_lr1e-2",
        "hold_start_pos_l1_ffpert__w1e5_lr3e-3",
        "hold_start_pos_l1_ffpert__w1e6_lr3e-3",
        "hold_start_pos_l2__w1e4",
        "hold_start_pos_l2__w1e6",
        "hold_start_pos_l2__w1e8",
        "hold_start_pos_l2_ffpert__w1e6_lr3e-3",
        "hold_start_pos_l2_ffpert__w1e8_lr1e-2",
        "hold_start_pos_l2_ffpert__w1e8_lr3e-3",
    ]
    assert [row.row_id for row in materialized.rows] == expected_rows
    assert len({row.planned_run_id for row in materialized.rows}) == 17
    assert "Row count: 17" in render_spec_lock_table(matrix, materialized)

    for row in materialized.rows:
        legacy = _legacy_flat_run(repo_root, row.row_id)
        assert row.payload["training_config"]["learning_rate"] == legacy["controller_lr"]
        assert _delayed_pre_go_semantics(
            row.payload["objective"]["payload"]["loss_summary"]
        ) == _delayed_pre_go_semantics(legacy["loss_summary"])
        assert row.payload["method_payload"]["payload"]["training_mode"] == legacy[
            "training_summary"
        ]["training_mode"]
