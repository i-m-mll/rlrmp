"""Tests for validation-selected GRU checkpoint recovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rlrmp.analysis.gru_checkpoint_selection import (
    active_loss_term_labels,
    select_validation_checkpoints_for_run,
    validation_objective_history,
)


def test_select_validation_checkpoints_ignores_zero_padding(tmp_path: Path) -> None:
    """Select by positive validation records scored at available checkpoints."""

    experiment = "issue123"
    run_id = "run_a"
    run_spec = {
        "hps": {
            "loss": {
                "weights": {
                    "effector_pos_running": 2.0,
                    "nn_output": 1.0,
                }
            }
        }
    }
    run_dir = tmp_path / "results" / experiment / "runs" / run_id
    artifact_dir = tmp_path / "_artifacts" / experiment / "runs" / run_id
    checkpoint_dir = artifact_dir / "checkpoints"
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(run_spec), encoding="utf-8")
    for checkpoint in (3, 6):
        (checkpoint_dir / f"checkpoint_{checkpoint:07d}").mkdir()

    zeros = np.zeros((6, 2), dtype=np.float64)
    training_pos = np.ones((6, 2), dtype=np.float64)
    training_control = np.ones((6, 2), dtype=np.float64)
    validation_pos = np.array(
        [
            [50.0, 50.0],
            [0.0, 0.0],
            [5.0, 0.25],
            [0.0, 0.0],
            [2.0, 0.5],
            [2.0, 10.0],
        ],
        dtype=np.float64,
    )
    validation_control = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    with (artifact_dir / "training_history.eqx").open("wb") as stream:
        stream.write(b"null\n")
        _write_loss_tree(stream, ((training_pos, 2.0), (training_control, 1.0)))
        _write_loss_tree(stream, ((validation_pos, 2.0), (validation_control, 1.0)))
        np.save(stream, zeros, allow_pickle=False)

    objective, valid_records = validation_objective_history(
        run_spec=run_spec,
        history_path=artifact_dir / "training_history.eqx",
    )
    assert objective[1, 0] == 0.0
    assert not valid_records[1, 0]

    selections = select_validation_checkpoints_for_run(
        experiment=experiment,
        run_id=run_id,
        repo_root=tmp_path,
    )
    assert [selection.checkpoint_batches for selection in selections] == [6, 3]
    assert [selection.scoring_validation_log_batch for selection in selections] == [6, 3]
    assert [selection.best_logged_validation_batch for selection in selections] == [5, 3]


def test_active_loss_term_labels_use_full_qrf_objective() -> None:
    run_spec = {
        "loss_objective": "full_analytical_qrf",
        "hps": {
            "loss": {
                "weights": {
                    "effector_pos_running": 1.0,
                    "nn_output": 1.0,
                }
            }
        },
    }

    assert active_loss_term_labels(run_spec) == ("full_analytical_qrf",)


def _write_loss_tree(stream: object, leaves: tuple[tuple[np.ndarray, float], ...]) -> None:
    for value, weight in leaves:
        np.save(stream, value, allow_pickle=False)
        np.save(stream, np.asarray(weight), allow_pickle=False)
    np.save(stream, np.asarray(1.0), allow_pickle=False)
