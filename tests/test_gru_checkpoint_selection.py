"""Tests for validation-selected GRU checkpoint recovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rlrmp.analysis.gru_checkpoint_selection import (
    FixedValidationBankSpec,
    active_loss_term_labels,
    materialize_fixed_bank_checkpoint_rescore_manifest,
    materialize_validation_selected_checkpoint_manifest,
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
        _write_loss_tree(
            stream,
            ((validation_pos, 2.0), (validation_control, 1.0)),
            branch_weight=np.ones_like(validation_pos),
        )
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
    assert [selection.selection_source for selection in selections] == [
        "sparse_history_fallback",
        "sparse_history_fallback",
    ]
    assert selections[0].final_vs_selected_validation_degradation == 0.0


def test_fixed_bank_rescore_manifest_scores_all_durable_checkpoints(tmp_path: Path) -> None:
    experiment = "issue123"
    run_id = "run_a"
    run_dir, artifact_dir = _make_checkpoint_fixture(
        tmp_path,
        experiment=experiment,
        run_id=run_id,
        n_replicates=2,
        checkpoints=(3, 6),
    )
    assert run_dir.exists()
    bank = FixedValidationBankSpec(
        bank_identity="fixed-validation-bank:test",
        scorer_identity="rollout_validation_objective:test",
        seed=123,
        n_trials=8,
        validation_role="generalized_held_out_perturbation_validation",
        selection_metric="aggregate_rollout_validation_objective",
        nominal_quality_role="reported_quality_sidecar_gate",
    )
    scores = {
        (0, 3): 10.0,
        (0, 6): 4.0,
        (1, 3): 2.0,
        (1, 6): 5.0,
    }

    def scorer(
        _run_id: str,
        replicate: int,
        checkpoint_batches: int,
        checkpoint_path: Path,
        _run_spec: dict[str, object],
        validation_bank: FixedValidationBankSpec,
    ) -> float:
        expected_path = artifact_dir / "checkpoints" / f"checkpoint_{checkpoint_batches:07d}"
        assert checkpoint_path == expected_path
        assert validation_bank.bank_identity == "fixed-validation-bank:test"
        return scores[(replicate, checkpoint_batches)]

    manifest = materialize_fixed_bank_checkpoint_rescore_manifest(
        experiment=experiment,
        run_ids=(run_id,),
        validation_bank=bank,
        scorer=scorer,
        repo_root=tmp_path,
    )

    assert manifest["materialization_status"] == "materialized"
    assert manifest["validation_bank"] == {
        "bank_identity": "fixed-validation-bank:test",
        "scorer_identity": "rollout_validation_objective:test",
        "seed": 123,
        "n_trials": 8,
        "validation_role": "generalized_held_out_perturbation_validation",
        "selection_metric": "aggregate_rollout_validation_objective",
        "nominal_quality_role": "reported_quality_sidecar_gate",
    }
    assert manifest["validation_role"] == "generalized_held_out_perturbation_validation"
    assert manifest["selection_metric"] == "aggregate_rollout_validation_objective"
    assert manifest["nominal_quality_role"] == "reported_quality_sidecar_gate"
    selections = manifest["runs"][run_id]
    assert [selection["checkpoint_batches"] for selection in selections] == [6, 3]
    assert [selection["scoring_validation_objective"] for selection in selections] == [4.0, 2.0]
    assert [selection["final_vs_selected_validation_degradation"] for selection in selections] == [
        0.0,
        3.0,
    ]

    selected = select_validation_checkpoints_for_run(
        experiment=experiment,
        run_id=run_id,
        repo_root=tmp_path,
    )
    assert [selection.checkpoint_batches for selection in selected] == [6, 3]
    assert [selection.selection_source for selection in selected] == [
        "fixed_bank_rescore",
        "fixed_bank_rescore",
    ]


def test_not_materialized_fixed_bank_manifest_falls_back_to_sparse_history(
    tmp_path: Path,
) -> None:
    experiment = "issue123"
    run_id = "run_a"
    _make_checkpoint_fixture(
        tmp_path,
        experiment=experiment,
        run_id=run_id,
        n_replicates=1,
        checkpoints=(3, 6),
    )
    _write_sparse_history_fixture(
        tmp_path,
        experiment=experiment,
        run_id=run_id,
        validation_values=np.array([[5.0], [0.0], [2.0], [0.0], [3.0], [4.0]]),
    )
    bank = FixedValidationBankSpec(
        bank_identity="fixed-validation-bank:test",
        scorer_identity="rollout_validation_objective:test",
        seed=123,
    )
    rescore_manifest = materialize_fixed_bank_checkpoint_rescore_manifest(
        experiment=experiment,
        run_ids=(run_id,),
        validation_bank=bank,
        repo_root=tmp_path,
    )
    assert rescore_manifest["materialization_status"] == "not_materialized"

    manifest = materialize_validation_selected_checkpoint_manifest(
        experiment=experiment,
        run_ids=(run_id,),
        repo_root=tmp_path,
    )

    assert manifest["selection_source"] == "sparse_history_fallback"
    assert manifest["runs"][run_id][0]["checkpoint_batches"] == 3


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


def test_active_loss_term_labels_include_force_filter_ablation_term() -> None:
    run_spec = {
        "loss_objective": "partial_net_output_force_filter",
        "hps": {
            "loss": {
                "weights": {
                    "mechanics_force_filter": 1.0 / 6.0,
                    "nn_output": 1.0,
                }
            }
        },
    }

    assert active_loss_term_labels(run_spec) == ("mechanics_force_filter", "nn_output")


def _write_loss_tree(
    stream: object,
    leaves: tuple[tuple[np.ndarray, float], ...],
    *,
    branch_weight: float | np.ndarray = 1.0,
) -> None:
    for value, weight in leaves:
        np.save(stream, value, allow_pickle=False)
        np.save(stream, np.asarray(weight), allow_pickle=False)
    np.save(stream, np.asarray(branch_weight), allow_pickle=False)


def _make_checkpoint_fixture(
    repo_root: Path,
    *,
    experiment: str,
    run_id: str,
    n_replicates: int,
    checkpoints: tuple[int, ...],
) -> tuple[Path, Path]:
    run_dir = repo_root / "results" / experiment / "runs" / run_id
    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    checkpoint_dir = artifact_dir / "checkpoints"
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "hps": {
                    "model": {"n_replicates": n_replicates},
                    "loss": {"weights": {"effector_pos_running": 1.0}},
                }
            }
        ),
        encoding="utf-8",
    )
    for checkpoint in checkpoints:
        (checkpoint_dir / f"checkpoint_{checkpoint:07d}").mkdir()
    return run_dir, artifact_dir


def _write_sparse_history_fixture(
    repo_root: Path,
    *,
    experiment: str,
    run_id: str,
    validation_values: np.ndarray,
) -> None:
    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    zeros = np.zeros_like(validation_values, dtype=np.float64)
    with (artifact_dir / "training_history.eqx").open("wb") as stream:
        stream.write(b"null\n")
        _write_loss_tree(stream, ((np.ones_like(validation_values), 1.0),))
        _write_loss_tree(stream, ((validation_values, 1.0),))
        np.save(stream, zeros, allow_pickle=False)
