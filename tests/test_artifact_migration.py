from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from rlrmp.artifact_migration import (
    load_migrated_model_artifact,
    minimax_args_from_run_spec,
)
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.eval.ensemble import eval_ensemble_on_trials
from rlrmp.eval.kinematics import compute_kinematics
from rlrmp.eval.minimax_io import load_model
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.minimax import build_hps


def _migrated_manifest_paths() -> list[Path]:
    return sorted(Path("results/b41c940/migrated").glob("*/*/model.artifact.manifest.json"))


def test_minimax_args_from_run_spec_normalizes_historical_cli_flags() -> None:
    args = minimax_args_from_run_spec(
        {
            "cli_flags": {
                "--hidden-type": "gru",
                "--n-warmup-batches": 12000,
                "--no-streaming-loss": True,
                "nn_hidden_derivative": 0.001,
            },
            "controller_lr": 1e-4,
        }
    )

    assert args.hidden_type == "gru"
    assert args.n_warmup_batches == 12000
    assert args.adversary_type == "gaussian_bump"
    assert args.streaming_loss is False
    assert args.nn_hidden_derivative == 0.001
    assert args.n_adversary_batches == 0


@pytest.mark.parametrize("manifest_path", _migrated_manifest_paths())
def test_migrated_artifact_behavior_matches_legacy_eqx(manifest_path: Path) -> None:
    if not manifest_path.exists():
        pytest.skip("b41c940 migrated artifact records are not present")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_run_spec_path = Path(manifest["provenance"]["parents"][0]["uri"])
    legacy_checkpoint_path = Path(manifest["provenance"]["parents"][1]["uri"])
    array_store_path = Path(manifest["parameter_store"]["uri"])
    if not legacy_checkpoint_path.exists() or not array_store_path.exists():
        pytest.skip("b41c940 bulk legacy/migrated artifacts are not present")

    run_spec = json.loads(legacy_run_spec_path.read_text(encoding="utf-8"))
    hps = build_hps(minimax_args_from_run_spec(run_spec))
    task_model_pair = setup_task_model_pair(hps, key=jr.PRNGKey(0))
    legacy_model = load_model(
        legacy_checkpoint_path.parent,
        legacy_checkpoint_path.name,
        hps,
        run_spec,
    )
    migrated_model = load_migrated_model_artifact(manifest_path, key=jr.PRNGKey(0))

    n_validation_trials = task_model_pair.task.validation_trials.intervene[
        PLANT_INTERVENOR_LABEL
    ].scale.shape[0]
    trial_specs = jtu.tree_map(
        lambda x: (
            x[:2] if eqx.is_array(x) and x.ndim > 0 and x.shape[0] == n_validation_trials else x
        ),
        task_model_pair.task.validation_trials,
    )

    eval_key = jr.PRNGKey(123)
    legacy_states = eval_ensemble_on_trials(
        task_model_pair.task,
        legacy_model,
        trial_specs,
        key=eval_key,
        n_replicates=5,
    )
    migrated_states = eval_ensemble_on_trials(
        task_model_pair.task,
        migrated_model,
        trial_specs,
        key=eval_key,
        n_replicates=5,
    )

    legacy_leaves = jtu.tree_leaves(eqx.filter(legacy_states, eqx.is_array))
    migrated_leaves = jtu.tree_leaves(eqx.filter(migrated_states, eqx.is_array))
    assert len(legacy_leaves) == len(migrated_leaves)
    for legacy_leaf, migrated_leaf in zip(legacy_leaves, migrated_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(migrated_leaf), np.asarray(legacy_leaf))

    legacy_kinematics = compute_kinematics(legacy_states, trial_specs)
    migrated_kinematics = compute_kinematics(migrated_states, trial_specs)
    assert legacy_kinematics.keys() == migrated_kinematics.keys()
    for name in legacy_kinematics:
        np.testing.assert_array_equal(migrated_kinematics[name], legacy_kinematics[name])
