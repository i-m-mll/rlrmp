"""Rescore GRU checkpoints on fixed delayed-reach no-catch/catch banks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.types import TreeNamespace, dict_to_namespace

from rlrmp.analysis.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.delayed_reach_eval_bank import (
    DEFAULT_DIRECTION_COUNT,
    DEFAULT_GO_CUE_STEPS,
    make_delayed_reach_eval_banks,
)
from rlrmp.analysis.gru_checkpoint_selection import (
    FIXED_BANK_CHECKPOINT_POLICY,
    FIXED_BANK_SCHEMA_VERSION,
    ReplicateCheckpointSelection,
    available_checkpoint_batches,
    checkpoint_path_for_batches,
)
from rlrmp.analysis.trial_alignment import (
    canonical_movement_horizon_from_metadata,
    infer_trial_count,
)
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p


DEFAULT_OUTPUT_NAME = "delayed_reach_fixed_bank_rescored_checkpoints.json"


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    manifest = materialize_delayed_reach_fixed_bank_checkpoint_manifest(
        experiment=args.experiment,
        run_ids=tuple(args.run_id),
        output_path=args.output_path,
        direction_count=args.direction_count,
        go_cue_steps=tuple(range(args.go_cue_min, args.go_cue_max + 1)),
        bank_kinds=tuple(args.bank_kind or ("no_catch", "catch")),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--direction-count", type=int, default=DEFAULT_DIRECTION_COUNT)
    parser.add_argument("--go-cue-min", type=int, default=min(DEFAULT_GO_CUE_STEPS))
    parser.add_argument("--go-cue-max", type=int, default=max(DEFAULT_GO_CUE_STEPS))
    parser.add_argument(
        "--bank-kind",
        action="append",
        choices=("no_catch", "catch"),
        default=None,
        help="Delayed eval bank(s) used for selection. Defaults to both.",
    )
    return parser


def materialize_delayed_reach_fixed_bank_checkpoint_manifest(
    *,
    experiment: str,
    run_ids: Sequence[str],
    output_path: Path | None = None,
    direction_count: int = DEFAULT_DIRECTION_COUNT,
    go_cue_steps: Sequence[int] = DEFAULT_GO_CUE_STEPS,
    bank_kinds: Sequence[str] = ("no_catch", "catch"),
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Write a fixed-bank checkpoint selection manifest for delayed reaches."""

    if not run_ids:
        raise ValueError("At least one run ID is required")
    if not bank_kinds:
        raise ValueError("At least one delayed eval bank kind is required")
    unsupported = set(bank_kinds) - {"no_catch", "catch"}
    if unsupported:
        raise ValueError(f"unsupported bank kinds: {sorted(unsupported)}")

    runs = {
        run_id: [
            selection.to_json(repo_root=repo_root)
            for selection in _score_run_checkpoints(
                experiment=experiment,
                run_id=run_id,
                direction_count=direction_count,
                go_cue_steps=go_cue_steps,
                bank_kinds=bank_kinds,
                repo_root=repo_root,
            )
        ]
        for run_id in run_ids
    }
    output_path = output_path or (
        repo_root / "results" / experiment / "notes" / DEFAULT_OUTPUT_NAME
    )
    manifest = {
        "schema_version": FIXED_BANK_SCHEMA_VERSION,
        "issue": experiment,
        "checkpoint_policy": FIXED_BANK_CHECKPOINT_POLICY,
        "selection_source": "delayed_reach_fixed_bank_rescore",
        "materialization_status": "materialized",
        "validation_bank": {
            "bank_identity": "delayed_reach_go_cue_grid_no_catch_catch",
            "scorer_identity": "feedbax_task_loss_mean_over_trials",
            "go_cue_steps": [int(step) for step in go_cue_steps],
            "go_cue_min": int(min(go_cue_steps)),
            "go_cue_max": int(max(go_cue_steps)),
            "direction_count": int(direction_count),
            "bank_kinds": list(bank_kinds),
        },
        "validation_role": "fixed_delayed_reach_no_catch_catch_rollout_validation",
        "selection_metric": "mean_task_loss_equal_weight_over_declared_banks",
        "selection_policy": (
            "per-replicate checkpoint selected by minimum mean Feedbax task loss "
            "on fixed delayed-reach evaluation banks spanning go cue timing and "
            "center-out directions; no-catch and catch banks are equally weighted"
        ),
        "runs": runs,
    }
    mkdir_p(output_path.parent)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _score_run_checkpoints(
    *,
    experiment: str,
    run_id: str,
    direction_count: int,
    go_cue_steps: Sequence[int],
    bank_kinds: Sequence[str],
    repo_root: Path,
) -> list[ReplicateCheckpointSelection]:
    run_spec_path = repo_root / "results" / experiment / "runs" / run_id / "run.json"
    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    horizon = canonical_movement_horizon_from_metadata(run_spec, default=None)
    all_banks = make_delayed_reach_eval_banks(
        pair.task.validation_trials,
        go_cue_steps=go_cue_steps,
        direction_count=direction_count,
        movement_horizon_steps=horizon,
    )
    banks = {kind: all_banks[kind] for kind in bank_kinds}
    checkpoint_batches = available_checkpoint_batches(artifact_dir)
    if not checkpoint_batches:
        raise FileNotFoundError(f"No checkpoints found under {artifact_dir / 'checkpoints'}")

    per_checkpoint_scores: dict[int, np.ndarray] = {}
    for checkpoint_batch in checkpoint_batches:
        checkpoint_path = checkpoint_path_for_batches(artifact_dir, checkpoint_batch)
        model = eqx.tree_deserialise_leaves(checkpoint_path / "model.eqx", pair.model)
        per_checkpoint_scores[checkpoint_batch] = _score_checkpoint_model(
            task=pair.task,
            model=model,
            n_replicates=n_replicates,
            banks=banks,
        )

    final_batch = checkpoint_batches[-1]
    final_scores = per_checkpoint_scores[final_batch]
    selections = []
    for replicate in range(n_replicates):
        selected_batch = min(
            checkpoint_batches,
            key=lambda batch: float(per_checkpoint_scores[batch][replicate]),
        )
        selected_score = float(per_checkpoint_scores[selected_batch][replicate])
        selections.append(
            ReplicateCheckpointSelection(
                replicate=replicate,
                checkpoint_batches=int(selected_batch),
                checkpoint_path=checkpoint_path_for_batches(artifact_dir, selected_batch),
                selection_source="delayed_reach_fixed_bank_rescore",
                scoring_validation_log_batch=int(selected_batch),
                scoring_validation_objective=selected_score,
                best_logged_validation_batch=int(selected_batch),
                best_logged_validation_objective=selected_score,
                final_validation_objective=float(final_scores[replicate]),
                final_vs_selected_validation_degradation=float(
                    final_scores[replicate] - selected_score
                ),
            )
        )
    return selections


def _score_checkpoint_model(
    *,
    task: Any,
    model: Any,
    n_replicates: int,
    banks: Mapping[str, Any],
) -> np.ndarray:
    """Return mean fixed-bank task loss per replicate for one checkpoint."""

    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates,
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> jnp.ndarray:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        scores = []
        for bank_name, bank in banks.items():
            del bank_name
            n_trials = infer_trial_count(bank.trial_specs)
            _states, losses = task.eval_trials_with_loss(
                replicate_model,
                bank.trial_specs,
                jr.split(key, n_trials),
            )
            scores.append(jnp.mean(_term_tree_value(losses)))
        return jnp.mean(jnp.stack(scores))

    scores = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(0), n_replicates),
    )
    return np.asarray(scores, dtype=np.float64)


def _term_tree_value(term_tree: Any) -> jnp.ndarray:
    """Return weighted scalar/array value for a Feedbax TermTree."""

    if getattr(term_tree, "value", None) is not None:
        return jnp.asarray(term_tree.value) * float(getattr(term_tree, "weight", 1.0))
    children = getattr(term_tree, "children", ())
    if not children:
        raise ValueError(f"TermTree {getattr(term_tree, 'label', '<unknown>')} has no value")
    child_values = [_term_tree_value(child) for child in children]
    return jnp.sum(jnp.stack(child_values), axis=0) * float(
        getattr(term_tree, "weight", 1.0)
    )


if __name__ == "__main__":
    main()
