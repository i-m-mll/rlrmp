"""Validation-selected checkpoint recovery for C&S GRU pilot artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.types import TreeNamespace, dict_to_namespace

from rlrmp.analysis.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p


@dataclass(frozen=True)
class ReplicateCheckpointSelection:
    """Recoverable validation-selected checkpoint for one replicate."""

    replicate: int
    checkpoint_batches: int
    checkpoint_path: Path
    scoring_validation_log_batch: int
    scoring_validation_objective: float
    best_logged_validation_batch: int
    best_logged_validation_objective: float
    final_validation_objective: float

    def to_json(self, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
        return {
            "replicate": self.replicate,
            "checkpoint_batches": self.checkpoint_batches,
            "checkpoint_path": _repo_relative(self.checkpoint_path, repo_root=repo_root),
            "scoring_validation_log_batch": self.scoring_validation_log_batch,
            "scoring_validation_objective": self.scoring_validation_objective,
            "best_logged_validation_batch": self.best_logged_validation_batch,
            "best_logged_validation_objective": self.best_logged_validation_objective,
            "final_validation_objective": self.final_validation_objective,
        }


def materialize_validation_selected_checkpoint_manifest(
    *,
    experiment: str,
    run_ids: Sequence[str],
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Write a JSON manifest of recoverable validation-selected checkpoints."""

    selections = {
        run_id: [
            selection.to_json(repo_root=repo_root)
            for selection in select_validation_checkpoints_for_run(
                experiment=experiment,
                run_id=run_id,
                repo_root=repo_root,
            )
        ]
        for run_id in run_ids
    }
    manifest = {
        "schema_version": "rlrmp.validation_selected_gru_checkpoints.v1",
        "issue": experiment,
        "selection_policy": (
            "per-replicate checkpoint selected by minimum positive rollout validation "
            "objective among available durable checkpoints; analytical action and I/O "
            "metrics are audit-only"
        ),
        "history_validation_log_note": (
            "Validation history contains sparse positive records and zero padding. "
            "Durable models are only available at numbered checkpoints, so each "
            "checkpoint is scored by the most recent positive validation record at "
            "or before that checkpoint."
        ),
        "runs": selections,
    }
    output_path = output_path or (
        repo_root / "results" / experiment / "notes" / "validation_selected_checkpoints.json"
    )
    mkdir_p(output_path.parent)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def select_validation_checkpoints_for_run(
    *,
    experiment: str,
    run_id: str,
    repo_root: Path = REPO_ROOT,
) -> list[ReplicateCheckpointSelection]:
    """Select the best available checkpoint for each replicate in a run."""

    run_spec_path = repo_root / "results" / experiment / "runs" / run_id / "run.json"
    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    objective, valid_records = validation_objective_history(
        run_spec=run_spec,
        history_path=artifact_dir / "training_history.eqx",
    )
    checkpoint_batches = available_checkpoint_batches(artifact_dir)
    if not checkpoint_batches:
        raise FileNotFoundError(f"No numbered checkpoints found under {artifact_dir / 'checkpoints'}")

    selections: list[ReplicateCheckpointSelection] = []
    for replicate in range(objective.shape[1]):
        valid_batches = np.flatnonzero(valid_records[:, replicate]) + 1
        if valid_batches.size == 0:
            raise ValueError(f"No positive validation records for {run_id} replicate {replicate}")
        valid_values = objective[valid_batches - 1, replicate]
        best_logged_index = int(np.argmin(valid_values))
        best_logged_batch = int(valid_batches[best_logged_index])
        best_logged_value = float(valid_values[best_logged_index])

        checkpoint_scores: list[tuple[float, int, int]] = []
        for checkpoint_batch in checkpoint_batches:
            eligible_batches = valid_batches[valid_batches <= checkpoint_batch]
            if eligible_batches.size == 0:
                continue
            scoring_batch = int(eligible_batches[-1])
            checkpoint_scores.append(
                (
                    float(objective[scoring_batch - 1, replicate]),
                    int(checkpoint_batch),
                    scoring_batch,
                )
            )
        if not checkpoint_scores:
            raise ValueError(f"No validation-scored checkpoints for {run_id} replicate {replicate}")
        score, checkpoint_batch, scoring_batch = min(checkpoint_scores, key=lambda item: item[0])
        selections.append(
            ReplicateCheckpointSelection(
                replicate=replicate,
                checkpoint_batches=checkpoint_batch,
                checkpoint_path=checkpoint_path_for_batches(artifact_dir, checkpoint_batch),
                scoring_validation_log_batch=scoring_batch,
                scoring_validation_objective=score,
                best_logged_validation_batch=best_logged_batch,
                best_logged_validation_objective=best_logged_value,
                final_validation_objective=float(objective[-1, replicate]),
            )
        )
    return selections


def load_validation_selected_checkpoint_model(
    *,
    experiment: str,
    run_id: str,
    run_spec: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> tuple[Any, list[ReplicateCheckpointSelection]]:
    """Load a model ensemble assembled from per-replicate selected checkpoints."""

    selections = select_validation_checkpoints_for_run(
        experiment=experiment,
        run_id=run_id,
        repo_root=repo_root,
    )
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    if len(selections) != n_replicates:
        raise ValueError(
            f"Selection count {len(selections)} does not match n_replicates={n_replicates}"
        )
    seed = int(run_spec.get("seed", 42))
    template = setup_task_model_pair(hps, key=jr.PRNGKey(seed)).model
    models = [
        eqx.tree_deserialise_leaves(selection.checkpoint_path / "model.eqx", template)
        for selection in selections
    ]

    def select_leaf(*leaves: Any) -> Any:
        leaf = leaves[0]
        if eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates:
            return jnp.stack([replicate_leaf[idx] for idx, replicate_leaf in enumerate(leaves)])
        return leaf

    return jt.map(select_leaf, *models), selections


def validation_objective_history(
    *,
    run_spec: Mapping[str, Any],
    history_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted validation objective and real-record mask, shape ``(batch, rep)``."""

    labels = active_loss_term_labels(run_spec)
    components: list[np.ndarray] = []
    real_record_terms: list[np.ndarray] = []
    with history_path.open("rb") as stream:
        header = stream.readline()
        if header.strip() != b"null":
            raise ValueError(f"Expected null history metadata header in {history_path}")
        _skip_loss_tree(stream, labels)
        for _label in labels:
            value = np.load(stream, allow_pickle=False)
            weight = float(np.load(stream, allow_pickle=False))
            components.append(np.asarray(value, dtype=np.float64) * weight)
            real_record_terms.append(np.asarray(value) != 0)
        branch_weight = float(np.load(stream, allow_pickle=False))
    objective = np.sum(np.stack(components), axis=0) * branch_weight
    valid_records = np.any(np.stack(real_record_terms), axis=0)
    return objective, valid_records


def active_loss_term_labels(run_spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Return active loss labels in Feedbax's serialized term order."""

    weights = run_spec.get("hps", {}).get("loss", {}).get("weights", {})
    candidate_order = (
        "effector_pos_running",
        "effector_terminal_pos",
        "effector_terminal_vel",
        "effector_vel_running",
        "effector_hold_pos",
        "effector_hold_vel",
        "effector_pos_mid",
        "effector_vel_mid",
        "effector_pos_late",
        "effector_vel_late",
        "effector_final_vel",
        "goal_hit_in_window",
        "nn_hidden",
        "nn_hidden_derivative",
        "nn_output_jerk",
        "nn_output_pre_go",
        "nn_hidden_derivative_pre_go",
        "fix_readout_norm",
        "nn_output",
    )
    active = tuple(label for label in candidate_order if float(weights.get(label, 0.0) or 0.0) != 0.0)
    if not active:
        raise ValueError("Run spec has no active loss terms")
    return active


def available_checkpoint_batches(artifact_dir: Path) -> list[int]:
    """Return sorted numbered checkpoint batch counts for a run artifact directory."""

    checkpoint_root = artifact_dir / "checkpoints"
    batches = []
    for path in checkpoint_root.glob("checkpoint_[0-9]*"):
        try:
            batches.append(int(path.name.removeprefix("checkpoint_")))
        except ValueError:
            continue
    return sorted(batches)


def checkpoint_path_for_batches(artifact_dir: Path, completed_batches: int) -> Path:
    """Return the path for a numbered checkpoint."""

    return artifact_dir / "checkpoints" / f"checkpoint_{completed_batches:07d}"


def _skip_loss_tree(stream: Any, labels: Sequence[str]) -> None:
    for _label in labels:
        np.load(stream, allow_pickle=False)
        np.load(stream, allow_pickle=False)
    np.load(stream, allow_pickle=False)


def _repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "ReplicateCheckpointSelection",
    "active_loss_term_labels",
    "available_checkpoint_batches",
    "load_validation_selected_checkpoint_model",
    "materialize_validation_selected_checkpoint_manifest",
    "select_validation_checkpoints_for_run",
    "validation_objective_history",
]
