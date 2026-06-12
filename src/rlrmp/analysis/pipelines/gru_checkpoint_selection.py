"""Validation-selected checkpoint recovery for C&S GRU pilot artifacts."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.types import TreeNamespace, dict_to_namespace

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p


SPARSE_HISTORY_SCHEMA_VERSION = "rlrmp.validation_selected_gru_checkpoints.v1"
FIXED_BANK_SCHEMA_VERSION = "rlrmp.fixed_bank_gru_checkpoint_rescore.v1"
SPARSE_HISTORY_CHECKPOINT_POLICY = "validation_selected_per_replicate"
FIXED_BANK_CHECKPOINT_POLICY = "fixed_bank_rescored_per_replicate"
DEFAULT_FIXED_BANK_MANIFEST_NAME = "fixed_bank_rescored_checkpoints.json"
CheckpointSelectionMode = Literal["sparse_history", "fixed_bank_manifest"]


@dataclass(frozen=True)
class ReplicateCheckpointSelection:
    """Recoverable validation-selected checkpoint for one replicate."""

    replicate: int
    checkpoint_batches: int
    checkpoint_path: Path
    selection_source: str
    scoring_validation_log_batch: int
    scoring_validation_objective: float
    best_logged_validation_batch: int
    best_logged_validation_objective: float
    final_validation_objective: float
    final_vs_selected_validation_degradation: float

    def to_json(self, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
        return {
            "replicate": self.replicate,
            "checkpoint_batches": self.checkpoint_batches,
            "checkpoint_path": _repo_relative(self.checkpoint_path, repo_root=repo_root),
            "selection_source": self.selection_source,
            "scoring_validation_log_batch": self.scoring_validation_log_batch,
            "scoring_validation_objective": self.scoring_validation_objective,
            "best_logged_validation_batch": self.best_logged_validation_batch,
            "best_logged_validation_objective": self.best_logged_validation_objective,
            "final_validation_objective": self.final_validation_objective,
            "final_vs_selected_validation_degradation": (
                self.final_vs_selected_validation_degradation
            ),
        }


@dataclass(frozen=True)
class FixedValidationBankSpec:
    """Declared validation bank and scorer identity for post-hoc checkpoint rescoring."""

    bank_identity: str
    scorer_identity: str
    seed: int | None = None
    n_trials: int | None = None
    scorer_version: str | None = None
    validation_role: str | None = None
    selection_metric: str | None = None
    nominal_quality_role: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "bank_identity": self.bank_identity,
            "scorer_identity": self.scorer_identity,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.n_trials is not None:
            payload["n_trials"] = self.n_trials
        if self.scorer_version is not None:
            payload["scorer_version"] = self.scorer_version
        if self.validation_role is not None:
            payload["validation_role"] = self.validation_role
        if self.selection_metric is not None:
            payload["selection_metric"] = self.selection_metric
        if self.nominal_quality_role is not None:
            payload["nominal_quality_role"] = self.nominal_quality_role
        return payload


CheckpointScorer = Callable[
    [str, int, int, Path, Mapping[str, Any], FixedValidationBankSpec],
    float,
]


def materialize_validation_selected_checkpoint_manifest(
    *,
    experiment: str,
    run_ids: Sequence[str],
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    preferred_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
) -> dict[str, Any]:
    """Write a JSON manifest of recoverable validation-selected checkpoints.

    The default mode selects from sparse training-history validation records.
    Callers that deliberately want a supplied fixed-bank rescore manifest must
    pass ``checkpoint_selection_mode="fixed_bank_manifest"``.
    """

    preferred_manifest = _resolve_checkpoint_selection_manifest(
        checkpoint_selection_mode=checkpoint_selection_mode,
        experiment=experiment,
        run_ids=run_ids,
        repo_root=repo_root,
        preferred_manifest_path=preferred_manifest_path,
    )
    selections = {
        run_id: [
            selection.to_json(repo_root=repo_root)
            for selection in select_validation_checkpoints_for_run(
                experiment=experiment,
                run_id=run_id,
                repo_root=repo_root,
                preferred_manifest=preferred_manifest,
                checkpoint_selection_mode=checkpoint_selection_mode,
            )
        ]
        for run_id in run_ids
    }
    manifest = (
        fixed_bank_manifest_for_runs(preferred_manifest, run_ids=run_ids, repo_root=repo_root)
        if preferred_manifest is not None
        else None
    )
    if manifest is None:
        manifest = {
            "schema_version": SPARSE_HISTORY_SCHEMA_VERSION,
            "issue": experiment,
            "checkpoint_policy": SPARSE_HISTORY_CHECKPOINT_POLICY,
            "selection_source": "sparse_history_fallback",
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
    else:
        manifest = manifest | {"runs": selections}
    output_path = output_path or (
        repo_root / "results" / experiment / "notes" / "validation_selected_checkpoints.json"
    )
    mkdir_p(output_path.parent)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def plan_fixed_bank_checkpoint_rescore(
    *,
    experiment: str,
    run_ids: Sequence[str],
    validation_bank: FixedValidationBankSpec,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Return a fixed-bank rescore manifest plan without loading model checkpoints."""

    checkpoint_lists = {
        run_id: [
            {
                "checkpoint_batches": batches,
                "checkpoint_path": _repo_relative(
                    checkpoint_path_for_batches(
                        repo_root / "_artifacts" / experiment / "runs" / run_id,
                        batches,
                    ),
                    repo_root=repo_root,
                ),
            }
            for batches in available_checkpoint_batches(
                repo_root / "_artifacts" / experiment / "runs" / run_id
            )
        ]
        for run_id in run_ids
    }
    output_path = output_path or fixed_bank_manifest_path(experiment, repo_root=repo_root)
    return {
        "schema_version": FIXED_BANK_SCHEMA_VERSION,
        "issue": experiment,
        "checkpoint_policy": FIXED_BANK_CHECKPOINT_POLICY,
        "selection_source": "fixed_bank_rescore",
        "materialization_status": "planned",
        "validation_bank": validation_bank.to_json(),
        "validation_role": (
            validation_bank.validation_role or "fixed_bank_rollout_validation"
        ),
        "selection_metric": (
            validation_bank.selection_metric or "aggregate_rollout_validation_objective"
        ),
        "nominal_quality_role": (
            validation_bank.nominal_quality_role or "reported_sidecar"
        ),
        "checkpoint_lists": checkpoint_lists,
        "output_path": _repo_relative(output_path, repo_root=repo_root),
        "selection_policy": (
            "per-replicate checkpoint selected by minimum rollout validation objective "
            "on the declared fixed validation bank; analytical action, I/O, perturbation, "
            "and objective-comparator metrics are audit-only"
        ),
    }


def materialize_fixed_bank_checkpoint_rescore_manifest(
    *,
    experiment: str,
    run_ids: Sequence[str],
    validation_bank: FixedValidationBankSpec,
    scorer: CheckpointScorer | None = None,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize or plan a fixed-bank checkpoint rescore manifest.

    The scorer receives ``(run_id, replicate, checkpoint_batches, checkpoint_path,
    run_spec, validation_bank)`` and must return the rollout validation objective
    for that checkpoint on the fixed bank. When no scorer is supplied, this
    writes an explicit ``not_materialized`` manifest that downstream postrun
    materialization can identify and fall back from.
    """

    plan = plan_fixed_bank_checkpoint_rescore(
        experiment=experiment,
        run_ids=run_ids,
        validation_bank=validation_bank,
        repo_root=repo_root,
        output_path=output_path,
    )
    output_path = output_path or fixed_bank_manifest_path(experiment, repo_root=repo_root)
    if scorer is None:
        manifest = plan | {
            "materialization_status": "not_materialized",
            "not_materialized_reason": "no_fixed_bank_checkpoint_scorer_supplied",
            "fallback_checkpoint_policy": SPARSE_HISTORY_CHECKPOINT_POLICY,
            "runs": {},
        }
    else:
        manifest = plan | {
            "materialization_status": "materialized",
            "runs": {
                run_id: [
                    selection.to_json(repo_root=repo_root)
                    for selection in score_fixed_bank_checkpoints_for_run(
                        experiment=experiment,
                        run_id=run_id,
                        validation_bank=validation_bank,
                        scorer=scorer,
                        repo_root=repo_root,
                    )
                ]
                for run_id in run_ids
            },
        }
    mkdir_p(output_path.parent)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def score_fixed_bank_checkpoints_for_run(
    *,
    experiment: str,
    run_id: str,
    validation_bank: FixedValidationBankSpec,
    scorer: CheckpointScorer,
    repo_root: Path = REPO_ROOT,
) -> list[ReplicateCheckpointSelection]:
    """Score all durable checkpoints for a run and select the best per replicate."""

    run_spec_path = repo_root / "results" / experiment / "runs" / run_id / "run.json"
    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    checkpoint_batches = available_checkpoint_batches(artifact_dir)
    if not checkpoint_batches:
        raise FileNotFoundError(
            f"No numbered checkpoints found under {artifact_dir / 'checkpoints'}"
        )
    n_replicates = _n_replicates_from_run_spec(run_spec)
    selections: list[ReplicateCheckpointSelection] = []
    for replicate in range(n_replicates):
        checkpoint_scores = [
            (
                float(
                    scorer(
                        run_id,
                        replicate,
                        checkpoint_batch,
                        checkpoint_path_for_batches(artifact_dir, checkpoint_batch),
                        run_spec,
                        validation_bank,
                    )
                ),
                checkpoint_batch,
            )
            for checkpoint_batch in checkpoint_batches
        ]
        selected_score, selected_batch = min(checkpoint_scores, key=lambda item: item[0])
        final_batch = checkpoint_batches[-1]
        final_score = next(score for score, batch in checkpoint_scores if batch == final_batch)
        selections.append(
            ReplicateCheckpointSelection(
                replicate=replicate,
                checkpoint_batches=selected_batch,
                checkpoint_path=checkpoint_path_for_batches(artifact_dir, selected_batch),
                selection_source="fixed_bank_rescore",
                scoring_validation_log_batch=selected_batch,
                scoring_validation_objective=selected_score,
                best_logged_validation_batch=selected_batch,
                best_logged_validation_objective=selected_score,
                final_validation_objective=float(final_score),
                final_vs_selected_validation_degradation=float(final_score - selected_score),
            )
        )
    return selections


def select_validation_checkpoints_for_run(
    *,
    experiment: str,
    run_id: str,
    repo_root: Path = REPO_ROOT,
    preferred_manifest: Mapping[str, Any] | None = None,
    preferred_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
) -> list[ReplicateCheckpointSelection]:
    """Select the best available checkpoint for each replicate in a run.

    ``sparse_history`` uses positive validation-history records only.
    ``fixed_bank_manifest`` uses a supplied materialized fixed-bank manifest and
    never falls back silently.
    """

    effective_selection_mode = checkpoint_selection_mode
    effective_manifest = preferred_manifest
    if effective_manifest is None:
        effective_manifest = load_materialized_fixed_bank_manifest(
            experiment=experiment,
            repo_root=repo_root,
            manifest_path=preferred_manifest_path,
        )
    if effective_selection_mode == "sparse_history" and effective_manifest is not None:
        effective_selection_mode = "fixed_bank_manifest"

    if effective_selection_mode == "sparse_history":
        return select_sparse_history_validation_checkpoints_for_run(
            experiment=experiment,
            run_id=run_id,
            repo_root=repo_root,
        )
    if effective_selection_mode != "fixed_bank_manifest":
        raise ValueError(f"unsupported checkpoint selection mode {checkpoint_selection_mode!r}")

    manifest = _resolve_checkpoint_selection_manifest(
        checkpoint_selection_mode=effective_selection_mode,
        experiment=experiment,
        run_ids=(run_id,),
        repo_root=repo_root,
        preferred_manifest=effective_manifest,
        preferred_manifest_path=preferred_manifest_path,
    )
    if manifest is not None and run_id in manifest.get("runs", {}):
        return selections_from_manifest_run(
            manifest["runs"][run_id],
            experiment=experiment,
            run_id=run_id,
            repo_root=repo_root,
        )

    raise ValueError(
        f"Fixed-bank checkpoint manifest does not contain run {run_id!r} for "
        f"experiment {experiment!r}"
    )


def select_sparse_history_validation_checkpoints_for_run(
    *,
    experiment: str,
    run_id: str,
    repo_root: Path = REPO_ROOT,
) -> list[ReplicateCheckpointSelection]:
    """Select checkpoints from sparse logged validation records."""

    run_spec_path = repo_root / "results" / experiment / "runs" / run_id / "run.json"
    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    objective, valid_records = validation_objective_history(
        run_spec=run_spec,
        history_path=artifact_dir / "training_history.eqx",
    )
    checkpoint_batches = available_checkpoint_batches(artifact_dir)
    if not checkpoint_batches:
        raise FileNotFoundError(
            f"No numbered checkpoints found under {artifact_dir / 'checkpoints'}"
        )

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
                selection_source="sparse_history_fallback",
                scoring_validation_log_batch=scoring_batch,
                scoring_validation_objective=score,
                best_logged_validation_batch=best_logged_batch,
                best_logged_validation_objective=best_logged_value,
                final_validation_objective=float(objective[-1, replicate]),
                final_vs_selected_validation_degradation=float(objective[-1, replicate] - score),
            )
        )
    return selections


def load_validation_selected_checkpoint_model(
    *,
    experiment: str,
    run_id: str,
    run_spec: Mapping[str, Any],
    preferred_manifest: Mapping[str, Any] | None = None,
    preferred_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
    repo_root: Path = REPO_ROOT,
) -> tuple[Any, list[ReplicateCheckpointSelection]]:
    """Load a model ensemble assembled from per-replicate selected checkpoints."""

    effective_selection_mode = checkpoint_selection_mode
    if effective_selection_mode == "sparse_history" and (
        preferred_manifest is not None or preferred_manifest_path is not None
    ):
        effective_selection_mode = "fixed_bank_manifest"
    selections = select_validation_checkpoints_for_run(
        experiment=experiment,
        run_id=run_id,
        preferred_manifest=preferred_manifest,
        preferred_manifest_path=preferred_manifest_path,
        checkpoint_selection_mode=effective_selection_mode,
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
        branch_weight = _scalar_weight(np.load(stream, allow_pickle=False))
    objective = np.sum(np.stack(components), axis=0) * branch_weight
    valid_records = np.any(np.stack(real_record_terms), axis=0)
    return objective, valid_records


def active_loss_term_labels(run_spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Return active loss labels in Feedbax's serialized term order."""

    loss_objective = str(run_spec.get("loss_objective") or "")
    if loss_objective == "full_analytical_qrf":
        return ("full_analytical_qrf",)

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
        "mechanics_force_filter",
        "nn_output",
    )
    active = tuple(
        label for label in candidate_order if float(weights.get(label, 0.0) or 0.0) != 0.0
    )
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


def fixed_bank_manifest_path(experiment: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """Return the default fixed-bank checkpoint rescore manifest path."""

    return repo_root / "results" / experiment / "notes" / DEFAULT_FIXED_BANK_MANIFEST_NAME


def load_materialized_fixed_bank_manifest(
    *,
    experiment: str,
    repo_root: Path = REPO_ROOT,
    manifest_path: Path | None = None,
) -> dict[str, Any] | None:
    """Load a fixed-bank rescore manifest only when it has materialized scores."""

    candidate_paths = [manifest_path] if manifest_path is not None else [
        fixed_bank_manifest_path(experiment, repo_root=repo_root),
        repo_root / "results" / experiment / "notes" / "validation_selected_checkpoints.json",
    ]
    existing_path = next(
        (path for path in candidate_paths if path is not None and path.exists()),
        None,
    )
    if existing_path is None:
        return None
    manifest = json.loads(existing_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FIXED_BANK_SCHEMA_VERSION:
        if manifest_path is None:
            return None
        raise ValueError(f"Expected {FIXED_BANK_SCHEMA_VERSION} in {existing_path}")
    if manifest.get("materialization_status") != "materialized":
        return None
    return manifest


def fixed_bank_manifest_for_runs(
    manifest: Mapping[str, Any] | None,
    *,
    run_ids: Sequence[str],
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any] | None:
    """Return a fixed-bank manifest subset for the requested runs."""

    if manifest is None:
        return None
    runs = manifest.get("runs", {})
    if not isinstance(runs, Mapping) or any(run_id not in runs for run_id in run_ids):
        return None
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"runs", "output_path"}
    } | {
        "runs": {
            run_id: [
                selection.to_json(repo_root=repo_root)
                for selection in selections_from_manifest_run(
                    runs[run_id],
                    experiment=str(manifest["issue"]),
                    run_id=run_id,
                    repo_root=repo_root,
                )
            ]
            for run_id in run_ids
        }
    }


def _resolve_checkpoint_selection_manifest(
    *,
    checkpoint_selection_mode: CheckpointSelectionMode,
    experiment: str,
    run_ids: Sequence[str],
    repo_root: Path = REPO_ROOT,
    preferred_manifest: Mapping[str, Any] | None = None,
    preferred_manifest_path: Path | None = None,
) -> Mapping[str, Any] | None:
    """Return the explicit fixed-bank manifest for ``run_ids`` when requested."""

    if checkpoint_selection_mode == "sparse_history":
        return None
    if checkpoint_selection_mode != "fixed_bank_manifest":
        raise ValueError(f"unsupported checkpoint selection mode {checkpoint_selection_mode!r}")
    manifest = preferred_manifest or load_materialized_fixed_bank_manifest(
        experiment=experiment,
        repo_root=repo_root,
        manifest_path=preferred_manifest_path,
    )
    if manifest is None:
        raise ValueError(
            "checkpoint_selection_mode='fixed_bank_manifest' requires a materialized "
            "fixed-bank checkpoint manifest"
        )
    runs = manifest.get("runs", {})
    if not isinstance(runs, Mapping):
        raise ValueError("fixed-bank checkpoint manifest has no run mapping")
    missing = [run_id for run_id in run_ids if run_id not in runs]
    if missing:
        raise ValueError(
            "fixed-bank checkpoint manifest is missing requested run(s): "
            + ", ".join(missing)
        )
    return manifest


def selections_from_manifest_run(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
    run_id: str,
    repo_root: Path = REPO_ROOT,
) -> list[ReplicateCheckpointSelection]:
    """Convert serialized selected-checkpoint rows into selection objects."""

    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    selections: list[ReplicateCheckpointSelection] = []
    for row in rows:
        checkpoint_batches = int(row["checkpoint_batches"])
        raw_path = Path(str(row.get("checkpoint_path") or ""))
        checkpoint_path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        if not str(row.get("checkpoint_path") or ""):
            checkpoint_path = checkpoint_path_for_batches(artifact_dir, checkpoint_batches)
        selected_score = float(row["scoring_validation_objective"])
        final_score = float(row.get("final_validation_objective", selected_score))
        selections.append(
            ReplicateCheckpointSelection(
                replicate=int(row["replicate"]),
                checkpoint_batches=checkpoint_batches,
                checkpoint_path=checkpoint_path,
                selection_source=str(row.get("selection_source") or "fixed_bank_rescore"),
                scoring_validation_log_batch=int(
                    row.get("scoring_validation_log_batch", checkpoint_batches)
                ),
                scoring_validation_objective=selected_score,
                best_logged_validation_batch=int(
                    row.get("best_logged_validation_batch", checkpoint_batches)
                ),
                best_logged_validation_objective=float(
                    row.get("best_logged_validation_objective", selected_score)
                ),
                final_validation_objective=final_score,
                final_vs_selected_validation_degradation=float(
                    row.get(
                        "final_vs_selected_validation_degradation",
                        final_score - selected_score,
                    )
                ),
            )
        )
    return selections


def _n_replicates_from_run_spec(run_spec: Mapping[str, Any]) -> int:
    hps = run_spec.get("hps", {})
    model = hps.get("model", {}) if isinstance(hps, Mapping) else {}
    if "n_replicates" not in model:
        raise ValueError("Run spec hps.model.n_replicates is required for fixed-bank rescoring")
    return int(model["n_replicates"])


def _skip_loss_tree(stream: Any, labels: Sequence[str]) -> None:
    for _label in labels:
        np.load(stream, allow_pickle=False)
        np.load(stream, allow_pickle=False)
    np.load(stream, allow_pickle=False)


def _scalar_weight(value: np.ndarray) -> float:
    """Return a scalar weight from Feedbax history scalar or broadcast array records."""

    array = np.asarray(value)
    if array.size == 1:
        return float(array.reshape(()))
    nonzero = array[array != 0]
    if nonzero.size == 0:
        return 0.0
    first = float(nonzero.reshape(-1)[0])
    if not np.allclose(nonzero, first):
        raise ValueError(f"Expected scalar or broadcast history weight, got shape {array.shape}")
    return first


def _repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "CheckpointSelectionMode",
    "ReplicateCheckpointSelection",
    "FixedValidationBankSpec",
    "active_loss_term_labels",
    "available_checkpoint_batches",
    "fixed_bank_manifest_path",
    "load_validation_selected_checkpoint_model",
    "materialize_fixed_bank_checkpoint_rescore_manifest",
    "materialize_validation_selected_checkpoint_manifest",
    "plan_fixed_bank_checkpoint_rescore",
    "score_fixed_bank_checkpoints_for_run",
    "select_validation_checkpoints_for_run",
    "select_sparse_history_validation_checkpoints_for_run",
    "validation_objective_history",
]
