"""Validation-selected checkpoint recovery for C&S GRU pilot artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace

from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.cs_nominal_gru import (
    CS_DELAYED_REACH_TASK_PRESET,
    CS_DELAYED_REACH_TASK_TYPE,
    CS_STAGE_COUNT,
    DEFAULT_DELAYED_GO_CUE_MAX_STEP,
    DEFAULT_DELAYED_GO_CUE_MIN_STEP,
    DELAYED_REACH_TRAINING_MODE,
)
from rlrmp.train.cs_perturbation_training import (
    TargetRelativeMultiTargetTrainingConfig,
    target_relative_input_contract,
    target_relative_target_support_config,
    target_relative_validation_bins,
)
from rlrmp.paths import REPO_ROOT, mkdir_p, resolve_run_artifact_path
from rlrmp.runtime.run_specs import resolve_run_record


SPARSE_HISTORY_SCHEMA_VERSION = "rlrmp.validation_selected_gru_checkpoints.v1"
FIXED_BANK_SCHEMA_VERSION = "rlrmp.fixed_bank_gru_checkpoint_rescore.v1"
DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION = "rlrmp.delayed_reach_eval_bank.v2"
SPARSE_HISTORY_CHECKPOINT_POLICY = "validation_selected_per_replicate"
FIXED_BANK_CHECKPOINT_POLICY = "fixed_bank_rescored_per_replicate"
DEFAULT_FIXED_BANK_MANIFEST_NAME = "fixed_bank_rescored_checkpoints.json"
DEFAULT_DELAYED_REACH_FIXED_BANK_MANIFEST_NAME = (
    "delayed_reach_fixed_bank_rescored_checkpoints.json"
)
DEFAULT_DELAYED_REACH_GO_CUE_STEPS = tuple(
    range(DEFAULT_DELAYED_GO_CUE_MIN_STEP, DEFAULT_DELAYED_GO_CUE_MAX_STEP + 1)
)
DEFAULT_DELAYED_REACH_DIRECTION_COUNT = 20
DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M = 0.15
CheckpointSelectionMode = Literal["sparse_history", "fixed_bank_manifest"]
DelayedReachCatchBank = Literal["no_catch", "catch"]
DelayedReachDirectionSource = Literal["uniform_grid", "validation_targets"]
_VALIDATION_SELECTED_MODEL_CACHE: dict[
    tuple[Any, ...],
    tuple[Any, list["ReplicateCheckpointSelection"]],
] = {}


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
    bank_spec: Mapping[str, Any] | None = None

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
        if self.bank_spec is not None:
            payload["bank_spec"] = _json_ready(self.bank_spec)
        return payload


@dataclass(frozen=True)
class DelayedReachEvalBankSpec:
    """Reusable fixed validation-bank contract for delayed target-relative GRU runs."""

    bank_role: DelayedReachCatchBank
    target_config: TargetRelativeMultiTargetTrainingConfig
    p_catch_trial: float
    direction_source: DelayedReachDirectionSource = "uniform_grid"
    direction_count: int = DEFAULT_DELAYED_REACH_DIRECTION_COUNT
    reach_length_m: float = DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M
    reach_length_explicit: bool = False
    go_cue_min_step: int = DEFAULT_DELAYED_GO_CUE_MIN_STEP
    go_cue_max_step: int = DEFAULT_DELAYED_GO_CUE_MAX_STEP
    seed: int | None = None

    @property
    def bank_identity(self) -> str:
        catch_label = "catch" if self.bank_role == "catch" else "no-catch"
        target_label = (
            "target-relative-force-filter"
            if self.target_config.force_filter_feedback
            else ("target-relative")
        )
        return (
            "rlrmp.delayed-reach.fixed-eval-bank:"
            f"{catch_label}:go{self.go_cue_min_step}-{self.go_cue_max_step}:"
            f"{self.direction_source}:directions{self.direction_count}:"
            f"reach{self.reach_length_m:g}:{target_label}"
        )

    @property
    def go_cue_steps(self) -> tuple[int, ...]:
        return tuple(range(int(self.go_cue_min_step), int(self.go_cue_max_step) + 1))

    @property
    def n_target_conditions(self) -> int:
        return len(self._targets())

    @property
    def trial_count(self) -> int:
        return len(self.go_cue_steps) * self.n_target_conditions

    def to_json(self) -> dict[str, Any]:
        target_config = self.target_config
        target_distribution = target_config.to_json()["target_distribution"]
        target_geometry = _target_geometry_rows(
            target_config,
            direction_source=self.direction_source,
            direction_count=self.direction_count,
            reach_length_m=self.reach_length_m,
        )
        return {
            "schema_version": DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION,
            "bank_identity": self.bank_identity,
            "bank_family": "delayed_reach_fixed_eval_bank",
            "kind": self.bank_role,
            "catch": self.bank_role == "catch",
            "go_cue_min": int(self.go_cue_min_step),
            "go_cue_max": int(self.go_cue_max_step),
            "go_cue_steps": [int(step) for step in self.go_cue_steps],
            "direction_source": self.direction_source,
            "direction_count": self.n_target_conditions,
            "requested_direction_count": int(self.direction_count),
            "trial_count": self.trial_count,
            "movement_horizon_steps": CS_STAGE_COUNT,
            "reach_length_m": float(self.reach_length_m),
            "reach_length_source": (
                "explicit" if self.reach_length_explicit else f"{self.direction_source}_default"
            ),
            "reach_length_m_explicit": bool(self.reach_length_explicit),
            "direction_source_inferred_from_validation_targets": (
                self.direction_source == "validation_targets"
            ),
            "duplicate_direction_count": _duplicate_direction_count(target_geometry),
            "target_radii_m": [float(row["target_radius_m"]) for row in target_geometry],
            "target_angles_rad": [float(row["target_angle_rad"]) for row in target_geometry],
            "source_trial_indices": [int(row["source_trial_index"]) for row in target_geometry],
            "task": {
                "mode": DELAYED_REACH_TRAINING_MODE,
                "task_type": CS_DELAYED_REACH_TASK_TYPE,
                "task_preset": CS_DELAYED_REACH_TASK_PRESET,
                "reach_length_m": float(self.reach_length_m),
                "target_visibility": "visible_from_trial_start",
                "movement_epoch": {
                    "kind": "delayed_reach_movement_epoch",
                    "start_transition": "sampled_go_cue_step",
                    "go_cue_min_step": int(self.go_cue_min_step),
                    "go_cue_max_step": int(self.go_cue_max_step),
                    "cs_horizon_steps": CS_STAGE_COUNT,
                    "cost_indexing": "movement_age_not_trial_age",
                },
                "go_cue_sampling": {
                    "min_step_inclusive": int(self.go_cue_min_step),
                    "max_step_inclusive": int(self.go_cue_max_step),
                    "distribution": "uniform_integer",
                },
                "catch_trials": {
                    "p_catch_trial": float(self.p_catch_trial),
                    "go_cue_value_for_catch": 0.0,
                    "semantics": (
                        "target remains visible, movement target is replaced by the "
                        "initial position, and the hold/go-cue input stays in prep state"
                    ),
                },
            },
            "direction_source_contract": {
                "kind": "target_relative_multitarget_static",
                "bank_direction_source": self.direction_source,
                "validation_targets_source": (
                    "TargetRelativeMultiTargetTrainingConfig.validation_targets_m"
                ),
                "controller_feedback_basis": target_relative_input_contract(
                    force_filter_feedback=target_config.force_filter_feedback
                ),
                "direction_semantics": (
                    "directions and radii are derived from actual static target "
                    "coordinates in metres, not from row labels"
                ),
            },
            "target_distribution": target_distribution,
            "validation_bins": target_relative_validation_bins(target_config),
            "validation_target_provenance": {
                "schema": "rlrmp.target_relative_multitarget_validation_targets.v1",
                "source": "target_config.original + seen_targets_m + held_out_targets_m",
                "dedupe_policy": (
                    "uniform_grid"
                    if self.direction_source == "uniform_grid"
                    else ("preserve_first_occurrence_by_cartesian_target")
                ),
                "duplicate_direction_metadata": _duplicate_target_metadata(
                    target_config,
                    direction_source=self.direction_source,
                    direction_count=self.direction_count,
                    reach_length_m=self.reach_length_m,
                ),
                "actual_targets": target_geometry,
                "n_target_conditions": self.n_target_conditions,
            },
            "selection_role": (
                "rollout loss over this delayed-reach fixed bank selects checkpoints; "
                "analytical action, I/O, perturbation, and objective-comparator metrics "
                "remain audit-only"
            ),
            "nominal_quality_role": (
                "original anchor, seen targets, held-out targets, and catch/no-catch "
                "bank separation remain reported sidecars"
            ),
            "seed": self.seed,
        }

    def _targets(self) -> tuple[tuple[float, float], ...]:
        if self.direction_source == "uniform_grid":
            return _uniform_targets(
                direction_count=self.direction_count,
                reach_length_m=self.reach_length_m,
            )
        if self.direction_source == "validation_targets":
            targets = tuple(self.target_config.validation_targets_m[: self.direction_count])
            if len(targets) < int(self.direction_count):
                raise ValueError(
                    "validation_targets direction source requires at least "
                    f"direction_count={self.direction_count} targets; got {len(targets)}"
                )
            return targets
        raise ValueError(f"Unsupported direction_source {self.direction_source!r}")

    def to_fixed_validation_bank_spec(
        self,
        *,
        scorer_identity: str = "rlrmp.delayed_reach.fixed_bank_rollout_objective",
        scorer_version: str | None = None,
    ) -> FixedValidationBankSpec:
        """Return a checkpoint-rescore bank wrapper for this delayed bank."""

        return FixedValidationBankSpec(
            bank_identity=self.bank_identity,
            scorer_identity=scorer_identity,
            seed=self.seed,
            n_trials=self.trial_count,
            scorer_version=scorer_version,
            validation_role="delayed_reach_fixed_bank_rollout_validation",
            selection_metric="aggregate_rollout_validation_objective",
            nominal_quality_role=("reported_sidecar_for_target_geometry_and_catch_bank_quality"),
            bank_spec=self.to_json(),
        )


CheckpointScorer = Callable[
    [str, int, int, Path, Mapping[str, Any], FixedValidationBankSpec],
    float,
]


def delayed_reach_eval_bank_spec(
    *,
    bank_role: DelayedReachCatchBank,
    p_catch_trial: float | None = None,
    target_config: TargetRelativeMultiTargetTrainingConfig | None = None,
    force_filter_feedback: bool = True,
    direction_source: DelayedReachDirectionSource = "uniform_grid",
    direction_count: int = DEFAULT_DELAYED_REACH_DIRECTION_COUNT,
    go_cue_min_step: int = DEFAULT_DELAYED_GO_CUE_MIN_STEP,
    go_cue_max_step: int = DEFAULT_DELAYED_GO_CUE_MAX_STEP,
    reach_length_m: float = DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M,
    reach_length_explicit: bool = False,
    seed: int | None = None,
) -> DelayedReachEvalBankSpec:
    """Return a reusable delayed-reach fixed validation-bank spec."""

    if bank_role not in {"no_catch", "catch"}:
        raise ValueError(f"Unsupported delayed-reach bank role {bank_role!r}")
    if p_catch_trial is None:
        p_catch_trial = 1.0 if bank_role == "catch" else 0.0
    config = target_config or target_relative_target_support_config(
        enabled=True,
        force_filter_feedback=force_filter_feedback,
    )
    return DelayedReachEvalBankSpec(
        bank_role=bank_role,
        target_config=config,
        p_catch_trial=float(p_catch_trial),
        direction_source=direction_source,
        direction_count=int(direction_count),
        reach_length_m=float(reach_length_m),
        reach_length_explicit=bool(reach_length_explicit),
        go_cue_min_step=int(go_cue_min_step),
        go_cue_max_step=int(go_cue_max_step),
        seed=seed,
    )


def delayed_reach_fixed_eval_bank_specs(
    *,
    target_config: TargetRelativeMultiTargetTrainingConfig | None = None,
    force_filter_feedback: bool = True,
    direction_source: DelayedReachDirectionSource = "uniform_grid",
    direction_count: int = DEFAULT_DELAYED_REACH_DIRECTION_COUNT,
    go_cue_min_step: int = DEFAULT_DELAYED_GO_CUE_MIN_STEP,
    go_cue_max_step: int = DEFAULT_DELAYED_GO_CUE_MAX_STEP,
    reach_length_m: float = DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M,
    reach_length_explicit: bool = False,
    seed: int | None = None,
) -> tuple[DelayedReachEvalBankSpec, DelayedReachEvalBankSpec]:
    """Return explicit no-catch and catch delayed-reach evaluation banks."""

    return (
        delayed_reach_eval_bank_spec(
            bank_role="no_catch",
            p_catch_trial=0.0,
            target_config=target_config,
            force_filter_feedback=force_filter_feedback,
            direction_source=direction_source,
            direction_count=direction_count,
            go_cue_min_step=go_cue_min_step,
            go_cue_max_step=go_cue_max_step,
            reach_length_m=reach_length_m,
            reach_length_explicit=reach_length_explicit,
            seed=seed,
        ),
        delayed_reach_eval_bank_spec(
            bank_role="catch",
            p_catch_trial=1.0,
            target_config=target_config,
            force_filter_feedback=force_filter_feedback,
            direction_source=direction_source,
            direction_count=direction_count,
            go_cue_min_step=go_cue_min_step,
            go_cue_max_step=go_cue_max_step,
            reach_length_m=reach_length_m,
            reach_length_explicit=reach_length_explicit,
            seed=seed,
        ),
    )


def delayed_reach_fixed_rescore_bank_spec(
    *,
    target_config: TargetRelativeMultiTargetTrainingConfig | None = None,
    force_filter_feedback: bool = True,
    direction_source: DelayedReachDirectionSource = "uniform_grid",
    direction_count: int = DEFAULT_DELAYED_REACH_DIRECTION_COUNT,
    go_cue_min_step: int = DEFAULT_DELAYED_GO_CUE_MIN_STEP,
    go_cue_max_step: int = DEFAULT_DELAYED_GO_CUE_MAX_STEP,
    reach_length_m: float = DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M,
    reach_length_explicit: bool = False,
    seed: int | None = None,
    scorer_identity: str = "feedbax_task_loss_mean_over_trials",
    scorer_version: str | None = None,
) -> FixedValidationBankSpec:
    """Return the combined no-catch/catch delayed fixed-bank rescore spec."""

    banks = delayed_reach_fixed_eval_bank_specs(
        target_config=target_config,
        force_filter_feedback=force_filter_feedback,
        direction_source=direction_source,
        direction_count=direction_count,
        go_cue_min_step=go_cue_min_step,
        go_cue_max_step=go_cue_max_step,
        reach_length_m=reach_length_m,
        reach_length_explicit=reach_length_explicit,
        seed=seed,
    )
    bank_spec = {
        "schema_version": "rlrmp.delayed_reach_fixed_bank_rescore_bank.v1",
        "bank_identity": "delayed_reach_go_cue_grid_no_catch_catch",
        "bank_kinds": [bank.bank_role for bank in banks],
        "go_cue_steps": [int(step) for step in banks[0].go_cue_steps],
        "go_cue_min": int(banks[0].go_cue_min_step),
        "go_cue_max": int(banks[0].go_cue_max_step),
        "direction_source": direction_source,
        "direction_count": int(banks[0].n_target_conditions),
        "requested_direction_count": int(direction_count),
        "reach_length_m": float(reach_length_m),
        "selection_source": "delayed_reach_fixed_bank_rescore",
        "selection_metric": "mean_task_loss_equal_weight_over_declared_banks",
        "bank_weighting": "equal_weight_over_declared_banks",
        "banks": [bank.to_json() for bank in banks],
    }
    return FixedValidationBankSpec(
        bank_identity="delayed_reach_go_cue_grid_no_catch_catch",
        scorer_identity=scorer_identity,
        seed=seed,
        n_trials=sum(bank.trial_count for bank in banks),
        scorer_version=scorer_version,
        validation_role="fixed_delayed_reach_no_catch_catch_rollout_validation",
        selection_metric="mean_task_loss_equal_weight_over_declared_banks",
        nominal_quality_role="reported_sidecar_for_no_catch_and_catch_bank_quality",
        bank_spec=bank_spec,
    )


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
        "selection_source": _fixed_bank_selection_source(validation_bank),
        "materialization_status": "planned",
        "validation_bank": validation_bank.to_json(),
        "validation_role": (validation_bank.validation_role or "fixed_bank_rollout_validation"),
        "selection_metric": (
            validation_bank.selection_metric or "aggregate_rollout_validation_objective"
        ),
        "nominal_quality_role": (validation_bank.nominal_quality_role or "reported_sidecar"),
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

    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    run_spec = resolve_run_record(experiment, run_id, repo_root=repo_root)
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
                selection_source=_fixed_bank_selection_source(validation_bank),
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
        return select_sparse_history_validation_checkpoints_for_run(
            experiment=experiment,
            run_id=run_id,
            repo_root=repo_root,
        )

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

    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    run_spec = resolve_run_record(experiment, run_id, repo_root=repo_root)
    objective, valid_records = validation_objective_history(
        run_spec=run_spec,
        history_path=resolve_run_artifact_path(artifact_dir, "training_history.eqx"),
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
    cache_key = _validation_selected_model_cache_key(
        experiment=experiment,
        run_id=run_id,
        run_spec=run_spec,
        preferred_manifest=preferred_manifest,
        preferred_manifest_path=preferred_manifest_path,
        checkpoint_selection_mode=effective_selection_mode,
        repo_root=repo_root,
    )
    if cache_key is not None and cache_key in _VALIDATION_SELECTED_MODEL_CACHE:
        return _VALIDATION_SELECTED_MODEL_CACHE[cache_key]
    selections = select_validation_checkpoints_for_run(
        experiment=experiment,
        run_id=run_id,
        preferred_manifest=preferred_manifest,
        preferred_manifest_path=preferred_manifest_path,
        checkpoint_selection_mode=effective_selection_mode,
        repo_root=repo_root,
    )
    selected_model_key = _selected_model_cache_key(
        run_spec=run_spec,
        selections=selections,
        repo_root=repo_root,
    )
    if selected_model_key in _VALIDATION_SELECTED_MODEL_CACHE:
        return _VALIDATION_SELECTED_MODEL_CACHE[selected_model_key]
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

    result = jt.map(select_leaf, *models), selections
    if cache_key is not None:
        _VALIDATION_SELECTED_MODEL_CACHE[cache_key] = result
    _VALIDATION_SELECTED_MODEL_CACHE[selected_model_key] = result
    return result


def _validation_selected_model_cache_key(
    *,
    experiment: str,
    run_id: str,
    run_spec: Mapping[str, Any],
    preferred_manifest: Mapping[str, Any] | None,
    preferred_manifest_path: Path | None,
    checkpoint_selection_mode: CheckpointSelectionMode,
    repo_root: Path,
) -> tuple[Any, ...] | None:
    """Return a stable key for per-process selected-model reuse."""

    if preferred_manifest is not None:
        return None
    manifest_fingerprint: tuple[str, int, int] | None = None
    if preferred_manifest_path is not None:
        manifest_path = (
            preferred_manifest_path
            if preferred_manifest_path.is_absolute()
            else repo_root / preferred_manifest_path
        )
        try:
            stat = manifest_path.stat()
        except FileNotFoundError:
            return None
        manifest_fingerprint = (
            str(manifest_path.resolve()),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
    run_spec_hash = hashlib.sha256(
        json.dumps(run_spec, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return (
        str(Path(repo_root).resolve()),
        str(experiment),
        str(run_id),
        str(checkpoint_selection_mode),
        manifest_fingerprint,
        run_spec_hash,
    )


def _selected_model_cache_key(
    *,
    run_spec: Mapping[str, Any],
    selections: Sequence[ReplicateCheckpointSelection],
    repo_root: Path,
) -> tuple[Any, ...]:
    """Return a cache key based on the concrete selected checkpoint files."""

    run_spec_hash = hashlib.sha256(
        json.dumps(run_spec, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    model_files = []
    for selection in selections:
        model_path = selection.checkpoint_path / "model.eqx"
        stat = model_path.stat()
        model_files.append(
            (
                int(selection.replicate),
                str(model_path.resolve()),
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        )
    return (
        "selected_model_files",
        str(Path(repo_root).resolve()),
        run_spec_hash,
        tuple(model_files),
    )


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
        arrays = _read_history_arrays(stream)
    if len(arrays) < 7 or (len(arrays) - 1) % 2:
        raise ValueError(f"Unexpected GRU history array count {len(arrays)} in {history_path}")
    arrays_per_loss_tree = (len(arrays) - 1) // 2
    _labels = _history_term_labels(labels, arrays_per_loss_tree)
    validation_arrays = arrays[arrays_per_loss_tree : 2 * arrays_per_loss_tree]
    for idx, _label in enumerate(_labels):
        value = validation_arrays[2 * idx]
        weight = _scalar_weight(validation_arrays[2 * idx + 1])
        components.append(np.asarray(value, dtype=np.float64) * weight)
        real_record_terms.append(np.asarray(value) != 0)
    branch_weight = _scalar_weight(validation_arrays[-1])
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

    checkpoint_root = resolve_run_artifact_path(artifact_dir, "checkpoints")
    batches = []
    for path in checkpoint_root.glob("checkpoint_[0-9]*"):
        try:
            batches.append(int(path.name.removeprefix("checkpoint_")))
        except ValueError:
            continue
    return sorted(batches)


def checkpoint_path_for_batches(artifact_dir: Path, completed_batches: int) -> Path:
    """Return the path for a numbered checkpoint."""

    return resolve_run_artifact_path(
        artifact_dir,
        "checkpoints",
        f"checkpoint_{completed_batches:07d}",
    )


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

    candidate_paths = (
        [manifest_path]
        if manifest_path is not None
        else [
            fixed_bank_manifest_path(experiment, repo_root=repo_root),
            repo_root / "results" / experiment / "notes" / "validation_selected_checkpoints.json",
        ]
    )
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
    return {key: value for key, value in manifest.items() if key not in {"runs", "output_path"}} | {
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
            "fixed-bank checkpoint manifest is missing requested run(s): " + ", ".join(missing)
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fixed_bank_selection_source(validation_bank: FixedValidationBankSpec) -> str:
    bank_spec = validation_bank.bank_spec or {}
    source = bank_spec.get("selection_source") if isinstance(bank_spec, Mapping) else None
    return str(source or "fixed_bank_rescore")


def _uniform_targets(
    *,
    direction_count: int,
    reach_length_m: float,
) -> tuple[tuple[float, float], ...]:
    if int(direction_count) < 2:
        raise ValueError("direction_count must be at least 2 for uniform center-out evaluation")
    targets = []
    for index in range(int(direction_count)):
        angle = 2.0 * math.pi * index / int(direction_count)
        targets.append(
            (
                _clean_float(float(reach_length_m) * math.cos(angle)),
                _clean_float(float(reach_length_m) * math.sin(angle)),
            )
        )
    return tuple(targets)


def _target_geometry_rows(
    config: TargetRelativeMultiTargetTrainingConfig,
    *,
    direction_source: DelayedReachDirectionSource,
    direction_count: int,
    reach_length_m: float,
) -> list[dict[str, Any]]:
    if direction_source == "uniform_grid":
        targets = _uniform_targets(
            direction_count=direction_count,
            reach_length_m=reach_length_m,
        )
        roles = ["uniform_grid"] * len(targets)
        source_angles = [0.0] * len(targets)
        source_indices = list(range(len(targets)))
    elif direction_source == "validation_targets":
        sources = _validation_target_geometry_sources(config)
        targets = tuple(source["target"] for source in sources[: int(direction_count)])
        if len(targets) < int(direction_count):
            raise ValueError(
                "validation_targets direction source requires at least "
                f"direction_count={direction_count} targets; got {len(targets)}"
            )
        roles = [str(source["role"]) for source in sources[: len(targets)]]
        source_angles = [float(source["angle_rad"]) for source in sources[: len(targets)]]
        source_indices = list(range(len(targets)))
    else:
        raise ValueError(f"Unsupported direction_source {direction_source!r}")

    rows = []
    for index, (target, role, source_index) in enumerate(zip(targets, roles, source_indices)):
        x, y = (float(target[0]), float(target[1]))
        radius = float(math.hypot(x, y))
        angle = (
            source_angles[index]
            if direction_source == "validation_targets"
            else float(math.atan2(y, x) % (2.0 * math.pi))
        )
        rows.append(
            {
                "index": int(index),
                "target_role": str(role),
                "target_m": [_clean_float(x), _clean_float(y)],
                "target_radius_m": radius,
                "target_angle_rad": angle,
                "target_angle_deg": math.degrees(angle),
                "source_trial_index": int(source_index),
            }
        )
    return rows


def _validation_target_geometry_sources(
    config: TargetRelativeMultiTargetTrainingConfig,
) -> list[dict[str, Any]]:
    seen: set[tuple[float, float]] = set()
    rows: list[dict[str, Any]] = []

    def append(
        *,
        target: tuple[float, float],
        role: str,
        direction_deg: float,
    ) -> None:
        key = (round(float(target[0]), 12), round(float(target[1]), 12))
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "target": key,
                "role": role,
                "angle_rad": math.radians(float(direction_deg)) % (2.0 * math.pi),
            }
        )

    append(
        target=tuple(float(value) for value in config.original_target_anchor_m),
        role="original_anchor",
        direction_deg=0.0,
    )
    for amplitude in config.seen_amplitudes_m:
        for direction in config.seen_directions_deg:
            append(
                target=_target_from_polar(amplitude_m=amplitude, direction_deg=direction),
                role="seen_training_support",
                direction_deg=direction,
            )
    for amplitude in config.held_out_amplitudes_m:
        for direction in config.held_out_directions_deg:
            append(
                target=_target_from_polar(amplitude_m=amplitude, direction_deg=direction),
                role="held_out_validation_support",
                direction_deg=direction,
            )
    return rows


def _target_from_polar(*, amplitude_m: float, direction_deg: float) -> tuple[float, float]:
    angle = math.radians(float(direction_deg))
    return (
        round(float(amplitude_m) * math.cos(angle), 12),
        round(float(amplitude_m) * math.sin(angle), 12),
    )


def _duplicate_target_metadata(
    config: TargetRelativeMultiTargetTrainingConfig,
    *,
    direction_source: DelayedReachDirectionSource,
    direction_count: int,
    reach_length_m: float,
) -> dict[str, Any]:
    rows = _target_geometry_rows(
        config,
        direction_source=direction_source,
        direction_count=direction_count,
        reach_length_m=reach_length_m,
    )
    angle_counts: dict[float, int] = {}
    for row in rows:
        key = round(float(row["target_angle_rad"]), 12)
        angle_counts[key] = angle_counts.get(key, 0) + 1
    duplicate_angles = {str(angle): count for angle, count in angle_counts.items() if count > 1}
    return {
        "angle_rounding_decimals": 12,
        "duplicate_direction_count": sum(count - 1 for count in duplicate_angles.values()),
        "duplicate_angles_rad": duplicate_angles,
    }


def _duplicate_direction_count(rows: Sequence[Mapping[str, Any]]) -> int:
    angle_counts: dict[float, int] = {}
    for row in rows:
        key = round(float(row["target_angle_rad"]), 12)
        angle_counts[key] = angle_counts.get(key, 0) + 1
    return sum(count - 1 for count in angle_counts.values() if count > 1)


def _clean_float(value: float) -> float:
    return 0.0 if abs(value) < 1e-15 else float(value)


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


def _read_history_arrays(stream: Any) -> list[np.ndarray]:
    """Read all NumPy arrays from a simple Feedbax history stream."""

    arrays: list[np.ndarray] = []
    while True:
        try:
            arrays.append(np.load(stream, allow_pickle=False))
        except (EOFError, ValueError):
            return arrays


def _history_term_labels(term_labels: Sequence[str], arrays_per_loss_tree: int) -> tuple[str, ...]:
    """Return labels matching the serialized loss-tree leaf count."""

    n_leaves = (arrays_per_loss_tree - 1) // 2
    labels = tuple(term_labels)
    if len(labels) == n_leaves:
        return labels
    if len(labels) == 1:
        return tuple(f"{labels[0]}_component_{idx}" for idx in range(n_leaves))
    return tuple(labels[:n_leaves]) + tuple(
        f"loss_component_{idx}" for idx in range(len(labels), n_leaves)
    )


def _scalar_weight(value: np.ndarray) -> float:
    """Return a scalar summary from Feedbax history weight records."""

    array = np.asarray(value)
    if array.size == 1:
        return float(array.reshape(()))
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0
    nonzero = finite[finite != 0]
    if nonzero.size == 0:
        return 0.0
    first = float(nonzero.reshape(-1)[0])
    if np.allclose(nonzero, first):
        return first
    return float(np.mean(nonzero))


def _repo_relative(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "CheckpointSelectionMode",
    "ReplicateCheckpointSelection",
    "FixedValidationBankSpec",
    "DelayedReachEvalBankSpec",
    "DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION",
    "DEFAULT_DELAYED_REACH_DIRECTION_COUNT",
    "DEFAULT_DELAYED_REACH_GO_CUE_STEPS",
    "DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M",
    "active_loss_term_labels",
    "available_checkpoint_batches",
    "delayed_reach_eval_bank_spec",
    "delayed_reach_fixed_eval_bank_specs",
    "delayed_reach_fixed_rescore_bank_spec",
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
