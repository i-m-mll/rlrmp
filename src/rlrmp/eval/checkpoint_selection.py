"""Checkpoint-selection capabilities for registered rlrmp evaluations."""

from __future__ import annotations

import hashlib
import importlib
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
from feedbax.contracts.manifest import (
    SCHEMA_VERSION as FEEDBAX_MANIFEST_SCHEMA_VERSION,
    ParentRef,
    load_manifest,
    spec_payload,
)

from rlrmp.analysis.gru_standard_certificate import normalize_gru_hps
from rlrmp.train.task_model import setup_task_model_pair
from rlrmp.train.cs_nominal_gru import (
    CS_DELAYED_REACH_TASK_PRESET,
    CS_DELAYED_REACH_TASK_TYPE,
    CS_STAGE_COUNT,
    DEFAULT_DELAYED_GO_CUE_MAX_STEP,
    DEFAULT_DELAYED_GO_CUE_MIN_STEP,
    DELAYED_REACH_TRAINING_MODE,
)
from rlrmp.analysis.data_products import load_analysis_parameter_preset
from rlrmp.train.cs_perturbation_training import (
    TargetRelativeMultiTargetTrainingConfig,
    target_relative_input_contract,
    target_relative_target_support_config,
    target_relative_validation_bins,
)
from rlrmp.paths import REPO_ROOT, resolve_run_artifact_path, run_spec_path
from rlrmp.runtime.run_spec_access import require_run_seed
from rlrmp.runtime.run_specs import resolve_run_record


CHECKPOINT_SELECTION_BANK_SCHEMA_ID = "feedbax.manifest.checkpoint_selection.bank"
DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION = FEEDBAX_MANIFEST_SCHEMA_VERSION
feedbax_manifest = importlib.import_module("feedbax.contracts.manifest")
CheckpointCandidateRef = feedbax_manifest.CheckpointCandidateRef
CheckpointScoreSummary = feedbax_manifest.CheckpointScoreSummary
CheckpointScorerIdentity = feedbax_manifest.CheckpointScorerIdentity
CheckpointSelectionBank = feedbax_manifest.CheckpointSelectionBank
CheckpointSelectionGroup = feedbax_manifest.CheckpointSelectionGroup
CheckpointSelectionManifest = feedbax_manifest.CheckpointSelectionManifest
CheckpointSelectionSpec = feedbax_manifest.CheckpointSelectionSpec
checkpoint_selection_manifest_id = feedbax_manifest.checkpoint_selection_manifest_id
SPARSE_HISTORY_CHECKPOINT_POLICY = "validation_selected_per_replicate"
FIXED_BANK_CHECKPOINT_POLICY = "fixed_bank_rescored_per_replicate"
DEFAULT_DELAYED_REACH_GO_CUE_STEPS = tuple(
    range(DEFAULT_DELAYED_GO_CUE_MIN_STEP, DEFAULT_DELAYED_GO_CUE_MAX_STEP + 1)
)
_ANALYSIS_PRESET = load_analysis_parameter_preset("gru_checkpoint_selection").parameters
DEFAULT_DELAYED_REACH_DIRECTION_COUNT = int(_ANALYSIS_PRESET["delayed_reach_direction_count"])
DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M = float(
    _ANALYSIS_PRESET["delayed_reach_uniform_reach_length_m"]
)
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
            "schema_id": CHECKPOINT_SELECTION_BANK_SCHEMA_ID,
            "schema_version": FEEDBAX_MANIFEST_SCHEMA_VERSION,
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


def build_validation_checkpoint_selection_manifest(
    *,
    experiment: str,
    run_ids: Sequence[str],
    repo_root: Path = REPO_ROOT,
    preferred_manifest: CheckpointSelectionManifest | None = None,
    preferred_manifest_path: Path | None = None,
    checkpoint_selection_mode: CheckpointSelectionMode = "sparse_history",
) -> CheckpointSelectionManifest:
    """Build a Feedbax manifest of recoverable validation-selected checkpoints.

    The default mode selects from sparse training-history validation records.
    Callers that deliberately want a supplied fixed-bank rescore manifest must
    pass ``checkpoint_selection_mode="fixed_bank_manifest"``.
    """

    if preferred_manifest is None and preferred_manifest_path is not None:
        preferred_manifest = load_materialized_fixed_bank_manifest(
            manifest_path=preferred_manifest_path
        )
    resolved_manifest = _resolve_checkpoint_selection_manifest(
        checkpoint_selection_mode=checkpoint_selection_mode,
        experiment=experiment,
        run_ids=run_ids,
        repo_root=repo_root,
        preferred_manifest=preferred_manifest,
    )
    selections = {
        run_id: [
            selection.to_json(repo_root=repo_root)
            for selection in select_validation_checkpoints_for_run(
                experiment=experiment,
                run_id=run_id,
                repo_root=repo_root,
                preferred_manifest=resolved_manifest,
                checkpoint_selection_mode=checkpoint_selection_mode,
            )
        ]
        for run_id in run_ids
    }
    source = (
        _fixed_bank_selection_source_from_manifest(resolved_manifest)
        if resolved_manifest is not None
        else "sparse_history_fallback"
    )
    return _checkpoint_selection_manifest_from_rows(
        experiment=experiment,
        checkpoint_policy=(
            FIXED_BANK_CHECKPOINT_POLICY
            if resolved_manifest is not None
            else SPARSE_HISTORY_CHECKPOINT_POLICY
        ),
        selection_source=source,
        selections=selections,
        repo_root=repo_root,
        bank=(resolved_manifest.bank if resolved_manifest is not None else None),
    )


def plan_fixed_bank_checkpoint_selection(
    *,
    experiment: str,
    run_ids: Sequence[str],
    validation_bank: FixedValidationBankSpec,
    repo_root: Path = REPO_ROOT,
) -> CheckpointSelectionSpec:
    """Return a Feedbax fixed-bank selection spec without loading checkpoints."""

    checkpoint_lists = {}
    for run_id in run_ids:
        artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
        available_batches = available_checkpoint_batches(artifact_dir)
        if not available_batches:
            checkpoint_lists[run_id] = []
            continue
        run_spec = resolve_run_record(experiment, run_id, repo_root=repo_root)
        checkpoint_lists[run_id] = [
            {
                "replicate": replicate,
                "checkpoint_batches": batches,
                "checkpoint_path": _repo_relative(
                    checkpoint_path_for_batches(artifact_dir, batches),
                    repo_root=repo_root,
                ),
            }
            for replicate in range(_n_replicates_from_run_spec(run_spec))
            for batches in available_batches
        ]
    payload = {
        "issue": experiment,
        "checkpoint_policy": FIXED_BANK_CHECKPOINT_POLICY,
        "selection_source": _fixed_bank_selection_source(validation_bank),
        "validation_bank": validation_bank.to_json(),
        "validation_role": validation_bank.validation_role or "fixed_bank_rollout_validation",
        "selection_metric": validation_bank.selection_metric
        or "aggregate_rollout_validation_objective",
        "nominal_quality_role": validation_bank.nominal_quality_role or "reported_sidecar",
        "checkpoint_lists": checkpoint_lists,
    }
    return _checkpoint_selection_spec(payload, issue=experiment, repo_root=repo_root)


def build_fixed_bank_checkpoint_selection_manifest(
    *,
    experiment: str,
    run_ids: Sequence[str],
    validation_bank: FixedValidationBankSpec,
    scorer: CheckpointScorer | None = None,
    repo_root: Path = REPO_ROOT,
) -> CheckpointSelectionManifest:
    """Build a fixed-bank Feedbax checkpoint-selection manifest.

    The scorer receives ``(run_id, replicate, checkpoint_batches, checkpoint_path,
    run_spec, validation_bank)`` and must return the rollout validation objective
    for that checkpoint on the fixed bank. When no scorer is supplied, this
    returns an explicit failed manifest; no implicit filesystem path or direct
    writer is used.
    """

    spec = plan_fixed_bank_checkpoint_selection(
        experiment=experiment,
        run_ids=run_ids,
        validation_bank=validation_bank,
        repo_root=repo_root,
    )
    if scorer is None:
        return CheckpointSelectionManifest(
            id=checkpoint_selection_manifest_id(spec),
            status="failed",
            selection_status="failed",
            failure_reason="no_fixed_bank_checkpoint_scorer_supplied",
            selection_spec=spec_payload(
                "CheckpointSelectionSpec", spec.model_dump(mode="json", exclude_none=True)
            ),
            scorer=spec.scorer,
            bank=spec.bank,
            fallback_allowed=False,
            inputs=spec.inputs,
            selections=[],
            metadata={"rlrmp_issue": experiment, "checkpoint_policy": FIXED_BANK_CHECKPOINT_POLICY},
        )
    selections = {
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
    }
    return _checkpoint_selection_manifest_from_rows(
        experiment=experiment,
        checkpoint_policy=FIXED_BANK_CHECKPOINT_POLICY,
        selection_source=_fixed_bank_selection_source(validation_bank),
        selections=selections,
        repo_root=repo_root,
        bank=spec.bank,
        spec=spec,
    )


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
    preferred_manifest: CheckpointSelectionManifest | None = None,
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
    if effective_manifest is None and preferred_manifest_path is not None:
        effective_manifest = load_materialized_fixed_bank_manifest(
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
    )
    if manifest is not None and run_id in checkpoint_selection_rows(manifest):
        return selections_from_manifest_run(
            manifest,
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
    preferred_manifest: CheckpointSelectionManifest | None = None,
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
    seed = require_run_seed(
        run_spec,
        source=run_spec_path(experiment, run_id, repo_root=repo_root),
    )
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


def _checkpoint_selection_spec(
    payload: Mapping[str, Any],
    *,
    issue: str,
    repo_root: Path,
) -> CheckpointSelectionSpec:
    """Build the canonical Feedbax spec from capability inputs."""

    checkpoint_policy = str(payload.get("checkpoint_policy") or SPARSE_HISTORY_CHECKPOINT_POLICY)
    selection_source = str(payload.get("selection_source") or checkpoint_policy)
    selection_metric = str(
        payload.get("selection_metric")
        or _validation_bank_payload(payload).get("selection_metric")
        or "validation_objective"
    )
    scorer = _checkpoint_scorer_identity(payload, selection_metric=selection_metric)
    bank = _checkpoint_selection_bank(payload, issue=issue, repo_root=repo_root)
    groups = _checkpoint_selection_groups(payload, issue=issue, repo_root=repo_root)
    candidate_checkpoints = _unique_candidates(
        [candidate for group in groups for candidate in group.candidate_checkpoints]
        + _plan_candidate_checkpoints(payload, issue=issue, repo_root=repo_root)
    )
    inputs = _checkpoint_selection_inputs(payload, issue=issue, repo_root=repo_root)
    return CheckpointSelectionSpec(
        selection_type=selection_source,
        scorer=scorer,
        bank=bank,
        group_by="replicate",
        candidate_checkpoints=candidate_checkpoints,
        inputs=inputs,
        fallback_allowed=False,
        params={
            "checkpoint_policy": checkpoint_policy,
            "selection_policy": payload.get("selection_policy"),
            "selection_metric": selection_metric,
            "validation_role": payload.get("validation_role"),
            "nominal_quality_role": payload.get("nominal_quality_role"),
        },
        metadata={"rlrmp_issue": issue},
    )


def _checkpoint_selection_manifest_from_rows(
    *,
    experiment: str,
    checkpoint_policy: str,
    selection_source: str,
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    repo_root: Path,
    bank: CheckpointSelectionBank | None = None,
    spec: CheckpointSelectionSpec | None = None,
) -> CheckpointSelectionManifest:
    """Build a canonical Feedbax manifest from scored replicate rows."""

    payload = {
        "issue": experiment,
        "checkpoint_policy": checkpoint_policy,
        "selection_source": selection_source,
        "selection_metric": "validation_objective",
        "runs": selections,
    }
    effective_spec = spec or _checkpoint_selection_spec(
        payload, issue=experiment, repo_root=repo_root
    )
    if bank is not None and spec is None:
        effective_spec = effective_spec.model_copy(update={"bank": bank})
    groups = _checkpoint_selection_groups(payload, issue=experiment, repo_root=repo_root)
    selected = any(group.selected_checkpoint is not None for group in groups)
    failure_reason = _checkpoint_selection_failure_reason(payload, selected=selected)
    return CheckpointSelectionManifest(
        id=checkpoint_selection_manifest_id(effective_spec),
        status="completed" if selected else "failed",
        selection_status="selected" if selected else "failed",
        failure_reason=failure_reason,
        selection_spec=spec_payload(
            "CheckpointSelectionSpec",
            effective_spec.model_dump(mode="json", exclude_none=True),
        ),
        scorer=effective_spec.scorer,
        bank=effective_spec.bank,
        fallback_allowed=effective_spec.fallback_allowed,
        inputs=effective_spec.inputs,
        selections=groups,
        summary_metrics={
            "n_runs": len(selections),
            "n_selected_checkpoints": sum(
                1 for group in groups if group.selected_checkpoint is not None
            ),
        },
        metadata={
            "rlrmp_issue": experiment,
            "checkpoint_policy": checkpoint_policy,
            "selection_source": selection_source,
        },
    )


def checkpoint_selection_rows(
    manifest: CheckpointSelectionManifest,
) -> dict[str, list[dict[str, Any]]]:
    """Return selected rows grouped by run for presentation and model loading."""

    runs: dict[str, list[dict[str, Any]]] = {}
    for group in manifest.selections:
        if group.selected_checkpoint is None:
            continue
        row = dict(group.selected_checkpoint.metadata.get("rlrmp_selection_row") or {})
        if not row:
            row = _selection_row_from_feedbax_group(group, manifest)
        runs.setdefault(str(group.run_id), []).append(row)
    return {
        run_id: sorted(rows, key=lambda row: int(row.get("replicate", 0)))
        for run_id, rows in runs.items()
    }


def load_checkpoint_selection_manifest(path: Path | str) -> CheckpointSelectionManifest:
    """Load an explicit Feedbax checkpoint-selection manifest path."""

    manifest = load_manifest(path)
    if not isinstance(manifest, CheckpointSelectionManifest):
        raise TypeError(f"Expected CheckpointSelectionManifest in {path}")
    return manifest


def _checkpoint_scorer_identity(
    payload: Mapping[str, Any],
    *,
    selection_metric: str,
) -> CheckpointScorerIdentity:
    validation_bank = _validation_bank_payload(payload)
    scorer_id = str(
        validation_bank.get("scorer_identity")
        or payload.get("scorer_identity")
        or payload.get("selection_source")
        or "rlrmp.sparse_history.validation_objective"
    )
    return CheckpointScorerIdentity(
        scorer_id=scorer_id,
        version=(
            None
            if validation_bank.get("scorer_version") is None
            else str(validation_bank["scorer_version"])
        ),
        parameters={"primary_metric": selection_metric, "objective": "minimize"},
        metadata={
            "selection_source": payload.get("selection_source"),
            "validation_role": payload.get("validation_role"),
            "nominal_quality_role": payload.get("nominal_quality_role"),
        },
    )


def _checkpoint_selection_bank(
    payload: Mapping[str, Any],
    *,
    issue: str,
    repo_root: Path,
) -> CheckpointSelectionBank:
    validation_bank = _validation_bank_payload(payload)
    source_manifest = payload.get("source_feedback_ablation_manifest")
    if validation_bank:
        bank_id = str(validation_bank.get("bank_identity") or "rlrmp.fixed_bank")
        ref = ParentRef(
            kind="CheckpointSelectionBank",
            id=bank_id,
            role=str(payload.get("validation_role") or "fixed_validation_bank"),
            uri=_repo_uri(source_manifest) if source_manifest is not None else None,
        )
        return CheckpointSelectionBank(
            role="fixed",
            status="available",
            bank_id=bank_id,
            logical_name=str(payload.get("validation_role") or bank_id),
            ref=ref,
            metadata={
                "validation_bank": _json_ready(validation_bank),
                "selection_metric": payload.get("selection_metric"),
                "nominal_quality_role": payload.get("nominal_quality_role"),
            },
        )
    if source_manifest is not None:
        source = str(source_manifest)
        return CheckpointSelectionBank(
            role="fixed",
            status="available",
            bank_id=f"rlrmp.feedback_rescore_audit:{issue}",
            logical_name="feedback_rescore_audit",
            ref=ParentRef(
                kind="RLRMPGRUFeedbackAblation",
                id=f"rlrmp-feedback-ablation:{issue}",
                role="feedback_rescore_audit_bank",
                uri=_repo_uri(source),
            ),
            metadata={"selection_metric": payload.get("selection_metric")},
        )
    return CheckpointSelectionBank(
        role="validation",
        status="available",
        bank_id=f"rlrmp.sparse_history_validation:{issue}",
        logical_name="sparse_training_history_validation",
        ref=ParentRef(
            kind="RLRMPGRUTrainingHistoryBank",
            id=f"rlrmp-gru-training-history-bank:{issue}",
            role="validation_history_bank",
            uri=f"repo://{_repo_relative(repo_root / '_artifacts' / issue / 'runs', repo_root=repo_root)}/*/training_history.eqx",
        ),
        metadata={
            "history_validation_log_note": payload.get("history_validation_log_note"),
        },
    )


def _checkpoint_selection_groups(
    payload: Mapping[str, Any],
    *,
    issue: str,
    repo_root: Path,
) -> list[CheckpointSelectionGroup]:
    runs = payload.get("runs", {})
    if not isinstance(runs, Mapping):
        return []
    groups: list[CheckpointSelectionGroup] = []
    for run_id, rows in runs.items():
        if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            candidate = _candidate_ref_from_selection_row(
                row,
                issue=issue,
                run_id=str(run_id),
                repo_root=repo_root,
            )
            groups.append(
                CheckpointSelectionGroup(
                    scope="replicate",
                    run_id=str(run_id),
                    replicate_id=str(row.get("replicate")),
                    candidate_checkpoints=[candidate],
                    selected_checkpoint=candidate,
                    score_summaries=[
                        _score_summary_from_selection_row(
                            row,
                            candidate_id=candidate.id,
                            payload=payload,
                        )
                    ],
                    metadata={
                        "checkpoint_policy": payload.get("checkpoint_policy"),
                        "selection_source": row.get("selection_source")
                        or payload.get("selection_source"),
                    },
                )
            )
    return groups


def _candidate_ref_from_selection_row(
    row: Mapping[str, Any],
    *,
    issue: str,
    run_id: str,
    repo_root: Path,
) -> CheckpointCandidateRef:
    replicate = int(row.get("replicate", 0))
    checkpoint_batches = int(row["checkpoint_batches"])
    raw_path = Path(str(row.get("checkpoint_path") or ""))
    checkpoint_path = (
        raw_path
        if raw_path.is_absolute()
        else (repo_root / raw_path if str(raw_path) else repo_root)
    )
    rel_path = _repo_relative(checkpoint_path, repo_root=repo_root)
    candidate_id = f"{issue}:{run_id}:replicate-{replicate}:checkpoint-{checkpoint_batches}"
    return CheckpointCandidateRef(
        id=candidate_id,
        checkpoint=ParentRef(
            kind="RLRMPGRUCheckpoint",
            id=candidate_id,
            role="checkpoint",
            uri=_repo_uri(rel_path),
        ),
        run_id=run_id,
        replicate_id=str(replicate),
        step=checkpoint_batches,
        training_run=ParentRef(
            kind="TrainingRunManifest",
            id=f"rlrmp-training-run:{issue}:{run_id}",
            role="training_run",
            uri=_repo_uri(f"results/{issue}/runs/{run_id}.json"),
        ),
        metadata={
            "checkpoint_batches": checkpoint_batches,
            "checkpoint_path": rel_path,
            "rlrmp_selection_row": _json_ready(dict(row)),
        },
    )


def _score_summary_from_selection_row(
    row: Mapping[str, Any],
    *,
    candidate_id: str,
    payload: Mapping[str, Any],
) -> CheckpointScoreSummary:
    primary_metric = str(payload.get("selection_metric") or "validation_objective")
    primary_value = float(row.get("feedback_score", row.get("scoring_validation_objective", 0.0)))
    metrics = {
        "scoring_validation_objective": float(
            row.get("scoring_validation_objective", primary_value)
        ),
        "best_logged_validation_objective": float(
            row.get("best_logged_validation_objective", primary_value)
        ),
        "final_validation_objective": float(row.get("final_validation_objective", primary_value)),
        "final_vs_selected_validation_degradation": float(
            row.get("final_vs_selected_validation_degradation", 0.0)
        ),
    }
    if "feedback_score" in row:
        metrics["feedback_score"] = float(row["feedback_score"])
    return CheckpointScoreSummary(
        candidate_id=candidate_id,
        primary_metric=primary_metric,
        primary_value=primary_value,
        objective="minimize",
        metrics=metrics,
        metadata={
            "scoring_validation_log_batch": row.get("scoring_validation_log_batch"),
            "best_logged_validation_batch": row.get("best_logged_validation_batch"),
        },
    )


def _plan_candidate_checkpoints(
    payload: Mapping[str, Any],
    *,
    issue: str,
    repo_root: Path,
) -> list[CheckpointCandidateRef]:
    checkpoint_lists = payload.get("checkpoint_lists", {})
    if not isinstance(checkpoint_lists, Mapping):
        return []
    candidates: list[CheckpointCandidateRef] = []
    for run_id, rows in checkpoint_lists.items():
        if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            candidates.append(
                _candidate_ref_from_selection_row(
                    {"replicate": 0, **dict(row)},
                    issue=issue,
                    run_id=str(run_id),
                    repo_root=repo_root,
                )
            )
    return candidates


def _unique_candidates(
    candidates: Sequence[CheckpointCandidateRef],
) -> list[CheckpointCandidateRef]:
    unique: dict[str, CheckpointCandidateRef] = {}
    for candidate in candidates:
        unique.setdefault(candidate.id, candidate)
    return list(unique.values())


def _checkpoint_selection_inputs(
    payload: Mapping[str, Any],
    *,
    issue: str,
    repo_root: Path,
) -> list[ParentRef]:
    runs = payload.get("runs")
    if not isinstance(runs, Mapping):
        runs = payload.get("checkpoint_lists", {})
    run_ids = tuple(str(run_id) for run_id in runs) if isinstance(runs, Mapping) else ()
    refs = [
        ParentRef(
            kind="RLRMPExperiment",
            id=issue,
            role="experiment",
            uri=_repo_uri(f"results/{issue}"),
        )
    ]
    refs.extend(
        ParentRef(
            kind="TrainingRunManifest",
            id=f"rlrmp-training-run:{issue}:{run_id}",
            role="training_run",
            uri=_repo_uri(f"results/{issue}/runs/{run_id}.json"),
        )
        for run_id in run_ids
    )
    source_manifest = payload.get("source_feedback_ablation_manifest")
    if source_manifest is not None:
        refs.append(
            ParentRef(
                kind="RLRMPGRUFeedbackAblation",
                id=f"rlrmp-feedback-ablation:{issue}",
                role="feedback_ablation_manifest",
                uri=_repo_uri(str(source_manifest)),
            )
        )
    return refs


def _checkpoint_selection_failure_reason(
    payload: Mapping[str, Any],
    *,
    selected: bool,
) -> str | None:
    if selected:
        return None
    return str(
        payload.get("not_materialized_reason")
        or payload.get("failure_reason")
        or "no_checkpoint_selection_rows_materialized"
    )


def _selection_row_from_feedbax_group(
    group: CheckpointSelectionGroup,
    manifest: CheckpointSelectionManifest,
) -> dict[str, Any]:
    selected = group.selected_checkpoint
    if selected is None:
        return {}
    score = group.score_summaries[0] if group.score_summaries else None
    selected_score = 0.0 if score is None else float(score.primary_value)
    return {
        "replicate": int(group.replicate_id or 0),
        "checkpoint_batches": int(selected.step or 0),
        "checkpoint_path": selected.metadata.get("checkpoint_path"),
        "selection_source": manifest.metadata.get("selection_source"),
        "scoring_validation_log_batch": int(selected.step or 0),
        "scoring_validation_objective": selected_score,
        "best_logged_validation_batch": int(selected.step or 0),
        "best_logged_validation_objective": selected_score,
        "final_validation_objective": selected_score,
        "final_vs_selected_validation_degradation": 0.0,
    }


def _validation_bank_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    validation_bank = payload.get("validation_bank")
    return validation_bank if isinstance(validation_bank, Mapping) else {}


def _repo_uri(path: str | Path | None) -> str | None:
    if path is None:
        return None
    text = str(path)
    if text.startswith("repo://"):
        return text
    return f"repo://{text}"


def load_materialized_fixed_bank_manifest(
    *,
    manifest_path: Path | None,
) -> CheckpointSelectionManifest | None:
    """Load an explicitly referenced materialized fixed-bank manifest."""

    if manifest_path is None or not manifest_path.exists():
        return None
    manifest = load_checkpoint_selection_manifest(manifest_path)
    if manifest.metadata.get("checkpoint_policy") != FIXED_BANK_CHECKPOINT_POLICY:
        raise ValueError(f"Expected {FIXED_BANK_CHECKPOINT_POLICY} in {manifest_path}")
    if manifest.selection_status not in {"selected", "fallback_selected"}:
        return None
    return manifest


def _resolve_checkpoint_selection_manifest(
    *,
    checkpoint_selection_mode: CheckpointSelectionMode,
    experiment: str,
    run_ids: Sequence[str],
    repo_root: Path = REPO_ROOT,
    preferred_manifest: CheckpointSelectionManifest | None = None,
) -> CheckpointSelectionManifest | None:
    """Return the explicit fixed-bank manifest for ``run_ids`` when requested."""

    if checkpoint_selection_mode == "sparse_history":
        return None
    if checkpoint_selection_mode != "fixed_bank_manifest":
        raise ValueError(f"unsupported checkpoint selection mode {checkpoint_selection_mode!r}")
    del repo_root
    manifest = preferred_manifest
    if manifest is None:
        raise ValueError(
            "checkpoint_selection_mode='fixed_bank_manifest' requires a materialized "
            "fixed-bank checkpoint manifest"
        )
    runs = checkpoint_selection_rows(manifest)
    missing = [run_id for run_id in run_ids if run_id not in runs]
    if missing:
        raise ValueError(
            "fixed-bank checkpoint manifest is missing requested run(s): " + ", ".join(missing)
        )
    return manifest


def selections_from_manifest_run(
    manifest: CheckpointSelectionManifest,
    *,
    experiment: str,
    run_id: str,
    repo_root: Path = REPO_ROOT,
) -> list[ReplicateCheckpointSelection]:
    """Convert serialized selected-checkpoint rows into selection objects."""

    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    selections: list[ReplicateCheckpointSelection] = []
    for row in checkpoint_selection_rows(manifest).get(run_id, ()):
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


def _fixed_bank_selection_source_from_manifest(
    manifest: CheckpointSelectionManifest,
) -> str:
    return str(manifest.metadata.get("selection_source") or "fixed_bank_rescore")


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
    "CHECKPOINT_SELECTION_BANK_SCHEMA_ID",
    "DEFAULT_DELAYED_REACH_DIRECTION_COUNT",
    "DEFAULT_DELAYED_REACH_GO_CUE_STEPS",
    "DEFAULT_DELAYED_REACH_UNIFORM_REACH_LENGTH_M",
    "DELAYED_REACH_EVAL_BANK_SCHEMA_VERSION",
    "FEEDBAX_MANIFEST_SCHEMA_VERSION",
    "active_loss_term_labels",
    "available_checkpoint_batches",
    "delayed_reach_eval_bank_spec",
    "delayed_reach_fixed_eval_bank_specs",
    "delayed_reach_fixed_rescore_bank_spec",
    "checkpoint_selection_rows",
    "load_checkpoint_selection_manifest",
    "load_materialized_fixed_bank_manifest",
    "load_validation_selected_checkpoint_model",
    "build_fixed_bank_checkpoint_selection_manifest",
    "build_validation_checkpoint_selection_manifest",
    "plan_fixed_bank_checkpoint_selection",
    "score_fixed_bank_checkpoints_for_run",
    "select_validation_checkpoints_for_run",
    "select_sparse_history_validation_checkpoints_for_run",
    "validation_objective_history",
]
