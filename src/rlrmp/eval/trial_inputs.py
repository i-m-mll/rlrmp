"""Shared run and trial inputs for registered evaluation capabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.tree as jt
import numpy as np

from rlrmp.paths import REPO_ROOT, run_spec_path
from rlrmp.runtime.run_specs import resolve_run_record


@dataclass(frozen=True)
class EvaluationRunInputs:
    """Governed run inputs used by model-backed registered evaluations."""

    run_id: str
    label: str
    run_spec_path: Path
    artifact_dir: Path
    run_spec: dict[str, Any]


def resolve_evaluation_run_inputs(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None,
    repo_root: Path = REPO_ROOT,
) -> list[EvaluationRunInputs]:
    """Resolve governed run manifests and their artifact directories."""

    if not run_ids:
        raise ValueError("At least one run ID is required")
    resolved_labels = tuple(labels or (default_run_label(run_id) for run_id in run_ids))
    if len(resolved_labels) != len(run_ids):
        raise ValueError("labels must contain one value per run ID")

    runs = []
    for run_id, label in zip(run_ids, resolved_labels, strict=True):
        spec_path = run_spec_path(experiment, run_id, repo_root=repo_root)
        artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
        if not spec_path.exists():
            raise FileNotFoundError(f"Missing run spec: {spec_path}")
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Missing artifact directory: {artifact_dir}")
        runs.append(
            EvaluationRunInputs(
                run_id=run_id,
                label=label,
                run_spec_path=spec_path,
                artifact_dir=artifact_dir,
                run_spec=resolve_run_record(experiment, run_id, repo_root=repo_root),
            )
        )
    return runs


def default_run_label(run_id: str) -> str:
    """Return a compact display label from a governed run ID."""

    return run_id.split("__")[-1]


def initial_effector_velocity(trial_specs: Any) -> Any:
    """Return the trial initial effector velocity, shape ``(trials, 2)``."""

    for init_state in trial_specs.inits.values():
        velocity = getattr(init_state, "vel", None)
        if velocity is not None:
            return velocity
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 4:
            return jnp.asarray(init_state)[..., 2:4]
    raise ValueError("Trial spec does not include an effector velocity initial state")


def trial_effector_target_position(trial_specs: Any) -> np.ndarray:
    """Return target positions from current or delayed trial input layouts."""

    inputs = getattr(trial_specs, "inputs", {})
    target = inputs.get("effector_target") if isinstance(inputs, Mapping) else None
    if target is None and isinstance(inputs, Mapping):
        delayed_inputs = inputs.get("task")
        target = getattr(delayed_inputs, "effector_target", None)
    position = getattr(target, "pos", None)
    if position is None:
        raise KeyError("could not locate effector_target.pos in trial inputs")
    return np.asarray(position, dtype=np.float64)


def repeat_single_validation_trial(trial_specs: Any, n_trials: int) -> Any:
    """Repeat one representative validation trial along the trial axis."""

    source_trials = _trial_count_from_spec(trial_specs)

    def repeat_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] == source_trials:
            return jnp.repeat(leaf[:1], n_trials, axis=0)
        return leaf

    return jt.map(repeat_leaf, trial_specs)


def _trial_count_from_spec(trial_specs: Any) -> int:
    for target_spec in getattr(trial_specs, "targets", {}).values():
        value = getattr(target_spec, "value", None)
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    for init in getattr(trial_specs, "inits", {}).values():
        shape = getattr(init, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    inputs = getattr(trial_specs, "inputs", None)
    values = inputs.values() if isinstance(inputs, dict) else (inputs,)
    for value in values:
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    return 1


__all__ = [
    "EvaluationRunInputs",
    "default_run_label",
    "initial_effector_velocity",
    "repeat_single_validation_trial",
    "resolve_evaluation_run_inputs",
    "trial_effector_target_position",
]
