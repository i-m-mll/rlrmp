"""Typed authoring and CLI entry surface for native distillation methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from rlrmp.runtime.training_run_specs import (
    attach_distillation_training_specs,
    validate_distillation_training_run_spec,
)
from rlrmp.train.distillation_native.executor import (
    execute_distillation_training_run_spec_native,
)
from rlrmp.train.training_configs import (
    ClosedLoopDistillationConfig,
    GuidedDistillationConfig,
)

DistillationMethod = Literal["guided_distillation", "closed_loop_distillation"]


def _guided_spec_path(config: GuidedDistillationConfig) -> Path:
    if config.run_spec is None:
        raise ValueError("guided distillation requires an explicit tracked run_spec")
    return Path(config.run_spec)


def _closed_loop_spec_path(config: ClosedLoopDistillationConfig) -> Path:
    if config.run_spec is None:
        raise ValueError("closed-loop distillation requires an explicit tracked run_spec")
    return Path(config.run_spec)


def load_distillation_run_spec(
    config: GuidedDistillationConfig | ClosedLoopDistillationConfig,
    *,
    method: DistillationMethod,
) -> dict[str, Any]:
    """Load a tracked recipe and apply its typed execution overrides."""

    path = (
        _guided_spec_path(config)
        if method == "guided_distillation"
        else _closed_loop_spec_path(config)
    )
    if not path.is_file():
        raise FileNotFoundError(f"tracked {method} run spec not found at {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if method == "guided_distillation":
        assert isinstance(config, GuidedDistillationConfig)
        if config.run_id is not None:
            spec["run_id"] = config.run_id
        spec["seed"] = config.seed
        spec["n_train_batches"] = config.n_batches
        spec["batch_size"] = config.batch_size
        spec["controller_lr"] = config.controller_lr
        if config.teacher_package is not None:
            spec["teacher_contract"]["teacher_package"] = config.teacher_package
        if config.teacher_manifest is not None:
            spec["teacher_contract"]["teacher_manifest"] = config.teacher_manifest
        if config.teacher_gains_key is not None:
            spec["teacher_bank"]["teacher_gains_key"] = config.teacher_gains_key
        weights = spec["distillation_surface"]["config"]["weights"]
        weights.update(
            clean_action=config.clean_action_weight,
            perturbation_response=config.perturbation_response_weight,
            input_output_jvp=config.input_output_jvp_weight,
            student_forced_rollout_anchor=config.rollout_anchor_weight,
        )
        spec["distillation_surface"]["config"]["n_jvp_directions"] = config.n_jvp_directions
        spec["teacher_bank"]["horizon"] = config.horizon
        spec["model_contract"].update(
            n_replicates=config.n_replicates,
            hidden_size=config.hidden_size,
            trainable_dtype=config.trainable_dtype,
            population_mask_mode=config.population_mask_mode,
        )
        hps = spec.get("hps")
        if isinstance(hps, dict):
            hps["batch_size"] = config.batch_size
            hps["n_batches_condition"] = config.n_batches
            hps["learning_rate_0"] = config.controller_lr
            hps_model = hps.setdefault("model", {})
            hps_model.update(
                n_replicates=config.n_replicates,
                hidden_size=config.hidden_size,
                trainable_dtype=config.trainable_dtype,
                population_mask_mode=config.population_mask_mode,
            )
        spec["optimizer"].update(
            controller_lr=config.controller_lr,
            lr_warmup_batches=config.lr_warmup_batches,
            lr_warmup_init_fraction=config.lr_warmup_init_fraction,
            lr_cosine_alpha=config.lr_cosine_alpha,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        output_dir = _authored_output_dir(spec, config.output_dir)
    else:
        assert isinstance(config, ClosedLoopDistillationConfig)
        if config.run_id is not None:
            spec["run_id"] = config.run_id
        spec["seed"] = config.seed
        student = spec["student_contract"]
        student.update(
            n_replicates=config.n_replicates,
            hidden_size=config.hidden_size,
            batch_size=config.batch_size,
            n_train_batches=config.n_batches,
            controller_lr=config.controller_lr,
            lr_warmup_batches=config.lr_warmup_batches,
            lr_cosine_alpha=config.lr_cosine_alpha,
            gradient_clip_norm=config.gradient_clip_norm,
            trainable_dtype=config.trainable_dtype,
        )
        if config.teacher_package is not None:
            spec["teacher_contract"]["teacher_package"] = config.teacher_package
        if config.teacher_gains_key is not None:
            spec["teacher_contract"]["teacher_gains_key"] = config.teacher_gains_key
        spec["teacher_contract"]["horizon"] = config.horizon
        spec["loss_surface"]["weights"].update(
            kinematics_trajectory=config.kinematics_trajectory_weight,
            velocity=config.velocity_weight,
            endpoint=config.endpoint_weight,
            settling=config.settling_weight,
            action_force_trajectory=config.action_force_weight,
            perturbation_response_trajectory=config.perturbation_response_weight,
            directional_input_output_jvp=config.input_output_jvp_weight,
            task_qr_rollout=config.task_rollout_loss_weight,
        )
        output_dir = _authored_output_dir(spec, config.output_dir)
    if not spec.get("run_id"):
        raise ValueError(f"tracked {method} run spec must declare run_id")
    spec = attach_distillation_training_specs(
        spec,
        method=method,
        output_dir=output_dir,
        spec_path=path,
    )
    validate_distillation_training_run_spec(spec, method=method)
    return spec


def _authored_output_dir(spec: dict[str, Any], override: str | None) -> Path:
    """Resolve output custody only from the authored spec or an explicit override."""

    value = override or spec.get("artifact_output_dir")
    if not value:
        raise ValueError("tracked distillation run spec must declare artifact_output_dir")
    return Path(str(value))


def run_distillation_config(
    config: GuidedDistillationConfig | ClosedLoopDistillationConfig,
    *,
    method: DistillationMethod,
) -> Any:
    """Execute one validated config through the sole native executor."""

    spec = load_distillation_run_spec(config, method=method)
    if config.dry_run or not config.full_train:
        return spec
    if (
        method == "closed_loop_distillation"
        and isinstance(config, ClosedLoopDistillationConfig)
        and not (config.confirm_full_train and config.user_confirmed)
    ):
        raise PermissionError(
            "full closed-loop distillation requires both --confirm-full-train and --user-confirmed"
        )
    return execute_distillation_training_run_spec_native(
        spec,
        method=method,
        run_id=str(spec["run_id"]),
        resume=config.resume,
    )


__all__ = [
    "load_distillation_run_spec",
    "run_distillation_config",
]
