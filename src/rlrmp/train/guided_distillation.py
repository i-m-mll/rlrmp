"""CLI support for guided C&S GRU distillation run specs.

This module is intentionally a thin entry surface around
``rlrmp.train.distillation``. The reusable loss/JVP implementation lives there;
this file owns the first 6D h0 H-infinity no-launch run contract and a tiny
local smoke path that proves the CLI can call the intended loss surface.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from rlrmp.paths import mkdir_p
from rlrmp.train.distillation import (
    CSH0DistillationConfig,
    DistillationLossWeights,
    cs_h0_distillation_config,
    guided_distillation_loss,
)

SCHEMA_VERSION = "rlrmp.guided_distillation.training_entry.v1"
ISSUE_ID = "9727d79"
IMPLEMENTATION_ISSUE_ID = "c314267"
BASE_ISSUE_ID = "020a65b"
TEACHER_ISSUE_ID = "376d023"
RUN_ID = "h0_hinf_6d_guided_distillation"
DEFAULT_SPEC_PATH = Path(f"results/{ISSUE_ID}/runs/{RUN_ID}.json")
DEFAULT_OUTPUT_DIR = f"_artifacts/{ISSUE_ID}/runs/{RUN_ID}"
DEFAULT_TEACHER_PACKAGE = "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz"
DEFAULT_TEACHER_MANIFEST = (
    "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers_manifest.json"
)
BASE_RUN_ID = (
    "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64"
)
BASE_RUN_SPEC = f"results/{BASE_ISSUE_ID}/runs/{BASE_RUN_ID}/run.json"
REQUIRED_TEACHER_KEYS = (
    "plant_A",
    "plant_B",
    "x0",
    "hinf_controller_gains",
    "observation_matrix",
)


def default_distillation_command(*, spec_path: Path = DEFAULT_SPEC_PATH) -> list[str]:
    """Return the no-launch dry-run command for the first guided distillation row."""

    return [
        "env",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/train_guided_distillation.py",
        "--run-spec-output",
        str(spec_path),
        "--dry-run",
        "--smoke-loss",
    ]


def full_train_command(*, spec_path: Path = DEFAULT_SPEC_PATH) -> list[str]:
    """Return the full-train command for the first guided distillation row."""

    return [
        "env",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/train_guided_distillation.py",
        "--run-spec",
        str(spec_path),
        "--full-train",
    ]


def build_distillation_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Build the no-launch spec for the first 6D h0 H-infinity distillation row."""

    config = cs_h0_distillation_config(
        weights=DistillationLossWeights(
            clean_action=float(args.clean_action_weight),
            perturbation_response=float(args.perturbation_response_weight),
            input_output_jvp=float(args.input_output_jvp_weight),
            student_forced_rollout_anchor=float(args.rollout_anchor_weight),
        ),
        n_jvp_directions=int(args.n_jvp_directions),
    )
    run_spec_path = Path(args.run_spec_output)
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE_ID,
        "implementation_issue": IMPLEMENTATION_ISSUE_ID,
        "launch_status": "not_launched",
        "no_launch_boundary": (
            "No RunPod, Modal, GPU acquisition, full training launch, push, or "
            "protected-branch auth request is authorized by this spec."
        ),
        "training_entry": {
            "script": "scripts/train_guided_distillation.py",
            "module": "rlrmp.train.guided_distillation",
            "loss_module": "rlrmp.train.distillation",
            "loss_function": "rlrmp.train.distillation.guided_distillation_loss",
            "run_spec_path": str(run_spec_path),
            "artifact_output_dir": str(args.output_dir),
            "spec_command": default_distillation_command(spec_path=run_spec_path),
            "command": full_train_command(spec_path=run_spec_path),
            "full_train_status": "implemented_no_launch",
            "trainer": "rlrmp.train.guided_distillation.run_guided_distillation_training",
        },
        "base_contract": {
            "issue": BASE_ISSUE_ID,
            "run_id": BASE_RUN_ID,
            "run_spec": BASE_RUN_SPEC,
            "inherit": [
                "h0 encoder",
                "force-filter feedback",
                "target-relative multitarget distribution",
                "hidden size 180",
                "5 replicates",
                "batch size 64",
                "AdamW learning rate 3e-3",
                "global gradient clip 5.0",
                "500-batch warmup cosine schedule",
                "12000 train batches",
                "full-Q/R/Qf warm-cos loss weights",
                "cs2019-rollout stochastic preset",
                "no PGD adversarial inner maximizer",
            ],
        },
        "teacher_contract": {
            "issue": TEACHER_ISSUE_ID,
            "primary_teacher": "6d_output_feedback_hinf",
            "diagnostic_teacher": "6d_output_feedback_extlqg",
            "teacher_package": str(args.teacher_package),
            "teacher_manifest": str(args.teacher_manifest),
            "external_basis": {
                "feedback_history": "target_relative_delayed_feedback_plus_force_filter",
                "action_history": "controller_command_history",
                "action_output": "controller_command_history",
            },
        },
        "teacher_bank": {
            "materializer": "rlrmp.train.guided_distillation.materialize_teacher_batch",
            "source": "analytical linear teacher package",
            "teacher": "hinf_controller_gains",
            "horizon": 60,
            "sampled_initial_state_std": 0.02,
            "observation_perturbation_std": 0.05,
            "action_context": (
                "teacher action histories for teacher-forced phases and a "
                "stop-gradient student/teacher blend for mixed/student-forced phases"
            ),
            "approximation": (
                "The first executable trainer uses analytical plant rollouts and a "
                "local observation-space affine teacher derived from K_t @ pinv(C). "
                "It preserves the external 6D feedback and 2D action-history loss "
                "contract, but it is not a full Feedbax task rollout or a formal "
                "output-feedback certificate materializer."
            ),
        },
        "training_schedule": {
            "total_batches": 12000,
            "phases": [
                {
                    "name": "teacher_forced_warm_start",
                    "start_batch": 0,
                    "end_batch": 1500,
                    "teacher_forcing_fraction": 1.0,
                    "student_forcing_fraction": 0.0,
                },
                {
                    "name": "mixed_teacher_student_forcing",
                    "start_batch": 1500,
                    "end_batch": 4000,
                    "teacher_forcing_fraction": 0.5,
                    "student_forcing_fraction": 0.5,
                },
                {
                    "name": "mostly_student_forced",
                    "start_batch": 4000,
                    "end_batch": 12000,
                    "teacher_forcing_fraction": 0.1,
                    "student_forcing_fraction": 0.9,
                },
            ],
        },
        "optimizer": {
            "name": "adamw",
            "controller_lr": 3e-3,
            "lr_schedule": "warmup_cosine",
            "lr_warmup_batches": 500,
            "lr_warmup_init_fraction": 0.1,
            "lr_cosine_alpha": 0.1,
            "gradient_clip_norm": 5.0,
        },
        "model_contract": {
            "initial_hidden_encoder": True,
            "force_filter_feedback": True,
            "hidden_size": 180,
            "batch_size": 64,
            "n_replicates": 5,
            "plant_backend": "cs_lss",
            "stochastic_preset": "cs2019-rollout",
            "broad_epsilon_pgd_training": False,
        },
        "distillation_surface": {
            "config": config.summary(),
            "hidden_state_supervision": False,
            "components": {
                "clean_action": {
                    "enabled": config.weights.clean_action > 0.0,
                    "weight": config.weights.clean_action,
                    "description": (
                        "Match clean student and teacher command histories on the "
                        "declared target/observation bank."
                    ),
                },
                "perturbation_response": {
                    "enabled": config.weights.perturbation_response > 0.0,
                    "weight": config.weights.perturbation_response,
                    "description": (
                        "Match teacher-induced corrective command response under the "
                        "deterministic analytical observation-perturbation bank."
                    ),
                },
                "input_output_jvp": {
                    "enabled": config.weights.input_output_jvp > 0.0,
                    "weight": config.weights.input_output_jvp,
                    "n_directions": config.n_jvp_directions,
                    "direction_basis": config.jvp_direction_basis,
                    "implementation": (
                        "jax.linearize plus jax.vmap over banked directional probes; "
                        "no dense Jacobian materialization in the training path"
                    ),
                },
                "student_forced_rollout_anchor": {
                    "enabled": config.weights.student_forced_rollout_anchor > 0.0,
                    "weight": config.weights.student_forced_rollout_anchor,
                    "description": (
                        "Anchor student-forced rollout summaries to base-row/teacher "
                        "clean behavior without selecting on hidden coordinates."
                    ),
                },
            },
        },
        "evaluation_gates": [
            "standard certificate",
            "objective comparator",
            "perturbation-response diagnostics",
            "velocity and loss figures",
            "H-infinity phenotype sidecar where applicable",
            "all-replicate reporting",
            "teacher-guided versus rollout-discovery interpretation note",
        ],
        "local_acceptance_checks": [
            "scripts/train_guided_distillation.py --full-train --smoke-train runs "
            "the real trainer/loss path on tiny CPU-friendly shapes",
            "tests/test_distillation.py covers dense-Jacobian equivalence only on "
            "tiny diagnostic maps",
            "the full-train path fails fast only for missing teacher packages or "
            "invalid specs, not as a placeholder guard",
        ],
    }


class GuidedGRUPolicy(eqx.Module):
    """GRU student policy over external feedback and action histories."""

    h0_encoder: eqx.nn.Linear
    cell: eqx.nn.GRUCell
    readout: eqx.nn.Linear

    def __init__(
        self,
        *,
        feedback_dim: int,
        action_dim: int,
        hidden_size: int,
        key: jax.Array,
    ) -> None:
        h0_key, cell_key, readout_key = jr.split(key, 3)
        self.h0_encoder = eqx.nn.Linear(feedback_dim, hidden_size, key=h0_key)
        self.cell = eqx.nn.GRUCell(feedback_dim + action_dim, hidden_size, key=cell_key)
        self.readout = eqx.nn.Linear(hidden_size, action_dim, key=readout_key)

    def _single(self, feedback_history: jax.Array, action_history: jax.Array) -> jax.Array:
        hidden = jnp.tanh(self.h0_encoder(feedback_history[0]))
        inputs = jnp.concatenate([feedback_history, action_history], axis=-1)

        def step(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_hidden = self.cell(value, carry)
            action = self.readout(next_hidden)
            return next_hidden, action

        _, actions = jax.lax.scan(step, hidden, inputs)
        return actions

    def __call__(self, feedback_history: jax.Array, action_history: jax.Array) -> jax.Array:
        if feedback_history.ndim == 2:
            return self._single(feedback_history, action_history)
        return jax.vmap(self._single)(feedback_history, action_history)


class BankedAffineTeacher(eqx.Module):
    """Local affine analytical teacher over a sampled external-feedback bank."""

    base_feedback: jax.Array
    base_actions: jax.Array
    feedback_gains: jax.Array

    def __call__(self, feedback_history: jax.Array, action_history: jax.Array) -> jax.Array:
        del action_history
        feedback_delta = feedback_history - self.base_feedback
        return self.base_actions + jnp.einsum(
            "...tuf,...tf->...tu",
            self.feedback_gains,
            feedback_delta,
        )


def _require_teacher_package(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Teacher package not found at {path}. Generate or sync the "
            "376d023 analytical teacher package before running --full-train."
        )
    package = np.load(path)
    missing = [key for key in REQUIRED_TEACHER_KEYS if key not in package.files]
    if missing:
        raise ValueError(f"Teacher package {path} is missing required keys: {', '.join(missing)}.")
    return {key: package[key] for key in package.files}


def load_teacher_package(path: str | Path) -> dict[str, jax.Array]:
    """Load and validate the analytical teacher package."""

    arrays = _require_teacher_package(Path(path))
    plant_a = arrays["plant_A"]
    plant_b = arrays["plant_B"]
    controller_gains = arrays["hinf_controller_gains"]
    observation_matrix = arrays["observation_matrix"]
    if plant_a.ndim != 2 or plant_a.shape[0] != plant_a.shape[1]:
        raise ValueError("teacher plant_A must be square")
    if plant_b.ndim != 2 or plant_b.shape[0] != plant_a.shape[0]:
        raise ValueError("teacher plant_B must have shape (state_dim, action_dim)")
    if controller_gains.ndim != 3 or controller_gains.shape[1:] != (
        plant_b.shape[1],
        plant_a.shape[0],
    ):
        raise ValueError("hinf_controller_gains must have shape (time, action_dim, state_dim)")
    if observation_matrix.ndim != 2 or observation_matrix.shape[1] != plant_a.shape[0]:
        raise ValueError("observation_matrix must have shape (feedback_dim, state_dim)")
    return {key: jnp.asarray(value, dtype=jnp.float32) for key, value in arrays.items()}


def forcing_fraction_for_batch(spec: dict[str, Any], batch_index: int) -> float:
    """Return the student-forcing fraction encoded by the staged run spec."""

    for phase in spec["training_schedule"]["phases"]:
        if int(phase["start_batch"]) <= batch_index < int(phase["end_batch"]):
            return float(phase["student_forcing_fraction"])
    return float(spec["training_schedule"]["phases"][-1]["student_forcing_fraction"])


def _teacher_rollout(
    package: dict[str, jax.Array],
    initial_states: jax.Array,
    *,
    horizon: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    plant_a = package["plant_A"]
    plant_b = package["plant_B"]
    observation_matrix = package["observation_matrix"]
    gains = package["hinf_controller_gains"][:horizon]
    feedback_pinv = jnp.linalg.pinv(observation_matrix)
    feedback_gains = -jnp.einsum("tus,sf->tuf", gains, feedback_pinv)

    def step(states: jax.Array, gain: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        feedback = states @ observation_matrix.T
        actions = -jnp.einsum("us,bs->bu", gain, states)
        next_states = states @ plant_a.T + actions @ plant_b.T
        return next_states, (states, feedback, actions)

    _, (states, feedback, actions) = jax.lax.scan(step, initial_states, gains)
    gains_by_batch = jnp.broadcast_to(
        feedback_gains[None, ...],
        (initial_states.shape[0], *feedback_gains.shape),
    )
    return states.swapaxes(0, 1), feedback.swapaxes(0, 1), actions.swapaxes(0, 1), gains_by_batch


def materialize_teacher_batch(
    package: dict[str, jax.Array],
    *,
    key: jax.Array,
    batch_size: int,
    horizon: int,
    n_jvp_directions: int,
    initial_state_std: float = 0.02,
    observation_perturbation_std: float = 0.05,
) -> dict[str, jax.Array]:
    """Materialize one deterministic analytical rollout/probe batch."""

    state_key, perturb_key, direction_key = jr.split(key, 3)
    x0 = package["x0"]
    initial_states = x0 + initial_state_std * jr.normal(
        state_key,
        (batch_size, x0.shape[0]),
        dtype=x0.dtype,
    )
    states, feedback, teacher_actions, feedback_gains = _teacher_rollout(
        package,
        initial_states,
        horizon=horizon,
    )
    del states
    perturb_feedback = feedback + observation_perturbation_std * jr.normal(
        perturb_key,
        feedback.shape,
        dtype=feedback.dtype,
    )
    direction_keys = jr.split(direction_key, 2)
    feedback_directions = 0.01 * jr.normal(
        direction_keys[0],
        (n_jvp_directions, *feedback.shape),
        dtype=feedback.dtype,
    )
    action_directions = 0.01 * jr.normal(
        direction_keys[1],
        (n_jvp_directions, *teacher_actions.shape),
        dtype=teacher_actions.dtype,
    )
    return {
        "feedback_history": feedback,
        "teacher_actions": teacher_actions,
        "perturbation_feedback_history": perturb_feedback,
        "feedback_directions": feedback_directions,
        "action_directions": action_directions,
        "feedback_gains": feedback_gains,
    }


def _make_optimizer(
    *,
    learning_rate: float,
    n_batches: int,
    warmup_batches: int,
    warmup_init_fraction: float,
    cosine_alpha: float,
    gradient_clip_norm: float,
) -> optax.GradientTransformation:
    effective_warmup = max(0, min(warmup_batches, n_batches - 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * warmup_init_fraction,
        peak_value=learning_rate,
        warmup_steps=effective_warmup,
        decay_steps=max(1, n_batches),
        end_value=learning_rate * cosine_alpha,
    )
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.adamw(schedule, weight_decay=0.0),
    )


def _loss_for_batch(
    model: GuidedGRUPolicy,
    batch: dict[str, jax.Array],
    config: CSH0DistillationConfig,
    *,
    student_forcing_fraction: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    teacher_actions = batch["teacher_actions"]
    teacher_context = jnp.zeros_like(teacher_actions)
    teacher_context = teacher_context.at[:, 1:, :].set(teacher_actions[:, :-1, :])
    teacher_context = jax.lax.stop_gradient(teacher_context)
    teacher_forced_student = jax.lax.stop_gradient(
        model(batch["feedback_history"], teacher_context)
    )
    student_context = jnp.zeros_like(teacher_actions)
    student_context = student_context.at[:, 1:, :].set(teacher_forced_student[:, :-1, :])
    action_context = (
        1.0 - student_forcing_fraction
    ) * teacher_context + student_forcing_fraction * student_context
    perturbation_context = action_context
    teacher = BankedAffineTeacher(
        base_feedback=batch["feedback_history"],
        base_actions=teacher_actions,
        feedback_gains=batch["feedback_gains"],
    )
    result = guided_distillation_loss(
        student_policy=model,
        teacher_policy=teacher,
        feedback_history=batch["feedback_history"],
        action_history=action_context,
        config=config,
        perturbation_feedback_history=batch["perturbation_feedback_history"],
        perturbation_action_history=perturbation_context,
        feedback_directions=batch["feedback_directions"],
        action_directions=batch["action_directions"],
        student_forced_rollout=model(batch["feedback_history"], student_context),
        rollout_anchor=teacher_actions,
    )
    return result.total, result.components


@eqx.filter_jit
def _train_step(
    model: GuidedGRUPolicy,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    batch: dict[str, jax.Array],
    config: CSH0DistillationConfig,
    student_forcing_fraction: float,
) -> tuple[GuidedGRUPolicy, optax.OptState, jax.Array, dict[str, jax.Array]]:
    (loss, components), grads = eqx.filter_value_and_grad(_loss_for_batch, has_aux=True)(
        model,
        batch,
        config,
        student_forcing_fraction=student_forcing_fraction,
    )
    updates, optimizer_state = optimizer.update(
        grads,
        optimizer_state,
        eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    return model, optimizer_state, loss, components


def _write_training_outputs(
    *,
    output_dir: Path,
    spec: dict[str, Any],
    histories: list[list[dict[str, Any]]],
    models: list[GuidedGRUPolicy],
) -> None:
    mkdir_p(output_dir)
    (output_dir / "run_spec_snapshot.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "loss_history.json").write_text(
        json.dumps(histories, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_losses = [history[-1]["loss_total"] for history in histories if history]
    summary = {
        "schema_version": "rlrmp.guided_distillation.training_summary.v1",
        "issue": ISSUE_ID,
        "run_id": RUN_ID,
        "n_replicates": len(histories),
        "n_batches": len(histories[0]) if histories else 0,
        "final_loss_mean": float(np.mean(final_losses)) if final_losses else None,
        "final_loss_min": float(np.min(final_losses)) if final_losses else None,
        "final_loss_max": float(np.max(final_losses)) if final_losses else None,
        "artifacts": {
            "loss_history": "loss_history.json",
            "run_spec_snapshot": "run_spec_snapshot.json",
            "student_model_replicates": [
                f"student_model_rep{replicate_index}.eqx" for replicate_index in range(len(models))
            ],
        },
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for replicate_index, model in enumerate(models):
        eqx.tree_serialise_leaves(output_dir / f"student_model_rep{replicate_index}.eqx", model)


def run_guided_distillation_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run the executable analytical-teacher guided-distillation trainer."""

    spec = build_distillation_spec(args)
    if args.run_spec is not None:
        run_spec_path = Path(args.run_spec)
        if not run_spec_path.is_file():
            raise FileNotFoundError(f"Run spec not found at {run_spec_path}.")
        loaded_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        if loaded_spec.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Run spec {run_spec_path} has schema_version "
                f"{loaded_spec.get('schema_version')!r}; expected {SCHEMA_VERSION!r}."
            )
        spec = loaded_spec

    n_batches = int(args.n_batches)
    batch_size = int(args.batch_size)
    n_replicates = int(args.n_replicates)
    hidden_size = int(args.hidden_size)
    horizon = int(args.horizon)
    n_jvp_directions = int(args.n_jvp_directions)
    if args.smoke_train:
        n_batches = min(n_batches, 3)
        batch_size = min(batch_size, 2)
        n_replicates = min(n_replicates, 1)
        hidden_size = min(hidden_size, 8)
        horizon = min(horizon, 6)
        n_jvp_directions = min(n_jvp_directions, 2)

    if n_batches <= 0 or batch_size <= 0 or n_replicates <= 0:
        raise ValueError("n_batches, batch_size, and n_replicates must be positive.")
    package = load_teacher_package(args.teacher_package)
    teacher_horizon = int(package["hinf_controller_gains"].shape[0])
    if horizon > teacher_horizon:
        raise ValueError(f"Requested horizon {horizon} exceeds teacher horizon {teacher_horizon}.")
    feedback_dim = int(package["observation_matrix"].shape[0])
    action_dim = int(package["plant_B"].shape[1])
    config = cs_h0_distillation_config(
        weights=DistillationLossWeights(
            clean_action=float(args.clean_action_weight),
            perturbation_response=float(args.perturbation_response_weight),
            input_output_jvp=float(args.input_output_jvp_weight),
            student_forced_rollout_anchor=float(args.rollout_anchor_weight),
        ),
        n_jvp_directions=n_jvp_directions,
    )
    optimizer = _make_optimizer(
        learning_rate=float(args.controller_lr),
        n_batches=n_batches,
        warmup_batches=int(args.lr_warmup_batches),
        warmup_init_fraction=float(args.lr_warmup_init_fraction),
        cosine_alpha=float(args.lr_cosine_alpha),
        gradient_clip_norm=float(args.gradient_clip_norm),
    )

    root_key = jr.PRNGKey(int(args.seed))
    histories: list[list[dict[str, Any]]] = []
    models: list[GuidedGRUPolicy] = []
    started = time.perf_counter()
    log_step = max(1, int(args.log_step))
    for replicate_index in range(n_replicates):
        model_key = jr.fold_in(root_key, replicate_index)
        batch_key = jr.fold_in(root_key, 10_000 + replicate_index)
        model = GuidedGRUPolicy(
            feedback_dim=feedback_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            key=model_key,
        )
        optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))
        history: list[dict[str, Any]] = []
        for batch_index in range(n_batches):
            batch_key, materialize_key = jr.split(batch_key)
            batch = materialize_teacher_batch(
                package,
                key=materialize_key,
                batch_size=batch_size,
                horizon=horizon,
                n_jvp_directions=n_jvp_directions,
            )
            forcing_fraction = forcing_fraction_for_batch(spec, batch_index)
            model, optimizer_state, loss, components = _train_step(
                model,
                optimizer_state,
                optimizer,
                batch,
                config,
                forcing_fraction,
            )
            history.append(
                {
                    "replicate": replicate_index,
                    "batch": batch_index + 1,
                    "student_forcing_fraction": forcing_fraction,
                    "loss_total": float(loss),
                    "components": {name: float(value) for name, value in components.items()},
                }
            )
            if (
                args.smoke_train
                or batch_index == 0
                or (batch_index + 1) % log_step == 0
                or batch_index + 1 == n_batches
            ):
                elapsed = time.perf_counter() - started
                print(
                    "BATCH "
                    f"phase=guided_distillation_rep{replicate_index} "
                    f"batch={batch_index + 1}/{n_batches} "
                    f"loss={float(loss):.8g} "
                    f"elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
        histories.append(history)
        models.append(model)

    output_dir = Path(args.output_dir)
    _write_training_outputs(
        output_dir=output_dir,
        spec=spec,
        histories=histories,
        models=models,
    )
    final_losses = [history[-1]["loss_total"] for history in histories if history]
    return {
        "output_dir": str(output_dir),
        "n_replicates": n_replicates,
        "n_batches": n_batches,
        "final_loss_mean": float(np.mean(final_losses)) if final_losses else None,
        "backend": str(jax.default_backend()),
    }


def smoke_distillation_loss() -> dict[str, Any]:
    """Run a tiny loss call through the intended entry surface."""

    feedback = jnp.ones((2, 4, 6), dtype=jnp.float32)
    actions = jnp.zeros((2, 4, 2), dtype=jnp.float32)
    feedback_directions = jnp.ones((3, *feedback.shape), dtype=jnp.float32) * 0.01
    action_directions = jnp.ones((3, *actions.shape), dtype=jnp.float32) * 0.02
    perturbation_feedback = feedback.at[..., 0].add(0.1)
    perturbation_actions = actions.at[..., 1].add(0.05)

    def student_policy(obs, act):
        return 0.3 * obs[..., :2] + 0.2 * act

    def teacher_policy(obs, act):
        return 0.1 * obs[..., :2] - 0.1 * act

    loss = guided_distillation_loss(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        feedback_history=feedback,
        action_history=actions,
        perturbation_feedback_history=perturbation_feedback,
        perturbation_action_history=perturbation_actions,
        feedback_directions=feedback_directions,
        action_directions=action_directions,
        student_forced_rollout=jnp.ones((2, 4, 2), dtype=jnp.float32),
        rollout_anchor=jnp.zeros((2, 4, 2), dtype=jnp.float32),
    )
    return {
        "loss_total": float(loss.total),
        "components": {name: float(value) for name, value in loss.components.items()},
        "finite": bool(jnp.isfinite(loss.total)),
        "backend": str(jax.default_backend()),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the guided-distillation CLI parser."""

    parser = argparse.ArgumentParser(
        description="Prepare the 9727d79 guided C&S GRU distillation no-launch spec.",
    )
    parser.add_argument("--run-spec", default=None)
    parser.add_argument("--run-spec-output", default=str(DEFAULT_SPEC_PATH))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--teacher-package", default=DEFAULT_TEACHER_PACKAGE)
    parser.add_argument("--teacher-manifest", default=DEFAULT_TEACHER_MANIFEST)
    parser.add_argument("--clean-action-weight", type=float, default=1.0)
    parser.add_argument("--perturbation-response-weight", type=float, default=1.0)
    parser.add_argument("--input-output-jvp-weight", type=float, default=0.25)
    parser.add_argument("--rollout-anchor-weight", type=float, default=0.25)
    parser.add_argument("--n-jvp-directions", type=int, default=16)
    parser.add_argument("--n-batches", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller-lr", type=float, default=3e-3)
    parser.add_argument("--lr-warmup-batches", type=int, default=500)
    parser.add_argument("--lr-warmup-init-fraction", type=float, default=0.1)
    parser.add_argument("--lr-cosine-alpha", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--log-step", type=int, default=10)
    parser.add_argument("--smoke-loss", action="store_true")
    parser.add_argument("--smoke-train", action="store_true")
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_spec is not None:
        args.run_spec_output = args.run_spec
    if args.full_train:
        result = run_guided_distillation_training(args)
        print(json.dumps(result, indent=2, sort_keys=True), end="")
        return 0

    payload: dict[str, Any] = build_distillation_spec(args)
    if args.smoke_loss:
        payload["smoke_loss"] = smoke_distillation_loss()

    if args.dry_run:
        print(json.dumps({"run_spec": payload}, indent=2, sort_keys=True), end="")
        return 0

    spec_path = Path(args.run_spec_output)
    mkdir_p(spec_path.parent)
    spec_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_spec_path": str(spec_path)}, indent=2, sort_keys=True), end="")
    return 0


__all__ = [
    "BASE_RUN_ID",
    "DEFAULT_SPEC_PATH",
    "ISSUE_ID",
    "RUN_ID",
    "build_distillation_spec",
    "build_parser",
    "default_distillation_command",
    "full_train_command",
    "forcing_fraction_for_batch",
    "load_teacher_package",
    "main",
    "materialize_teacher_batch",
    "run_guided_distillation_training",
    "smoke_distillation_loss",
]
