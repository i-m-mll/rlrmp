"""CLI support for guided C&S GRU distillation run specs.

This module is intentionally a thin entry surface around
``rlrmp.train.distillation``. The reusable loss/JVP implementation lives there;
this file owns the first 6D h0 H-infinity no-launch run contract and a tiny
local smoke path that proves the CLI can call the intended loss surface.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax

from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from jax_cookbook import save as fbx_save
from jax_cookbook.tree import filter_spec_leaves
from rlrmp.paths import mkdir_p
from rlrmp.model.trainable import staged_network_trainable_parts
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
LEGACY_ACTION_HISTORY_RUN_ID = "h0_hinf_6d_guided_distillation"
RUN_ID = "h0_hinf_6d_standard_graph_distillation"
DEFAULT_SPEC_PATH = Path(f"results/{ISSUE_ID}/runs/{RUN_ID}.json")
DEFAULT_OUTPUT_DIR = f"_artifacts/{ISSUE_ID}/runs/{RUN_ID}"
DEFAULT_TEACHER_PACKAGE = "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz"
DEFAULT_TEACHER_MANIFEST = (
    "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers_manifest.json"
)
DEFAULT_CHECKPOINT_INTERVAL_BATCHES = 500
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


def run_spec_path_for(run_id: str) -> Path:
    """Return the tracked flat run-spec path for a 9727d79 run variant."""

    return Path(f"results/{ISSUE_ID}/runs/{run_id}.json")


def output_dir_for(run_id: str) -> str:
    """Return the ignored artifact directory for a 9727d79 run variant."""

    return f"_artifacts/{ISSUE_ID}/runs/{run_id}"


def _arg_value(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _resolve_run_id(args: argparse.Namespace, spec: dict[str, Any] | None = None) -> str:
    explicit = _arg_value(args, "run_id")
    if explicit:
        return str(explicit)
    if spec is not None and spec.get("run_id"):
        return str(spec["run_id"])
    run_spec_output = _arg_value(args, "run_spec_output")
    if run_spec_output:
        return Path(run_spec_output).stem
    return RUN_ID


def _resolve_run_spec_path(args: argparse.Namespace, run_id: str) -> Path:
    run_spec_output = _arg_value(args, "run_spec_output")
    if run_spec_output:
        return Path(run_spec_output)
    return run_spec_path_for(run_id)


def _resolve_output_dir(
    args: argparse.Namespace,
    run_id: str,
    spec: dict[str, Any] | None = None,
) -> str:
    output_dir = _arg_value(args, "output_dir")
    if output_dir:
        return str(output_dir)
    if spec is not None and spec.get("artifact_output_dir"):
        return str(spec["artifact_output_dir"])
    return output_dir_for(run_id)


def _spec_run_id(spec: dict[str, Any]) -> str:
    if spec.get("run_id"):
        return str(spec["run_id"])
    if spec.get("artifact_output_dir"):
        return Path(str(spec["artifact_output_dir"])).name
    return RUN_ID


def _dump_json_metadata_bytes(file: Any, hyperparameters: dict[str, Any] | None) -> None:
    file.write(json.dumps(hyperparameters, sort_keys=True).encode("utf-8") + b"\n")


def _save_pytree(path: Path, tree: Any, *, hyperparameters: dict[str, Any] | None = None) -> None:
    fbx_save(
        path,
        tree,
        hyperparameters=hyperparameters,
        dump_fn=_dump_json_metadata_bytes,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_serialized_hps(hps: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(hps)
    if normalized.get("hidden_type") == "equinox.nn._rnn.GRUCell":
        normalized["hidden_type"] = None
    return normalized


def _base_run_spec(path: str | Path = BASE_RUN_SPEC) -> dict[str, Any]:
    return _read_json(Path(path))


def _standard_hps_dict(
    *,
    base_spec_path: str | Path = BASE_RUN_SPEC,
    n_replicates: int = 5,
    hidden_size: int = 180,
    batch_size: int = 64,
    n_batches: int = 12000,
    controller_lr: float = 3e-3,
    lr_warmup_batches: int = 500,
    lr_warmup_init_fraction: float = 0.1,
    lr_cosine_alpha: float = 0.1,
    gradient_clip_norm: float = 5.0,
) -> dict[str, Any]:
    hps = _normalize_serialized_hps(_base_run_spec(base_spec_path)["hps"])
    hps["batch_size"] = int(batch_size)
    hps["n_batches_condition"] = int(n_batches)
    hps["learning_rate_0"] = float(controller_lr)
    hps["constant_lr_iterations"] = int(lr_warmup_batches)
    hps["warmup_init_fraction"] = float(lr_warmup_init_fraction)
    hps["cosine_annealing_alpha"] = float(lr_cosine_alpha)
    hps["gradient_clip_norm"] = float(gradient_clip_norm)
    model = hps.setdefault("model", {})
    model["n_replicates"] = int(n_replicates)
    model["hidden_size"] = int(hidden_size)
    population = model.setdefault("population_structure", {})
    population["n_input_only"] = 0
    population["n_readout_only"] = 0
    population["n_recurrent_only"] = 0
    population["n_input_readout"] = int(hidden_size)
    return hps


def _standard_hps_from_spec(
    spec: dict[str, Any],
    *,
    n_replicates: int,
    hidden_size: int,
    batch_size: int,
    n_batches: int,
    controller_lr: float,
    lr_warmup_batches: int,
    lr_warmup_init_fraction: float,
    lr_cosine_alpha: float,
    gradient_clip_norm: float,
) -> TreeNamespace:
    hps = spec.get("hps")
    if hps is None:
        hps = _standard_hps_dict(
            base_spec_path=spec.get("base_contract", {}).get("run_spec", BASE_RUN_SPEC),
            n_replicates=n_replicates,
            hidden_size=hidden_size,
            batch_size=batch_size,
            n_batches=n_batches,
            controller_lr=controller_lr,
            lr_warmup_batches=lr_warmup_batches,
            lr_warmup_init_fraction=lr_warmup_init_fraction,
            lr_cosine_alpha=lr_cosine_alpha,
            gradient_clip_norm=gradient_clip_norm,
        )
    else:
        hps = _normalize_serialized_hps(hps)
        hps["batch_size"] = int(batch_size)
        hps["n_batches_condition"] = int(n_batches)
        hps["learning_rate_0"] = float(controller_lr)
        hps["constant_lr_iterations"] = int(lr_warmup_batches)
        hps["warmup_init_fraction"] = float(lr_warmup_init_fraction)
        hps["cosine_annealing_alpha"] = float(lr_cosine_alpha)
        hps["gradient_clip_norm"] = float(gradient_clip_norm)
        hps["model"]["n_replicates"] = int(n_replicates)
        hps["model"]["hidden_size"] = int(hidden_size)
        population = hps["model"].setdefault("population_structure", {})
        population["n_input_only"] = 0
        population["n_readout_only"] = 0
        population["n_recurrent_only"] = 0
        population["n_input_readout"] = int(hidden_size)
    return dict_to_namespace(hps, to_type=TreeNamespace)


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
        "--resume",
    ]


def build_distillation_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Build the no-launch spec for the first 6D h0 H-infinity distillation row."""

    run_id = _resolve_run_id(args)
    run_spec_path = _resolve_run_spec_path(args, run_id)
    output_dir = _resolve_output_dir(args, run_id)
    config = cs_h0_distillation_config(
        weights=DistillationLossWeights(
            clean_action=float(args.clean_action_weight),
            perturbation_response=float(args.perturbation_response_weight),
            input_output_jvp=float(args.input_output_jvp_weight),
            student_forced_rollout_anchor=float(args.rollout_anchor_weight),
        ),
        n_jvp_directions=int(args.n_jvp_directions),
    )
    existing_spec = _read_json(run_spec_path) if run_spec_path.is_file() else {}
    hps = _standard_hps_dict(
        n_replicates=int(getattr(args, "n_replicates", 5)),
        hidden_size=int(getattr(args, "hidden_size", 180)),
        batch_size=int(getattr(args, "batch_size", 64)),
        n_batches=int(getattr(args, "n_batches", 12000)),
        controller_lr=float(getattr(args, "controller_lr", 3e-3)),
        lr_warmup_batches=int(getattr(args, "lr_warmup_batches", 500)),
        lr_warmup_init_fraction=float(getattr(args, "lr_warmup_init_fraction", 0.1)),
        lr_cosine_alpha=float(getattr(args, "lr_cosine_alpha", 0.1)),
        gradient_clip_norm=float(getattr(args, "gradient_clip_norm", 5.0)),
    )
    spec = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE_ID,
        "implementation_issue": IMPLEMENTATION_ISSUE_ID,
        "run_id": run_id,
        "seed": int(getattr(args, "seed", 0)),
        "batch_size": int(getattr(args, "batch_size", 64)),
        "n_train_batches": int(getattr(args, "n_batches", 12000)),
        "controller_lr": float(getattr(args, "controller_lr", 3e-3)),
        "artifact_output_dir": output_dir,
        "hps": hps,
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
            "artifact_output_dir": output_dir,
            "spec_command": default_distillation_command(spec_path=run_spec_path),
            "command": full_train_command(spec_path=run_spec_path),
            "full_train_status": "implemented_no_launch",
            "trainer": "rlrmp.train.guided_distillation.run_guided_distillation_training",
            "replicate_execution": "vectorized with eqx.filter_vmap over the replicate axis",
        },
        "checkpointing": {
            "enabled": bool(getattr(args, "checkpoint", True)),
            "default": "enabled",
            "interval_batches": int(
                getattr(args, "checkpoint_interval_batches", DEFAULT_CHECKPOINT_INTERVAL_BATCHES)
            ),
            "resume_flag": "--resume",
            "latest_pointer": "checkpoints/checkpoint_latest",
            "format": (
                "standard Feedbax graph checkpoints with model.eqx, optimizer_state.eqx, "
                "batch_keys.npy, loss_history.json, and metadata.json"
            ),
            "contents": [
                "model.eqx",
                "optimizer_state.eqx",
                "batch_keys.npy",
                "loss_history.json",
                "metadata.json",
            ],
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
            "student_architecture_boundary": (
                "Teacher/student action histories are supervision and JVP context only. "
                "The student Feedbax graph consumes the normal 6D controller feedback "
                "from setup_task_model_pair and no explicit previous-action input."
            ),
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
            "setup_function": "rlrmp.train.task_model.setup_task_model_pair",
            "checkpoint_format": "jax_cookbook.save/load_with_hyperparameters",
            "final_model": "trained_model.eqx",
            "checkpoint_model": "checkpoints/<checkpoint>/model.eqx",
            "controller_input_dim": 6,
            "student_action_history_input": False,
            "initial_hidden_encoder": True,
            "force_filter_feedback": True,
            "hidden_size": 180,
            "batch_size": 64,
            "n_replicates": 5,
            "vectorized_replicates": True,
            "plant_backend": "cs_lss",
            "stochastic_preset": "cs2019-rollout",
            "broad_epsilon_pgd_training": False,
        },
        "launch_ready_summary": {
            "status": "locked_preflight_no_launch",
            "requires_user_confirmation_before_billable_run": True,
            "recommended_gpu": "secure RunPod RTX 5090",
            "corrected_variant": (
                "Standard h0 Feedbax graph distillation rerun with 6 controller inputs; "
                "separate from the legacy action-history-input run."
            ),
            "output_dir": output_dir,
            "parity_table": [
                {
                    "field": "Base graph",
                    "020a65b_no_pgd": "setup_task_model_pair h0 GRU",
                    "corrected_distillation": "same",
                },
                {
                    "field": "Controller inputs",
                    "020a65b_no_pgd": "6D force-filter feedback",
                    "corrected_distillation": "same; no previous-action input",
                },
                {
                    "field": "Hidden size / replicates",
                    "020a65b_no_pgd": "180 / 5",
                    "corrected_distillation": "180 / 5",
                },
                {
                    "field": "Batch / lr / clip / schedule",
                    "020a65b_no_pgd": "64 / 3e-3 / 5 / warmup-cosine",
                    "corrected_distillation": "same",
                },
                {
                    "field": "Training batches / PGD",
                    "020a65b_no_pgd": "12000 / none",
                    "corrected_distillation": "12000 / none",
                },
                {
                    "field": "Only intended difference",
                    "020a65b_no_pgd": "standard full-Q/R/Qf rollout loss",
                    "corrected_distillation": "H-infinity teacher-guided distillation loss",
                },
            ],
        },
        "distillation_surface": {
            "config": config.summary(),
            "hidden_state_supervision": False,
            "student_action_history_input": False,
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
            "the real trainer/loss path on tiny CPU-friendly shapes with two "
            "replicates in one vectorized batch path",
            "tests/test_distillation.py covers dense-Jacobian equivalence only on "
            "tiny diagnostic maps",
            "the full-train path fails fast only for missing teacher packages or "
            "invalid specs, not as a placeholder guard",
        ],
    }
    if "post_run_provenance" in existing_spec:
        spec["post_run_provenance"] = existing_spec["post_run_provenance"]
    return spec


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


@dataclass(frozen=True)
class GuidedDistillationTrainingState:
    """Serializable state needed to resume guided distillation training."""

    model: Any
    optimizer_state: optax.OptState
    completed_batches: int
    batch_keys: jax.Array
    histories: list[list[dict[str, Any]]]


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


def _standard_model_actions(model: Any, feedback_history: jax.Array) -> jax.Array:
    net_node = model.nodes["net"]

    def single(feedback: jax.Array) -> jax.Array:
        hidden = net_node.h0_encoder(feedback[0])

        def step(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_hidden = net_node.net.hidden(value, carry)
            action = net_node.net.readout(next_hidden)
            return next_hidden, action

        _, actions = jax.lax.scan(step, hidden, feedback)
        return actions

    if feedback_history.ndim == 2:
        return single(feedback_history)
    return jax.vmap(single)(feedback_history)


def _where_train_fn(model: Any) -> tuple[Any, ...]:
    return staged_network_trainable_parts(model.nodes["net"])


def _where_train_spec(model: Any) -> Any:
    return filter_spec_leaves(model, _where_train_fn)


def _loss_for_batch(
    model: Any,
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
        _standard_model_actions(model, batch["feedback_history"])
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

    def student_policy(feedback_history: jax.Array, action_history: jax.Array) -> jax.Array:
        del action_history
        return _standard_model_actions(model, feedback_history)

    result = guided_distillation_loss(
        student_policy=student_policy,
        teacher_policy=teacher,
        feedback_history=batch["feedback_history"],
        action_history=action_context,
        config=config,
        perturbation_feedback_history=batch["perturbation_feedback_history"],
        perturbation_action_history=perturbation_context,
        feedback_directions=batch["feedback_directions"],
        action_directions=batch["action_directions"],
        student_forced_rollout=_standard_model_actions(model, batch["feedback_history"]),
        rollout_anchor=teacher_actions,
    )
    return result.total, result.components


@eqx.filter_jit
def _train_step(
    model: Any,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    where_train_spec: Any,
    batch: dict[str, jax.Array],
    config: CSH0DistillationConfig,
    student_forcing_fraction: float,
) -> tuple[Any, optax.OptState, jax.Array, dict[str, jax.Array]]:
    trainable, frozen = eqx.partition(model, where_train_spec)

    def loss_for_trainable(trainable_model: Any) -> tuple[jax.Array, dict[str, jax.Array]]:
        return _loss_for_batch(
            eqx.combine(trainable_model, frozen),
            batch,
            config,
            student_forcing_fraction=student_forcing_fraction,
        )

    (loss, components), grads = eqx.filter_value_and_grad(loss_for_trainable, has_aux=True)(
        trainable,
    )
    updates, optimizer_state = optimizer.update(
        grads,
        optimizer_state,
        trainable,
    )
    trainable = eqx.apply_updates(trainable, updates)
    model = eqx.combine(trainable, frozen)
    return model, optimizer_state, loss, components


def _init_standard_model_ensemble(
    *,
    hps: TreeNamespace,
    key: jax.Array,
) -> Any:
    import rlrmp.analysis  # noqa: F401
    from rlrmp.train.task_model import setup_task_model_pair

    return setup_task_model_pair(hps, key=key).model


def _init_optimizer_state(
    *,
    model: Any,
    optimizer: optax.GradientTransformation,
    where_train_spec: Any,
) -> optax.OptState:
    trainable, _frozen = eqx.partition(model, where_train_spec)
    return eqx.filter_vmap(optimizer.init)(trainable)


def _replicate_keys(root_key: jax.Array, *, offset: int, n_replicates: int) -> jax.Array:
    replicate_indices = jnp.arange(n_replicates, dtype=jnp.uint32)
    return jax.vmap(lambda index: jr.fold_in(root_key, index + offset))(replicate_indices)


def _materialize_replicate_batches(
    package: dict[str, jax.Array],
    *,
    keys: jax.Array,
    batch_size: int,
    horizon: int,
    n_jvp_directions: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    split_keys = jax.vmap(lambda key: jr.split(key, 2))(keys)
    next_keys = split_keys[:, 0]
    materialize_keys = split_keys[:, 1]
    batches = eqx.filter_vmap(
        lambda key: materialize_teacher_batch(
            package,
            key=key,
            batch_size=batch_size,
            horizon=horizon,
            n_jvp_directions=n_jvp_directions,
        )
    )(materialize_keys)
    return next_keys, batches


@eqx.filter_jit
def _batched_train_step(
    models: Any,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    where_train_spec: Any,
    batches: dict[str, jax.Array],
    config: CSH0DistillationConfig,
    student_forcing_fraction: float,
) -> tuple[Any, optax.OptState, jax.Array, dict[str, jax.Array]]:
    return eqx.filter_vmap(
        lambda model, state, batch: _train_step(
            model,
            state,
            optimizer,
            where_train_spec,
            batch,
            config,
            student_forcing_fraction,
        )
    )(models, optimizer_state, batches)


def _single_replicate_model(
    models: Any,
    *,
    replicate_index: int,
    n_replicates: int,
) -> Any:
    def select_leaf(leaf: Any) -> Any:
        if eqx.is_array(leaf) and leaf.ndim > 0 and int(leaf.shape[0]) == n_replicates:
            return leaf[replicate_index]
        return leaf

    return jt.map(select_leaf, models)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(_json_dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def latest_checkpoint_path(checkpoint_root: Path) -> Path:
    """Return the path used by the durable latest-checkpoint contract."""

    return checkpoint_root / "checkpoint_latest"


def _atomic_latest_link(checkpoint_root: Path, checkpoint_name: str) -> None:
    latest = latest_checkpoint_path(checkpoint_root)
    tmp_link = checkpoint_root / ".checkpoint_latest.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(checkpoint_name, tmp_link)
    os.replace(tmp_link, latest)


def _checkpoint_metadata(
    *,
    state: GuidedDistillationTrainingState,
    args: argparse.Namespace,
    spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_VERSION}.checkpoint.v1",
        "issue": ISSUE_ID,
        "run_id": _spec_run_id(spec),
        "completed_batches": int(state.completed_batches),
        "n_batches": int(args.n_batches),
        "n_replicates": int(args.n_replicates),
        "batch_size": int(args.batch_size),
        "hidden_size": int(args.hidden_size),
        "horizon": int(args.horizon),
        "n_jvp_directions": int(args.n_jvp_directions),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "seed": int(args.seed),
        "vectorized_replicates": True,
        "replicate_execution": "eqx.filter_vmap",
        "batch_keys_path": "batch_keys.npy",
        "loss_history_path": "loss_history.json",
        "model_path": "model.eqx",
        "optimizer_state_path": "optimizer_state.eqx",
        "run_spec": spec,
    }


def save_training_checkpoint(
    checkpoint_root: Path,
    state: GuidedDistillationTrainingState,
    *,
    args: argparse.Namespace,
    spec: dict[str, Any],
) -> Path:
    """Write a numbered checkpoint and atomically repoint ``checkpoint_latest``."""

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"checkpoint_{state.completed_batches:07d}"
    target = checkpoint_root / checkpoint_name
    tmp = checkpoint_root / f".{checkpoint_name}.tmp"
    if tmp.exists():
        _remove_tree(tmp)
    tmp.mkdir(parents=True)

    eqx.tree_serialise_leaves(tmp / "model.eqx", state.model)
    eqx.tree_serialise_leaves(tmp / "optimizer_state.eqx", state.optimizer_state)
    np.save(tmp / "batch_keys.npy", np.asarray(jax.device_get(state.batch_keys)))
    _atomic_write_json(tmp / "loss_history.json", state.histories)
    _atomic_write_json(
        tmp / "metadata.json", _checkpoint_metadata(state=state, args=args, spec=spec)
    )
    if target.exists():
        _remove_tree(target)
    os.replace(tmp, target)
    _atomic_latest_link(checkpoint_root, checkpoint_name)
    _atomic_write_json(
        checkpoint_root / "checkpoint_index.json",
        {
            "latest": checkpoint_name,
            "latest_path": str(latest_checkpoint_path(checkpoint_root)),
            "completed_batches": int(state.completed_batches),
            "n_replicates": int(args.n_replicates),
            "vectorized_replicates": True,
        },
    )
    return target


def load_latest_checkpoint(
    checkpoint_root: Path,
    *,
    model_template: Any,
    optimizer_state_template: optax.OptState,
    n_replicates: int,
) -> GuidedDistillationTrainingState:
    """Load ``checkpoint_latest`` using explicit batched model and optimizer templates."""

    checkpoint_path = latest_checkpoint_path(checkpoint_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_latest found under {checkpoint_root}")
    metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata["n_replicates"]) != int(n_replicates):
        raise ValueError(
            f"Checkpoint {checkpoint_path} has n_replicates={metadata['n_replicates']}; "
            f"expected {n_replicates}."
        )
    model = eqx.tree_deserialise_leaves(checkpoint_path / "model.eqx", model_template)
    optimizer_state = eqx.tree_deserialise_leaves(
        checkpoint_path / "optimizer_state.eqx",
        optimizer_state_template,
    )
    batch_keys = jnp.asarray(np.load(checkpoint_path / "batch_keys.npy"), dtype=jnp.uint32)
    histories = json.loads((checkpoint_path / "loss_history.json").read_text(encoding="utf-8"))
    return GuidedDistillationTrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=int(metadata["completed_batches"]),
        batch_keys=batch_keys,
        histories=histories,
    )


def _write_training_outputs(
    *,
    output_dir: Path,
    spec: dict[str, Any],
    histories: list[list[dict[str, Any]]],
    model: Any,
    completed_batches: int,
    requested_batches: int,
    checkpoint_root: Path,
    checkpoint_enabled: bool,
    latest_checkpoint: Path | None,
) -> None:
    mkdir_p(output_dir)
    _atomic_write_json(output_dir / "run_spec_snapshot.json", spec)
    _atomic_write_json(output_dir / "loss_history.json", histories)
    final_losses = [history[-1]["loss_total"] for history in histories if history]
    summary = {
        "schema_version": "rlrmp.guided_distillation.training_summary.v1",
        "issue": ISSUE_ID,
        "run_id": _spec_run_id(spec),
        "n_replicates": len(histories),
        "n_batches": completed_batches,
        "completed_batches": completed_batches,
        "requested_batches": requested_batches,
        "vectorized_replicates": True,
        "checkpointing": {
            "enabled": checkpoint_enabled,
            "checkpoint_root": str(checkpoint_root),
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
        },
        "final_loss_mean": float(np.mean(final_losses)) if final_losses else None,
        "final_loss_min": float(np.min(final_losses)) if final_losses else None,
        "final_loss_max": float(np.max(final_losses)) if final_losses else None,
        "artifacts": {
            "loss_history": "loss_history.json",
            "run_spec_snapshot": "run_spec_snapshot.json",
            "trained_model": "trained_model.eqx",
        },
    }
    _atomic_write_json(output_dir / "training_summary.json", summary)
    _save_pytree(output_dir / "trained_model.eqx", model, hyperparameters=spec)


def run_guided_distillation_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run the executable analytical-teacher guided-distillation trainer."""

    spec = build_distillation_spec(args)
    if _arg_value(args, "run_spec") is not None:
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
        n_replicates = min(n_replicates, 2)
        hidden_size = min(hidden_size, 8)
        horizon = min(horizon, 6)
        n_jvp_directions = min(n_jvp_directions, 2)

    if n_batches <= 0 or batch_size <= 0 or n_replicates <= 0:
        raise ValueError("n_batches, batch_size, and n_replicates must be positive.")
    checkpoint_interval_batches = int(args.checkpoint_interval_batches)
    checkpoint_enabled = bool(args.checkpoint)
    if checkpoint_enabled and checkpoint_interval_batches < 1:
        raise ValueError("--checkpoint-interval-batches must be positive when checkpointing.")
    stop_after_batches = None if args.stop_after_batches is None else int(args.stop_after_batches)
    if stop_after_batches is not None:
        if stop_after_batches < 1:
            raise ValueError("--stop-after-batches must be positive when provided.")
        if stop_after_batches > n_batches:
            raise ValueError("--stop-after-batches cannot exceed --n-batches.")
    target_batches = n_batches if stop_after_batches is None else stop_after_batches
    effective_args = argparse.Namespace(**vars(args))
    effective_args.n_batches = n_batches
    effective_args.batch_size = batch_size
    effective_args.n_replicates = n_replicates
    effective_args.hidden_size = hidden_size
    effective_args.horizon = horizon
    effective_args.n_jvp_directions = n_jvp_directions
    effective_args.checkpoint_interval_batches = checkpoint_interval_batches

    package = load_teacher_package(args.teacher_package)
    teacher_horizon = int(package["hinf_controller_gains"].shape[0])
    if horizon > teacher_horizon:
        raise ValueError(f"Requested horizon {horizon} exceeds teacher horizon {teacher_horizon}.")
    feedback_dim = int(package["observation_matrix"].shape[0])
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
    model_key, _unused = jr.split(root_key)
    batch_keys = _replicate_keys(root_key, offset=10_000, n_replicates=n_replicates)
    hps = _standard_hps_from_spec(
        spec,
        n_replicates=n_replicates,
        hidden_size=hidden_size,
        batch_size=batch_size,
        n_batches=n_batches,
        controller_lr=float(args.controller_lr),
        lr_warmup_batches=int(args.lr_warmup_batches),
        lr_warmup_init_fraction=float(args.lr_warmup_init_fraction),
        lr_cosine_alpha=float(args.lr_cosine_alpha),
        gradient_clip_norm=float(args.gradient_clip_norm),
    )
    model = _init_standard_model_ensemble(hps=hps, key=model_key)
    model_feedback_dim = int(model.nodes["net"].net.hidden.weight_ih.shape[-1])
    if feedback_dim != model_feedback_dim:
        raise ValueError(
            f"Teacher package feedback_dim={feedback_dim}, but the standard Feedbax graph "
            f"expects {model_feedback_dim} controller feedback channels."
        )
    where_train_spec = _where_train_spec(model)
    optimizer_state = _init_optimizer_state(
        model=model,
        optimizer=optimizer,
        where_train_spec=where_train_spec,
    )
    histories: list[list[dict[str, Any]]] = [[] for _ in range(n_replicates)]
    output_dir = Path(_resolve_output_dir(args, _spec_run_id(spec), spec))
    checkpoint_root = output_dir / "checkpoints"
    state = GuidedDistillationTrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=0,
        batch_keys=batch_keys,
        histories=histories,
    )
    resumed_from: Path | None = None
    if args.resume and latest_checkpoint_path(checkpoint_root).exists():
        state = load_latest_checkpoint(
            checkpoint_root,
            model_template=model,
            optimizer_state_template=optimizer_state,
            n_replicates=n_replicates,
        )
        resumed_from = latest_checkpoint_path(checkpoint_root)
        if state.completed_batches > n_batches:
            raise ValueError(
                f"Latest checkpoint completed {state.completed_batches} batches, "
                f"but --n-batches is {n_batches}."
            )

    started = time.perf_counter()
    log_step = max(1, int(args.log_step))
    latest_checkpoint: Path | None = (
        latest_checkpoint_path(checkpoint_root)
        if checkpoint_enabled and latest_checkpoint_path(checkpoint_root).exists()
        else None
    )
    for batch_index in range(state.completed_batches, target_batches):
        next_batch_keys, batches = _materialize_replicate_batches(
            package,
            keys=state.batch_keys,
            batch_size=batch_size,
            horizon=horizon,
            n_jvp_directions=n_jvp_directions,
        )
        forcing_fraction = forcing_fraction_for_batch(spec, batch_index)
        model, optimizer_state, losses, components = _batched_train_step(
            state.model,
            state.optimizer_state,
            optimizer,
            where_train_spec,
            batches,
            config,
            forcing_fraction,
        )
        losses_host = np.asarray(jax.device_get(losses))
        components_host = {
            name: np.asarray(jax.device_get(value)) for name, value in components.items()
        }
        for replicate_index in range(n_replicates):
            state.histories[replicate_index].append(
                {
                    "replicate": replicate_index,
                    "batch": batch_index + 1,
                    "student_forcing_fraction": forcing_fraction,
                    "loss_total": float(losses_host[replicate_index]),
                    "components": {
                        name: float(value[replicate_index])
                        for name, value in components_host.items()
                    },
                }
            )
        state = GuidedDistillationTrainingState(
            model=model,
            optimizer_state=optimizer_state,
            completed_batches=batch_index + 1,
            batch_keys=next_batch_keys,
            histories=state.histories,
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
                "phase=guided_distillation_vectorized "
                f"batch={batch_index + 1}/{n_batches} "
                f"loss={float(np.mean(losses_host)):.8g} "
                f"replicates={n_replicates} "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        if checkpoint_enabled and (
            state.completed_batches % checkpoint_interval_batches == 0
            or state.completed_batches == n_batches
            or state.completed_batches == target_batches
        ):
            latest_checkpoint = save_training_checkpoint(
                checkpoint_root,
                state,
                args=effective_args,
                spec=spec,
            )

    _write_training_outputs(
        output_dir=output_dir,
        spec=spec,
        histories=state.histories,
        model=state.model,
        completed_batches=state.completed_batches,
        requested_batches=n_batches,
        checkpoint_root=checkpoint_root,
        checkpoint_enabled=checkpoint_enabled,
        latest_checkpoint=latest_checkpoint,
    )
    final_losses = [history[-1]["loss_total"] for history in state.histories if history]
    return {
        "output_dir": str(output_dir),
        "n_replicates": n_replicates,
        "n_batches": state.completed_batches,
        "completed_batches": state.completed_batches,
        "requested_batches": n_batches,
        "vectorized_replicates": True,
        "checkpointing": checkpoint_enabled,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
        "resumed_from": str(resumed_from) if resumed_from is not None else None,
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
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--run-spec", default=None)
    parser.add_argument("--run-spec-output", default=None)
    parser.add_argument("--output-dir", default=None)
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
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-interval-batches",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL_BATCHES,
    )
    parser.add_argument("--stop-after-batches", type=int, default=None)
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

    spec_path = _resolve_run_spec_path(args, _resolve_run_id(args, payload))
    mkdir_p(spec_path.parent)
    spec_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_spec_path": str(spec_path)}, indent=2, sort_keys=True), end="")
    return 0


__all__ = [
    "BASE_RUN_ID",
    "DEFAULT_SPEC_PATH",
    "ISSUE_ID",
    "LEGACY_ACTION_HISTORY_RUN_ID",
    "RUN_ID",
    "build_distillation_spec",
    "build_parser",
    "default_distillation_command",
    "full_train_command",
    "forcing_fraction_for_batch",
    "load_teacher_package",
    "main",
    "materialize_teacher_batch",
    "output_dir_for",
    "run_guided_distillation_training",
    "run_spec_path_for",
    "smoke_distillation_loss",
]
