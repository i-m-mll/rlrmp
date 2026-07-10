"""Preflight surface for closed-loop extLQG distillation into the h0 GRU.

This module owns the issue a378b34 run/spec contract. It deliberately does not
reuse the older guided teacher-feedback-bank trainer: full training must happen
through a Feedbax closed-loop rollout where student actions update the plant
state that the student later observes.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.objectives.loss import AbstractLoss, TermTree
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.model.trainable import staged_network_trainable_parts
from rlrmp.runtime.training_run_specs import (
    CLOSED_LOOP_DISTILLATION_METHOD_REF,
    attach_distillation_training_specs,
    training_arg_parser,
    validate_distillation_training_run_spec,
    write_distillation_run_spec,
)
from rlrmp.train.distillation_native.losses import batched_directional_jvps
from rlrmp.train.training_configs import ClosedLoopDistillationConfig

SCHEMA_VERSION = "rlrmp.closed_loop_distillation.training_entry.v1"
ISSUE_ID = "a378b34"
TRACKER_ISSUE_ID = "7792ef1"
UMBRELLA_ISSUE_ID = "40e1911"
PRIOR_GUIDED_ISSUE_ID = "9727d79"
GUIDED_JVP_ISSUE_ID = "c314267"
BASE_ISSUE_ID = "020a65b"
TEACHER_ISSUE_ID = "376d023"

RUN_ID = "h0_extlqg_6d_closed_loop_distillation"
DEFAULT_SPEC_PATH = Path(f"results/{ISSUE_ID}/runs/{RUN_ID}.json")
DEFAULT_OUTPUT_DIR = f"_artifacts/{ISSUE_ID}/runs/{RUN_ID}"
DEFAULT_TEACHER_PACKAGE = "_artifacts/376d023/analytical_teachers/6d_output_feedback_teachers.npz"
DEFAULT_TEACHER_GAINS_KEY = "extlqg_controller_gains"
DEFAULT_TRAINABLE_DTYPE = "float32"
DEFAULT_CHECKPOINT_INTERVAL_BATCHES = 500
BASE_RUN_ID = (
    "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64"
)
BASE_RUN_SPEC = f"results/{BASE_ISSUE_ID}/runs/{BASE_RUN_ID}/run.json"


@dataclass(frozen=True)
class ClosedLoopLossWeights:
    """Weights for pure closed-loop extLQG distillation components."""

    kinematics_trajectory: float = 1.0
    velocity: float = 1.0
    endpoint: float = 0.0
    settling: float = 0.0
    action_force_trajectory: float = 1.0
    perturbation_response_trajectory: float = 1.0
    directional_input_output_jvp: float = 0.25
    task_qr_rollout: float = 0.0

    def summary(self) -> dict[str, float]:
        """Return a JSON-serializable weight summary."""

        return asdict(self)


class FullTrainingApprovalRequiredError(RuntimeError):
    """Raised when full training is requested without explicit launch approval."""


class ExtLQGClosedLoopReference(eqx.Module):
    """Analytical closed-loop extLQG reference over shared observable channels."""

    plant_a: jax.Array
    plant_b: jax.Array
    controller_gains: jax.Array
    observation_matrix: jax.Array
    feedback_gains: jax.Array
    state_dim: int = eqx.field(static=True)

    @classmethod
    def from_package(
        cls,
        path: str | Path,
        *,
        teacher_gains_key: str = DEFAULT_TEACHER_GAINS_KEY,
    ) -> "ExtLQGClosedLoopReference":
        """Load the extLQG package produced by the analytical-teacher lane."""

        package_path = Path(path)
        if not package_path.is_file():
            raise FileNotFoundError(
                f"Teacher package not found at {package_path}. Sync or materialize "
                f"[issue:{TEACHER_ISSUE_ID}] before training a378b34."
            )
        arrays = np.load(package_path)
        required = ("plant_A", "plant_B", "observation_matrix", teacher_gains_key)
        missing = [key for key in required if key not in arrays.files]
        if missing:
            raise ValueError(
                f"Teacher package {package_path} is missing required keys: {', '.join(missing)}."
            )
        plant_a = jnp.asarray(arrays["plant_A"], dtype=jnp.float32)
        plant_b = jnp.asarray(arrays["plant_B"], dtype=jnp.float32)
        controller_gains = jnp.asarray(arrays[teacher_gains_key], dtype=jnp.float32)
        observation_matrix = jnp.asarray(arrays["observation_matrix"], dtype=jnp.float32)
        if plant_a.ndim != 2 or plant_a.shape[0] != plant_a.shape[1]:
            raise ValueError("teacher plant_A must be square.")
        if plant_b.shape != (plant_a.shape[0], 2):
            raise ValueError("teacher plant_B must have shape (state_dim, 2).")
        if controller_gains.shape[1:] != (2, plant_a.shape[0]):
            raise ValueError(f"{teacher_gains_key} must have shape (time, 2, {plant_a.shape[0]}).")
        feedback_pinv = jnp.linalg.pinv(observation_matrix)
        feedback_gains = -jnp.einsum("tus,sf->tuf", controller_gains, feedback_pinv)
        return cls(
            plant_a=plant_a,
            plant_b=plant_b,
            controller_gains=controller_gains,
            observation_matrix=observation_matrix,
            feedback_gains=feedback_gains,
            state_dim=int(plant_a.shape[0]),
        )

    def rollout(
        self,
        *,
        initial_vector: jax.Array,
        target_pos: jax.Array,
        n_steps: int,
    ) -> dict[str, jax.Array]:
        """Roll the analytical controller from trial starts in target-centered coordinates."""

        x0 = self._initial_teacher_state(initial_vector, target_pos)
        indices = jnp.clip(jnp.arange(int(n_steps)), 0, self.controller_gains.shape[0] - 1)
        gains = self.controller_gains[indices]

        def step(state: jax.Array, gain: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
            action = -jnp.einsum("us,...s->...u", gain, state)
            next_state = state @ self.plant_a.T + action @ self.plant_b.T
            return next_state, (next_state, action)

        _, (relative_states_t, actions_t) = jax.lax.scan(step, x0, gains)
        relative_states = jnp.moveaxis(relative_states_t, 0, -2)
        actions = jnp.moveaxis(actions_t, 0, -2)
        target = jnp.expand_dims(target_pos, axis=-2)
        return {
            "position": relative_states[..., 0:2] + target,
            "velocity": relative_states[..., 2:4],
            "force_filter": relative_states[..., 4:6],
            "action": actions,
        }

    def feedback_policy(self, feedback_history: jax.Array) -> jax.Array:
        """Approximate local 6D feedback-to-action teacher map for JVP matching."""

        n_steps = int(feedback_history.shape[-2])
        indices = jnp.clip(jnp.arange(n_steps), 0, self.feedback_gains.shape[0] - 1)
        gains = self.feedback_gains[indices]
        return jnp.einsum("tuf,...tf->...tu", gains, feedback_history)

    def _initial_teacher_state(self, initial_vector: jax.Array, target_pos: jax.Array) -> jax.Array:
        initial_vector = jnp.asarray(initial_vector, dtype=jnp.float32)
        target_pos = jnp.asarray(target_pos, dtype=jnp.float32)
        batch_shape = jnp.broadcast_shapes(initial_vector.shape[:-1], target_pos.shape[:-1])
        initial_vector = jnp.broadcast_to(initial_vector, (*batch_shape, initial_vector.shape[-1]))
        target_pos = jnp.broadcast_to(target_pos, (*batch_shape, 2))
        teacher_state = jnp.zeros((*batch_shape, self.state_dim), dtype=initial_vector.dtype)
        shared = min(6, self.state_dim, initial_vector.shape[-1])
        teacher_state = teacher_state.at[..., :shared].set(initial_vector[..., :shared])
        if self.state_dim >= 2:
            teacher_state = teacher_state.at[..., 0:2].set(initial_vector[..., 0:2] - target_pos)
        return teacher_state


class ClosedLoopDistillationLoss(AbstractLoss):
    """Feedbax loss for pure closed-loop extLQG distillation.

    The loss is called by Feedbax after the normal closed-loop
    rollout. The analytical reference is rolled from the same batched trial
    initial states and targets; matching happens on shared observable channels
    rather than pretending the 36D teacher and 48D student state bases are the
    same object.
    """

    reference: ExtLQGClosedLoopReference
    weights: ClosedLoopLossWeights = eqx.field(static=True)
    label: str = eqx.field(default="closed_loop_extlqg_distillation", static=True)

    @jax.named_scope("rlrmp.ClosedLoopDistillationLoss")
    def __call__(self, states: Any, trial_specs: Any, model: Any) -> TermTree:
        components = closed_loop_distillation_components(
            states,
            trial_specs,
            model,
            reference=self.reference,
        )
        weights = self.weights.summary()
        leaves = {
            name: TermTree.leaf(name, value).with_weight(weights[name])
            for name, value in components.items()
            if weights.get(name, 0.0) > 0.0
        }
        if self.weights.task_qr_rollout > 0.0:
            raise ValueError(
                "task_qr_rollout is intentionally not mixed into the first pure "
                "closed-loop distillation row."
            )
        return TermTree.branch(self.label, leaves, originator=self)

    def skeleton(self, batch_dims: tuple[int, ...]) -> TermTree:
        weights = self.weights.summary()
        leaves = {
            name: TermTree.leaf(name, jnp.empty(batch_dims)).with_weight(weight)
            for name, weight in weights.items()
            if weight > 0.0 and name != "task_qr_rollout"
        }
        return TermTree.branch(self.label, leaves, originator=self)


def run_spec_path_for(run_id: str = RUN_ID) -> Path:
    """Return the tracked flat run-spec path for an a378b34 run variant."""

    return Path(f"results/{ISSUE_ID}/runs/{run_id}.json")


def output_dir_for(run_id: str = RUN_ID) -> str:
    """Return the ignored artifact directory for an a378b34 run variant."""

    return f"_artifacts/{ISSUE_ID}/runs/{run_id}"


def default_preflight_command(*, spec_path: Path = DEFAULT_SPEC_PATH) -> list[str]:
    """Return the no-launch command that writes and validates the run spec."""

    return [
        "env",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/train_closed_loop_distillation.py",
        "--run-spec-output",
        str(spec_path),
        "--write-run-spec",
        "--dry-run",
        "--smoke-preflight",
    ]


def full_train_command(*, spec_path: Path = DEFAULT_SPEC_PATH) -> list[str]:
    """Return the intended full-train command after user launch approval."""

    return [
        "env",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/train_closed_loop_distillation.py",
        "--run-spec",
        str(spec_path),
        "--full-train",
        "--confirm-full-train",
        "--resume",
    ]


def smoke_train_command(*, spec_path: Path = DEFAULT_SPEC_PATH) -> list[str]:
    """Return the local CPU smoke command for native closed-loop training."""

    return [
        "env",
        "PYTHONPATH=src",
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/train_closed_loop_distillation.py",
        "--run-spec",
        str(spec_path),
        "--smoke-train",
        "--smoke-n-batches",
        "1",
        "--smoke-batch-size",
        "1",
    ]


def _arg_value(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_closed_loop_distillation_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Build the no-launch spec for the pure closed-loop extLQG row."""

    args = argparse.Namespace(
        **ClosedLoopDistillationConfig.model_validate(vars(args)).model_dump(mode="python")
    )

    run_id = str(_arg_value(args, "run_id", RUN_ID))
    run_spec_path = Path(_arg_value(args, "run_spec_output", run_spec_path_for(run_id)))
    output_dir = str(_arg_value(args, "output_dir", output_dir_for(run_id)))
    trainable_dtype = str(_arg_value(args, "trainable_dtype", DEFAULT_TRAINABLE_DTYPE))
    weights = ClosedLoopLossWeights(
        kinematics_trajectory=float(_arg_value(args, "kinematics_trajectory_weight", 1.0)),
        velocity=float(_arg_value(args, "velocity_weight", 1.0)),
        endpoint=float(_arg_value(args, "endpoint_weight", 0.0)),
        settling=float(_arg_value(args, "settling_weight", 0.0)),
        action_force_trajectory=float(_arg_value(args, "action_force_weight", 1.0)),
        perturbation_response_trajectory=float(
            _arg_value(args, "perturbation_response_weight", 1.0)
        ),
        directional_input_output_jvp=float(_arg_value(args, "input_output_jvp_weight", 0.25)),
        task_qr_rollout=float(_arg_value(args, "task_rollout_loss_weight", 0.0)),
    )
    spec = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE_ID,
        "tracker_issue": TRACKER_ISSUE_ID,
        "umbrella_issue": UMBRELLA_ISSUE_ID,
        "prior_guided_issue": PRIOR_GUIDED_ISSUE_ID,
        "guided_jvp_issue": GUIDED_JVP_ISSUE_ID,
        "run_id": run_id,
        "method_ref": CLOSED_LOOP_DISTILLATION_METHOD_REF,
        "seed": int(_arg_value(args, "seed", 0)),
        "user_confirmed": bool(_arg_value(args, "user_confirmed", False)),
        "artifact_output_dir": output_dir,
        "launch_status": "implemented_no_launch_pending_user_approval",
        "no_launch_boundary": (
            "This spec does not authorize RunPod, Modal, GPU acquisition, full training, "
            "push, or protected-branch auth."
        ),
        "training_entry": {
            "script": "scripts/train_closed_loop_distillation.py",
            "module": "rlrmp.train.closed_loop_distillation",
            "run_spec_path": str(run_spec_path),
            "artifact_output_dir": output_dir,
            "preflight_command": default_preflight_command(spec_path=run_spec_path),
            "smoke_train_command": smoke_train_command(spec_path=run_spec_path),
            "intended_full_train_command": full_train_command(spec_path=run_spec_path),
            "full_train_status": "implemented_no_launch_pending_user_approval",
            "failure_guard": (
                "native closed-loop distillation training requires explicit "
                "--confirm-full-train approval before a non-smoke full run."
            ),
            "trainer_path": (
                "rlrmp.train.distillation_native.execute_distillation_training_run_spec_native"
            ),
        },
        "student_contract": {
            "setup_function": "rlrmp.train.task_model.setup_task_model_pair",
            "graph_source": "standard h0 Feedbax GraphSpec via setup_task_model_pair",
            "feedback_input_basis": "target_relative_delayed_feedback_plus_force_filter",
            "controller_input_dim": 6,
            "force_filter_feedback": True,
            "initial_hidden_encoder": True,
            "hidden_size": int(_arg_value(args, "hidden_size", 180)),
            "n_replicates": int(_arg_value(args, "n_replicates", 5)),
            "batch_size": int(_arg_value(args, "batch_size", 64)),
            "n_train_batches": int(_arg_value(args, "n_batches", 12000)),
            "controller_lr": float(_arg_value(args, "controller_lr", 3e-3)),
            "lr_schedule": "warmup_cosine",
            "lr_warmup_batches": int(_arg_value(args, "lr_warmup_batches", 500)),
            "lr_cosine_alpha": float(_arg_value(args, "lr_cosine_alpha", 0.01)),
            "gradient_clip_norm": float(_arg_value(args, "gradient_clip_norm", 5.0)),
            "broad_epsilon_pgd_training": False,
            "trainable_dtype": trainable_dtype,
        },
        "teacher_contract": {
            "issue": TEACHER_ISSUE_ID,
            "controller": "6d_output_feedback_extlqg",
            "teacher_package": str(_arg_value(args, "teacher_package", DEFAULT_TEACHER_PACKAGE)),
            "teacher_gains_key": str(
                _arg_value(args, "teacher_gains_key", DEFAULT_TEACHER_GAINS_KEY)
            ),
            "required_package_key": DEFAULT_TEACHER_GAINS_KEY,
            "horizon": 60,
            "matched_inputs": [
                "seeds",
                "trial specifications",
                "initial conditions",
                "targets",
                "process and observation noise",
                "perturbation schedule",
            ],
            "feedback_basis": "target_relative_delayed_feedback_plus_force_filter",
            "reference_basis_note": (
                "The extLQG package is 36D while the standard h0 student state is 48D; "
                "the training loss matches shared observable behavior: position, velocity, "
                "command/action, force-filter, perturbation response, and the full local "
                "2x6 feedback-to-action Jacobian computed by coordinate-basis JVPs."
            ),
        },
        "closed_loop_semantics": {
            "student_rollout": "normal_feedbax_tasktrainer_train_pair_run_component",
            "student_actions_feed_future_observations": True,
            "teacher_forced_feedback_bank_imitation": False,
            "old_guided_trainer_is_main_path": False,
            "trainer_contract": (
                "Use the native Feedbax executor closed-loop rollout; student actions are "
                "generated by the standard h0 graph and feed future plant "
                "state/observations through the normal task surface."
            ),
        },
        "base_contract": {
            "issue": BASE_ISSUE_ID,
            "run_id": BASE_RUN_ID,
            "run_spec": BASE_RUN_SPEC,
            "inherit": [
                "h0 encoder",
                "6D force-filter feedback input",
                "target-relative multitarget task distribution",
                "hidden size 180",
                "5 replicates",
                "batch size 64",
                "AdamW learning rate 3e-3",
                "gradient clip 5",
                "warmup/cosine schedule alpha 0.01",
                "12000 train batches",
                "no PGD adversarial inner maximizer",
            ],
        },
        "loss_surface": {
            "weights": weights.summary(),
            "default_task_qr_rollout_loss": "off",
            "task_qr_rollout_loss_can_be_enabled_later": True,
            "components": {
                "closed_loop_kinematics_trajectory": {
                    "enabled": weights.kinematics_trajectory > 0.0,
                    "weight": weights.kinematics_trajectory,
                },
                "velocity_trajectory": {
                    "enabled": weights.velocity > 0.0,
                    "weight": weights.velocity,
                },
                "action_force_trajectory": {
                    "enabled": weights.action_force_trajectory > 0.0,
                    "weight": weights.action_force_trajectory,
                },
                "perturbation_response_trajectory": {
                    "enabled": weights.perturbation_response_trajectory > 0.0,
                    "weight": weights.perturbation_response_trajectory,
                },
                "directional_input_output_jvp": {
                    "enabled": weights.directional_input_output_jvp > 0.0,
                    "weight": weights.directional_input_output_jvp,
                    "basis": "full_6d_coordinate_basis",
                    "jacobian_shape": [2, 6],
                    "implementation": (
                        "jax.jvp plus jax.vmap over the six coordinate-basis directions "
                        "of the controller-visible feedback input, yielding the full "
                        "local 2x6 feedback-to-action Jacobian without dense Jacobian "
                        "materialization in the training path"
                    ),
                },
                "task_qr_rollout": {
                    "enabled": weights.task_qr_rollout > 0.0,
                    "weight": weights.task_qr_rollout,
                },
            },
        },
        "expected_artifacts": {
            "tracked_run_spec": str(run_spec_path),
            "bulk_output_dir": output_dir,
            "training_summary": f"{output_dir}/training_summary.json",
            "checkpoints": f"{output_dir}/checkpoints/",
            "final_model": f"{output_dir}/trained_model.eqx",
            "post_run_notes": f"results/{ISSUE_ID}/notes/{run_id}.md",
        },
        "checkpointing": {
            "enabled": True,
            "interval_batches": int(
                _arg_value(args, "checkpoint_interval_batches", DEFAULT_CHECKPOINT_INTERVAL_BATCHES)
            ),
            "resume_flag": "--resume",
            "latest_pointer": f"{output_dir}/checkpoints/checkpoint_latest",
            "format": "Feedbax graph checkpoint plus training summary and locked run spec",
        },
        "execution_target": {
            "cloud": "RunPod secure cloud",
            "gpu": "RTX 5090 preferred if available, otherwise secure RTX 4090",
            "billable_launch_authorized": False,
            "requires_explicit_user_confirmation": True,
        },
        "stopping_and_debug": {
            "stop_on": [
                "NaN/Inf loss",
                "teacher package/reference shape mismatch",
                "smoke-train failure",
                "no position/velocity trajectory improvement in early monitored losses",
            ],
            "monitoring": [
                "first JIT/compile completion",
                "BATCH progress lines or trainer iteration losses",
                "component losses for position, velocity, action/force, response, local Jacobian",
            ],
        },
        "post_run_analyses": [
            "closed-loop position and velocity summaries versus extLQG",
            "action/force trajectory mismatch",
            "perturbation-response trajectory mismatch",
            "full local 2x6 feedback-to-action Jacobian mismatch",
            "standard certificate-style post-run diagnostics where applicable",
        ],
        "locked_spec_summary": {
            "teacher": "6D extLQG analytical controller from 376d023 package",
            "graph_contract": "standard h0 setup_task_model_pair Feedbax graph",
            "n_batches": int(_arg_value(args, "n_batches", 12000)),
            "batch_size": int(_arg_value(args, "batch_size", 64)),
            "n_replicates": int(_arg_value(args, "n_replicates", 5)),
            "seed": int(_arg_value(args, "seed", 0)),
            "dtype": trainable_dtype,
        },
        "local_acceptance_checks": [
            "JSON run spec parses and validates with rlrmp.train.closed_loop_distillation.",
            "Dry-run smoke checks full local-Jacobian basis-JVP code path.",
            "--smoke-train exercises native closed-loop training with a custom loss.",
            "--full-train requires explicit user launch approval, not a Feedbax hook blocker.",
        ],
    }
    return attach_distillation_training_specs(
        spec,
        method="closed_loop_distillation",
        output_dir=Path(output_dir),
        spec_path=run_spec_path,
    )


def validate_run_spec(spec: dict[str, Any]) -> None:
    """Validate invariants that prevent accidental fallback to the old trainer."""

    if spec.get("issue") != ISSUE_ID:
        raise ValueError(f"Expected issue {ISSUE_ID}, got {spec.get('issue')!r}.")
    if spec.get("run_id") != RUN_ID:
        raise ValueError(f"Expected run_id {RUN_ID}, got {spec.get('run_id')!r}.")
    if spec.get("artifact_output_dir") != DEFAULT_OUTPUT_DIR:
        raise ValueError("Closed-loop distillation artifact output path drifted.")
    teacher = spec.get("teacher_contract", {})
    if teacher.get("teacher_gains_key") != DEFAULT_TEACHER_GAINS_KEY:
        raise ValueError("The first a378b34 row must use extlqg_controller_gains.")
    student = spec.get("student_contract", {})
    if student.get("controller_input_dim") != 6 or not student.get("force_filter_feedback"):
        raise ValueError("Student contract must use the 6D force-filter feedback input.")
    semantics = spec.get("closed_loop_semantics", {})
    if not semantics.get("student_actions_feed_future_observations"):
        raise ValueError("Spec must enforce closed-loop student action/state coupling.")
    if semantics.get("teacher_forced_feedback_bank_imitation"):
        raise ValueError("Teacher-forced feedback-bank imitation is forbidden for a378b34.")
    entry = spec.get("training_entry", {})
    if entry.get("full_train_status") != "implemented_no_launch_pending_user_approval":
        raise ValueError("Full-train entry must remain pending explicit user launch approval.")
    trainer_path = str(entry.get("trainer_path", ""))
    if "execute_distillation_training_run_spec_native" not in trainer_path:
        raise ValueError("Training entry must use the native distillation executor.")
    validate_distillation_training_run_spec(spec, method="closed_loop_distillation")


def _base_run_spec(path: str | Path = BASE_RUN_SPEC) -> dict[str, Any]:
    return _read_json(Path(path))


def _normalize_serialized_hps(hps: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(hps)
    if normalized.get("hidden_type") == "equinox.nn._rnn.GRUCell":
        normalized["hidden_type"] = None
    if "intervention_scaleup_batches" not in normalized:
        normalized["intervention_scaleup_batches"] = [0, 0]
    pgd = normalized.get("broad_epsilon_pgd_training")
    if isinstance(pgd, dict) and not pgd.get("enabled", False):
        budget_contract = pgd.get("budget_contract")
        if isinstance(budget_contract, dict) and budget_contract.get("effective_l2_radius_15cm"):
            budget_contract.setdefault(
                "budget_source",
                {
                    "key": "disabled_closed_loop_distillation_no_pgd",
                    "note": (
                        "Closed-loop distillation disables PGD; retained radius is provenance only."
                    ),
                },
            )
    return normalized


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
    trainable_dtype: str,
) -> TreeNamespace:
    hps = spec.get("hps")
    if hps is None:
        hps = _normalize_serialized_hps(
            _base_run_spec(spec.get("base_contract", {}).get("run_spec", BASE_RUN_SPEC))["hps"]
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
    model = hps.setdefault("model", {})
    model["n_replicates"] = int(n_replicates)
    model["hidden_size"] = int(hidden_size)
    population = model.setdefault("population_structure", {})
    population["n_input_only"] = 0
    population["n_readout_only"] = 0
    population["n_recurrent_only"] = 0
    population["n_input_readout"] = int(hidden_size)
    model["trainable_dtype"] = str(trainable_dtype)
    return dict_to_namespace(hps, to_type=TreeNamespace)


def _where_train_fn(model: Any) -> tuple[Any, ...]:
    return staged_network_trainable_parts(model.nodes["net"])


def _target_position(trial_specs: Any, states: Any) -> jax.Array:
    target_spec = trial_specs.targets.get("mechanics.effector.pos", None)
    if target_spec is None or not hasattr(target_spec, "value"):
        pos = jnp.asarray(states.mechanics.effector.pos)
        return jnp.zeros((*pos.shape[:-2], 2), dtype=pos.dtype)
    target_value = jnp.asarray(target_spec.value, dtype=jnp.float32)
    return target_value[..., -1, :]


def _initial_vector(trial_specs: Any, states: Any) -> jax.Array:
    vector = jnp.asarray(states.mechanics.vector, dtype=jnp.float32)
    if "mechanics.vector" in trial_specs.inits:
        initial = jnp.asarray(trial_specs.inits["mechanics.vector"], dtype=vector.dtype)
        return jnp.broadcast_to(initial, (*vector.shape[:-2], vector.shape[-1]))
    return jnp.zeros((*vector.shape[:-2], vector.shape[-1]), dtype=vector.dtype)


def _per_trial_mse(diff: jax.Array) -> jax.Array:
    diff = jnp.asarray(diff, dtype=jnp.float32)
    if diff.ndim <= 1:
        return jnp.mean(jnp.square(diff))
    return jnp.mean(jnp.square(diff), axis=tuple(range(1, diff.ndim)))


def _last_window(values: jax.Array, width: int = 10) -> jax.Array:
    n_time = int(values.shape[-2])
    start = max(0, n_time - int(width))
    return values[..., start:, :]


def _coordinate_feedback_directions(feedback_history: jax.Array) -> jax.Array:
    """Return full coordinate-basis directions for the local 6D feedback input."""

    feedback_history = jnp.asarray(feedback_history, dtype=jnp.float32)
    feedback_dim = int(feedback_history.shape[-1])
    basis = jnp.eye(feedback_dim, dtype=feedback_history.dtype)
    direction_shape = (feedback_dim, *feedback_history.shape)
    return jnp.broadcast_to(
        basis.reshape((feedback_dim, *([1] * (feedback_history.ndim - 1)), feedback_dim)),
        direction_shape,
    )


def _model_feedback_policy(model: Any, feedback_history: jax.Array) -> jax.Array:
    """Run the standard h0 controller on a controller-visible feedback history."""

    net_node = model.nodes["net"]

    def single(feedback: jax.Array) -> jax.Array:
        hidden = net_node.h0_encoder(feedback[0])

        def step(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_hidden = net_node.net.hidden(value, carry)
            action = net_node.net.readout(next_hidden)
            return next_hidden, action

        _, actions = jax.lax.scan(step, hidden, feedback)
        return actions

    return jax.vmap(single)(feedback_history)


def _model_local_feedback_jvps(model: Any, feedback_history: jax.Array) -> jax.Array:
    """Return full local feedback-to-action Jacobian columns by basis JVPs.

    The local map is the per-step controller update ``feedback_t -> action_t``
    with the recurrent carry entering that step held fixed. The returned tensor
    has shape ``(feedback_dim, batch, time, action_dim)``.
    """

    net_node = model.nodes["net"]
    feedback_history = jnp.asarray(feedback_history, dtype=jnp.float32)
    feedback_dim = int(feedback_history.shape[-1])
    basis = jnp.eye(feedback_dim, dtype=feedback_history.dtype)

    def sequence_jvps(feedback: jax.Array) -> jax.Array:
        hidden0 = net_node.h0_encoder(feedback[0])

        def collect_pre_hidden(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_hidden = net_node.net.hidden(value, carry)
            return next_hidden, carry

        _, hidden_before = jax.lax.scan(collect_pre_hidden, hidden0, feedback)

        def step_jvps(hidden: jax.Array, value: jax.Array) -> jax.Array:
            def action_for_feedback(local_feedback: jax.Array) -> jax.Array:
                return net_node.net.readout(net_node.net.hidden(local_feedback, hidden))

            return jax.vmap(
                lambda direction: jax.jvp(action_for_feedback, (value,), (direction,))[1]
            )(basis)

        time_major = jax.vmap(step_jvps)(hidden_before, feedback)
        return jnp.moveaxis(time_major, 0, 1)

    return jax.vmap(sequence_jvps)(feedback_history).transpose(1, 0, 2, 3)


def _teacher_local_feedback_jvps(
    reference: ExtLQGClosedLoopReference,
    feedback_history: jax.Array,
) -> jax.Array:
    """Return full local teacher feedback-to-action Jacobian columns."""

    n_steps = int(feedback_history.shape[-2])
    feedback_dim = int(feedback_history.shape[-1])
    indices = jnp.clip(jnp.arange(n_steps), 0, reference.feedback_gains.shape[0] - 1)
    gains = reference.feedback_gains[indices]
    columns = jnp.moveaxis(gains[:, :, :feedback_dim], -1, 0)
    return jnp.broadcast_to(
        columns[:, None, :, :], (feedback_dim, feedback_history.shape[0], n_steps, 2)
    )


def _full_local_jacobian_component(
    *,
    model: Any,
    reference: ExtLQGClosedLoopReference,
    feedback_history: jax.Array,
) -> jax.Array:
    if feedback_history.shape[-1] != reference.observation_matrix.shape[0]:
        return jnp.zeros(feedback_history.shape[0], dtype=jnp.float32)
    student_jvps = _model_local_feedback_jvps(model, feedback_history)
    teacher_jvps = _teacher_local_feedback_jvps(reference, feedback_history)
    return jnp.mean(jnp.square(student_jvps - teacher_jvps), axis=(0, 2, 3))


def closed_loop_distillation_components(
    states: Any,
    trial_specs: Any,
    model: Any,
    *,
    reference: ExtLQGClosedLoopReference,
) -> dict[str, jax.Array]:
    """Compute unweighted per-trial closed-loop distillation components."""

    pos = jnp.asarray(states.mechanics.effector.pos, dtype=jnp.float32)
    vel = jnp.asarray(states.mechanics.effector.vel, dtype=jnp.float32)
    vector = jnp.asarray(states.mechanics.vector, dtype=jnp.float32)
    command = jnp.asarray(states.net.output, dtype=jnp.float32)
    target_pos = _target_position(trial_specs, states)
    initial_vector = _initial_vector(trial_specs, states)
    teacher = reference.rollout(
        initial_vector=initial_vector,
        target_pos=target_pos,
        n_steps=int(pos.shape[-2]),
    )
    force_filter = vector[..., 4:6]
    feedback_history = jnp.asarray(states.net.input, dtype=jnp.float32)
    return {
        "kinematics_trajectory": _per_trial_mse(pos - teacher["position"]),
        "velocity": _per_trial_mse(vel - teacher["velocity"]),
        "endpoint": _per_trial_mse(pos[..., -1, :] - teacher["position"][..., -1, :]),
        "settling": _per_trial_mse(_last_window(pos) - _last_window(teacher["position"]))
        + _per_trial_mse(_last_window(vel) - _last_window(teacher["velocity"])),
        "action_force_trajectory": _per_trial_mse(command - teacher["action"])
        + _per_trial_mse(force_filter - teacher["force_filter"]),
        "perturbation_response_trajectory": _per_trial_mse(
            (pos - pos[..., :1, :]) - (teacher["position"] - teacher["position"][..., :1, :])
        ),
        "directional_input_output_jvp": _full_local_jacobian_component(
            model=model,
            reference=reference,
            feedback_history=feedback_history,
        ),
    }


def build_closed_loop_loss(
    spec: dict[str, Any],
    *,
    reference: ExtLQGClosedLoopReference | None = None,
) -> ClosedLoopDistillationLoss:
    """Build the custom Feedbax loss from the run-spec weights."""

    weights = spec.get("loss_surface", {}).get("weights", {})
    teacher = spec["teacher_contract"]
    reference = reference or ExtLQGClosedLoopReference.from_package(
        teacher["teacher_package"],
        teacher_gains_key=teacher["teacher_gains_key"],
    )
    return ClosedLoopDistillationLoss(
        reference=reference,
        weights=ClosedLoopLossWeights(
            kinematics_trajectory=float(weights.get("kinematics_trajectory", 1.0)),
            velocity=float(weights.get("velocity", 1.0)),
            endpoint=float(weights.get("endpoint", 1.0)),
            settling=float(weights.get("settling", 0.5)),
            action_force_trajectory=float(weights.get("action_force_trajectory", 1.0)),
            perturbation_response_trajectory=float(
                weights.get("perturbation_response_trajectory", 1.0)
            ),
            directional_input_output_jvp=float(weights.get("directional_input_output_jvp", 0.25)),
            task_qr_rollout=float(weights.get("task_qr_rollout", 0.0)),
        ),
    )


def build_closed_loop_trainer(spec: dict[str, Any], *, n_batches: int | None = None) -> None:
    """Fail closed for the retired legacy closed-loop trainer path."""

    del spec, n_batches
    raise RuntimeError(
        "Closed-loop distillation no longer constructs the retired Feedbax trainer; "
        "use run_closed_loop_distillation_training_native or inject an executor-backed "
        "training function in tests."
    )


def _training_hps_from_spec(
    spec: dict[str, Any],
    *,
    n_batches: int | None = None,
    batch_size: int | None = None,
    n_replicates: int | None = None,
    hidden_size: int | None = None,
) -> TreeNamespace:
    student = spec["student_contract"]
    return _standard_hps_from_spec(
        spec,
        n_replicates=int(n_replicates or student["n_replicates"]),
        hidden_size=int(hidden_size or student["hidden_size"]),
        batch_size=int(batch_size or student["batch_size"]),
        n_batches=int(n_batches or student["n_train_batches"]),
        controller_lr=float(student["controller_lr"]),
        lr_warmup_batches=int(student["lr_warmup_batches"]),
        lr_warmup_init_fraction=0.1,
        lr_cosine_alpha=float(student["lr_cosine_alpha"]),
        gradient_clip_norm=float(student["gradient_clip_norm"]),
        trainable_dtype=str(student["trainable_dtype"]),
    )


def smoke_directional_jvp() -> dict[str, Any]:
    """Run a tiny full-local-Jacobian smoke using coordinate-basis JVPs."""

    feedback = jnp.arange(12, dtype=jnp.float32).reshape(2, 1, 6) / 10.0
    actions = jnp.zeros((2, 1, 2), dtype=jnp.float32)
    feedback_dirs = _coordinate_feedback_directions(feedback)
    action_dirs = jnp.zeros((6, *actions.shape), dtype=jnp.float32)

    def policy(feedback_history: jax.Array, action_history: jax.Array) -> jax.Array:
        del action_history
        return feedback_history[..., :2] * 0.5 + feedback_history[..., 4:6] * 0.25

    jvps = batched_directional_jvps(
        policy,
        feedback,
        actions,
        feedback_dirs,
        action_dirs,
    )
    return {
        "finite": bool(jnp.all(jnp.isfinite(jvps))),
        "shape": list(jvps.shape),
        "implementation": "full_local_jacobian_basis_jvp_vmap",
    }


def write_run_spec(path: Path, spec: dict[str, Any]) -> None:
    """Write a stable, sorted JSON run spec."""

    write_distillation_run_spec(path, spec, method="closed_loop_distillation")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_closed_loop_distillation_training_native(
    *,
    spec: dict[str, Any],
    args: argparse.Namespace,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run closed-loop distillation through Feedbax's native training executor."""

    validate_run_spec(spec)
    if not smoke and not bool(args.confirm_full_train):
        raise FullTrainingApprovalRequiredError(
            "Full closed-loop extLQG distillation is wired through the native "
            "Feedbax training executor but requires explicit user launch approval. "
            "No Feedbax hook blocker is being asserted."
        )
    train_batches = int(
        args.smoke_n_batches if smoke else spec["student_contract"]["n_train_batches"]
    )
    train_batch_size = int(
        args.smoke_batch_size if smoke else spec["student_contract"]["batch_size"]
    )
    train_replicates = int(1 if smoke else spec["student_contract"]["n_replicates"])
    train_hidden = int(6 if smoke else spec["student_contract"]["hidden_size"])
    if train_batches <= 0 or train_batch_size <= 0 or train_replicates <= 0:
        raise ValueError("n_batches, batch_size, and n_replicates must be positive.")

    source_spec = copy.deepcopy(spec)
    output_dir = mkdir_p(Path(str(source_spec.get("artifact_output_dir", args.output_dir))))
    source_spec["artifact_output_dir"] = str(output_dir)
    source_spec["student_contract"] = {
        **source_spec["student_contract"],
        "n_train_batches": train_batches,
        "batch_size": train_batch_size,
        "n_replicates": train_replicates,
        "hidden_size": train_hidden,
    }
    source_spec["checkpointing"] = {
        **source_spec["checkpointing"],
        "interval_batches": int(args.checkpoint_interval_batches),
    }
    source_spec["training_entry"] = {
        **source_spec["training_entry"],
        "full_train_status": "native_executor_implemented_no_launch_pending_user_approval",
        "trainer_path": (
            "rlrmp.train.distillation_native.execute_distillation_training_run_spec_native"
        ),
    }
    spec_path = Path(args.run_spec) if args.run_spec is not None else Path(args.run_spec_output)
    native_spec = attach_distillation_training_specs(
        source_spec,
        method="closed_loop_distillation",
        output_dir=output_dir,
        spec_path=spec_path,
    )
    from rlrmp.train.distillation_native.executor import (
        execute_distillation_training_run_spec_native,
    )

    run_hash = hashlib.sha256(str(output_dir.resolve()).encode()).hexdigest()[:8]
    started = time.perf_counter()
    execution = execute_distillation_training_run_spec_native(
        native_spec,
        method="closed_loop_distillation",
        run_id=f"{source_spec['run_id']}-{run_hash}",
        manifest_root=REPO_ROOT / "_artifacts" / "feedbax_runs",
        checkpoint_root=output_dir / "checkpoints",
        resume=bool(args.resume),
        manifest_conflict_policy="reuse-identical",
        issues=[ISSUE_ID, TRACKER_ISSUE_ID],
    )
    duration = time.perf_counter() - started
    result = {
        "run_id": source_spec["run_id"],
        "mode": "smoke_train" if smoke else "full_train",
        "trainer_path": (
            "rlrmp.train.distillation_native.execute_distillation_training_run_spec_native"
        ),
        "native_executor": "feedbax.training.executor.execute_training_run_spec",
        "completed_batches": int(execution.final_slots["completed_batches"]),
        "batch_size": train_batch_size,
        "n_replicates": train_replicates,
        "hidden_size": train_hidden,
        "final_loss_mean": float(execution.final_slots["train_loss"]),
        "output_dir": str(output_dir),
        "latest_checkpoint": str(output_dir / "checkpoints" / "latest.json"),
        "training_manifest_path": str(execution.manifest_path),
        "training_duration_seconds": duration,
    }
    _write_json(output_dir / "run_spec_snapshot.json", native_spec)
    _write_json(output_dir / "training_summary.json", result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = training_arg_parser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--run-spec", type=Path)
    parser.add_argument("--run-spec-output", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--teacher-package", default=DEFAULT_TEACHER_PACKAGE)
    parser.add_argument("--teacher-gains-key", default=DEFAULT_TEACHER_GAINS_KEY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-batches", type=int, default=12000)
    parser.add_argument("--controller-lr", type=float, default=3e-3)
    parser.add_argument("--lr-warmup-batches", type=int, default=500)
    parser.add_argument("--lr-cosine-alpha", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--checkpoint-interval-batches",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL_BATCHES,
    )
    parser.add_argument("--trainable-dtype", default=DEFAULT_TRAINABLE_DTYPE)
    parser.add_argument("--kinematics-trajectory-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-weight", type=float, default=0.0)
    parser.add_argument("--settling-weight", type=float, default=0.0)
    parser.add_argument("--action-force-weight", type=float, default=1.0)
    parser.add_argument("--perturbation-response-weight", type=float, default=1.0)
    parser.add_argument("--input-output-jvp-weight", type=float, default=0.25)
    parser.add_argument("--task-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--write-run-spec", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-preflight", action="store_true")
    parser.add_argument("--smoke-train", action="store_true")
    parser.add_argument("--smoke-n-batches", type=int, default=1)
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--confirm-full-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--user-confirmed", action="store_true")
    parser.set_defaults(**ClosedLoopDistillationConfig().model_dump(mode="python"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for preflight and fail-closed training commands."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    spec = (
        _read_json(args.run_spec)
        if args.run_spec is not None
        else build_closed_loop_distillation_spec(args)
    )
    validate_run_spec(spec)
    smoke = smoke_directional_jvp() if args.smoke_preflight else None
    if args.write_run_spec:
        write_run_spec(args.run_spec_output, spec)
    train_result = None
    if args.smoke_train:
        train_result = run_closed_loop_distillation_training_native(
            spec=spec,
            args=args,
            smoke=True,
        )
    if args.full_train:
        try:
            train_result = run_closed_loop_distillation_training_native(
                spec=spec,
                args=args,
            )
        except FullTrainingApprovalRequiredError as exc:
            print(
                json.dumps(
                    {
                        "error": "full_training_requires_user_approval",
                        "message": str(exc),
                        "run_spec": spec,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
    if (
        args.dry_run
        or args.smoke_preflight
        or args.write_run_spec
        or args.smoke_train
        or args.full_train
    ):
        payload: dict[str, Any] = {"run_spec": spec}
        if smoke is not None:
            payload["smoke_preflight"] = smoke
        if train_result is not None:
            payload["smoke_train" if args.smoke_train else "training"] = train_result
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
