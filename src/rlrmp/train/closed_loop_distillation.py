"""Preflight surface for closed-loop extLQG distillation into the h0 GRU.

This module owns the issue a378b34 run/spec contract. It deliberately does not
reuse the older guided teacher-feedback-bank trainer: full training must happen
through a Feedbax closed-loop rollout where student actions update the plant
state that the student later observes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from rlrmp.train.distillation import batched_directional_jvps

SCHEMA_VERSION = "rlrmp.closed_loop_distillation.preflight.v1"
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
BASE_RUN_ID = (
    "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64"
)
BASE_RUN_SPEC = f"results/{BASE_ISSUE_ID}/runs/{BASE_RUN_ID}/run.json"


@dataclass(frozen=True)
class ClosedLoopLossWeights:
    """Weights for pure closed-loop extLQG distillation components."""

    kinematics_trajectory: float = 1.0
    velocity: float = 1.0
    endpoint: float = 1.0
    settling: float = 0.5
    action_force_trajectory: float = 1.0
    perturbation_response_trajectory: float = 1.0
    directional_input_output_jvp: float = 0.25
    task_qr_rollout: float = 0.0

    def summary(self) -> dict[str, float]:
        """Return a JSON-serializable weight summary."""

        return asdict(self)


class ClosedLoopTrainingUnavailableError(RuntimeError):
    """Raised when a caller tries to launch before the closed-loop hook exists."""


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
    """Return the intended full-train command once the Feedbax hook exists."""

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
        "--resume",
    ]


def _arg_value(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_closed_loop_distillation_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Build the no-launch spec for the pure closed-loop extLQG row."""

    run_id = str(_arg_value(args, "run_id", RUN_ID))
    run_spec_path = Path(_arg_value(args, "run_spec_output", run_spec_path_for(run_id)))
    output_dir = str(_arg_value(args, "output_dir", output_dir_for(run_id)))
    trainable_dtype = str(_arg_value(args, "trainable_dtype", DEFAULT_TRAINABLE_DTYPE))
    n_jvp_directions = int(_arg_value(args, "n_jvp_directions", 16))
    weights = ClosedLoopLossWeights(
        kinematics_trajectory=float(_arg_value(args, "kinematics_trajectory_weight", 1.0)),
        velocity=float(_arg_value(args, "velocity_weight", 1.0)),
        endpoint=float(_arg_value(args, "endpoint_weight", 1.0)),
        settling=float(_arg_value(args, "settling_weight", 0.5)),
        action_force_trajectory=float(_arg_value(args, "action_force_weight", 1.0)),
        perturbation_response_trajectory=float(
            _arg_value(args, "perturbation_response_weight", 1.0)
        ),
        directional_input_output_jvp=float(_arg_value(args, "input_output_jvp_weight", 0.25)),
        task_qr_rollout=float(_arg_value(args, "task_rollout_loss_weight", 0.0)),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE_ID,
        "tracker_issue": TRACKER_ISSUE_ID,
        "umbrella_issue": UMBRELLA_ISSUE_ID,
        "prior_guided_issue": PRIOR_GUIDED_ISSUE_ID,
        "guided_jvp_issue": GUIDED_JVP_ISSUE_ID,
        "run_id": run_id,
        "seed": int(_arg_value(args, "seed", 0)),
        "artifact_output_dir": output_dir,
        "launch_status": "preflight_only_not_launched",
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
            "intended_full_train_command": full_train_command(spec_path=run_spec_path),
            "full_train_status": "fail_closed_pending_feedbax_closed_loop_hook",
            "failure_guard": (
                "run_closed_loop_distillation_training raises "
                "ClosedLoopTrainingUnavailableError until the differentiable Feedbax "
                "closed-loop rollout hook is implemented."
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
            "matched_inputs": [
                "seeds",
                "trial specifications",
                "initial conditions",
                "targets",
                "process and observation noise",
                "perturbation schedule",
            ],
            "feedback_basis": "target_relative_delayed_feedback_plus_force_filter",
        },
        "closed_loop_semantics": {
            "student_rollout": "normal_feedbax_plant_task_surface",
            "student_actions_feed_future_observations": True,
            "teacher_forced_feedback_bank_imitation": False,
            "old_guided_trainer_is_main_path": False,
            "required_hook": (
                "Differentiate a Feedbax rollout where the student action at t drives the "
                "plant transition and the resulting state determines later observations."
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
                "velocity_endpoint_settling": {
                    "enabled": any(
                        weight > 0.0
                        for weight in (weights.velocity, weights.endpoint, weights.settling)
                    ),
                    "velocity_weight": weights.velocity,
                    "endpoint_weight": weights.endpoint,
                    "settling_weight": weights.settling,
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
                    "n_directions": n_jvp_directions,
                    "implementation": (
                        "jax.linearize plus jax.vmap over directional probes; dense "
                        "Jacobian materialization is forbidden in the hot path"
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
        },
        "local_acceptance_checks": [
            "JSON run spec parses and validates with rlrmp.train.closed_loop_distillation.",
            "Dry-run smoke checks directional-JVP code path without dense Jacobians.",
            "--full-train fails closed until the Feedbax closed-loop hook is implemented.",
        ],
    }


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
    if entry.get("full_train_status") != "fail_closed_pending_feedbax_closed_loop_hook":
        raise ValueError("Full-train entry must fail closed until the Feedbax hook lands.")


def smoke_directional_jvp() -> dict[str, Any]:
    """Run a tiny directional-JVP smoke without materializing dense Jacobians."""

    feedback = jnp.arange(12, dtype=jnp.float32).reshape(2, 1, 6) / 10.0
    actions = jnp.zeros((2, 1, 2), dtype=jnp.float32)
    feedback_dirs = jnp.ones((3, *feedback.shape), dtype=jnp.float32) * 0.01
    action_dirs = jnp.zeros((3, *actions.shape), dtype=jnp.float32)

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
        "implementation": "directional_jvp_vmap",
    }


def write_run_spec(path: Path, spec: dict[str, Any]) -> None:
    """Write a stable, sorted JSON run spec."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_closed_loop_distillation_training(*, spec: dict[str, Any]) -> None:
    """Fail closed until the Feedbax differentiable closed-loop hook is present."""

    validate_run_spec(spec)
    raise ClosedLoopTrainingUnavailableError(
        "Full closed-loop extLQG distillation is not implemented in this preflight pass. "
        "The required Feedbax hook must roll out the standard h0 graph through the plant "
        "so student actions feed future plant states and observations. Refusing to fall "
        "back to the older teacher-feedback-bank/action/JVP imitation trainer."
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--trainable-dtype", default=DEFAULT_TRAINABLE_DTYPE)
    parser.add_argument("--kinematics-trajectory-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--settling-weight", type=float, default=0.5)
    parser.add_argument("--action-force-weight", type=float, default=1.0)
    parser.add_argument("--perturbation-response-weight", type=float, default=1.0)
    parser.add_argument("--input-output-jvp-weight", type=float, default=0.25)
    parser.add_argument("--task-rollout-loss-weight", type=float, default=0.0)
    parser.add_argument("--n-jvp-directions", type=int, default=16)
    parser.add_argument("--write-run-spec", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-preflight", action="store_true")
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
    if args.full_train:
        try:
            run_closed_loop_distillation_training(spec=spec)
        except ClosedLoopTrainingUnavailableError as exc:
            print(
                json.dumps(
                    {
                        "error": "closed_loop_training_unavailable",
                        "message": str(exc),
                        "run_spec": spec,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
    if args.dry_run or args.smoke_preflight or args.write_run_spec:
        payload: dict[str, Any] = {"run_spec": spec}
        if smoke is not None:
            payload["smoke_preflight"] = smoke
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
