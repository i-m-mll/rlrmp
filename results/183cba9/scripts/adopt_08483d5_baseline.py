"""Adopt the 08483d5 legacy C&S baseline into Feedbax checkpoint custody."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr


REPO = Path(__file__).resolve().parents[3]
FEEDBAX_TOOL = Path(
    "/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax"
)
LEGACY_FEEDBAX_API = Path(
    "/Users/mll/Main/10 Projects/10 PhD/20 Feedbax/feedbax/"
    "worktrees/feature__f9a8524-migration-trainability"
)
LEGACY_COMMIT = "9f919c65e52b0042181d615d4a40e1cc6fab5d0b"
LEGACY_SPEC = REPO / "results/08483d5/runs/h0_6d_no_pgd_const_band16_cpu.json"
LEGACY_CHECKPOINT = (
    REPO
    / "_artifacts/08483d5/runs/h0_6d_no_pgd_const_band16_cpu/checkpoints/checkpoint_0012000"
)
MANIFEST_PATH = LEGACY_CHECKPOINT / "leaf_manifest.json"
RUN_SPEC_PATH = REPO / "results/3cd018b/runs/ramp3500_to1000.json"
FEEDBAX_SPEC_PATH = (
    REPO / "results/3cd018b/runs/ramp3500_to1000/feedbax_training_run_spec.json"
)
CURRENT_SLOTS_SUMMARY_PATH = REPO / "results/3cd018b/notes/current_slot_summary.json"
ADOPTION_CONTEXT_PATH = REPO / "results/183cba9/notes/adoption_context.json"
ADOPTION_RESULT_PATH = REPO / "results/3cd018b/notes/legacy_baseline_adoption.json"
CHECKPOINT_ROOT = REPO / "_artifacts/3cd018b/runs/ramp3500_to1000/checkpoints"

COMPLETED_BATCHES = 12000
TARGET_N_TRAIN_BATCHES = 16500
SMOKE_STOP_AFTER_BATCHES = 12500


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--skip-adopt", action="store_true")
    args = parser.parse_args()

    _write_current_run_spec()
    current_slots = _build_current_slots()
    if not args.skip_manifest:
        _dump_leaf_manifest()
    if not args.skip_adopt:
        _adopt_checkpoint(current_slots)
    _write_adoption_summary()
    return 0


def _write_current_run_spec() -> None:
    from rlrmp.train.cs_nominal_gru import build_parser, write_run_spec, _args_values_from_run_spec

    run_spec = json.loads(LEGACY_SPEC.read_text(encoding="utf-8"))
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in _args_values_from_run_spec(run_spec).items():
        setattr(args, key, value)
    overrides = {
        "output_dir": "_artifacts/3cd018b/runs/ramp3500_to1000",
        "spec_dir": "results/3cd018b/runs/ramp3500_to1000",
        "issue": "3cd018b",
        "n_train_batches": TARGET_N_TRAIN_BATCHES,
        "broad_epsilon_pgd_training": True,
        "broad_epsilon_pgd_objective": "soft_energy",
        "broad_epsilon_pgd_energy_lambda": 281032999.21861446,
        "broad_epsilon_pgd_inner_optimizer_method": "adam",
        "broad_epsilon_pgd_adam_lr": 2e-5,
        "broad_epsilon_pgd_steps": 12,
        "adaptive_epsilon_curriculum": True,
        "adaptive_epsilon_damage_start": 0.0,
        "adaptive_epsilon_damage_peak": 3500.0,
        "adaptive_epsilon_damage_final": 1000.0,
        "adaptive_epsilon_damage_ramp_batches": 2500,
        "adaptive_epsilon_damage_anneal_batches": 5000,
        "adaptive_epsilon_update_interval_batches": 50,
        "adaptive_epsilon_ema_alpha": 0.1,
        "adaptive_epsilon_eta": 0.1,
        "adaptive_epsilon_deadband_frac": 0.1,
        "adaptive_epsilon_max_log_step": 0.1,
        "adaptive_epsilon_outer_weight_start": 0.0,
        "adaptive_epsilon_outer_weight_final": 1.0,
        "adaptive_epsilon_outer_weight_ramp_batches": 2500,
        "checkpoint_interval_batches": 500,
        "log_step": 100,
        "resume": True,
        "stop_after_batches": SMOKE_STOP_AFTER_BATCHES,
        "dry_run": False,
        "full_train": True,
    }
    for key, value in overrides.items():
        setattr(args, key, value)
    result = write_run_spec(args)
    payload = json.loads(Path(result["run_spec_path"]).read_text(encoding="utf-8"))
    FEEDBAX_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEEDBAX_SPEC_PATH.write_text(
        json.dumps(payload["feedbax_training_run_spec"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ADOPTION_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADOPTION_CONTEXT_PATH.write_text(
        json.dumps(
            {
                "legacy_commit": LEGACY_COMMIT,
                "legacy_spec": _rel(LEGACY_SPEC),
                "legacy_checkpoint": _rel(LEGACY_CHECKPOINT),
                "manifest": _rel(MANIFEST_PATH),
                "target_n_train_batches": TARGET_N_TRAIN_BATCHES,
                "completed_batches": COMPLETED_BATCHES,
                "stop_after_batches": SMOKE_STOP_AFTER_BATCHES,
                "continuation_batches": SMOKE_STOP_AFTER_BATCHES - COMPLETED_BATCHES,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_current_slots() -> dict[str, Any]:
    from rlrmp.train.cs_nominal_gru import (
        _args_values_from_run_spec,
        _build_trainer,
        _initial_adaptive_epsilon_state,
        _initial_adaptive_epsilon_zero_guard,
        _initial_training_state,
        _where_train,
        build_hps,
        build_parser,
        setup_task_model_pair,
    )
    from rlrmp.train.adaptive_epsilon_native import _adaptive_state_slot, _json_slot
    from rlrmp.train.executor.initial_slots import split_initial_keys
    from rlrmp.train.executor.slots import (
        ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES,
        ADAPTIVE_EPSILON_STATE,
        COMPLETED_BATCHES as COMPLETED_BATCHES_SLOT,
        DAMAGE_METRIC,
        EPSILON_SCALE,
        HISTORY_CHUNK_BYTES,
        MODEL,
        OBJECTIVE,
        OPTIMIZER,
        PRNG,
        TRAIN_LOSS,
        ZERO_ADVERSARY_GUARD,
    )

    run_spec = json.loads(RUN_SPEC_PATH.read_text(encoding="utf-8"))
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in _args_values_from_run_spec(run_spec).items():
        setattr(args, key, value)
    hps = build_hps(args)
    key_init, key_train, _key_adversary = split_initial_keys(jr.PRNGKey(int(args.seed)))
    metadata = json.loads((LEGACY_CHECKPOINT / "metadata.json").read_text(encoding="utf-8"))
    next_prng_key = jnp.asarray(metadata["next_prng_key"], dtype=jnp.uint32)
    pair = setup_task_model_pair(hps, key=key_init)
    trainer = _build_trainer(hps)
    state = _initial_training_state(
        model=pair.model,
        trainer=trainer,
        where_train=_where_train()[0],
        key=key_train,
    )
    adaptive_state = _initial_adaptive_epsilon_state(
        hps,
        schedule_start_batch=COMPLETED_BATCHES,
    )
    zero_guard = _initial_adaptive_epsilon_zero_guard(enabled=True)
    slots = {
        MODEL: state.model,
        OPTIMIZER: state.optimizer_state,
        PRNG: next_prng_key,
        COMPLETED_BATCHES_SLOT: jnp.asarray(COMPLETED_BATCHES, dtype=jnp.int32),
        ADAPTIVE_EPSILON_STATE: _adaptive_state_slot(adaptive_state),
        ZERO_ADVERSARY_GUARD: _json_slot(zero_guard),
        OBJECTIVE: None,
        TRAIN_LOSS: 0.0,
        DAMAGE_METRIC: 0.0,
        EPSILON_SCALE: 0.0,
        HISTORY_CHUNK_BYTES: b"",
        ADAPTIVE_EPSILON_DIAGNOSTICS_BYTES: b"",
    }
    CURRENT_SLOTS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_SLOTS_SUMMARY_PATH.write_text(
        json.dumps(
            {
                "slots": sorted(slots),
                "completed_batches": COMPLETED_BATCHES,
                "adaptive_epsilon_state_json": adaptive_state.to_json()
                if adaptive_state is not None
                else None,
                "optimizer_diagnostic_target_batches": TARGET_N_TRAIN_BATCHES,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return slots


def _dump_leaf_manifest() -> None:
    env = os.environ.copy()
    site_packages = next((FEEDBAX_TOOL / ".venv/lib").glob("python*/site-packages"))
    env["PYTHONPATH"] = (
        str(REPO / "results/183cba9/scripts")
        + os.pathsep
        + str(LEGACY_FEEDBAX_API)
        + os.pathsep
        + str(site_packages)
    )
    env["UV_PROJECT_ENVIRONMENT"] = str(FEEDBAX_TOOL / ".venv")
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "feedbax",
        "adopt-legacy-checkpoint",
        "dump-manifest",
        "--repo",
        str(REPO),
        "--commit",
        LEGACY_COMMIT,
        "--spec",
        str(LEGACY_SPEC),
        "--builder",
        "legacy_checkpoint_builders:cs_nominal_gru_model_optimizer",
        "--output",
        str(MANIFEST_PATH),
        "--skip-uv-sync",
    ]
    _run(command, cwd=FEEDBAX_TOOL, env=env)


def _adopt_checkpoint(current_slots: dict[str, Any]) -> None:
    if str(FEEDBAX_TOOL) not in sys.path:
        sys.path.insert(0, str(FEEDBAX_TOOL))
    from feedbax.contracts.training import TrainingRunSpec
    from feedbax.contracts.worker import ProgressCoordinate
    from feedbax.training.legacy_checkpoint_adoption import (
        adopt_legacy_checkpoint,
        load_leaf_manifest,
    )
    from legacy_checkpoint_builders import adaptive_epsilon_adoption_resume_transform

    run_spec = TrainingRunSpec.model_validate(
        json.loads(FEEDBAX_SPEC_PATH.read_text(encoding="utf-8"))
    )
    result = adopt_legacy_checkpoint(
        checkpoint_root=CHECKPOINT_ROOT,
        run_spec=run_spec,
        phase_program=run_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_adaptive_epsilon_train_chunk",
        coordinate=ProgressCoordinate(
            run_id="rlrmp-3cd018b-ramp3500-to1000",
            phase="adaptive_epsilon_train_chunk",
            global_step=COMPLETED_BATCHES,
            completed_barrier="after_adaptive_epsilon_train_chunk",
        ),
        current_slots=current_slots,
        leaf_manifest=load_leaf_manifest(MANIFEST_PATH),
        model_stream=LEGACY_CHECKPOINT / "model.eqx",
        optimizer_stream=LEGACY_CHECKPOINT / "optimizer_state.eqx",
        resume_slot_transform=adaptive_epsilon_adoption_resume_transform,
    )
    print(
        json.dumps(
            {
                "transaction_id": result.write.manifest.transaction_id,
                "manifest_path": str(result.write.manifest_path),
                "latest_pointer_path": str(result.write.latest_pointer_path),
                "model_assigned_paths": len(result.model_report.assigned_paths),
                "optimizer_assigned_paths": (
                    len(result.optimizer_report.assigned_paths)
                    if result.optimizer_report is not None
                    else 0
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _write_adoption_summary() -> None:
    latest = CHECKPOINT_ROOT / "latest.json"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    latest_payload = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {}
    payload = {
        "status": "adopted" if latest.exists() else "not_adopted",
        "legacy_commit": LEGACY_COMMIT,
        "legacy_checkpoint": _rel(LEGACY_CHECKPOINT),
        "leaf_manifest": _rel(MANIFEST_PATH),
        "leaf_manifest_model_entries": len(manifest.get("model", [])),
        "leaf_manifest_optimizer_entries": len(manifest.get("optimizer", [])),
        "run_spec": _rel(RUN_SPEC_PATH),
        "feedbax_training_run_spec": _rel(FEEDBAX_SPEC_PATH),
        "current_slot_summary": _rel(CURRENT_SLOTS_SUMMARY_PATH),
        "checkpoint_root": _rel(CHECKPOINT_ROOT),
        "latest_pointer": _rel(latest) if latest.exists() else None,
        "latest_transaction_id": latest_payload.get("transaction_id"),
        "completed_batches": COMPLETED_BATCHES,
        "target_n_train_batches": TARGET_N_TRAIN_BATCHES,
        "stop_after_batches": SMOKE_STOP_AFTER_BATCHES,
        "continuation_batches": SMOKE_STOP_AFTER_BATCHES - COMPLETED_BATCHES,
    }
    ADOPTION_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ADOPTION_RESULT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), file=sys.stderr, flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
