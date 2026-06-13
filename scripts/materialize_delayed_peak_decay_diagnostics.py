"""Materialize delayed no-PGD peak/support-decay diagnostics.

This is a standalone artifact materializer for the delayed 8D no-PGD
``p_catch_trial=0.5``, ``nn_output_pre_go=1e5`` row. It evaluates the corrected
uniform 20-direction, 0.15 m delayed fixed banks and compares go-aligned GRU
profiles to the matched deterministic extLQG reference.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax._io import load_with_hyperparameters
from feedbax.types import TreeNamespace, dict_to_namespace

from rlrmp.analysis.cs_game_card import materialize_reference
from rlrmp.analysis.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.cs_released_simulation import (
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    simulate_lqg_released_forward,
    zero_forward_noise_draws,
)
from rlrmp.analysis.delayed_reach_eval_bank import (
    DEFAULT_DIRECTION_COUNT,
    DEFAULT_GO_CUE_STEPS,
    DEFAULT_UNIFORM_REACH_LENGTH_M,
    DelayedReachEvalBank,
    make_delayed_reach_eval_banks,
)
from rlrmp.analysis.gru_checkpoint_selection import (
    available_checkpoint_batches,
    checkpoint_path_for_batches,
)
from rlrmp.analysis.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
)
from rlrmp.analysis.trial_alignment import canonical_movement_horizon_from_metadata
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p


DEFAULT_EXPERIMENT = "6c36536"
DEFAULT_RUN_ID = "delayed_8d_no_pgd_catch0p5_prego1e5_lr3e-3_clip5_b64_seed42"
SCHEMA_VERSION = "rlrmp.delayed_peak_decay_diagnostics.v1"
THRESHOLDS = (0.95, 0.90, 0.85)
SUPPORT_WINDOWS = ((0, 5), (5, 10), (10, 15), (15, 21), (21, 31))


@dataclass(frozen=True)
class RunContext:
    """Resolved run inputs."""

    experiment: str
    run_id: str
    run_spec_path: Path
    artifact_dir: Path
    run_spec: dict[str, Any]
    hps: TreeNamespace
    n_replicates: int
    seed: int
    pair: Any
    movement_horizon_steps: int
    dt: float


@dataclass(frozen=True)
class EvaluatedBank:
    """Extracted rollout arrays for one checkpoint and one bank."""

    checkpoint_label: str
    bank_kind: str
    position: np.ndarray
    velocity: np.ndarray
    command: np.ndarray
    efferent: np.ndarray
    force: np.ndarray
    mechanics_vector: np.ndarray
    target_position: np.ndarray
    initial_position: np.ndarray
    reach_direction: np.ndarray
    go_index: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ExtLqgReference:
    """Deterministic matched extLQG reference arrays."""

    command: np.ndarray
    state: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    force_filter: np.ndarray
    acceleration: np.ndarray
    dt: float
    peak_velocity: float
    time_to_peak_step: int
    terminal_position_error: float
    metadata: dict[str, Any]


def main() -> None:
    """CLI entry point."""

    args = build_parser().parse_args()
    started = time.time()
    payload = materialize_diagnostics(
        experiment=args.experiment,
        run_id=args.run_id,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        direction_count=args.direction_count,
        reach_length_m=args.reach_length_m,
        go_cue_steps=tuple(range(args.go_cue_min, args.go_cue_max + 1)),
        checkpoint_stride=args.checkpoint_stride,
        include_final=not args.no_final,
        repo_root=REPO_ROOT,
    )
    elapsed = time.time() - started
    print(json.dumps(payload["outputs"], indent=2, sort_keys=True))
    print(f"materialized {len(payload['checkpoint_sweep']['rows'])} rows in {elapsed:.1f}s")


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--direction-count", type=int, default=DEFAULT_DIRECTION_COUNT)
    parser.add_argument("--reach-length-m", type=float, default=DEFAULT_UNIFORM_REACH_LENGTH_M)
    parser.add_argument("--go-cue-min", type=int, default=min(DEFAULT_GO_CUE_STEPS))
    parser.add_argument("--go-cue-max", type=int, default=max(DEFAULT_GO_CUE_STEPS))
    parser.add_argument("--checkpoint-stride", type=int, default=1000)
    parser.add_argument("--no-final", action="store_true")
    return parser


def materialize_diagnostics(
    *,
    experiment: str,
    run_id: str,
    output_dir: Path | None,
    summary_path: Path | None,
    direction_count: int,
    reach_length_m: float,
    go_cue_steps: Sequence[int],
    checkpoint_stride: int,
    include_final: bool,
    repo_root: Path,
) -> dict[str, Any]:
    """Evaluate checkpoints and write diagnostic artifacts."""

    context = resolve_context(experiment=experiment, run_id=run_id, repo_root=repo_root)
    output_dir = output_dir or repo_root / "_artifacts" / experiment / "diagnostics" / (
        "delayed_peak_decay"
    )
    summary_path = summary_path or repo_root / "results" / experiment / "notes" / (
        "delayed_peak_decay_diagnostics.md"
    )
    mkdir_p(output_dir)
    mkdir_p(summary_path.parent)

    banks = make_delayed_reach_eval_banks(
        context.pair.task.validation_trials,
        go_cue_steps=go_cue_steps,
        direction_count=direction_count,
        movement_horizon_steps=context.movement_horizon_steps,
        reach_length_m=reach_length_m,
    )
    reference = build_deterministic_extlqg_reference()

    checkpoint_specs = checkpoint_specs_for_run(
        context.artifact_dir,
        stride=checkpoint_stride,
        include_final=include_final,
    )
    rows = []
    no_catch_velocity_profiles = {}
    final_no_catch = None
    final_catch = None
    for index, (checkpoint_label, model_path) in enumerate(checkpoint_specs, start=1):
        print(f"[{index}/{len(checkpoint_specs)}] evaluating {checkpoint_label}")
        model = load_model(context, checkpoint_label=checkpoint_label, model_path=model_path)
        no_catch = evaluate_bank(
            context,
            model=model,
            bank=banks["no_catch"],
            bank_kind="no_catch",
            checkpoint_label=checkpoint_label,
        )
        catch = evaluate_bank(
            context,
            model=model,
            bank=banks["catch"],
            bank_kind="catch",
            checkpoint_label=checkpoint_label,
        )
        row = checkpoint_metrics(
            checkpoint_label=checkpoint_label,
            no_catch=no_catch,
            catch=catch,
            reference=reference,
            movement_horizon_steps=context.movement_horizon_steps,
            dt=context.dt,
        )
        rows.append(row)
        no_catch_velocity_profiles[checkpoint_label] = mean_movement_profile(
            no_catch,
            "velocity",
            context.movement_horizon_steps,
        )
        if checkpoint_label == "final":
            final_no_catch = no_catch
            final_catch = catch
        elif checkpoint_label == f"checkpoint_{max_available_batch(context.artifact_dir):07d}":
            final_no_catch = no_catch if final_no_catch is None else final_no_catch
            final_catch = catch if final_catch is None else final_catch

    if final_no_catch is None or final_catch is None:
        raise RuntimeError("No final/latest checkpoint evaluation was available")

    command_decay = support_decay_summary(
        name="command_radial_positive",
        gru_profile=mean_movement_profile(
            final_no_catch,
            "command",
            context.movement_horizon_steps,
        ),
        ext_profile=reference.command,
        baseline_window=(0, 5),
        threshold_start_step=5,
        max_step=30,
    )
    support_decay = {
        "force_filter_radial_positive": support_decay_summary(
            name="force_filter_radial_positive",
            gru_profile=mean_movement_profile(
                final_no_catch,
                "force_filter",
                context.movement_horizon_steps,
            ),
            ext_profile=reference.force_filter,
            baseline_window=(1, 6),
            threshold_start_step=6,
            max_step=30,
        ),
        "acceleration_radial_positive": support_decay_summary(
            name="acceleration_radial_positive",
            gru_profile=mean_acceleration_profile(final_no_catch, context.dt),
            ext_profile=reference.acceleration,
            baseline_window=(0, 5),
            threshold_start_step=5,
            max_step=30,
        ),
        "velocity_radial_positive": support_decay_summary(
            name="velocity_radial_positive",
            gru_profile=mean_movement_profile(
                final_no_catch,
                "velocity",
                context.movement_horizon_steps,
            ),
            ext_profile=reference.velocity,
            baseline_window=(1, 6),
            threshold_start_step=6,
            max_step=30,
        ),
        "effector_force_radial_positive": support_decay_summary(
            name="effector_force_radial_positive",
            gru_profile=mean_movement_profile(
                final_no_catch,
                "force",
                context.movement_horizon_steps,
            ),
            ext_profile=reference.force_filter,
            baseline_window=(1, 6),
            threshold_start_step=6,
            max_step=30,
        ),
    }
    checkpoint_summary = summarize_checkpoint_choice(rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "issue": experiment,
        "run_id": run_id,
        "run_spec": repo_relative(context.run_spec_path, repo_root),
        "artifact_dir": repo_relative(context.artifact_dir, repo_root),
        "evaluation": {
            "bank": {
                "direction_source": "uniform_grid",
                "direction_count": int(direction_count),
                "reach_length_m": float(reach_length_m),
                "go_cue_steps": [int(step) for step in go_cue_steps],
                "no_catch_trial_count": int(banks["no_catch"].metadata["trial_count"]),
                "catch_trial_count": int(banks["catch"].metadata["trial_count"]),
            },
            "gru_rollouts": (
                "one seeded stochastic Feedbax rollout for each fixed bank trial and "
                "replicate; pooled over replicates, go cues, and directions"
            ),
            "reference": reference.metadata,
            "movement_horizon_steps": int(context.movement_horizon_steps),
            "dt": float(context.dt),
        },
        "command_decay": command_decay,
        "support_decay": support_decay,
        "checkpoint_sweep": {
            "selection_note": (
                "shape metrics are audit-only; checkpoint choice should also require "
                "catch/pre-go leakage to remain small"
            ),
            "best_shape_row": checkpoint_summary["best_shape_row"],
            "final_row": checkpoint_summary["final_row"],
            "recommendation": checkpoint_summary["recommendation"],
            "rows": rows,
        },
    }

    json_path = output_dir / "delayed_peak_decay_diagnostics.json"
    arrays_path = output_dir / "delayed_peak_decay_profiles.npz"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        arrays_path,
        ext_command=reference.command,
        ext_velocity=reference.velocity,
        ext_force_filter=reference.force_filter,
        ext_acceleration=reference.acceleration,
        final_gru_command=mean_movement_profile(
            final_no_catch,
            "command",
            context.movement_horizon_steps,
        ),
        final_gru_velocity=mean_movement_profile(
            final_no_catch,
            "velocity",
            context.movement_horizon_steps,
        ),
        final_gru_force_filter=mean_movement_profile(
            final_no_catch,
            "force_filter",
            context.movement_horizon_steps,
        ),
        final_gru_acceleration=mean_acceleration_profile(final_no_catch, context.dt),
        checkpoint_labels=np.asarray(list(no_catch_velocity_profiles), dtype=object),
        checkpoint_velocity_profiles=np.stack(list(no_catch_velocity_profiles.values())),
    )
    payload["outputs"] = {
        "json": repo_relative(json_path, repo_root),
        "arrays_npz": repo_relative(arrays_path, repo_root),
        "summary_markdown": repo_relative(summary_path, repo_root),
    }
    summary_path.write_text(render_markdown_summary(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def resolve_context(*, experiment: str, run_id: str, repo_root: Path) -> RunContext:
    """Resolve run spec, artifact directory, hps, and task/model template."""

    run_spec_path = repo_root / "results" / experiment / "runs" / run_id / "run.json"
    artifact_dir = repo_root / "_artifacts" / experiment / "runs" / run_id
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    movement_horizon = canonical_movement_horizon_from_metadata(run_spec, default=None)
    if movement_horizon is None:
        raise ValueError("run spec does not declare a canonical movement horizon")
    dt = float(run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    return RunContext(
        experiment=experiment,
        run_id=run_id,
        run_spec_path=run_spec_path,
        artifact_dir=artifact_dir,
        run_spec=run_spec,
        hps=hps,
        n_replicates=int(hps.model.n_replicates),
        seed=seed,
        pair=pair,
        movement_horizon_steps=int(movement_horizon),
        dt=dt,
    )


def checkpoint_specs_for_run(
    artifact_dir: Path,
    *,
    stride: int,
    include_final: bool,
) -> list[tuple[str, Path]]:
    """Return numbered checkpoint model paths plus optional final model."""

    batches = [
        batch for batch in available_checkpoint_batches(artifact_dir) if batch % stride == 0
    ]
    specs = [
        (
            f"checkpoint_{batch:07d}",
            checkpoint_path_for_batches(artifact_dir, batch) / "model.eqx",
        )
        for batch in batches
    ]
    if include_final:
        specs.append(("final", artifact_dir / "trained_model.eqx"))
    return specs


def max_available_batch(artifact_dir: Path) -> int:
    """Return the largest available numbered checkpoint batch."""

    batches = available_checkpoint_batches(artifact_dir)
    if not batches:
        raise FileNotFoundError(f"No checkpoints found under {artifact_dir / 'checkpoints'}")
    return int(max(batches))


def load_model(context: RunContext, *, checkpoint_label: str, model_path: Path) -> Any:
    """Load either a numbered checkpoint or final Feedbax checkpoint."""

    if checkpoint_label == "final":
        model, _hps = load_with_hyperparameters(
            model_path,
            setup_func=lambda key, **_kwargs: setup_task_model_pair(
                context.hps,
                key=key,
            ).model,
        )
        return model
    return eqx.tree_deserialise_leaves(model_path, context.pair.model)


def evaluate_bank(
    context: RunContext,
    *,
    model: Any,
    bank: DelayedReachEvalBank,
    bank_kind: str,
    checkpoint_label: str,
) -> EvaluatedBank:
    """Evaluate all model replicates on one fixed delayed bank."""

    trial_specs = bank.trial_specs
    n_trials = int(bank.metadata["trial_count"])
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: eqx.is_array(leaf)
        and leaf.ndim >= 1
        and leaf.shape[0] == context.n_replicates,
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return context.pair.task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, n_trials),
        )

    states = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(
            jr.PRNGKey(stable_eval_seed(context.seed, checkpoint_label, bank_kind)),
            context.n_replicates,
        ),
    )
    position = np.asarray(states.mechanics.effector.pos, dtype=np.float64)
    target_position = target_position_sequence(trial_specs)
    initial_position = initial_position_array(trial_specs)
    direction = unit_vectors(target_position[:, -1, :] - initial_position)
    go_index = go_index_array(trial_specs)
    return EvaluatedBank(
        checkpoint_label=checkpoint_label,
        bank_kind=bank_kind,
        position=position,
        velocity=np.asarray(states.mechanics.effector.vel, dtype=np.float64),
        command=np.asarray(states.net.output, dtype=np.float64),
        efferent=np.asarray(states.efferent.output, dtype=np.float64),
        force=np.asarray(states.mechanics.effector.force, dtype=np.float64),
        mechanics_vector=np.asarray(states.mechanics.vector, dtype=np.float64),
        target_position=target_position,
        initial_position=initial_position,
        reach_direction=direction,
        go_index=go_index,
        metadata=bank.metadata,
    )


def stable_eval_seed(seed: int, checkpoint_label: str, bank_kind: str) -> int:
    """Return a deterministic small positive seed for a checkpoint/bank."""

    label_value = sum(ord(char) for char in f"{checkpoint_label}:{bank_kind}")
    return int((seed + 104729 + label_value) % (2**31 - 1))


def build_deterministic_extlqg_reference() -> ExtLqgReference:
    """Return the matched 0.15 m deterministic extLQG output-feedback path."""

    reference = materialize_reference()
    config = OutputFeedbackConfig()
    covariances = default_cs_noise_covariances(reference.plant, config)
    comparator = build_extlqg_comparator_path(
        reference.plant,
        reference.lqr_solution.K,
        covariances,
        schedule=reference.schedule,
        config=config,
    )
    rollout = simulate_lqg_released_forward(
        reference.plant,
        comparator.controller_gains,
        make_cs_output_feedback_initial_state(reference.plant, config),
        draws=zero_forward_noise_draws(
            T=reference.schedule.T,
            plant=reference.plant,
            config=config,
        ),
        covariances=covariances,
        estimator_gains=comparator.estimator_gains,
        config=config,
    )
    x = np.asarray(rollout.x, dtype=np.float64)
    command = np.asarray(rollout.u_command, dtype=np.float64)[:, 0]
    velocity = x[: int(reference.schedule.T), reference.plant.vel_slice[0]]
    position = x[: int(reference.schedule.T), reference.plant.pos_slice[0]]
    force_filter = x[: int(reference.schedule.T), 4]
    acceleration = np.diff(velocity) / float(reference.plant.dt)
    return ExtLqgReference(
        command=command,
        state=x,
        position=position,
        velocity=velocity,
        force_filter=force_filter,
        acceleration=acceleration,
        dt=float(reference.plant.dt),
        peak_velocity=float(rollout.peak_forward_velocity),
        time_to_peak_step=int(rollout.peak_forward_velocity_idx),
        terminal_position_error=float(rollout.terminal_position_error),
        metadata={
            "label": "deterministic_extLQG_output_feedback_8D",
            "noise": (
                "zero rollout draws with standard C&S covariance-derived Kalman "
                "gains; deterministic expected-path comparator"
            ),
            "reach_length_m": DEFAULT_UNIFORM_REACH_LENGTH_M,
            "observation_channel": "oldest_delayed_physical_block_full_8d",
            "parity_status": comparator.parity_status,
            "peak_velocity_m_s": float(rollout.peak_forward_velocity),
            "time_to_peak_step": int(rollout.peak_forward_velocity_idx),
            "terminal_position_error_m": float(rollout.terminal_position_error),
        },
    )


def checkpoint_metrics(
    *,
    checkpoint_label: str,
    no_catch: EvaluatedBank,
    catch: EvaluatedBank,
    reference: ExtLqgReference,
    movement_horizon_steps: int,
    dt: float,
) -> dict[str, Any]:
    """Return compact no-catch/catch checkpoint metrics."""

    velocity = mean_movement_profile(no_catch, "velocity", movement_horizon_steps)
    peak_step = int(np.argmax(velocity))
    rmse = float(np.sqrt(np.mean((velocity - reference.velocity[: velocity.shape[0]]) ** 2)))
    shape_error = scaled_shape_rmse(velocity, reference.velocity[: velocity.shape[0]])
    endpoint_error = endpoint_error_at_step(no_catch, movement_horizon_steps - 1)
    no_catch_leak = pre_go_leakage(no_catch)
    catch_leak = catch_leakage(catch, movement_horizon_steps)
    return {
        "checkpoint": checkpoint_label,
        "no_catch_peak_velocity_m_s": float(velocity[peak_step]),
        "no_catch_time_to_peak_step": peak_step,
        "no_catch_time_to_peak_s": float(peak_step * dt),
        "velocity_rmse_vs_extlqg_m_s": rmse,
        "velocity_shape_rmse_scaled_m_s": shape_error["rmse_m_s"],
        "velocity_shape_rmse_scaled_by_ext_peak": shape_error["rmse_over_ext_peak"],
        "velocity_shape_scale_to_ext": shape_error["scale_to_ext"],
        "endpoint_error_go_plus_59_m": endpoint_error["mean"],
        "endpoint_error_go_plus_59_p95_m": endpoint_error["p95"],
        "pre_go_peak_abs_command": no_catch_leak["peak_abs_command"],
        "pre_go_peak_abs_velocity_m_s": no_catch_leak["peak_abs_velocity_m_s"],
        "pre_go_endpoint_drift_m": no_catch_leak["endpoint_drift_m"],
        "catch_peak_abs_command": catch_leak["peak_abs_command"],
        "catch_peak_abs_velocity_m_s": catch_leak["peak_abs_velocity_m_s"],
        "catch_endpoint_drift_go_plus_59_m": catch_leak["endpoint_drift_go_plus_59_m"],
    }


def mean_movement_profile(
    bank: EvaluatedBank,
    signal: str,
    movement_horizon_steps: int,
) -> np.ndarray:
    """Return pooled mean radial movement profile for one signal."""

    if signal == "command":
        array = bank.command
    elif signal == "efferent":
        array = bank.efferent
    elif signal == "velocity":
        array = bank.velocity
    elif signal == "force":
        array = bank.force
    elif signal == "force_filter":
        array = bank.mechanics_vector[..., 4:6]
    else:
        raise ValueError(f"unsupported signal {signal!r}")
    aligned = align_movement(array, bank.go_index, movement_horizon_steps)
    radial = radial_component(aligned, bank.reach_direction)
    return np.mean(radial.reshape(-1, radial.shape[-1]), axis=0)


def mean_acceleration_profile(bank: EvaluatedBank, dt: float) -> np.ndarray:
    """Return pooled mean radial finite-difference acceleration profile."""

    velocity = align_movement(bank.velocity, bank.go_index, 60)
    radial_velocity = radial_component(velocity, bank.reach_direction)
    acceleration = np.diff(radial_velocity, axis=-1) / float(dt)
    return np.mean(acceleration.reshape(-1, acceleration.shape[-1]), axis=0)


def support_decay_summary(
    *,
    name: str,
    gru_profile: np.ndarray,
    ext_profile: np.ndarray,
    baseline_window: tuple[int, int],
    threshold_start_step: int,
    max_step: int,
) -> dict[str, Any]:
    """Summarize normalized positive-support decay crossings."""

    stop = min(int(max_step) + 1, gru_profile.shape[0], ext_profile.shape[0])
    gru_support = np.maximum(gru_profile[:stop], 0.0)
    ext_support = np.maximum(ext_profile[:stop], 0.0)
    base_start, base_stop = baseline_window
    base_stop = min(base_stop, stop)
    gru_base = float(np.mean(gru_support[base_start:base_stop]))
    ext_base = float(np.mean(ext_support[base_start:base_stop]))
    ratio = np.full(stop, np.nan, dtype=np.float64)
    if abs(gru_base) > 1e-12 and abs(ext_base) > 1e-12:
        normalized_gru = gru_support / gru_base
        normalized_ext = ext_support / ext_base
        valid = normalized_ext > 1e-9
        ratio[valid] = normalized_gru[valid] / normalized_ext[valid]
    crossings = {
        f"below_{int(threshold * 100)}pct": first_crossing(
            ratio,
            threshold=threshold,
            start_step=threshold_start_step,
        )
        for threshold in THRESHOLDS
    }
    sustained = {
        f"below_{int(threshold * 100)}pct_sustained_3": first_crossing(
            ratio,
            threshold=threshold,
            start_step=threshold_start_step,
            sustained_steps=3,
        )
        for threshold in THRESHOLDS
    }
    window_ratios = {}
    for start, end in SUPPORT_WINDOWS:
        end = min(end, stop)
        if end <= start:
            continue
        values = ratio[start:end]
        window_ratios[f"steps_{start}_{end - 1}"] = finite_mean(values)
    return {
        "metric": name,
        "support_definition": "positive target-radial component, pooled mean profile",
        "normalization": {
            "baseline_window_steps": [int(base_start), int(base_stop - 1)],
            "gru_baseline": gru_base,
            "extlqg_baseline": ext_base,
        },
        "raw_initial_support_ratio": safe_ratio(gru_base, ext_base),
        "threshold_crossings": crossings,
        "sustained_threshold_crossings": sustained,
        "early_mid_window_ratios": window_ratios,
        "normalized_ratio_steps_0_to_30": [
            None if not np.isfinite(value) else float(value) for value in ratio
        ],
        "gru_profile_steps_0_to_30": [float(value) for value in gru_profile[:stop]],
        "extlqg_profile_steps_0_to_30": [float(value) for value in ext_profile[:stop]],
    }


def first_crossing(
    ratio: np.ndarray,
    *,
    threshold: float,
    start_step: int,
    sustained_steps: int = 1,
) -> int | None:
    """Return first step where ratio falls below threshold."""

    for step in range(int(start_step), ratio.shape[0] - sustained_steps + 1):
        values = ratio[step : step + sustained_steps]
        if np.all(np.isfinite(values)) and np.all(values < threshold):
            return int(step)
    return None


def endpoint_error_at_step(bank: EvaluatedBank, step: int) -> dict[str, float]:
    """Return endpoint error at a go-aligned movement step."""

    aligned_position = align_movement(bank.position, bank.go_index, step + 1)
    endpoint = aligned_position[..., step, :]
    target = bank.target_position[:, -1, :]
    error = np.linalg.norm(endpoint - target[None, :, :], axis=-1)
    return summary_stats(error)


def pre_go_leakage(bank: EvaluatedBank) -> dict[str, float]:
    """Return prep-window no-catch leakage metrics."""

    command_values = []
    velocity_values = []
    drift_values = []
    for trial, go_step in enumerate(bank.go_index):
        if go_step <= 0:
            continue
        direction = bank.reach_direction[trial]
        command = np.sum(bank.command[:, trial, :go_step, :] * direction, axis=-1)
        velocity = np.sum(bank.velocity[:, trial, :go_step, :] * direction, axis=-1)
        drift = np.linalg.norm(
            bank.position[:, trial, go_step - 1, :] - bank.initial_position[trial],
            axis=-1,
        )
        command_values.append(np.abs(command).reshape(-1))
        velocity_values.append(np.abs(velocity).reshape(-1))
        drift_values.append(drift.reshape(-1))
    return {
        "peak_abs_command": finite_max_concat(command_values),
        "peak_abs_velocity_m_s": finite_max_concat(velocity_values),
        "endpoint_drift_m": finite_mean_concat(drift_values),
    }


def catch_leakage(bank: EvaluatedBank, movement_horizon_steps: int) -> dict[str, float]:
    """Return catch-trial leakage metrics over the movement window."""

    command = align_movement(bank.command, bank.go_index, movement_horizon_steps)
    velocity = align_movement(bank.velocity, bank.go_index, movement_horizon_steps)
    position = align_movement(bank.position, bank.go_index, movement_horizon_steps)
    radial_command = radial_component(command, bank.reach_direction)
    radial_velocity = radial_component(velocity, bank.reach_direction)
    endpoint_drift = np.linalg.norm(
        position[..., movement_horizon_steps - 1, :] - bank.initial_position[None, :, :],
        axis=-1,
    )
    return {
        "peak_abs_command": float(np.max(np.abs(radial_command))),
        "peak_abs_velocity_m_s": float(np.max(np.abs(radial_velocity))),
        "endpoint_drift_go_plus_59_m": float(np.mean(endpoint_drift)),
    }


def summarize_checkpoint_choice(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return best shape row and recommendation text."""

    if not rows:
        raise ValueError("checkpoint rows are empty")
    final = next((row for row in rows if row["checkpoint"] == "final"), rows[-1])
    best_shape = min(rows, key=lambda row: row["velocity_shape_rmse_scaled_by_ext_peak"])
    acceptable = [
        row
        for row in rows
        if row["catch_peak_abs_velocity_m_s"] < 0.03
        and row["pre_go_peak_abs_velocity_m_s"] < 0.03
        and row["catch_endpoint_drift_go_plus_59_m"] < 0.005
    ]
    best_acceptable = (
        min(acceptable, key=lambda row: row["velocity_shape_rmse_scaled_by_ext_peak"])
        if acceptable
        else None
    )
    if best_acceptable is None:
        recommendation = (
            "No checkpoint passes the conservative leakage screen used by this "
            "diagnostic; do not select by trajectory shape alone."
        )
    else:
        improvement = (
            final["velocity_shape_rmse_scaled_by_ext_peak"]
            - best_acceptable["velocity_shape_rmse_scaled_by_ext_peak"]
        )
        if best_acceptable["checkpoint"] == final["checkpoint"] or improvement < 0.01:
            recommendation = (
                "Checkpoint choice is not the main explanation: the final/latest "
                "checkpoint is effectively as close in velocity shape as the best "
                "leakage-acceptable checkpoint."
            )
        else:
            recommendation = (
                f"{best_acceptable['checkpoint']} is materially closer in scaled "
                f"velocity shape than final by {improvement:.4f} ext-peak units while "
                "passing the leakage screen."
            )
    return {
        "best_shape_row": dict(best_shape),
        "final_row": dict(final),
        "best_leakage_acceptable_row": dict(best_acceptable) if best_acceptable else None,
        "recommendation": recommendation,
    }


def render_markdown_summary(payload: Mapping[str, Any]) -> str:
    """Render the concise Markdown summary."""

    command = payload["command_decay"]
    support = payload["support_decay"]
    sweep = payload["checkpoint_sweep"]
    final = sweep["final_row"]
    best = sweep["best_shape_row"]
    lines = [
        "# Delayed Peak/Support-Decay Diagnostics",
        "",
        f"Run: `{payload['run_id']}`",
        "",
        "Evaluation lens: corrected uniform 20-direction, 0.15 m delayed fixed banks "
        "over go cues 10..30. GRU values are one seeded stochastic rollout per bank "
        "trial and replicate, pooled over 5 replicates, 21 go cues, and 20 directions. "
        "The extLQG reference is deterministic: zero rollout draws with the standard "
        "C&S covariance-derived Kalman gains.",
        "",
        "## Support-Decay Onset",
        "",
        "| Metric | raw initial GRU/ext | first <95% | first <90% | first <85% | ratio steps 5-9 | ratio steps 10-14 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        support_row("command", command),
        support_row("force/filter", support["force_filter_radial_positive"]),
        support_row("acceleration", support["acceleration_radial_positive"]),
        support_row("velocity", support["velocity_radial_positive"]),
        support_row("effector force", support["effector_force_radial_positive"]),
        "",
        "Threshold ratios use positive target-radial support profiles after normalizing "
        "each GRU/extLQG profile by its own initial-launch window. Command uses steps "
        "0..4; force/filter and velocity use steps 1..5 because support starts near "
        "zero at movement onset.",
        "",
        "## Checkpoint Sweep",
        "",
        "| Checkpoint | peak vel | t_peak | vel RMSE | shape err/ext peak | endpoint@go+59 | pre-go peak vel | catch peak vel | catch endpoint drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["checkpoint_sweep"]["rows"]:
        lines.append(
            f"| `{row['checkpoint']}` | "
            f"{row['no_catch_peak_velocity_m_s']:.4f} | "
            f"{row['no_catch_time_to_peak_step']} | "
            f"{row['velocity_rmse_vs_extlqg_m_s']:.4f} | "
            f"{row['velocity_shape_rmse_scaled_by_ext_peak']:.4f} | "
            f"{row['endpoint_error_go_plus_59_m']:.5f} | "
            f"{row['pre_go_peak_abs_velocity_m_s']:.5f} | "
            f"{row['catch_peak_abs_velocity_m_s']:.5f} | "
            f"{row['catch_endpoint_drift_go_plus_59_m']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            f"- Final checkpoint: peak `{final['no_catch_peak_velocity_m_s']:.4f} m/s`, "
            f"time-to-peak step `{final['no_catch_time_to_peak_step']}`, "
            f"shape error `{final['velocity_shape_rmse_scaled_by_ext_peak']:.4f}` ext-peak units.",
            f"- Best shape checkpoint: `{best['checkpoint']}` with shape error "
            f"`{best['velocity_shape_rmse_scaled_by_ext_peak']:.4f}` ext-peak units.",
            f"- Recommendation: {sweep['recommendation']}",
            "",
            "Outputs:",
            f"- JSON: `{payload['outputs']['json']}`",
            f"- Arrays: `{payload['outputs']['arrays_npz']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def support_row(label: str, summary: Mapping[str, Any]) -> str:
    """Render one support-decay table row."""

    crossings = summary["threshold_crossings"]
    windows = summary["early_mid_window_ratios"]
    return (
        f"| {label} | {fmt(summary['raw_initial_support_ratio'])} | "
        f"{fmt_step(crossings['below_95pct'])} | "
        f"{fmt_step(crossings['below_90pct'])} | "
        f"{fmt_step(crossings['below_85pct'])} | "
        f"{fmt(windows.get('steps_5_9'))} | "
        f"{fmt(windows.get('steps_10_14'))} |"
    )


def align_movement(array: np.ndarray, go_index: np.ndarray, n_steps: int) -> np.ndarray:
    """Gather go-aligned movement samples from ``array``."""

    reps, trials, _time, dim = array.shape
    steps = np.arange(n_steps, dtype=np.int64)
    indices = go_index[None, :, None] + steps[None, None, :]
    indices = np.broadcast_to(indices, (reps, trials, n_steps))
    return np.take_along_axis(array, indices[..., None], axis=2).reshape(
        reps,
        trials,
        n_steps,
        dim,
    )


def radial_component(array: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project vector array onto each trial's reach direction."""

    return np.sum(array * direction[None, :, None, :], axis=-1)


def target_position_sequence(trial_specs: Any) -> np.ndarray:
    """Return visible target sequence from delayed trial specs."""

    inputs = getattr(trial_specs, "inputs", None)
    task_inputs = getattr(inputs, "task", None) if isinstance(inputs, Mapping) else inputs
    effector_target = getattr(task_inputs, "effector_target", None)
    position = getattr(effector_target, "pos", None)
    if position is not None:
        return np.asarray(position, dtype=np.float64)
    if isinstance(inputs, Mapping) and "target" in inputs:
        return np.asarray(inputs["target"], dtype=np.float64)
    for target_spec in getattr(trial_specs, "targets", {}).values():
        value = getattr(target_spec, "value", None)
        if getattr(value, "shape", None) is not None and value.shape[-1] == 2:
            return np.asarray(value, dtype=np.float64)
    raise ValueError("Could not resolve visible target position sequence")


def initial_position_array(trial_specs: Any) -> np.ndarray:
    """Return trial initial effector positions."""

    for init_state in trial_specs.inits.values():
        position = getattr(init_state, "pos", None)
        if position is not None:
            return np.asarray(position, dtype=np.float64)
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 2:
            return np.asarray(init_state, dtype=np.float64)[..., 0:2]
    raise ValueError("Could not resolve initial effector positions")


def go_index_array(trial_specs: Any) -> np.ndarray:
    """Return movement epoch start index for each delayed trial."""

    bounds = np.asarray(trial_specs.timeline.epoch_bounds, dtype=np.int64)
    if bounds.ndim != 2 or bounds.shape[1] < 2:
        raise ValueError(f"Unexpected epoch_bounds shape {bounds.shape}")
    return bounds[:, 1]


def unit_vectors(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise unit vectors."""

    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("Cannot normalize zero reach direction")
    return vectors / norm


def scaled_shape_rmse(profile: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return least-squares amplitude-scaled shape RMSE."""

    denom = float(np.dot(reference, reference))
    scale = float(np.dot(profile, reference) / denom) if denom > 0.0 else math.nan
    residual = profile - scale * reference
    rmse = float(np.sqrt(np.mean(residual * residual)))
    ext_peak = float(np.max(reference))
    return {
        "scale_to_ext": scale,
        "rmse_m_s": rmse,
        "rmse_over_ext_peak": safe_ratio(rmse, ext_peak),
    }


def summary_stats(values: Any) -> dict[str, float]:
    """Return basic mean/p95 stats."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def finite_mean(values: Any) -> float | None:
    """Return finite mean or ``None``."""

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def finite_max_concat(values: Sequence[np.ndarray]) -> float:
    """Return max over a list of arrays."""

    if not values:
        return 0.0
    return float(np.max(np.concatenate(values)))


def finite_mean_concat(values: Sequence[np.ndarray]) -> float:
    """Return mean over a list of arrays."""

    if not values:
        return 0.0
    return float(np.mean(np.concatenate(values)))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite ratio or ``None``."""

    if abs(float(denominator)) <= 1e-12:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def fmt(value: Any) -> str:
    """Format compact table values."""

    if value is None:
        return "n/a"
    return f"{float(value):.3g}"


def fmt_step(value: Any) -> str:
    """Format a step index."""

    return "n/a" if value is None else str(int(value))


def repo_relative(path: Path, repo_root: Path) -> str:
    """Return repo-relative path when possible."""

    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
