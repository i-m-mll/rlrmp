"""Materialize no-delay direction-aligned velocity figures for the 3e-3 run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import plotly.graph_objects as go
from feedbax._io import load_with_hyperparameters
from feedbax._mapping import WhereDict
from feedbax.task import TaskTrialSpec
from feedbax.types import TreeNamespace, dict_to_namespace

from rlrmp.analysis.cs_game_card import OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
from rlrmp.analysis.cs_game_card import materialize_reference
from rlrmp.analysis.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.cs_released_simulation import (
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
)
from rlrmp.analysis.gru_checkpoint_selection import (
    ReplicateCheckpointSelection,
    load_validation_selected_checkpoint_model,
)
from rlrmp.analysis.gru_pilot_figures import (
    RunFigureInputs,
    _initial_effector_position,
    _rgba,
    _reach_direction,
    _target_position_sequence,
    initial_effector_velocity,
    resolve_run_inputs,
)
from rlrmp.analysis.output_feedback import OutputFeedbackConfig
from rlrmp.analysis.trial_alignment import canonical_movement_horizon_from_metadata
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p


EXPERIMENT = "ba82f3d"
RUN_ID = "target_relative_multitarget_fullqrf_warmcos__lr3e-3_clip5_b64"
RUN_LABEL = "target-relative multitarget full Q/R/Qf, lr=3e-3"
TOPIC = "no_delay_direction_aligned_velocity_lr3e-3_validation_selected"
DEFAULT_DIRECTION_COUNT = 20
DEFAULT_REACH_LENGTH_M = 0.15
DEFAULT_REFERENCE_SAMPLES = 2100


@dataclass(frozen=True)
class DirectionAlignedBank:
    """No-delay target bank and metadata."""

    trial_specs: TaskTrialSpec
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DirectionVelocityProfile:
    """Target-aligned velocity statistics for one direction."""

    direction_index: int
    angle_deg: float
    unit_vector: tuple[float, float]
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    peak_mean_velocity_m_s: float
    time_of_peak_mean_velocity_s: float


@dataclass(frozen=True)
class PooledVelocityProfile:
    """Target-aligned velocity statistics pooled across the direction bank."""

    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_replicates: int
    n_directions: int
    n_samples: int
    peak_mean_velocity_m_s: float
    time_of_peak_mean_velocity_s: float


@dataclass(frozen=True)
class MatchedReferenceProfile:
    """Reach-matched extLQG velocity reference."""

    label: str
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    reach_length_m: float
    peak_mean_velocity_m_s: float
    time_of_peak_mean_velocity_s: float
    terminal_position_error_m: float
    parity_status: str


def main() -> None:
    """Generate the requested figure path and metadata sidecars."""

    args = build_parser().parse_args()
    output_dir = args.output_dir or REPO_ROOT / "_artifacts" / EXPERIMENT / "figures" / args.topic
    spec_dir = args.spec_dir or REPO_ROOT / "results" / EXPERIMENT / "figures" / args.topic
    note_dir = REPO_ROOT / "results" / EXPERIMENT / "notes"
    mkdir_p(output_dir)
    mkdir_p(spec_dir)
    mkdir_p(note_dir)

    run = resolve_run_inputs(
        experiment=EXPERIMENT,
        run_ids=(args.run_id,),
        labels=(args.label,),
    )[0]
    bank, source_directions = make_uniform_no_delay_direction_bank(
        run,
        direction_count=args.direction_count,
        reach_length_m=args.reach_length_m,
    )
    pooled, per_direction, selections = evaluate_direction_aligned_velocity(
        run,
        bank=bank,
        checkpoint_policy=args.checkpoint_policy,
    )
    reference = build_matched_extlqg_reference(
        reach_length_m=args.reach_length_m,
        n_samples=args.reference_samples,
    )

    pooled_figure = output_dir / "pooled_direction_aligned_velocity_with_matched_extlqg.html"
    per_direction_figure = output_dir / "per_direction_aligned_velocity_with_matched_extlqg.html"
    write_pooled_figure(
        run=run,
        pooled=pooled,
        reference=reference,
        figure_path=pooled_figure,
    )
    write_per_direction_figure(
        run=run,
        profiles=per_direction,
        reference=reference,
        figure_path=per_direction_figure,
    )

    summary = build_summary(
        run=run,
        bank=bank,
        source_directions=source_directions,
        pooled=pooled,
        per_direction=per_direction,
        reference=reference,
        selections=selections,
        checkpoint_policy=args.checkpoint_policy,
        pooled_figure=pooled_figure,
        per_direction_figure=per_direction_figure,
    )
    summary_path = output_dir / "summary.json"
    summary["figures"]["summary"] = repo_relative(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(
        json.dumps(build_spec(args=args, summary=summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    note_path = note_dir / f"{args.topic}.md"
    note_path.write_text(render_note(summary), encoding="utf-8")

    print(
        json.dumps(
            {
                "pooled_figure": str(pooled_figure),
                "per_direction_figure": str(per_direction_figure),
                "summary": str(summary_path),
                "spec": str(spec_path),
                "note": str(note_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--label", default=RUN_LABEL)
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--spec-dir", type=Path)
    parser.add_argument("--direction-count", type=int, default=DEFAULT_DIRECTION_COUNT)
    parser.add_argument("--reach-length-m", type=float, default=DEFAULT_REACH_LENGTH_M)
    parser.add_argument("--reference-samples", type=int, default=DEFAULT_REFERENCE_SAMPLES)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("validation_selected_per_replicate", "final_checkpoint"),
        default="validation_selected_per_replicate",
    )
    return parser


def make_uniform_no_delay_direction_bank(
    run: RunFigureInputs,
    *,
    direction_count: int,
    reach_length_m: float,
) -> tuple[DirectionAlignedBank, list[dict[str, Any]]]:
    """Return a real no-delay 20-direction target bank."""

    if direction_count < 1:
        raise ValueError("direction_count must be positive")
    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    seed = int(run.run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    source_trial_specs = pair.task.validation_trials
    n_time_steps = infer_time_steps(source_trial_specs)
    source_directions = direction_table_from_trial_specs(source_trial_specs)
    base_initial_position = _initial_effector_position(source_trial_specs)[0].astype(np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, direction_count, endpoint=False)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    targets = base_initial_position[None, :] + float(reach_length_m) * directions
    target_sequence = np.broadcast_to(
        targets[:, None, :],
        (direction_count, n_time_steps, 2),
    ).astype(np.float64)
    initial_position = np.broadcast_to(
        base_initial_position[None, :],
        (direction_count, 2),
    ).astype(np.float64)
    trial_specs = replace_trial_targets(
        source_trial_specs,
        target_sequence=jnp.asarray(target_sequence),
        initial_position=jnp.asarray(initial_position),
    )
    movement_horizon = canonical_movement_horizon_from_metadata(
        run.run_spec,
        default=n_time_steps,
    )
    metadata = {
        "schema_version": "rlrmp.no_delay_direction_aligned_bank.v1",
        "bank_kind": "uniform_static_targets",
        "direction_count": int(direction_count),
        "reach_length_m": float(reach_length_m),
        "target_angles_deg": [
            float((np.degrees(angle) + 360.0) % 360.0) for angle in angles
        ],
        "initial_position_m": [float(x) for x in base_initial_position],
        "n_time_steps": int(n_time_steps),
        "movement_horizon_steps": int(movement_horizon),
        "source_validation_trial_count": int(len(source_directions)),
        "source_validation_direction_table": source_directions,
        "construction_note": (
            "Uniform no-delay direction bank built locally in this script; no delayed "
            "eval-bank helper was imported or edited."
        ),
    }
    return DirectionAlignedBank(trial_specs=trial_specs, metadata=metadata), source_directions


def replace_trial_targets(
    trial_specs: TaskTrialSpec,
    *,
    target_sequence: jnp.ndarray,
    initial_position: jnp.ndarray,
) -> TaskTrialSpec:
    """Return ``trial_specs`` with static target arrays replaced."""

    inputs = dict(trial_specs.inputs)
    if "target" in inputs:
        inputs["target"] = target_sequence
    if "effector_target" in inputs:
        current = inputs["effector_target"]
        vel = getattr(current, "vel", None)
        if vel is not None:
            inputs["effector_target"] = eqx.tree_at(
                lambda state: (state.pos, state.vel),
                current,
                (target_sequence, jnp.zeros_like(target_sequence, dtype=vel.dtype)),
            )
        else:
            inputs["effector_target"] = eqx.tree_at(
                lambda state: state.pos,
                current,
                target_sequence,
            )

    targets: dict[str, Any] = {}
    for key, target_spec in trial_specs.targets.items():
        value = getattr(target_spec, "value", None)
        if getattr(value, "shape", None) is not None and value.shape[-1] == 2:
            targets[key] = eqx.tree_at(lambda spec: spec.value, target_spec, target_sequence)
        else:
            targets[key] = target_spec

    inits = trial_specs.inits
    updated_inits: dict[str, Any] = {}
    for key, init_state in trial_specs.inits.items():
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 2 and shape[-1] >= 2:
            value = jnp.broadcast_to(init_state[0], (target_sequence.shape[0], shape[-1]))
            value = value.at[:, 0:2].set(initial_position)
            updated_inits[key] = value
        else:
            updated_inits[key] = init_state
    if updated_inits:
        inits = WhereDict(updated_inits)

    return TaskTrialSpec(
        inits=inits,
        inputs=inputs,
        targets=WhereDict(targets),
        intervene=trial_specs.intervene,
        timeline=trial_specs.timeline,
        extra={
            **dict(trial_specs.extra or {}),
            "no_delay_direction_aligned_bank": {
                "direction_count": int(target_sequence.shape[0]),
                "reach_length_m": float(
                    np.linalg.norm(np.asarray(target_sequence[0, -1]) - np.asarray(initial_position[0]))
                ),
            },
        },
    )


def evaluate_direction_aligned_velocity(
    run: RunFigureInputs,
    *,
    bank: DirectionAlignedBank,
    checkpoint_policy: str,
) -> tuple[
    PooledVelocityProfile,
    list[DirectionVelocityProfile],
    tuple[ReplicateCheckpointSelection, ...],
]:
    """Evaluate the GRU on the bank and summarize target-radial velocity."""

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run.run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    if checkpoint_policy == "validation_selected_per_replicate":
        model, selections = load_validation_selected_checkpoint_model(
            experiment=EXPERIMENT,
            run_id=run.run_id,
            run_spec=run.run_spec,
        )
    elif checkpoint_policy == "final_checkpoint":
        model, _hyperparameters = load_with_hyperparameters(
            run.artifact_dir / "trained_model.eqx",
            setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
        )
        selections = []
    else:
        raise ValueError(f"Unsupported checkpoint policy {checkpoint_policy!r}")

    trial_specs = bank.trial_specs
    eval_trial_count = infer_trial_count_from_targets(trial_specs)
    initial_velocity = initial_effector_velocity(trial_specs)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates,
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        states = pair.task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, eval_trial_count),
        )
        return jnp.concatenate(
            [initial_velocity[:, None, :], states.mechanics.effector.vel],
            axis=1,
        )

    velocity = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(0), n_replicates),
    )
    velocity_np = np.asarray(velocity, dtype=np.float64)
    target_position = _target_position_sequence(trial_specs)
    initial_position = _initial_effector_position(trial_specs)
    direction, distance = _reach_direction(initial_position, target_position[:, -1, :])
    if not np.allclose(distance, bank.metadata["reach_length_m"], atol=1e-10, rtol=0.0):
        raise ValueError(f"Unexpected target distances in generated bank: {distance}")
    forward = np.sum(velocity_np * direction[None, :, None, :], axis=-1)
    dt = float(run.run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    time_s = np.arange(forward.shape[-1], dtype=np.float64) * dt
    pooled_flat = forward.reshape(n_replicates * forward.shape[1], forward.shape[-1])
    pooled_mean = np.mean(pooled_flat, axis=0)
    pooled_std = np.std(pooled_flat, axis=0)
    pooled_peak_idx = int(np.argmax(pooled_mean))
    pooled = PooledVelocityProfile(
        time_s=time_s,
        mean=pooled_mean,
        std=pooled_std,
        n_replicates=n_replicates,
        n_directions=int(forward.shape[1]),
        n_samples=int(pooled_flat.shape[0]),
        peak_mean_velocity_m_s=float(pooled_mean[pooled_peak_idx]),
        time_of_peak_mean_velocity_s=float(time_s[pooled_peak_idx]),
    )
    per_direction = []
    for direction_idx, unit in enumerate(direction):
        samples = forward[:, direction_idx, :]
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0)
        peak_idx = int(np.argmax(mean))
        per_direction.append(
            DirectionVelocityProfile(
                direction_index=int(direction_idx),
                angle_deg=float(
                    (np.degrees(np.arctan2(unit[1], unit[0])) + 360.0) % 360.0
                ),
                unit_vector=(float(unit[0]), float(unit[1])),
                time_s=time_s,
                mean=mean,
                std=std,
                n_samples=int(samples.shape[0]),
                peak_mean_velocity_m_s=float(mean[peak_idx]),
                time_of_peak_mean_velocity_s=float(time_s[peak_idx]),
            )
        )
    return pooled, per_direction, tuple(selections)


def build_matched_extlqg_reference(
    *,
    reach_length_m: float,
    n_samples: int,
) -> MatchedReferenceProfile:
    """Simulate the extLQG comparator from a target-centered reach state."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    config = OutputFeedbackConfig()
    covariances = default_cs_noise_covariances(reference.plant, config)
    comparator = build_extlqg_comparator_path(
        reference.plant,
        reference.lqr_solution.K,
        covariances,
        schedule=reference.schedule,
        config=config,
    )
    x0 = output_feedback_initial_state_for_target(
        reference.plant,
        config,
        target_pos=jnp.asarray([reach_length_m, 0.0], dtype=jnp.float64),
    )
    rollouts = [
        simulate_lqg_released_forward(
            reference.plant,
            comparator.controller_gains,
            x0,
            draws=sample_forward_noise_draws(
                sample_key,
                T=reference.schedule.T,
                covariances=covariances,
            ),
            covariances=covariances,
            estimator_gains=comparator.estimator_gains,
            config=config,
        )
        for sample_key in jr.split(jr.PRNGKey(0), n_samples)
    ]
    x = np.stack([np.asarray(rollout.x, dtype=np.float64) for rollout in rollouts], axis=0)
    vel_lo = reference.plant.vel_slice[0]
    pos_lo, pos_hi = reference.plant.pos_slice
    forward = x[:, :, vel_lo]
    mean = np.mean(forward, axis=0)
    std = np.std(forward, axis=0)
    peak_idx = int(np.argmax(mean))
    terminal_error = np.linalg.norm(x[:, -1, pos_lo:pos_hi], axis=-1)
    return MatchedReferenceProfile(
        label="C&S extLQG/output-feedback 8D, reach matched to 0.15 m bank",
        time_s=np.arange(mean.shape[0], dtype=np.float64) * float(reference.plant.dt),
        mean=mean,
        std=std,
        n_samples=n_samples,
        reach_length_m=float(reach_length_m),
        peak_mean_velocity_m_s=float(mean[peak_idx]),
        time_of_peak_mean_velocity_s=float(peak_idx * float(reference.plant.dt)),
        terminal_position_error_m=float(np.mean(terminal_error)),
        parity_status=comparator.parity_status,
    )


def output_feedback_initial_state_for_target(
    plant: Any,
    config: OutputFeedbackConfig,
    *,
    target_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Return delay-augmented output-feedback state for a custom reach length."""

    x_phys = jnp.zeros((config.n_phys,), dtype=jnp.float64)
    pos_lo, pos_hi = plant.pos_slice
    x_phys = x_phys.at[pos_lo:pos_hi].set(-target_pos.astype(jnp.float64))
    return jnp.tile(x_phys, config.delay_steps + 1)


def write_pooled_figure(
    *,
    run: RunFigureInputs,
    pooled: PooledVelocityProfile,
    reference: MatchedReferenceProfile,
    figure_path: Path,
) -> None:
    """Write the pooled bank velocity figure."""

    fig = go.Figure()
    add_band_trace(
        fig,
        time_s=pooled.time_s,
        mean=pooled.mean,
        std=pooled.std,
        color="#2563eb",
        name="GRU pooled directions",
        legendgroup="gru",
    )
    add_line_trace(
        fig,
        time_s=pooled.time_s,
        mean=pooled.mean,
        color="#2563eb",
        name="GRU pooled directions",
        legendgroup="gru",
    )
    add_band_trace(
        fig,
        time_s=reference.time_s,
        mean=reference.mean,
        std=reference.std,
        color="#111827",
        name="matched extLQG",
        legendgroup="extlqg",
        alpha=0.09,
    )
    add_line_trace(
        fig,
        time_s=reference.time_s,
        mean=reference.mean,
        color="#111827",
        name="matched extLQG",
        legendgroup="extlqg",
        dash="dash",
    )
    fig.update_layout(
        title="No-delay direction-aligned velocity",
        width=860,
        height=560,
        margin={"l": 72, "r": 24, "t": 72, "b": 104},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": -0.18,
            "yanchor": "top",
        },
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Target-aligned velocity (m/s)", zeroline=True)
    fig.write_html(figure_path)


def write_per_direction_figure(
    *,
    run: RunFigureInputs,
    profiles: Sequence[DirectionVelocityProfile],
    reference: MatchedReferenceProfile,
    figure_path: Path,
) -> None:
    """Write one figure with direction-resolved mean traces."""

    fig = go.Figure()
    palette = (
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4b5563",
    )
    for idx, profile in enumerate(profiles):
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=profile.time_s,
                y=profile.mean,
                mode="lines",
                line={"color": color, "width": 1.5},
                name=f"{profile.direction_index}: {profile.angle_deg:.0f} deg",
            )
        )
    add_line_trace(
        fig,
        time_s=reference.time_s,
        mean=reference.mean,
        color="#111827",
        name="matched extLQG",
        legendgroup="extlqg",
        dash="dash",
        width=2.8,
    )
    fig.update_layout(
        title="No-delay per-direction velocity",
        width=1040,
        height=560,
        margin={"l": 72, "r": 180, "t": 72, "b": 64},
        hovermode="x unified",
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Target-aligned velocity (m/s)", zeroline=True)
    fig.write_html(figure_path)


def add_band_trace(
    fig: go.Figure,
    *,
    time_s: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    name: str,
    legendgroup: str,
    alpha: float = 0.16,
) -> None:
    """Add a mean +/- SD filled band to ``fig``."""

    upper = mean + std
    lower = mean - std
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([time_s, time_s[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor=_rgba(color, alpha),
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name=f"{name} mean +/- 1 SD",
            legendgroup=legendgroup,
            showlegend=False,
        )
    )


def add_line_trace(
    fig: go.Figure,
    *,
    time_s: np.ndarray,
    mean: np.ndarray,
    color: str,
    name: str,
    legendgroup: str,
    dash: str | None = None,
    width: float = 2.4,
) -> None:
    """Add a mean line to ``fig``."""

    line: dict[str, Any] = {"color": color, "width": width}
    if dash is not None:
        line["dash"] = dash
    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=mean,
            mode="lines",
            line=line,
            name=name,
            legendgroup=legendgroup,
        )
    )


def build_summary(
    *,
    run: RunFigureInputs,
    bank: DirectionAlignedBank,
    source_directions: list[dict[str, Any]],
    pooled: PooledVelocityProfile,
    per_direction: Sequence[DirectionVelocityProfile],
    reference: MatchedReferenceProfile,
    selections: Sequence[ReplicateCheckpointSelection],
    checkpoint_policy: str,
    pooled_figure: Path,
    per_direction_figure: Path,
) -> dict[str, Any]:
    """Return JSON-compatible materialization metadata."""

    return {
        "schema_version": "rlrmp.no_delay_direction_aligned_velocity.v1",
        "issue": EXPERIMENT,
        "topic": TOPIC,
        "run": {
            "run_id": run.run_id,
            "label": run.label,
            "run_spec_path": repo_relative(run.run_spec_path),
            "artifact_dir": repo_relative(run.artifact_dir),
        },
        "figures": {
            "pooled": repo_relative(pooled_figure),
            "per_direction": repo_relative(per_direction_figure),
        },
        "checkpoint_policy": checkpoint_policy,
        "checkpoint_source": (
            "per-replicate validation-selected numbered checkpoints from sparse "
            "training-history validation records"
            if checkpoint_policy == "validation_selected_per_replicate"
            else "final trained_model.eqx"
        ),
        "checkpoint_selection": [
            selection.to_json(repo_root=REPO_ROOT) for selection in selections
        ],
        "bank": bank.metadata,
        "source_validation_directions": source_directions,
        "projection": (
            "target-aligned velocity: dot(effector velocity, "
            "unit(target - initial_position))"
        ),
        "error_band": (
            "GRU pooled band is mean +/- 1 SD over replicate x direction samples; "
            "per-direction summaries use mean +/- 1 SD over replicates; extLQG band "
            "is mean +/- 1 SD over stochastic analytical rollouts."
        ),
        "pooled_profile": {
            "n_replicates": pooled.n_replicates,
            "n_directions": pooled.n_directions,
            "n_samples": pooled.n_samples,
            "peak_mean_velocity_m_s": pooled.peak_mean_velocity_m_s,
            "time_of_peak_mean_velocity_s": pooled.time_of_peak_mean_velocity_s,
            "time_start_s": float(pooled.time_s[0]),
            "time_stop_s": float(pooled.time_s[-1]),
        },
        "direction_profiles": [
            {
                "direction_index": profile.direction_index,
                "angle_deg": profile.angle_deg,
                "unit_vector": list(profile.unit_vector),
                "n_samples": profile.n_samples,
                "peak_mean_velocity_m_s": profile.peak_mean_velocity_m_s,
                "time_of_peak_mean_velocity_s": profile.time_of_peak_mean_velocity_s,
            }
            for profile in per_direction
        ],
        "extlqg_reference": {
            "label": reference.label,
            "reach_length_m": reference.reach_length_m,
            "n_samples": reference.n_samples,
            "peak_mean_velocity_m_s": reference.peak_mean_velocity_m_s,
            "time_of_peak_mean_velocity_s": reference.time_of_peak_mean_velocity_s,
            "terminal_position_error_m": reference.terminal_position_error_m,
            "parity_status": reference.parity_status,
            "time_start_s": float(reference.time_s[0]),
            "time_stop_s": float(reference.time_s[-1]),
        },
    }


def build_spec(*, args: argparse.Namespace, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact tracked regeneration spec."""

    return {
        "schema_version": "rlrmp.figure_regeneration_spec.v1",
        "script": "scripts/materialize_no_delay_direction_aligned_velocity.py",
        "args": {
            "run_id": args.run_id,
            "label": args.label,
            "topic": args.topic,
            "direction_count": args.direction_count,
            "reach_length_m": args.reach_length_m,
            "reference_samples": args.reference_samples,
            "checkpoint_policy": args.checkpoint_policy,
        },
        "outputs": summary["figures"],
        "summary": summary["figures"]["summary"],
    }


def render_note(summary: Mapping[str, Any]) -> str:
    """Render a short Markdown note for the generated figure path."""

    bank = summary["bank"]
    pooled = summary["pooled_profile"]
    ext = summary["extlqg_reference"]
    lines = [
        "# No-delay direction-aligned velocity metadata",
        "",
        f"Run: `{summary['run']['run_id']}`",
        "",
        f"Checkpoint policy: `{summary['checkpoint_policy']}`",
        "",
        summary["checkpoint_source"],
        "",
        "## Outputs",
        "",
        f"- Pooled figure: `{summary['figures']['pooled']}`",
        f"- Per-direction figure: `{summary['figures']['per_direction']}`",
        "",
        "## Bank",
        "",
        f"- Bank kind: `{bank['bank_kind']}`",
        f"- Directions: {bank['direction_count']} uniformly spaced angles",
        f"- Reach length: {bank['reach_length_m']:.12g} m",
        f"- Movement horizon: {bank['movement_horizon_steps']} steps",
        "- The bank is constructed in the no-delay materializer; the delayed "
        "eval-bank helper is not imported or edited.",
        "",
        "## Projection",
        "",
        summary["projection"],
        "",
        "## Summary",
        "",
        f"- GRU pooled samples: {pooled['n_samples']} "
        f"({pooled['n_replicates']} replicates x {pooled['n_directions']} directions)",
        f"- GRU pooled peak mean velocity: {pooled['peak_mean_velocity_m_s']:.6g} m/s "
        f"at {pooled['time_of_peak_mean_velocity_s']:.6g} s",
        f"- extLQG reach length: {ext['reach_length_m']:.12g} m",
        f"- extLQG samples: {ext['n_samples']}",
        f"- extLQG peak mean velocity: {ext['peak_mean_velocity_m_s']:.6g} m/s "
        f"at {ext['time_of_peak_mean_velocity_s']:.6g} s",
        f"- extLQG parity: {ext['parity_status']}",
        "",
        "## Direction Table",
        "",
        "| Direction | Angle (deg) | Unit vector | Peak mean velocity (m/s) | Peak time (s) |",
        "|---:|---:|---|---:|---:|",
    ]
    for profile in summary["direction_profiles"]:
        unit = profile["unit_vector"]
        lines.append(
            f"| {profile['direction_index']} | {profile['angle_deg']:.6g} | "
            f"[{unit[0]:.6g}, {unit[1]:.6g}] | "
            f"{profile['peak_mean_velocity_m_s']:.6g} | "
            f"{profile['time_of_peak_mean_velocity_s']:.6g} |"
        )
    lines.append("")
    return "\n".join(lines)


def direction_table_from_trial_specs(trial_specs: Any) -> list[dict[str, Any]]:
    """Return target direction metadata for an existing trial spec."""

    target_position = _target_position_sequence(trial_specs)
    initial_position = _initial_effector_position(trial_specs)
    direction, distance = _reach_direction(initial_position, target_position[:, -1, :])
    return [
        {
            "index": int(idx),
            "angle_deg": float((np.degrees(np.arctan2(vec[1], vec[0])) + 360.0) % 360.0),
            "unit_vector": [float(vec[0]), float(vec[1])],
            "target_distance_m": float(distance[idx]),
        }
        for idx, vec in enumerate(direction)
    ]


def infer_time_steps(trial_specs: Any) -> int:
    """Infer the trial time dimension from target arrays."""

    target_position = _target_position_sequence(trial_specs)
    return int(target_position.shape[1])


def infer_trial_count_from_targets(trial_specs: Any) -> int:
    """Infer trial count from target arrays."""

    target_position = _target_position_sequence(trial_specs)
    return int(target_position.shape[0])


def repo_relative(path: Path) -> str:
    """Return a repo-relative path when ``path`` is inside ``REPO_ROOT``."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
