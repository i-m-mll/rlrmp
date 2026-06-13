"""Materialize delayed no-PGD no-catch direction-split velocity profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import plotly.graph_objects as go
from feedbax._io import load_with_hyperparameters
from feedbax.types import TreeNamespace, dict_to_namespace
from plotly.subplots import make_subplots

from rlrmp.analysis.cs_game_card import OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR
from rlrmp.analysis.cs_game_card import materialize_reference
from rlrmp.analysis.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.cs_released_simulation import (
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
)
from rlrmp.analysis.delayed_reach_eval_bank import (
    DelayedReachEvalBank,
    make_delayed_reach_eval_bank,
)
from rlrmp.analysis.gru_pilot_figures import (
    RunFigureInputs,
    _initial_effector_position,
    _reach_direction,
    _rgba,
    _target_position_sequence,
    resolve_run_inputs,
)
from rlrmp.analysis.output_feedback import OutputFeedbackConfig
from rlrmp.analysis.trial_alignment import (
    align_trials,
    canonical_movement_horizon_from_metadata,
    pooled_trial_mean_with_band,
    trial_timing_from_specs,
    trim_to_full_support,
)
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p


EXPERIMENT = "6c36536"
TOPIC = "delayed_no_pgd_catch_prego1e5_no_catch_direction_split_final_checkpoints"
RUN_IDS = (
    "delayed_8d_no_pgd_catch0p5_prego1e5_lr3e-3_clip5_b64_seed42",
    "delayed_8d_no_pgd_catch0p5_prego1e5_lr1e-3_clip5_b64_seed42",
)
LABELS = (
    "p_catch=0.5, pre-go loss=1e5, lr=3e-3",
    "p_catch=0.5, pre-go loss=1e5, lr=1e-3",
)
GOOD_DIRECTION_INDICES = tuple(range(12))
BAD_DIRECTION_INDICES = tuple(range(12, 20))
PRE_GO_CONTEXT_STEPS = 10
REFERENCE_N_SAMPLES = 2100
BANK_VARIANTS = (
    {
        "key": "inferred_validation_list_bank",
        "description": (
            "Existing fixed-bank convention: directions are inferred from the first "
            "20 validation targets and rescaled to the median validation-target "
            "distance."
        ),
        "direction_source": "validation_targets",
        "uniform_reach_length_m": None,
    },
    {
        "key": "uniform_20dir_0p15m_bank",
        "description": (
            "Corrected diagnostic bank: 20 evenly spaced directions at 0.15 m, "
            "with the same go-cue grid and no-catch semantics."
        ),
        "direction_source": "uniform_grid",
        "uniform_reach_length_m": 0.15,
    },
)


@dataclass(frozen=True)
class DirectionGroupProfile:
    """Go-aligned target-radial velocity summary for one run and direction group."""

    run_id: str
    label: str
    group_name: str
    direction_indices: tuple[int, ...]
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_replicates: int
    n_trials_per_replicate: int
    n_pooled_samples: int
    peak_mean_forward_velocity_m_s: float
    time_of_peak_mean_forward_velocity_s: float


@dataclass(frozen=True)
class MatchedReferenceProfile:
    """Reach-matched extLQG velocity reference."""

    label: str
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    reach_length_m: float
    peak_mean_forward_velocity_m_s: float
    time_of_peak_mean_forward_velocity_s: float
    terminal_position_error_m: float
    parity_status: str


def main() -> None:
    """Generate the direction-split figure and sidecars."""

    output_dir = REPO_ROOT / "_artifacts" / EXPERIMENT / "figures" / TOPIC
    mkdir_p(output_dir)
    runs = resolve_run_inputs(experiment=EXPERIMENT, run_ids=RUN_IDS, labels=LABELS)
    outputs = []
    for variant in BANK_VARIANTS:
        all_profiles: list[DirectionGroupProfile] = []
        per_run_metadata: dict[str, Any] = {}
        reach_lengths = []
        direction_table: list[dict[str, Any]] | None = None
        for run in runs:
            profiles, metadata = evaluate_run_direction_groups(run, bank_variant=variant)
            all_profiles.extend(profiles)
            per_run_metadata[run.label] = metadata
            reach_lengths.append(float(metadata["evaluation_bank"]["reach_length_m"]))
            direction_table = metadata["directions"]

        if not np.allclose(reach_lengths, reach_lengths[0], atol=1e-12, rtol=0.0):
            raise ValueError(f"Expected one shared bank reach length; got {reach_lengths}")
        bank_reach_length = float(reach_lengths[0])
        reference = build_matched_extlqg_reference(
            reach_length_m=bank_reach_length,
            n_samples=REFERENCE_N_SAMPLES,
        )

        key = str(variant["key"])
        figure_path = output_dir / f"{key}_forward_velocity_direction_split_with_matched_extlqg.html"
        write_direction_split_figure(
            profiles=all_profiles,
            reference=reference,
            figure_path=figure_path,
            title_suffix=str(variant["description"]),
        )
        summary = build_summary(
            figure_path=figure_path,
            profiles=all_profiles,
            reference=reference,
            per_run_metadata=per_run_metadata,
            direction_table=direction_table or [],
            bank_variant=variant,
        )
        summary_path = output_dir / f"{key}_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata_path = output_dir / f"{key}_metadata.md"
        metadata_path.write_text(render_metadata(summary), encoding="utf-8")
        outputs.append(
            {
                "bank_variant": key,
                "figure": str(figure_path),
                "metadata": str(metadata_path),
                "summary": str(summary_path),
            }
        )
    print(json.dumps({"outputs": outputs}, indent=2, sort_keys=True))


def evaluate_run_direction_groups(
    run: RunFigureInputs,
    *,
    bank_variant: dict[str, Any],
) -> tuple[list[DirectionGroupProfile], dict[str, Any]]:
    """Evaluate one final checkpoint on the fixed no-catch delayed bank."""

    hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run.run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    horizon = canonical_movement_horizon_from_metadata(run.run_spec, default=None)
    source_direction_table = source_validation_direction_table(
        pair.task.validation_trials,
        direction_count=20,
    )
    bank = make_bank_for_variant(
        pair.task.validation_trials,
        bank_variant=bank_variant,
        movement_horizon_steps=horizon,
        source_direction_table=source_direction_table,
    )
    model, _hyperparameters = load_with_hyperparameters(
        run.artifact_dir / "trained_model.eqx",
        setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
    )
    trial_specs = bank.trial_specs
    eval_trial_count = infer_trial_count_from_targets(trial_specs)
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
        return states.mechanics.effector.vel

    velocity = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(0), n_replicates),
    )
    velocity_np = np.asarray(velocity, dtype=np.float64)
    target_position = _target_position_sequence(trial_specs)
    initial_position = _initial_effector_position(trial_specs)
    direction, distance = _reach_direction(initial_position, target_position[:, -1, :])
    forward = np.sum(velocity_np * direction[None, :, None, :], axis=-1)
    timing = trial_timing_from_specs(
        trial_specs,
        n_time_steps=int(forward.shape[-1]),
        movement_horizon_steps=horizon,
    )
    direction_count = int(bank.metadata["direction_count"])
    go_cue_steps = tuple(int(step) for step in bank.metadata["go_cue_steps"])
    direction_index = np.tile(np.arange(direction_count, dtype=np.int64), len(go_cue_steps))
    if direction_index.shape[0] != eval_trial_count:
        raise ValueError(
            f"Direction index length {direction_index.shape[0]} does not match "
            f"trial count {eval_trial_count}"
        )

    dt = float(run.run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    profiles = [
        summarize_group(
            run=run,
            group_name="good directions 0-11",
            direction_indices=GOOD_DIRECTION_INDICES,
            forward=forward,
            go_index=timing.go_index,
            direction_index=direction_index,
            dt=dt,
            movement_horizon_steps=int(timing.movement_horizon_steps),
            n_replicates=n_replicates,
        ),
        summarize_group(
            run=run,
            group_name="bad directions 12-19",
            direction_indices=BAD_DIRECTION_INDICES,
            forward=forward,
            go_index=timing.go_index,
            direction_index=direction_index,
            dt=dt,
            movement_horizon_steps=int(timing.movement_horizon_steps),
            n_replicates=n_replicates,
        ),
    ]
    directions = direction[:direction_count]
    distances = distance[:direction_count]
    metadata = {
        "run_id": run.run_id,
        "label": run.label,
        "bank_variant": bank_variant["key"],
        "bank_variant_description": bank_variant["description"],
        "checkpoint_source": str(run.artifact_dir / "trained_model.eqx"),
        "checkpoint_policy": "final_checkpoint",
        "evaluation_bank": bank.metadata,
        "projection": "target-aligned radial velocity: dot(effector velocity, unit(target - initial_position))",
        "directions": [
            {
                "index": int(idx),
                "angle_deg": float((np.degrees(np.arctan2(vec[1], vec[0])) + 360.0) % 360.0),
                "unit_vector": [float(vec[0]), float(vec[1])],
                "bank_target_distance_m": float(distances[idx]),
                "source_validation_target_distance_m": (
                    source_direction_table[idx]["source_validation_target_distance_m"]
                    if idx < len(source_direction_table)
                    else None
                ),
                "source_validation_angle_deg": (
                    source_direction_table[idx]["angle_deg"]
                    if idx < len(source_direction_table)
                    else None
                ),
            }
            for idx, vec in enumerate(directions)
        ],
    }
    return profiles, metadata


def make_bank_for_variant(
    source_trial_specs: Any,
    *,
    bank_variant: dict[str, Any],
    movement_horizon_steps: int | None,
    source_direction_table: list[dict[str, Any]],
) -> DelayedReachEvalBank:
    """Return the requested delayed no-catch evaluation bank."""

    uniform_reach = bank_variant["uniform_reach_length_m"]
    bank = make_delayed_reach_eval_bank(
        source_trial_specs,
        catch=False,
        movement_horizon_steps=movement_horizon_steps,
        direction_source=str(bank_variant["direction_source"]),
        reach_length_m=None if uniform_reach is None else float(uniform_reach),
    )
    return DelayedReachEvalBank(
        trial_specs=bank.trial_specs,
        metadata={
            **bank.metadata,
            "bank_variant": str(bank_variant["key"]),
            "bank_variant_description": str(bank_variant["description"]),
            "source_validation_direction_table": source_direction_table,
        },
    )


def summarize_group(
    *,
    run: RunFigureInputs,
    group_name: str,
    direction_indices: tuple[int, ...],
    forward: np.ndarray,
    go_index: np.ndarray,
    direction_index: np.ndarray,
    dt: float,
    movement_horizon_steps: int,
    n_replicates: int,
) -> DirectionGroupProfile:
    """Return pooled go-aligned mean and SD for a direction group."""

    mask = np.isin(direction_index, np.asarray(direction_indices, dtype=np.int64))
    if not np.any(mask):
        raise ValueError(f"No trials selected for {group_name}")
    group_forward = forward[:, mask, :]
    group_go = go_index[mask]
    aligned_forward, center = align_trials(group_forward, group_go)
    _support, support_slice = trim_to_full_support(aligned_forward)
    support_start = int(support_slice.start or 0)
    support_stop = int(support_slice.stop or aligned_forward.shape[-1])
    requested_start = max(0, int(center) - PRE_GO_CONTEXT_STEPS)
    requested_stop = min(
        int(aligned_forward.shape[-1]),
        int(center) + int(movement_horizon_steps),
    )
    start = max(support_start, requested_start)
    stop = min(support_stop, requested_stop)
    if stop <= start:
        raise ValueError(f"Go-aligned profile for {group_name} has no full-support samples")
    window = aligned_forward[..., start:stop]
    mean, lower, upper = pooled_trial_mean_with_band(window, band="sd", trim=False)
    std = (upper - lower) / 2.0
    time_s = (np.arange(start, stop, dtype=np.float64) - float(center)) * float(dt)
    peak_idx = int(np.argmax(mean))
    return DirectionGroupProfile(
        run_id=run.run_id,
        label=run.label,
        group_name=group_name,
        direction_indices=direction_indices,
        time_s=time_s,
        mean=mean,
        std=std,
        n_replicates=n_replicates,
        n_trials_per_replicate=int(group_forward.shape[1]),
        n_pooled_samples=int(n_replicates * group_forward.shape[1]),
        peak_mean_forward_velocity_m_s=float(mean[peak_idx]),
        time_of_peak_mean_forward_velocity_s=float(time_s[peak_idx]),
    )


def source_validation_direction_table(
    trial_specs: Any,
    *,
    direction_count: int,
) -> list[dict[str, Any]]:
    """Return source validation-target directions before fixed-bank rescaling."""

    target_position = _target_position_sequence(trial_specs)
    initial_position = _initial_effector_position(trial_specs)
    direction, distance = _reach_direction(initial_position, target_position[:, -1, :])
    rows = []
    for idx, vec in enumerate(direction[:direction_count]):
        rows.append(
            {
                "index": int(idx),
                "angle_deg": float((np.degrees(np.arctan2(vec[1], vec[0])) + 360.0) % 360.0),
                "unit_vector": [float(vec[0]), float(vec[1])],
                "source_validation_target_distance_m": float(distance[idx]),
            }
        )
    return rows


def build_matched_extlqg_reference(
    *,
    reach_length_m: float,
    n_samples: int,
) -> MatchedReferenceProfile:
    """Simulate the extLQG comparator from a 13.5 cm target-centered initial state."""

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
        label="C&S extLQG/output-feedback 8D, reach matched to GRU bank",
        time_s=np.arange(mean.shape[0], dtype=np.float64) * float(reference.plant.dt),
        mean=mean,
        std=std,
        n_samples=n_samples,
        reach_length_m=float(reach_length_m),
        peak_mean_forward_velocity_m_s=float(mean[peak_idx]),
        time_of_peak_mean_forward_velocity_s=float(peak_idx * float(reference.plant.dt)),
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


def write_direction_split_figure(
    *,
    profiles: list[DirectionGroupProfile],
    reference: MatchedReferenceProfile,
    figure_path: Path,
    title_suffix: str,
) -> None:
    """Write one two-panel Plotly figure."""

    labels = list(dict.fromkeys(profile.label for profile in profiles))
    fig = make_subplots(
        rows=len(labels),
        cols=1,
        shared_xaxes=True,
        subplot_titles=labels,
        vertical_spacing=0.08,
    )
    colors = {
        "good directions 0-11": "#2563eb",
        "bad directions 12-19": "#dc2626",
    }
    for row_idx, label in enumerate(labels, start=1):
        row_profiles = [profile for profile in profiles if profile.label == label]
        for profile in row_profiles:
            color = colors[profile.group_name]
            upper = profile.mean + profile.std
            lower = profile.mean - profile.std
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=_rgba(color, 0.14),
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    name=f"{profile.group_name} mean +/- 1 SD",
                    legendgroup=profile.group_name,
                    showlegend=row_idx == 1,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=profile.time_s,
                    y=profile.mean,
                    mode="lines",
                    line={"color": color, "width": 2.4},
                    name=profile.group_name,
                    legendgroup=profile.group_name,
                    showlegend=row_idx == 1,
                ),
                row=row_idx,
                col=1,
            )
        upper = reference.mean + reference.std
        lower = reference.mean - reference.std
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([reference.time_s, reference.time_s[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=_rgba("#111827", 0.09),
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                name="extLQG mean +/- 1 SD",
                legendgroup="extlqg",
                showlegend=row_idx == 1,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=reference.time_s,
                y=reference.mean,
                mode="lines",
                line={"color": "#111827", "width": 2.4, "dash": "dash"},
                name="matched-reach extLQG",
                legendgroup="extlqg",
                showlegend=row_idx == 1,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_vline(
            x=0.0,
            line={"color": "black", "dash": "dash", "width": 1},
            row=row_idx,
            col=1,
        )
    fig.update_layout(
        title="Delayed no-PGD direction-split velocity (final checkpoints)",
        width=920,
        height=860,
        margin={"l": 72, "r": 24, "t": 78, "b": 112},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "x": 0.0,
            "xanchor": "left",
            "y": -0.16,
            "yanchor": "top",
        },
    )
    fig.update_xaxes(title_text="Time relative to go cue (s)", row=len(labels), col=1)
    fig.update_yaxes(title_text="Target-radial velocity (m/s)", zeroline=True)
    fig.write_html(figure_path)


def build_summary(
    *,
    figure_path: Path,
    profiles: list[DirectionGroupProfile],
    reference: MatchedReferenceProfile,
    per_run_metadata: dict[str, Any],
    direction_table: list[dict[str, Any]],
    bank_variant: dict[str, Any],
) -> dict[str, Any]:
    """Return JSON-compatible materialization summary."""

    return {
        "schema_version": "rlrmp.delayed_no_pgd_direction_split_velocity.v1",
        "issue": EXPERIMENT,
        "topic": TOPIC,
        "bank_variant": {
            "key": bank_variant["key"],
            "description": bank_variant["description"],
            "uniform_reach_length_m": bank_variant["uniform_reach_length_m"],
        },
        "figure": str(figure_path),
        "checkpoint_policy": "final_checkpoint",
        "checkpoint_source": "trained_model.eqx final output in each run artifact directory",
        "direction_groups": {
            "good directions 0-11": list(GOOD_DIRECTION_INDICES),
            "bad directions 12-19": list(BAD_DIRECTION_INDICES),
        },
        "directions": direction_table,
        "projection": "target-aligned radial velocity: dot(effector velocity, unit(target - initial_position))",
        "error_band": (
            "GRU bands are mean +/- 1 SD over pooled replicate-trials "
            "(replicates x fixed-bank go-cue/direction trials); extLQG band is "
            "mean +/- 1 SD over stochastic analytical rollouts"
        ),
        "runs": per_run_metadata,
        "profiles": [
            {
                "run_id": profile.run_id,
                "label": profile.label,
                "group_name": profile.group_name,
                "direction_indices": list(profile.direction_indices),
                "n_replicates": profile.n_replicates,
                "n_trials_per_replicate": profile.n_trials_per_replicate,
                "n_pooled_samples": profile.n_pooled_samples,
                "peak_mean_forward_velocity_m_s": profile.peak_mean_forward_velocity_m_s,
                "time_of_peak_mean_forward_velocity_s": (
                    profile.time_of_peak_mean_forward_velocity_s
                ),
                "time_start_s": float(profile.time_s[0]),
                "time_stop_s": float(profile.time_s[-1]),
            }
            for profile in profiles
        ],
        "extlqg_reference": {
            "label": reference.label,
            "reach_length_m": reference.reach_length_m,
            "n_samples": reference.n_samples,
            "peak_mean_forward_velocity_m_s": reference.peak_mean_forward_velocity_m_s,
            "time_of_peak_mean_forward_velocity_s": reference.time_of_peak_mean_forward_velocity_s,
            "terminal_position_error_m": reference.terminal_position_error_m,
            "parity_status": reference.parity_status,
            "time_start_s": float(reference.time_s[0]),
            "time_stop_s": float(reference.time_s[-1]),
        },
    }


def render_metadata(summary: dict[str, Any]) -> str:
    """Render the requested human-readable metadata note."""

    ext = summary["extlqg_reference"]
    first_run = next(iter(summary["runs"].values()))
    bank = first_run["evaluation_bank"]
    variant = summary["bank_variant"]
    directions = summary["directions"]
    good = summary["direction_groups"]["good directions 0-11"]
    bad = summary["direction_groups"]["bad directions 12-19"]
    lines = [
        "# Delayed no-PGD no-catch direction-split velocity metadata",
        "",
        f"Figure: `{summary['figure']}`",
        "",
        f"Bank variant: `{variant['key']}`",
        "",
        variant["description"],
        "",
        "## Direction groups",
        "",
        f"- Good directions: {good}",
        f"- Bad directions: {bad}",
        "",
        "The good/bad labels preserve the previous diagnostics' index split. "
        "For the uniform 20-direction bank this is a continuity split, not a "
        "fresh per-direction failure classification.",
        "",
        "Direction indices and target angles are from the generated fixed no-catch "
        "evaluation bank. Source validation angle/distance columns record the "
        "mixed validation-target list that the original inferred bank uses.",
        "",
        "| Direction index | Bank angle (deg) | Bank unit vector | Bank target distance (m) | Source validation angle (deg) | Source validation distance (m) |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for item in directions:
        unit = item["unit_vector"]
        source_angle = item.get("source_validation_angle_deg")
        source_distance = item.get("source_validation_target_distance_m")
        lines.append(
            f"| {item['index']} | {item['angle_deg']:.6g} | "
            f"[{unit[0]:.6g}, {unit[1]:.6g}] | "
            f"{item['bank_target_distance_m']:.12g} | "
            f"{source_angle:.6g} | "
            f"{source_distance:.12g} |"
        )
    lines.extend(
        [
            "",
            "## Reach length and comparator parity",
            "",
            f"- GRU no-catch eval-bank reach length: {bank['reach_length_m']:.16g} m",
            f"- extLQG comparator reach length used in this figure: {ext['reach_length_m']:.16g} m",
            "- extLQG comparator: C&S output-feedback 8D fixed-point path, "
            f"{ext['parity_status']}",
            f"- extLQG samples: {ext['n_samples']}",
            f"- extLQG peak mean target-radial velocity: "
            f"{ext['peak_mean_forward_velocity_m_s']:.6g} m/s at "
            f"{ext['time_of_peak_mean_forward_velocity_s']:.6g} s",
            "",
            "## Checkpoints and samples",
            "",
            "- Checkpoint source: final `trained_model.eqx` for each run; no "
            "validation-selected checkpoint assembly was used.",
            "- Projection: target-aligned radial velocity, computed as "
            "`dot(effector velocity, unit(target - initial_position))`.",
            "- GRU error bands: mean +/- 1 SD over pooled replicate-trials, where "
            "samples are replicate x fixed-bank go-cue/direction trials. "
            "Replicates are not averaged before computing the band.",
            "- extLQG error band: mean +/- 1 SD over stochastic analytical rollouts.",
            "",
            "| Run | Group | Direction indices | Replicates | Trials per replicate | Pooled samples | Peak mean (m/s) | Peak time (s) |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for profile in summary["profiles"]:
        lines.append(
            f"| {profile['label']} | {profile['group_name']} | "
            f"{profile['direction_indices']} | {profile['n_replicates']} | "
            f"{profile['n_trials_per_replicate']} | {profile['n_pooled_samples']} | "
            f"{profile['peak_mean_forward_velocity_m_s']:.6g} | "
            f"{profile['time_of_peak_mean_forward_velocity_s']:.6g} |"
        )
    lines.append("")
    return "\n".join(lines)


def infer_trial_count_from_targets(trial_specs: Any) -> int:
    """Infer trial count from target arrays."""

    for target_spec in getattr(trial_specs, "targets", {}).values():
        value = getattr(target_spec, "value", None)
        if getattr(value, "shape", None) is not None and len(value.shape) >= 1:
            return int(value.shape[0])
    raise ValueError("Could not infer trial count from trial specs")


if __name__ == "__main__":
    main()
