"""Materialize the e148f33 nominal velocity-profile comparison figure."""

from __future__ import annotations

# ruff: noqa: E402

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.plot import save_figure

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    build_no_integrator_game,
)
from rlrmp.analysis.math.cs_released_simulation import (
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
    sample_forward_noise_draws,
    simulate_lqg_released_forward,
    simulate_robust_released_forward,
)
from rlrmp.analysis.math.hinf_riccati import find_gamma_star, solve_hinf_riccati, solve_lqr
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_output_feedback_gains,
)
from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
from rlrmp.analysis.pipelines.gru_pilot_figures import (
    DEFAULT_N_ROLLOUT_TRIALS,
    repeat_single_validation_trial,
)
from rlrmp.paths import REPO_ROOT
from rlrmp.train.task_model import setup_task_model_pair


ISSUE_ID = "e148f33"
TOPIC = "nominal_velocity_profile_comparison"
RUN_ID = "adaptive_curriculum_3500to1000"
BASELINE_ISSUE_ID = "08483d5"
BASELINE_RUN_ID = "h0_6d_no_pgd_const_band16_cpu"
N_ROLLOUT_TRIALS = DEFAULT_N_ROLLOUT_TRIALS
ANALYTICAL_SEED = 376023
GRU_ROLLOUT_SEED = 0

RUN_SPEC_PATH = REPO_ROOT / "results" / ISSUE_ID / "runs" / f"{RUN_ID}.json"
RUN_CHECKPOINT = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE_ID
    / "runs"
    / RUN_ID
    / "checkpoints"
    / "checkpoint_0019500"
)
BASELINE_RUN_SPEC_PATH = (
    REPO_ROOT / "results" / BASELINE_ISSUE_ID / "runs" / f"{BASELINE_RUN_ID}.json"
)
BASELINE_CHECKPOINT = (
    REPO_ROOT
    / "_artifacts"
    / BASELINE_ISSUE_ID
    / "runs"
    / BASELINE_RUN_ID
    / "checkpoints"
    / "checkpoint_0012000"
)


@dataclass(frozen=True)
class Profile:
    """One nominal forward-velocity profile."""

    label: str
    kind: str
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    source: str
    line_color: str
    line_dash: str = "solid"
    checkpoint: str | None = None
    run_spec: str | None = None
    terminal_position_error_m: float | None = None
    parity_status: str | None = None

    @property
    def peak_forward_velocity_m_s(self) -> float:
        """Return peak mean forward velocity."""

        return float(np.max(self.mean))

    @property
    def time_of_peak_forward_velocity_s(self) -> float:
        """Return time of peak mean forward velocity."""

        return float(self.time_s[int(np.argmax(self.mean))])


def main() -> None:
    """Write the figure, sidecars, and summary."""

    adaptive = evaluate_gru_checkpoint_profile(
        label="adaptive_curriculum_3500to1000 final",
        kind="gru_adaptive_curriculum_final_checkpoint",
        run_spec_path=RUN_SPEC_PATH,
        checkpoint_dir=RUN_CHECKPOINT,
        line_color="#2563eb",
    )
    baseline = evaluate_gru_checkpoint_profile(
        label="08483d5 no-PGD 12k baseline",
        kind="gru_no_pgd_pre_adversary_checkpoint",
        run_spec_path=BASELINE_RUN_SPEC_PATH,
        checkpoint_dir=BASELINE_CHECKPOINT,
        line_color="#059669",
    )
    analytical = materialize_analytical_profiles(n_samples=max(adaptive.n_samples, baseline.n_samples))
    profiles = (adaptive, baseline, *analytical)

    fig = make_plotly_figure(profiles)
    summary = build_summary(profiles)
    spec = build_figure_spec(summary)
    saved = save_figure(
        fig=fig,
        spec=spec,
        package="rlrmp",
        experiment=ISSUE_ID,
        topic=TOPIC,
        extra_packages=["rlrmp"],
    )
    render_dir = require_path(saved["render_path"]).parent
    csv_path = write_profile_csv(profiles, render_dir=render_dir)
    png_path = write_png(profiles, render_dir=render_dir)
    summary_path = render_dir / "figure_summary.json"
    summary["outputs"] = {
        "html": repo_ref(require_path(saved["render_path"])),
        "png": repo_ref(png_path),
        "csv": repo_ref(csv_path),
        "summary": repo_ref(summary_path),
        "spec": repo_ref(require_path(saved["spec_path"])),
        "spec_symlink": (
            repo_ref(require_path(saved["symlink_path"])) if saved.get("symlink_path") else None
        ),
    }
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))


def evaluate_gru_checkpoint_profile(
    *,
    label: str,
    kind: str,
    run_spec_path: Path,
    checkpoint_dir: Path,
    line_color: str,
) -> Profile:
    """Evaluate one GRU checkpoint on the fixed nominal validation reach."""

    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    model = eqx.tree_deserialise_leaves(checkpoint_dir / "model.eqx", pair.model)
    trial_specs = repeat_single_validation_trial(pair.task.validation_trials, N_ROLLOUT_TRIALS)
    initial_velocity = initial_effector_velocity(trial_specs)
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: is_replicate_array(leaf, n_replicates),
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        states = pair.task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, N_ROLLOUT_TRIALS),
        )
        return jnp.concatenate(
            [initial_velocity[:, None, :], states.mechanics.effector.vel],
            axis=1,
        )

    velocity = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(GRU_ROLLOUT_SEED), n_replicates),
    )
    velocity_np = np.asarray(velocity, dtype=np.float64)
    forward = velocity_np[..., 0]
    pooled = forward.reshape(n_replicates * N_ROLLOUT_TRIALS, forward.shape[-1])
    dt = float(run_spec.get("game_card", {}).get("dt", getattr(hps, "dt", 0.01)))
    return Profile(
        label=label,
        kind=kind,
        time_s=np.arange(pooled.shape[-1], dtype=np.float64) * dt,
        mean=np.mean(pooled, axis=0),
        std=np.std(pooled, axis=0),
        n_samples=int(pooled.shape[0]),
        source="gru_checkpoint_nominal_rollout",
        checkpoint=repo_ref(checkpoint_dir),
        run_spec=repo_ref(run_spec_path),
        line_color=line_color,
    )


def materialize_analytical_profiles(*, n_samples: int) -> tuple[Profile, Profile]:
    """Build 6D extLQG and H-infinity nominal profiles with common random draws."""

    plant, schedule = build_no_integrator_game()
    config = OutputFeedbackConfig(n_phys=6)
    gamma_star = find_gamma_star(plant, schedule)
    lqr_solution = solve_lqr(plant, schedule)
    hinf_solution = solve_hinf_riccati(
        plant,
        schedule,
        OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR * gamma_star,
    )
    x0 = make_cs_output_feedback_initial_state(plant, config)
    covariances = default_cs_noise_covariances(plant, config)
    extlqg_path = build_extlqg_comparator_path(
        plant,
        lqr_solution.K,
        covariances,
        schedule=schedule,
        config=config,
    )
    robust_covariances = robust_estimator_covariances(
        plant,
        schedule,
        hinf_solution.gamma,
        config,
    )
    robust_gains = robust_output_feedback_gains(
        plant,
        schedule,
        hinf_solution,
        robust_covariances,
        config,
    )

    extlqg_rollouts = []
    hinf_rollouts = []
    for key in jr.split(jr.PRNGKey(ANALYTICAL_SEED), n_samples):
        draws = sample_forward_noise_draws(key, T=schedule.T, covariances=covariances)
        extlqg_rollouts.append(
            simulate_lqg_released_forward(
                plant,
                extlqg_path.controller_gains,
                x0,
                draws=draws,
                covariances=covariances,
                estimator_gains=extlqg_path.estimator_gains,
                config=config,
            )
        )
        hinf_rollouts.append(
            simulate_robust_released_forward(
                plant,
                schedule,
                hinf_solution,
                x0,
                draws=draws,
                covariances=covariances,
                gains=robust_gains,
                config=config,
            )
        )

    vel_lo, _vel_hi = plant.vel_slice
    extlqg_forward = np.stack(
        [np.asarray(rollout.x[:, vel_lo], dtype=np.float64) for rollout in extlqg_rollouts],
        axis=0,
    )
    hinf_forward = np.stack(
        [np.asarray(rollout.x[:, vel_lo], dtype=np.float64) for rollout in hinf_rollouts],
        axis=0,
    )
    time_s = np.arange(schedule.T + 1, dtype=np.float64) * float(plant.dt)
    return (
        Profile(
            label="6D analytical extLQG nominal",
            kind="analytical_extlqg_6d_output_feedback",
            time_s=time_s,
            mean=np.mean(extlqg_forward, axis=0),
            std=np.std(extlqg_forward, axis=0),
            n_samples=n_samples,
            source="rlrmp.analysis.math.cs_released_simulation",
            line_color="#111827",
            line_dash="dash",
            terminal_position_error_m=float(
                np.mean([rollout.terminal_position_error for rollout in extlqg_rollouts])
            ),
            parity_status=extlqg_path.parity_status,
        ),
        Profile(
            label="output-feedback H-infinity nominal",
            kind="analytical_hinf_6d_output_feedback",
            time_s=time_s,
            mean=np.mean(hinf_forward, axis=0),
            std=np.std(hinf_forward, axis=0),
            n_samples=n_samples,
            source="rlrmp.analysis.math.cs_released_simulation",
            line_color="#dc2626",
            line_dash="dot",
            terminal_position_error_m=float(
                np.mean([rollout.terminal_position_error for rollout in hinf_rollouts])
            ),
            parity_status=(
                "6D no-integrator output-feedback robust estimator/controller; "
                f"gamma_factor={OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR:g}"
            ),
        ),
    )


def make_plotly_figure(profiles: tuple[Profile, ...]) -> go.Figure:
    """Build the interactive overlay figure."""

    fig = go.Figure()
    for profile in profiles:
        upper = profile.mean + profile.std
        lower = profile.mean - profile.std
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=rgba(profile.line_color, 0.12),
                line={"color": "rgba(0,0,0,0)"},
                hoverinfo="skip",
                name=f"{profile.label} mean +/- 1 SD",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=profile.time_s,
                y=profile.mean,
                mode="lines",
                line={
                    "color": profile.line_color,
                    "width": 2.6,
                    "dash": profile.line_dash,
                },
                name=profile.label,
            )
        )
    fig.update_layout(
        title="Nominal forward velocity comparison for e148f33",
        width=980,
        height=600,
        margin={"l": 72, "r": 28, "t": 72, "b": 64},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_xaxes(title_text="Time (s)", range=[0.0, 0.6], zeroline=False)
    fig.update_yaxes(title_text="Forward velocity (m/s)", zeroline=True)
    return fig


def write_png(profiles: tuple[Profile, ...], *, render_dir: Path) -> Path:
    """Write a static PNG sidecar using the same profile arrays."""

    path = render_dir / "figure.png"
    fig, ax = plt.subplots(figsize=(9.8, 6.0), dpi=160)
    for profile in profiles:
        ax.fill_between(
            profile.time_s,
            profile.mean - profile.std,
            profile.mean + profile.std,
            color=profile.line_color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            profile.time_s,
            profile.mean,
            color=profile.line_color,
            linestyle=plotly_dash_to_mpl(profile.line_dash),
            linewidth=2.0,
            label=profile.label,
        )
    ax.set_xlim(0.0, 0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Forward velocity (m/s)")
    ax.set_title("Nominal forward velocity comparison for e148f33")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_profile_csv(profiles: tuple[Profile, ...], *, render_dir: Path) -> Path:
    """Write a wide CSV sidecar for the plotted arrays."""

    time_s = profiles[0].time_s
    columns = [time_s]
    headers = ["time_s"]
    for profile in profiles:
        if profile.time_s.shape != time_s.shape or not np.allclose(profile.time_s, time_s):
            raise ValueError(f"Profile {profile.label} has a nonmatching time axis")
        slug = slugify(profile.label)
        columns.extend([profile.mean, profile.std])
        headers.extend([f"{slug}_mean_m_s", f"{slug}_std_m_s"])
    path = render_dir / "velocity_profile_comparison.csv"
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
    return path


def build_summary(profiles: tuple[Profile, ...]) -> dict[str, Any]:
    """Return the JSON-compatible figure summary without output paths."""

    return {
        "schema_version": "rlrmp.e148f33.nominal_velocity_profile_comparison.v1",
        "issue": ISSUE_ID,
        "topic": TOPIC,
        "materializer": repo_ref(Path(__file__).resolve()),
        "plot_contract": {
            "quantity": "target-axis forward velocity",
            "condition": "nominal fixed +x validation reach",
            "bands": "mean +/- 1 SD",
            "gru_rollout_trials_per_replicate": N_ROLLOUT_TRIALS,
            "gru_rollout_seed": GRU_ROLLOUT_SEED,
            "analytical_seed": ANALYTICAL_SEED,
            "time_axis": "0.00s through 0.60s, including initial state sample",
        },
        "profiles": {
            profile.label: {
                "kind": profile.kind,
                "source": profile.source,
                "checkpoint": profile.checkpoint,
                "run_spec": profile.run_spec,
                "n_samples": profile.n_samples,
                "peak_mean_forward_velocity_m_s": profile.peak_forward_velocity_m_s,
                "time_of_peak_mean_forward_velocity_s": profile.time_of_peak_forward_velocity_s,
                "terminal_position_error_m": profile.terminal_position_error_m,
                "parity_status": profile.parity_status,
            }
            for profile in profiles
        },
        "limitations": [
            "The final e148f33 training_summary.json and training_diagnostics sidecars are known stale from the 12500-batch gate, so this figure evaluates checkpoint_0019500 directly.",
            "The GRU profiles are stochastic nominal rollouts of the first validation reach, pooled over five replicates and 64 repeats per replicate.",
            "The analytical profiles are 6D no-integrator output-feedback comparators under the same noise draws; they are not trained model checkpoints.",
        ],
        "no_training_or_remote_gpu": True,
    }


def build_figure_spec(summary: dict[str, Any]) -> dict[str, Any]:
    """Build the tracked Feedbax figure spec."""

    return {
        "schema_version": "rlrmp.e148f33.figure_spec.v1",
        "issue": ISSUE_ID,
        "topic": TOPIC,
        "materializer": summary["materializer"],
        "inputs": [
            {"role": "adaptive_run_spec", "path": repo_ref(RUN_SPEC_PATH)},
            {"role": "adaptive_checkpoint_model", "path": repo_ref(RUN_CHECKPOINT / "model.eqx")},
            {"role": "adaptive_checkpoint_metadata", "path": repo_ref(RUN_CHECKPOINT / "metadata.json")},
            {"role": "baseline_run_spec", "path": repo_ref(BASELINE_RUN_SPEC_PATH)},
            {"role": "baseline_checkpoint_model", "path": repo_ref(BASELINE_CHECKPOINT / "model.eqx")},
            {"role": "baseline_checkpoint_metadata", "path": repo_ref(BASELINE_CHECKPOINT / "metadata.json")},
        ],
        "analytical_provenance": {
            "analytical_builders": [
                "rlrmp.analysis.math.cs_game_card.build_no_integrator_game",
                "rlrmp.analysis.math.cs_released_simulation",
                "rlrmp.analysis.math.output_feedback",
            ],
        },
        "plot_contract": summary["plot_contract"],
        "profile_summaries": summary["profiles"],
        "no_training_or_remote_gpu": True,
    }


def initial_effector_velocity(trial_specs: Any) -> jnp.ndarray:
    """Return the trial initial effector velocity array."""

    for init_state in trial_specs.inits.values():
        velocity = getattr(init_state, "vel", None)
        if velocity is not None:
            return velocity
        shape = getattr(init_state, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[-1] >= 4:
            return jnp.asarray(init_state)[..., 2:4]
    raise ValueError("Trial spec does not include an effector velocity initial state")


def is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    """Return true when an array's leading axis indexes model replicates."""

    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


def repo_ref(path: Path) -> str:
    """Return a repo-relative path string."""

    return str(path.relative_to(REPO_ROOT))


def require_path(path: Any) -> Path:
    """Return a non-null Path from Feedbax save output."""

    if path is None:
        raise ValueError("Expected Feedbax save_figure to return a path")
    return Path(path)


def rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba CSS syntax."""

    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def plotly_dash_to_mpl(dash: str) -> str:
    """Map Plotly dash labels to Matplotlib line styles."""

    return {"solid": "-", "dash": "--", "dot": ":"}.get(dash, "-")


def slugify(label: str) -> str:
    """Return a compact CSV-safe label slug."""

    return (
        label.lower()
        .replace("08483d5 ", "")
        .replace("6d ", "six_d_")
        .replace("h-infinity", "hinf")
        .replace(" ", "_")
        .replace("-", "_")
    )


def json_ready(value: Any) -> Any:
    """Convert paths and NumPy scalars for JSON output."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    main()
