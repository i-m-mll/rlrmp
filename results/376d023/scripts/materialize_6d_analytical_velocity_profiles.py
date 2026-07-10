"""Materialize 6D analytical comparators for issue 376d023.

This script is intentionally experiment-local. It compares 6D no-integrator
analytical output-feedback models against the two latest h0 rows from 020a65b
without launching training or touching remote resources.
"""

from __future__ import annotations
from rlrmp.paths import portable_repo_path
from rlrmp.viz.colors import hex_to_rgba

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.random as jr
import numpy as np
import plotly.graph_objects as go

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
    delayed_observation_matrix,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_output_feedback_gains,
)
from rlrmp.io import update_marked_section


jax.config.update("jax_enable_x64", True)

CHECKOUT_ROOT = Path(__file__).resolve().parents[3]
ISSUE_ID = "376d023"
SOURCE_ISSUE_ID = "020a65b"
OUTPUT_NAME = "6d_analytical_velocity_profiles"
DT_S = 0.01
NO_PGD_RUN_ID = (
    "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_no_pgd_lr3e-3_clip5_b64"
)
PGD_RUN_ID = (
    "target_relative_multitarget_h0_fullqrf_warmcos__proprio_cal_small_pgd_ofb_lr3e-3_clip5_b64"
)
NO_PGD_NPZ = (
    CHECKOUT_ROOT
    / "_artifacts"
    / SOURCE_ISSUE_ID
    / "evaluation_diagnostics"
    / "gru_h0_pgd_bank_two_rows_validation_selected"
    / f"{NO_PGD_RUN_ID}.npz"
)
PGD_NPZ = NO_PGD_NPZ.with_name(f"{PGD_RUN_ID}.npz")
NO_PGD_RUN_SPEC = CHECKOUT_ROOT / "results" / SOURCE_ISSUE_ID / "runs" / NO_PGD_RUN_ID / "run.json"
TRACKED_FIGURE_DIR = CHECKOUT_ROOT / "results" / ISSUE_ID / "figures" / OUTPUT_NAME
TRACKED_RUN_DIR = CHECKOUT_ROOT / "results" / ISSUE_ID / "runs"
NOTE_PATH = CHECKOUT_ROOT / "results" / ISSUE_ID / "notes" / f"{OUTPUT_NAME}.md"
BULK_FIGURE_DIR = CHECKOUT_ROOT / "_artifacts" / ISSUE_ID / "figures" / OUTPUT_NAME
TEACHER_DIR = CHECKOUT_ROOT / "_artifacts" / ISSUE_ID / "analytical_teachers"


@dataclass(frozen=True)
class Profile:
    """One velocity profile with pooled stochastic bands."""

    label: str
    kind: str
    time_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    run_id: str | None = None
    terminal_position_error_m: float | None = None
    parity_status: str | None = None

    @property
    def peak_forward_velocity_m_s(self) -> float:
        """Return peak mean forward velocity."""

        return float(np.max(self.mean))

    @property
    def time_of_peak_forward_velocity_s(self) -> float:
        """Return the time of peak mean forward velocity."""

        return float(self.time_s[int(np.argmax(self.mean))])


def main() -> None:
    """Write all issue outputs."""

    tracked_paths = (TRACKED_FIGURE_DIR, TRACKED_RUN_DIR, NOTE_PATH.parent)
    bulk_paths = (BULK_FIGURE_DIR, TEACHER_DIR)
    for path in (*tracked_paths, *bulk_paths):
        path.mkdir(parents=True, exist_ok=True)

    no_pgd = load_gru_profile(
        NO_PGD_NPZ,
        run_id=NO_PGD_RUN_ID,
        label="020a65b h0 no-PGD",
    )
    pgd = load_gru_profile(
        PGD_NPZ,
        run_id=PGD_RUN_ID,
        label="020a65b h0 PGD",
    )
    n_samples = max(no_pgd.n_samples, pgd.n_samples)
    extlqg, hinf, teacher_manifest = materialize_analytical_profiles(n_samples=n_samples)
    profiles = (extlqg, hinf, no_pgd, pgd)

    html_path = write_velocity_plot(profiles)
    csv_path = write_profile_csv(profiles)
    summary = write_summary(profiles, html_path=html_path, csv_path=csv_path)
    spec_path = write_figure_spec(summary, teacher_manifest)
    distillation_plan_path = write_distillation_plan(teacher_manifest, summary)
    write_note(summary, spec_path, distillation_plan_path)

    print(json.dumps(summary, indent=2, sort_keys=True))


def load_gru_profile(path: Path, *, run_id: str, label: str) -> Profile:
    """Load a stored h0 velocity diagnostic as a 61-sample profile."""

    data = np.load(path, allow_pickle=False)
    velocity = np.asarray(data["velocity"], dtype=np.float64)
    forward = velocity[..., 0].reshape(-1, velocity.shape[-2])
    initial = np.zeros((forward.shape[0], 1), dtype=np.float64)
    forward = np.concatenate([initial, forward], axis=1)
    return Profile(
        label=label,
        kind="gru_h0_validation_selected",
        run_id=run_id,
        time_s=np.arange(forward.shape[1], dtype=np.float64) * DT_S,
        mean=np.mean(forward, axis=0),
        std=np.std(forward, axis=0),
        n_samples=int(forward.shape[0]),
    )


def materialize_analytical_profiles(*, n_samples: int) -> tuple[Profile, Profile, dict[str, Any]]:
    """Build 6D extLQG and H-infinity profiles with common random numbers."""

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
    for key in jr.split(jr.PRNGKey(376023), n_samples):
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

    teacher_package = TEACHER_DIR / "6d_output_feedback_teachers.npz"
    np.savez(
        teacher_package,
        plant_A=np.asarray(plant.A),
        plant_B=np.asarray(plant.B),
        plant_Bw=np.asarray(plant.Bw),
        schedule_Q=np.asarray(schedule.Q),
        schedule_R=np.asarray(schedule.R),
        schedule_Q_f=np.asarray(schedule.Q_f),
        x0=np.asarray(x0),
        extlqg_controller_gains=np.asarray(extlqg_path.controller_gains),
        extlqg_estimator_gains=np.asarray(extlqg_path.estimator_gains),
        extlqg_state_covariances=np.asarray(extlqg_path.state_covariances),
        hinf_controller_gains=np.asarray(robust_gains),
        hinf_estimator_covariances=np.asarray(robust_covariances),
        hinf_P=np.asarray(hinf_solution.P),
        gamma_star=np.asarray(gamma_star),
        hinf_gamma=np.asarray(hinf_solution.gamma),
        observation_matrix=np.asarray(delayed_observation_matrix(plant, config)),
    )
    manifest = {
        "schema_version": "rlrmp.376d023.analytical_teachers.v1",
        "issue": ISSUE_ID,
        "teacher_package": repo_ref(teacher_package),
        "plant": {
            "state_dim": int(plant.n),
            "physical_state_dim": 6,
            "disturbance_dim": int(plant.m_w),
            "control_dim": int(plant.m_u),
            "delay_steps": config.delay_steps,
            "dt": float(plant.dt),
            "disturbance_integrators_exposed": False,
        },
        "observation_contract": {
            "basis": "oldest_delayed_physical_block_6d_force_filter",
            "dimension": int(delayed_observation_matrix(plant, config).shape[0]),
            "observed_physical_indices": list(range(config.n_phys)),
        },
        "extlqg": {
            "controller": "local extLQG fixed-point output-feedback path",
            "parity_status": extlqg_path.parity_status,
            "n_iterations": int(extlqg_path.n_iterations),
            "expected_cost": extlqg_path.expected_cost,
        },
        "h_infinity": {
            "controller": "output-feedback H-infinity robust estimator/controller",
            "gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
            "gamma_star": float(gamma_star),
            "gamma": float(hinf_solution.gamma),
            "admissible": bool(hinf_solution.admissible),
        },
    }
    manifest_path = TEACHER_DIR / "6d_output_feedback_teachers_manifest.json"
    manifest["manifest"] = repo_ref(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return (
        Profile(
            label="6D extLQG analytical",
            kind="analytical_extlqg_6d_output_feedback",
            time_s=time_s,
            mean=np.mean(extlqg_forward, axis=0),
            std=np.std(extlqg_forward, axis=0),
            n_samples=n_samples,
            terminal_position_error_m=float(
                np.mean([rollout.terminal_position_error for rollout in extlqg_rollouts])
            ),
            parity_status=extlqg_path.parity_status,
        ),
        Profile(
            label="6D output-feedback H-infinity analytical",
            kind="analytical_hinf_6d_output_feedback",
            time_s=time_s,
            mean=np.mean(hinf_forward, axis=0),
            std=np.std(hinf_forward, axis=0),
            n_samples=n_samples,
            terminal_position_error_m=float(
                np.mean([rollout.terminal_position_error for rollout in hinf_rollouts])
            ),
            parity_status="6D no-integrator output-feedback robust estimator/controller",
        ),
        manifest,
    )


def write_velocity_plot(profiles: tuple[Profile, ...]) -> Path:
    """Write the overlay HTML plot."""

    colors = {
        "6D extLQG analytical": "#111827",
        "6D output-feedback H-infinity analytical": "#7c3aed",
        "020a65b h0 no-PGD": "#2563eb",
        "020a65b h0 PGD": "#dc2626",
    }
    dashes = {
        "6D extLQG analytical": "dash",
        "6D output-feedback H-infinity analytical": "dot",
        "020a65b h0 no-PGD": "solid",
        "020a65b h0 PGD": "solid",
    }
    fig = go.Figure()
    for profile in profiles:
        color = colors[profile.label]
        upper = profile.mean + profile.std
        lower = profile.mean - profile.std
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([profile.time_s, profile.time_s[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=rgba(color, 0.12),
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
                line={"color": color, "width": 2.4, "dash": dashes[profile.label]},
                name=profile.label,
            )
        )
    fig.update_layout(
        title="6D analytical output-feedback vs 020a65b h0 nominal velocity",
        width=920,
        height=560,
        margin={"l": 72, "r": 24, "t": 68, "b": 64},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_xaxes(title_text="Time (s)", range=[0.0, 0.6], zeroline=False)
    fig.update_yaxes(title_text="Forward velocity (m/s)", zeroline=True)
    path = BULK_FIGURE_DIR / "velocity_profile_overlay.html"
    fig.write_html(path)
    return path


def write_profile_csv(profiles: tuple[Profile, ...]) -> Path:
    """Write a wide CSV sidecar for the plotted profiles."""

    time_s = profiles[0].time_s
    columns = [time_s]
    headers = ["time_s"]
    for profile in profiles:
        if profile.time_s.shape != time_s.shape or not np.allclose(profile.time_s, time_s):
            raise ValueError(f"Profile {profile.label} has a nonmatching time axis")
        slug = (
            profile.label.lower()
            .replace("020a65b ", "")
            .replace("6d ", "six_d_")
            .replace(" ", "_")
            .replace("-", "_")
        )
        columns.extend([profile.mean, profile.std])
        headers.extend([f"{slug}_mean_m_s", f"{slug}_std_m_s"])
    path = BULK_FIGURE_DIR / "velocity_profile_overlay.csv"
    np.savetxt(
        path,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
    return path


def write_summary(
    profiles: tuple[Profile, ...],
    *,
    html_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    """Write and return the JSON plot summary."""

    summaries = {
        profile.label: {
            "kind": profile.kind,
            "run_id": profile.run_id,
            "n_samples": profile.n_samples,
            "peak_mean_forward_velocity_m_s": profile.peak_forward_velocity_m_s,
            "time_of_peak_mean_forward_velocity_s": profile.time_of_peak_forward_velocity_s,
            "terminal_position_error_m": profile.terminal_position_error_m,
            "parity_status": profile.parity_status,
        }
        for profile in profiles
    }
    ext_peak = summaries["6D extLQG analytical"]["peak_mean_forward_velocity_m_s"]
    hinf_peak = summaries["6D output-feedback H-infinity analytical"][
        "peak_mean_forward_velocity_m_s"
    ]
    summary = {
        "schema_version": "rlrmp.376d023.velocity_profile_overlay.v1",
        "issue": ISSUE_ID,
        "source_issue": SOURCE_ISSUE_ID,
        "html": repo_ref(html_path),
        "csv": repo_ref(csv_path),
        "profiles": summaries,
        "source_rows": {
            "h0_no_pgd": NO_PGD_RUN_ID,
            "h0_pgd": PGD_RUN_ID,
        },
        "interpretation": (
            "The 6D H-infinity output-feedback arm preserves the expected "
            "robustification signature relative to 6D extLQG: higher/faster "
            "nominal forward velocity under the same force-filter feedback contract."
        ),
        "robustification_delta_peak_m_s": float(hinf_peak - ext_peak),
        "no_training_or_remote_gpu": True,
    }
    path = BULK_FIGURE_DIR / "figure_summary.json"
    summary["summary"] = repo_ref(path)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def write_figure_spec(summary: dict[str, Any], teacher_manifest: dict[str, Any]) -> Path:
    """Write the tracked figure spec."""

    path = TRACKED_FIGURE_DIR / "spec.json"
    spec = {
        "schema_version": "rlrmp.376d023.figure_spec.v1",
        "issue": ISSUE_ID,
        "materializer": repo_ref(Path(__file__).resolve()),
        "outputs": {
            "html": summary["html"],
            "csv": summary["csv"],
            "summary": summary["summary"],
            "teacher_manifest": teacher_manifest["manifest"],
            "teacher_package": teacher_manifest["teacher_package"],
        },
        "inputs": {
            "h0_no_pgd_npz": repo_ref(NO_PGD_NPZ),
            "h0_pgd_npz": repo_ref(PGD_NPZ),
            "h0_no_pgd_run_spec": repo_ref(NO_PGD_RUN_SPEC),
        },
        "plot_contract": {
            "time_axis": "0.00s through 0.60s; GRU diagnostics prepend zero initial velocity",
            "bands": "mean +/- 1 SD over pooled stochastic samples",
            "analytical_samples": summary["profiles"]["6D extLQG analytical"]["n_samples"],
            "gru_samples_per_row": summary["profiles"]["020a65b h0 no-PGD"]["n_samples"],
        },
        "no_training_or_remote_gpu": True,
    }
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return path


def write_distillation_plan(
    teacher_manifest: dict[str, Any],
    summary: dict[str, Any],
) -> Path:
    """Write the no-launch distillation plan based on the h0 no-PGD row."""

    base_spec = json.loads(NO_PGD_RUN_SPEC.read_text(encoding="utf-8"))
    path = TRACKED_RUN_DIR / "proposed_h0_no_pgd_distillation_6d_teacher.json"
    plan = {
        "schema_version": "rlrmp.376d023.distillation_plan.v1",
        "status": "planned_not_launched",
        "issue": ISSUE_ID,
        "base_issue": SOURCE_ISSUE_ID,
        "base_run_id": NO_PGD_RUN_ID,
        "base_run_spec": repo_ref(NO_PGD_RUN_SPEC),
        "preserve_from_base_run": {
            "training_script_family": base_spec["training_script"],
            "n_train_batches": base_spec["n_train_batches"],
            "batch_size": base_spec["batch_size"],
            "controller_lr": base_spec["controller_lr"],
            "optimizer": base_spec["optimizer"],
            "model_summary": {
                key: base_spec["model_summary"][key]
                for key in (
                    "controller_kind",
                    "hidden_size",
                    "n_replicates",
                    "controller_input_dimension",
                    "initial_hidden_encoder",
                    "population_structure",
                )
            },
            "training_distribution": {
                key: base_spec["training_distribution"][key]
                for key in (
                    "mode",
                    "force_filter_feedback",
                    "target_relative_multitarget",
                    "initial_hidden_encoder",
                    "validation_bins",
                )
                if key in base_spec["training_distribution"]
            },
            "stochastic_preset": base_spec["stochastic_preset"],
            "loss_anchor_weights": base_spec["hps"]["loss"]["weights"],
        },
        "teacher_contract": {
            "primary_teacher": "6D output-feedback H-infinity analytical",
            "diagnostic_control_teacher": "6D extLQG analytical",
            "teacher_manifest": teacher_manifest["manifest"],
            "teacher_package": teacher_manifest["teacher_package"],
            "information_contract": teacher_manifest["observation_contract"],
            "plant_contract": teacher_manifest["plant"],
        },
        "required_changes": {
            "training_structure": (
                "Add a distillation-capable runner derived from the 020a65b "
                "h0 no-PGD nominal GRU runner; keep PGD off for this first "
                "teacher-distillation run."
            ),
            "loss_terms": [
                "clean_action_imitation: match H-infinity teacher actions on nominal and target-bank observation histories",
                "perturbation_response_imitation: match teacher action deltas and recovery trajectories under calibrated small perturbations",
                "student_forced_rollout_anchor: keep endpoint, velocity profile, and full QRF task quality near the base row",
                "io_jvp_match: match feedback/observation-history to action-history JVPs on a sampled local basis",
                "extlqg_sidecar: evaluate but do not train to the extLQG teacher unless an ablation is requested",
            ],
            "non_certificate_terms": [
                "hidden_state_matching is optional for optimization diagnostics only",
                "GRU hidden-state to action Jacobians are not certificate evidence",
            ],
            "teacher_student_schedule": [
                {
                    "batches": "0-1500",
                    "mode": "teacher_forced",
                    "role": "stabilize action imitation and h0 encoder under teacher observation histories",
                },
                {
                    "batches": "1500-4000",
                    "mode": "mixed_teacher_forced_and_student_forced",
                    "role": "introduce perturbation-response imitation and short closed-loop rollouts",
                },
                {
                    "batches": "4000-12000",
                    "mode": "mostly_student_forced",
                    "role": "optimize closed-loop behavior, action match, perturbation match, and I/O-map JVP gates",
                },
            ],
        },
        "certificate_and_eval_gates": [
            "clean action match on original, seen, and held-out target bins",
            "perturbation-response match under calibrated initial-state, sensory, process/load, and command-input perturbations",
            "input-output JVP/Jacobian match from feedback/observation history to action history",
            "student-forced nominal velocity, endpoint, and task-cost parity with the base h0 no-PGD row",
            "student-vs-6D-H-infinity standard certificate report; extLQG remains a diagnostic comparator",
        ],
        "no_launch_boundary": "No training, RunPod, Modal, GPU acquisition, auth request, or push is authorized by this plan.",
        "supporting_velocity_plot": summary["html"],
    }
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return path


def write_note(summary: dict[str, Any], spec_path: Path, distillation_plan_path: Path) -> None:
    """Write a concise tracked Markdown note."""

    lines = [
        "# 6D analytical output-feedback comparators",
        "",
        "Generated the no-integrator 6D extLQG and output-feedback H-infinity analytical "
        "models under the same delayed force-filter feedback contract used by the h0 rows.",
        "",
        "| Row | Peak mean forward velocity (m/s) | Time of peak (s) | Samples |",
        "|---|---:|---:|---:|",
    ]
    for label, item in summary["profiles"].items():
        lines.append(
            "| "
            f"{label} | "
            f"{item['peak_mean_forward_velocity_m_s']:.6f} | "
            f"{item['time_of_peak_mean_forward_velocity_s']:.2f} | "
            f"{item['n_samples']} |"
        )
    lines.extend(
        [
            "",
            f"Interpretation: {summary['interpretation']}",
            "",
            "Artifacts:",
            f"- Plot: `{summary['html']}`",
            f"- Summary: `{summary['summary']}`",
            f"- Figure spec: `{repo_ref(spec_path)}`",
            f"- Distillation plan: `{repo_ref(distillation_plan_path)}`",
            "",
            "Rows used:",
            f"- h0 no-PGD: `{NO_PGD_RUN_ID}`",
            f"- h0 PGD: `{PGD_RUN_ID}`",
            "",
        ]
    )
    update_marked_section(NOTE_PATH, "six_d_velocity_profiles", "\n".join(lines) + "\n")


repo_ref = portable_repo_path


rgba = hex_to_rgba


if __name__ == "__main__":
    main()
