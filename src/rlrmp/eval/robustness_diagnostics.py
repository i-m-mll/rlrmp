"""Shared context and orchestration primitives for robustness diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.random as jr
import numpy as np

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    build_no_integrator_game,
)
from rlrmp.analysis.math.cs_released_simulation import (
    simulate_robust_released_forward,
    zero_forward_noise_draws,
    zero_noise_covariances,
)
from rlrmp.analysis.math.hinf_riccati import find_gamma_star, solve_hinf_riccati
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    make_cs_output_feedback_initial_state,
    robust_estimator_covariances,
    robust_output_feedback_gains,
)


def build_robust_output_feedback_6d_context(
    *,
    evaluation_from_rollout: Callable[..., Any],
    gamma_factor: float = OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
) -> dict[str, Any]:
    """Build the deterministic 6D no-integrator output-feedback H-infinity context."""

    plant, schedule = build_no_integrator_game()
    config = OutputFeedbackConfig(n_phys=6)
    gamma_star = find_gamma_star(plant, schedule)
    solution = solve_hinf_riccati(plant, schedule, gamma_factor * gamma_star)
    covariances = robust_estimator_covariances(
        plant,
        schedule,
        solution.gamma,
        config,
    )
    gains = robust_output_feedback_gains(
        plant,
        schedule,
        solution,
        covariances,
        config,
    )
    x0 = make_cs_output_feedback_initial_state(plant, config)
    base_rollout = simulate_robust_released_forward(
        plant,
        schedule,
        solution,
        x0,
        draws=zero_forward_noise_draws(T=schedule.T, plant=plant, config=config),
        covariances=zero_noise_covariances(plant, config),
        gains=gains,
        config=config,
    )
    if int(plant.n) != 36 or int(config.n_phys) != 6:
        raise ValueError(
            f"unexpected H-inf context dimensions: plant.n={plant.n}, "
            f"n_phys={config.n_phys}"
        )
    return {
        "plant": plant,
        "schedule": schedule,
        "config": config,
        "solution": solution,
        "gains": gains,
        "gamma_factor": gamma_factor,
        "gamma": float(solution.gamma),
        "gamma_star": float(gamma_star),
        "base_initial_state": np.asarray(x0, dtype=np.float64),
        "base_evaluation": evaluation_from_rollout(base_rollout, initial_state=x0),
        "contract": {
            "label": "6D output-feedback H-infinity",
            "state_dim": int(plant.n),
            "physical_dim": int(config.n_phys),
            "disturbance_dim": int(plant.m_w),
            "control_dim": int(plant.m_u),
            "delay_steps": int(config.delay_steps),
            "disturbance_integrators_exposed": False,
            "game_source": "rlrmp.analysis.math.cs_game_card.build_no_integrator_game",
            "config": "rlrmp.analysis.math.output_feedback.OutputFeedbackConfig(n_phys=6)",
            "gamma_factor": float(gamma_factor),
            "gamma_star": float(gamma_star),
            "gamma": float(solution.gamma),
            "admissible": bool(solution.admissible),
        },
    }


def evaluate_stabilization_row(
    row_spec: Any,
    *,
    repo_root: Path,
    hooks: Mapping[str, Any] | Any,
    source_experiment: str,
    row_metadata: Callable[[Any], Mapping[str, Any]],
    allowed_missing_families: Sequence[str] = (),
    run_spec_transform: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint on the canonical stabilization probe bank.

    ``hooks`` supplies the experiment-specific probe/summary vocabulary;
    ``row_metadata`` and ``allowed_missing_families`` are explicit schema
    extension points for the three historical result families. Runtime imports
    stay local to avoid widening the import cycle through training setup.
    """

    from feedbax.config.namespace import TreeNamespace, dict_to_namespace

    from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
    from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
        load_validation_selected_checkpoint_model,
    )
    from rlrmp.analysis.pipelines.gru_perturbation_bank import (
        apply_perturbation_to_trial_specs,
    )
    from rlrmp.analysis.pipelines.gru_pilot_figures import (
        repeat_single_validation_trial,
        resolve_run_inputs,
    )
    from rlrmp.analysis.pipelines.gru_steady_state_perturbation_bank import (
        DEFAULT_N_ROLLOUT_TRIALS,
        DEFAULT_POST_ONSET_FIGURE_STEPS,
        DEFAULT_PULSE_DURATION_STEPS,
        _evaluate_model_on_trial_specs,
        _expected_feedback_dim_from_hps,
        _feedback_dim,
        _target_position,
        make_steady_state_trial_specs,
        pad_feedback_offset_inputs,
        washin_diagnostics,
    )
    from rlrmp.analysis.pipelines.sisu_spectrum_diagnostics import zero_disturbance_payload
    from rlrmp.train.task_model import setup_task_model_pair

    def hook(name: str) -> Any:
        return _runtime_value(hooks, name)

    run = resolve_run_inputs(
        experiment=source_experiment,
        run_ids=[row_spec.run_id],
        labels=[row_spec.run_id],
        repo_root=repo_root,
    )[0]
    run_spec = run.run_spec
    if run_spec_transform is not None:
        run_spec = run_spec_transform(run_spec)
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    seed = int(run_spec.get("seed", 42))
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(seed))
    n_replicates = int(hps.model.n_replicates)
    model, checkpoint_selection = load_validation_selected_checkpoint_model(
        experiment=source_experiment,
        run_id=run.run_id,
        run_spec=run_spec,
        checkpoint_selection_mode="sparse_history",
        repo_root=repo_root,
    )
    base_trials = repeat_single_validation_trial(
        pair.task.validation_trials,
        DEFAULT_N_ROLLOUT_TRIALS,
    )
    steady_trials, timing = make_steady_state_trial_specs(
        base_trials,
        delayed=False,
        target_position=np.asarray(
            _target_position(run, base_trials),
            dtype=np.float64,
        ),
        pulse_duration_steps=DEFAULT_PULSE_DURATION_STEPS,
        min_post_onset_steps=DEFAULT_POST_ONSET_FIGURE_STEPS,
    )
    steady_trials = pad_feedback_offset_inputs(
        steady_trials,
        expected_feedback_dim=_expected_feedback_dim_from_hps(hps),
    )
    steady_trials = zero_disturbance_payload(steady_trials)
    feedback_dim = _feedback_dim(steady_trials)
    probes = hook("build_probes")(
        feedback_dim=feedback_dim,
        pulse_start=int(timing["pulse_start_step"]),
        pulse_duration=int(timing["pulse_duration_steps"]),
    )
    base = _evaluate_model_on_trial_specs(
        model=model,
        task=pair.task,
        trial_specs=steady_trials,
        n_replicates=n_replicates,
        seed=0,
    )
    details = []
    for probe in probes:
        adapter = apply_perturbation_to_trial_specs(
            steady_trials,
            probe.row,
            model=model,
        )
        if adapter.status != "evaluated":
            details.append(
                {
                    "perturbation_id": probe.perturbation_id,
                    "group": probe.group,
                    "family": probe.family,
                    "status": adapter.status,
                    "reason": adapter.reason,
                    "adapter": adapter.to_json(),
                }
            )
            continue
        perturbed = _evaluate_model_on_trial_specs(
            model=adapter.model if adapter.model is not None else model,
            task=pair.task,
            trial_specs=adapter.trial_specs,
            n_replicates=n_replicates,
            seed=0,
        )
        details.append(
            hook("summarize_probe")(
                probe=probe,
                base=base,
                perturbed=perturbed,
                pulse_start=int(timing["pulse_start_step"]),
            )
            | {"status": "evaluated", "adapter": adapter.to_json()}
        )

    family_summary = hook("summarize_by_family")(details)
    group_summary = hook("summarize_by_group")(details)
    missing = set(allowed_missing_families)
    washin = washin_diagnostics(base, pulse_start=timing["pulse_start_step"])

    def family_metric(family: str, metric: str) -> Any:
        if family in missing:
            return family_summary.get(family, {}).get(metric)
        return family_summary[family][metric]

    result = {
        **dict(row_metadata(row_spec)),
        "run_spec_path": hook("repo_relative")(run.run_spec_path, repo_root),
        "artifact_dir": hook("repo_relative")(run.artifact_dir, repo_root),
        "checkpoint_selection_summary": hook("checkpoint_selection_summary")(
            checkpoint_selection
        ),
        "response_label": hook("response_label")(washin),
        "dt_s": float(base.dt),
        "timing": timing,
        "n_replicates": int(base.command.shape[0]),
        "n_rollout_trials_per_replicate": int(base.command.shape[1]),
        "feedback_dim": int(feedback_dim),
        "washin": washin,
        "feedback_auc_mm_s": group_summary["feedback"]["auc_displacement_mm_s_mean"],
        "mechanical_auc_mm_s": group_summary["mechanical"]["auc_displacement_mm_s_mean"],
        "command_input_auc_mm_s": family_metric(
            "command_input_pulse", "auc_displacement_mm_s_mean"
        ),
        "process_force_auc_mm_s": family_metric(
            "process_epsilon_force_state_xy", "auc_displacement_mm_s_mean"
        ),
        "feedback_peak_mm": group_summary["feedback"]["peak_displacement_mm_mean"],
        "mechanical_peak_mm": group_summary["mechanical"]["peak_displacement_mm_mean"],
        "family_summary": family_summary,
        "per_probe_detail": details,
    }
    if missing:
        result["missing_families"] = [
            family for family in allowed_missing_families if family not in family_summary
        ]
    return result


def run_feedback_robustness_diagnostics(
    *,
    hooks: Mapping[str, Any] | Any,
    paths: Mapping[str, Path],
    output_dirs: Sequence[Path],
    issue: str,
    repo_root: Path,
    run_ids: Sequence[str],
    labels: Sequence[str],
    evaluation_bulk_dir: Path,
    perturbation_bulk_dir: Path,
    feedback_scope: str,
    build_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    build_summary_payload: Callable[
        [Sequence[Mapping[str, Any]], Mapping[str, Any]], dict[str, Any]
    ],
    write_outputs: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], None],
    materialize_extensions: Callable[
        [Mapping[str, Path], Mapping[str, Any]], Mapping[str, Any]
    ] = lambda _paths, _components: {},
    n_rollout_trials: int = 64,
    write_evaluation_bulk_arrays: bool = True,
) -> dict[str, Any]:
    """Run the common feedback/perturbation diagnostic orchestration."""

    from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
        materialize_validation_selected_checkpoint_manifest,
    )
    from rlrmp.analysis.pipelines.gru_evaluation_diagnostics import (
        materialize_gru_evaluation_diagnostics,
    )
    from rlrmp.analysis.pipelines.gru_feedback_ablation import (
        materialize_gru_feedback_ablation,
    )
    from rlrmp.analysis.pipelines.gru_perturbation_bank import (
        materialize_gru_perturbation_response,
    )
    from rlrmp.paths import mkdir_p

    def hook(name: str) -> Any:
        return _runtime_value(hooks, name)

    for directory in output_dirs:
        mkdir_p(directory)
    checkpoint_manifest = (
        hook("load_json")(paths["checkpoint_manifest"])
        if paths["checkpoint_manifest"].exists()
        else materialize_validation_selected_checkpoint_manifest(
            experiment=issue,
            run_ids=run_ids,
            output_path=paths["checkpoint_manifest"],
            repo_root=repo_root,
        )
    )
    evaluation = (
        hook("load_json")(paths["evaluation"])
        if paths["evaluation"].exists()
        else materialize_gru_evaluation_diagnostics(
            experiment=issue,
            run_ids=run_ids,
            labels=labels,
            output_path=paths["evaluation"],
            bulk_dir=evaluation_bulk_dir,
            n_rollout_trials=n_rollout_trials,
            write_bulk_arrays=write_evaluation_bulk_arrays,
            regeneration_spec_path=paths["evaluation_regeneration_spec"],
            repo_root=repo_root,
        )
    )
    perturbation = (
        hook("load_json")(paths["perturbation"])
        if hook("perturbation_output_is_current")(
            paths["perturbation"], expected_trials=n_rollout_trials
        )
        else materialize_gru_perturbation_response(
            source_experiment=issue,
            result_experiment=issue,
            run_ids=run_ids,
            labels=labels,
            n_rollout_trials=n_rollout_trials,
            output_path=paths["perturbation"],
            note_path=paths["perturbation_note"],
            bulk_dir=perturbation_bulk_dir,
            regeneration_spec_path=paths["perturbation_regeneration_spec"],
            bank_mode="calibrated",
            calibration_level="moderate",
            calibration_reach=0.15,
            feedback_scale_manifest_path=paths["evaluation"],
            extlqg_physical_dim=6,
            write_bulk_arrays=False,
            repo_root=repo_root,
        )
    )
    feedback = (
        hook("load_json")(paths["feedback"])
        if hook("run_output_is_current")(
            paths["feedback"], expected_trials=n_rollout_trials
        )
        else materialize_gru_feedback_ablation(
            source_experiment=issue,
            result_experiment=issue,
            scope=feedback_scope,
            run_ids=run_ids,
            labels=labels,
            n_rollout_trials=n_rollout_trials,
            bank_mode="calibrated",
            calibration_level="moderate",
            calibration_reach=0.15,
            feedback_selection_level="moderate",
            feedback_scale_manifest_path=paths["evaluation"],
            output_path=paths["feedback"],
            note_path=paths["feedback_note"],
            regeneration_spec_path=paths["feedback_regeneration_spec"],
            repo_root=repo_root,
        )
    )
    components: dict[str, Any] = {
        "checkpoint_manifest": checkpoint_manifest,
        "evaluation": evaluation,
        "perturbation": perturbation,
        "feedback": feedback,
        "perturbation_detail": hook("load_json")(
            Path(perturbation["bulk_detail_manifest"]["path"])
        ),
    }
    components.update(materialize_extensions(paths, components))
    rows = build_rows(components)
    summary = build_summary_payload(rows, components)
    write_outputs(summary, rows)
    return {"summary": summary, "rows": rows}


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema_version: str,
    issue: str,
    scope: str,
    row_order: Sequence[str],
    paths: Mapping[str, Path],
    repo_relative: Callable[[Path], str],
    components: Mapping[str, Mapping[str, Any]],
    component_schema_names: Sequence[str],
    extensions: Mapping[str, Any] | None = None,
    source_output_extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a diagnostic summary with explicit schema-specific extensions."""

    return {
        "schema_version": schema_version,
        "issue": issue,
        "scope": scope,
        "row_order": list(row_order),
        "rows": list(rows),
        **dict(extensions or {}),
        "source_outputs": {key: repo_relative(path) for key, path in paths.items()}
        | dict(source_output_extensions or {}),
        "component_schemas": {
            name: components[name].get("schema_version") for name in component_schema_names
        },
    }


def _runtime_value(runtime: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(runtime, Mapping):
        return runtime[name]
    return getattr(runtime, name)
