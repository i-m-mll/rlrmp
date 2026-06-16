"""Standard-certificate materialization for C&S nominal GRU pilot artifacts."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from jax_cookbook import load_with_hyperparameters
from feedbax.runtime.channel import Channel
from feedbax.runtime.graph import init_state_from_component
from feedbax.config.namespace import TreeNamespace, dict_to_namespace

from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.cs_released_simulation import (
    build_extlqg_comparator_path,
    default_cs_noise_covariances,
)
from rlrmp.analysis.math.output_feedback import (
    OutputFeedbackConfig,
    delayed_observation_matrix,
    make_cs_output_feedback_initial_state,
    position_velocity_observation_config,
    rollout_with_kalman_estimator,
)
from rlrmp.analysis.pipelines.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    DISTURBANCE_HISTORY_TO_ACTION_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_COST_QUADRATIC,
    DISTURBANCE_HISTORY_TO_OUTPUT_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_STATE_MAP_MISMATCH,
    MEASUREMENT_HISTORY_TO_ACTION_MAP_MISMATCH,
    MEASUREMENT_HISTORY_TO_OUTPUT_MAP_MISMATCH,
    OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
)
from rlrmp.analysis.pipelines.bridge_contracts import (
    BridgeCertificateComponent,
    BridgeRunManifest,
    BridgeRunSpec,
    make_bridge_run_id,
)
from rlrmp.analysis.pipelines.diagnostic_provenance import write_regeneration_spec
from rlrmp.analysis.pipelines.failure_decomposition import failure_diagnostic_from_standard_row
from rlrmp.analysis.pipelines.standard_certificate_materialization import (
    StandardCertificateRowRequest,
    build_standard_certificate_manifest,
    materialization_summary,
    repo_relative,
)
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.runtime.spec_migrations import (
    CS_GRU_STANDARD_CERTIFICATES_KIND,
    CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION,
    stamp_current_schema,
)
from rlrmp.model.stochastic_runtime import (
    PLANT_PROCESS_FORCE_NOISE_LABEL,
    add_plant_process_force_noise,
)
from rlrmp.train.task_model import setup_task_model_pair

MATERIALIZER_ISSUE_ID = "e6a32b8"
SOURCE_ISSUE_ID = "30f2313"
SCHEMA_VERSION = CS_GRU_STANDARD_CERTIFICATES_SCHEMA_VERSION
RUN_IDS = (
    "cs_stochastic_gru__no_hidden_penalty",
    "cs_stochastic_gru__hidden_penalty",
)
DEFAULT_RESPONSE_MAP_ROLLOUT_TRIALS = 16
RESULT_RUN_ROOT = REPO_ROOT / "results" / SOURCE_ISSUE_ID / "runs"
ARTIFACT_RUN_ROOT = REPO_ROOT / "_artifacts" / SOURCE_ISSUE_ID / "runs"
NOTE_PATH = REPO_ROOT / "results" / SOURCE_ISSUE_ID / "notes" / "gru_standard_certificates.md"
MANIFEST_PATH = (
    REPO_ROOT / "results" / SOURCE_ISSUE_ID / "notes" / "gru_standard_certificates_manifest.json"
)

_BLOCKED_RESPONSE_COMPONENTS = {
    OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH,
    MEASUREMENT_HISTORY_TO_ACTION_MAP_MISMATCH,
    MEASUREMENT_HISTORY_TO_OUTPUT_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_ACTION_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_STATE_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_OUTPUT_MAP_MISMATCH,
    DISTURBANCE_HISTORY_TO_COST_QUADRATIC,
}
_NO_ACTION_EVIDENCE_REASON = (
    "candidate action samples were not evaluated; load a trained GRU model and "
    "run the clean Feedbax rollout before claiming action mismatch evidence"
)


def materialize_gru_standard_result(
    *,
    run_ids: tuple[str, ...] = RUN_IDS,
    load_models: bool = True,
    experiment: str = SOURCE_ISSUE_ID,
    materializer_issue_id: str = MATERIALIZER_ISSUE_ID,
    use_validation_selected_checkpoints: bool = False,
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return standard rows and companion failure diagnostics for GRU pilots."""

    selection_manifest = None
    if use_validation_selected_checkpoints:
        from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
            materialize_validation_selected_checkpoint_manifest,
        )

        selection_manifest = materialize_validation_selected_checkpoint_manifest(
            experiment=experiment,
            run_ids=run_ids,
            preferred_manifest_path=preferred_checkpoint_manifest_path,
            checkpoint_selection_mode=(
                "fixed_bank_manifest"
                if preferred_checkpoint_manifest_path is not None
                else "sparse_history"
            ),
            repo_root=repo_root,
        )
    result_run_root = repo_root / "results" / experiment / "runs"
    rows = [
        materialize_gru_standard_row(
            run_id,
            load_model=load_models,
            experiment=experiment,
            materializer_issue_id=materializer_issue_id,
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
            repo_root=repo_root,
        )
        for run_id in run_ids
    ]
    row_dicts = [row.to_json_dict() for row in rows]
    failure_rows = [
        failure_diagnostic_from_standard_row(
            row,
            source_group="cs_stochastic_gru",
            row_parameters=row["spec"]["parameters"],
        )
        for row in row_dicts
    ]
    blockers = sorted(
        {
            row["metrics"]["io_response_map_blocker"]
            for row in row_dicts
            if row["metrics"].get("io_response_map_blocker")
        }
    )
    effective_checkpoint_policy = (
        str(selection_manifest.get("checkpoint_policy"))
        if isinstance(selection_manifest, Mapping)
        and selection_manifest.get("checkpoint_policy") is not None
        else (
            "validation_selected_per_replicate"
            if use_validation_selected_checkpoints
            else "final_checkpoint"
        )
    )
    return stamp_current_schema(
        CS_GRU_STANDARD_CERTIFICATES_KIND,
        {
            "format": SCHEMA_VERSION,
            "issue": materializer_issue_id,
            "source_issue": experiment,
            "checkpoint_policy": effective_checkpoint_policy,
            "source_manifests": {
                run_id: repo_relative(result_run_root / run_id / "run.json", repo_root=repo_root)
                for run_id in run_ids
            },
            "source_artifacts": {
                run_id: repo_relative(
                    _default_model_path(run_id, experiment=experiment, repo_root=repo_root),
                    repo_root=repo_root,
                )
                for run_id in run_ids
            },
            "checkpoint_selection": selection_manifest,
            "summary": materialization_summary(rows)
            | {
                "failure_classification_counts": _classification_counts(failure_rows),
                "blockers": blockers,
            },
            "scope": (
                f"C&S stochastic GRU pilot rows from `{experiment}`. The "
                "standard certificate is materialized in empirical_nonlinear mode: "
                "clean rollout action behavior is available, same-coordinate "
                "transition/value/Bellman components are not applicable, and response "
                "map rows remain missing until the observation projection contract is "
                "defined."
            ),
            "rows": row_dicts,
            "failure_decomposition": {"rows": failure_rows},
        },
    )


def materialize_gru_standard_row(
    run_id: str,
    *,
    load_model: bool = True,
    experiment: str = SOURCE_ISSUE_ID,
    materializer_issue_id: str = MATERIALIZER_ISSUE_ID,
    use_validation_selected_checkpoints: bool = False,
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> BridgeRunManifest:
    """Materialize one GRU pilot standard row."""

    run_spec_path = repo_root / "results" / experiment / "runs" / run_id / "run.json"
    run_spec = _read_json(run_spec_path)
    training_summary_path = _default_training_summary_path(
        run_id,
        experiment=experiment,
        repo_root=repo_root,
    )
    training_summary = _read_json(training_summary_path) if training_summary_path.exists() else {}
    reference_actions, reference_metadata = cs_output_feedback_reference_actions()
    reference_map, response_reference_metadata = cs_output_feedback_observation_action_map()
    action_weight = reference_metadata["action_weight"]
    serializable_reference_metadata = {
        key: value for key, value in reference_metadata.items() if key != "action_weight"
    }
    serializable_reference_metadata["io_response_map"] = response_reference_metadata
    if load_model:
        candidate_actions, candidate_map, evaluation_metadata = evaluate_gru_clean_actions(
            run_id,
            run_spec=run_spec,
            experiment=experiment,
            use_validation_selected_checkpoints=use_validation_selected_checkpoints,
            preferred_checkpoint_manifest_path=preferred_checkpoint_manifest_path,
            repo_root=repo_root,
        )
        observation_history_covariance = evaluation_metadata.pop(
            "_observation_history_covariance_array",
            None,
        )
    else:
        candidate_actions = np.zeros((0, reference_actions.shape[0], reference_actions.shape[1]))
        candidate_map = None
        observation_history_covariance = None
        evaluation_metadata = {
            "status": "not_evaluated",
            "reason": "model loading was disabled",
            "observation_history_covariance": (
                _missing_observation_history_covariance_metadata("model loading was disabled")
            ),
        }
    if candidate_actions.size:
        reference_batch = np.broadcast_to(reference_actions[None, :, :], candidate_actions.shape)
    else:
        reference_batch = candidate_actions
    reference_response_map = None
    if candidate_map is not None:
        expected_shape = candidate_map.shape[1:]
        if reference_map.shape == expected_shape:
            reference_response_map = np.broadcast_to(
                reference_map[None, :, :, :], candidate_map.shape
            )
        else:
            evaluation_metadata["io_response_map"]["reference_status"] = "shape_mismatch_blocked"
            evaluation_metadata["io_response_map"]["candidate_map_shape"] = [
                int(dim) for dim in candidate_map.shape
            ]
            evaluation_metadata["io_response_map"]["reference_map_shape"] = [
                int(dim) for dim in reference_map.shape
            ]
    return build_gru_standard_manifest_from_actions(
        run_id=run_id,
        run_spec=run_spec,
        training_summary=training_summary,
        candidate_actions=candidate_actions,
        reference_actions=reference_batch,
        action_weight=action_weight,
        candidate_observation_to_action_map=candidate_map,
        reference_observation_to_action_map=reference_response_map,
        observation_history_covariance=observation_history_covariance,
        observation_history_covariance_metadata=evaluation_metadata.get(
            "observation_history_covariance"
        ),
        evaluation_metadata=evaluation_metadata,
        run_spec_path=run_spec_path,
        model_path=_default_model_path(run_id, experiment=experiment, repo_root=repo_root),
        training_summary_path=training_summary_path,
        reference_metadata=serializable_reference_metadata,
        source_issue_id=experiment,
        materializer_issue_id=materializer_issue_id,
        repo_root=repo_root,
    )


def build_gru_standard_manifest_from_actions(
    *,
    run_id: str,
    run_spec: dict[str, Any],
    training_summary: dict[str, Any],
    candidate_actions: np.ndarray,
    reference_actions: np.ndarray,
    action_weight: np.ndarray,
    candidate_observation_to_action_map: np.ndarray | None = None,
    reference_observation_to_action_map: np.ndarray | None = None,
    observation_history_covariance: np.ndarray | None = None,
    observation_history_covariance_metadata: dict[str, Any] | None = None,
    evaluation_metadata: dict[str, Any] | None = None,
    run_spec_path: Path | None = None,
    model_path: Path | None = None,
    training_summary_path: Path | None = None,
    reference_metadata: dict[str, Any] | None = None,
    source_issue_id: str = SOURCE_ISSUE_ID,
    materializer_issue_id: str = MATERIALIZER_ISSUE_ID,
    repo_root: Path = REPO_ROOT,
) -> BridgeRunManifest:
    """Build a GRU empirical/nonlinear standard row from clean action traces."""

    candidate = np.asarray(candidate_actions, dtype=np.float64)
    reference = np.asarray(reference_actions, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference action batches must have the same shape")
    if candidate.ndim != 3:
        raise ValueError("actions must have shape (batch, horizon, action)")
    action_evidence_available = candidate.shape[0] > 0 and reference.shape[0] > 0
    response_map_evidence_available = (
        candidate_observation_to_action_map is not None
        and reference_observation_to_action_map is not None
    )
    covariance_metadata = observation_history_covariance_metadata or (
        _missing_observation_history_covariance_metadata(
            "sampled observation histories were not supplied"
        )
    )
    blocker = (
        None
        if response_map_evidence_available
        else gru_io_response_map_blocker(run_spec, source_issue_id=source_issue_id)
    )
    model_hps = run_spec.get("hps", {}).get("model", {})
    response_map_status = (
        "available_4d_observation_contract"
        if response_map_evidence_available
        else "blocked_missing_observation_contract"
    )
    spec = BridgeRunSpec(
        issue_id=materializer_issue_id,
        run_id=make_bridge_run_id(run_id, "nominal_clean"),
        objective="optimal",
        architecture="gru",
        controller_label=run_id,
        optimizer_label="adamw_nominal_gru",
        training_distribution="nominal",
        evaluation_lane="deterministic",
        reference_controller="analytical_lqr_kalman_output_feedback",
        seed=run_spec.get("seed"),
        gamma_factor=OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        parameters={
            "source_issue": source_issue_id,
            "source_run_id": run_id,
            "evaluation_lens": "nominal_clean",
            "certificate_mode": "empirical_nonlinear",
            "stochastic_preset": run_spec.get("stochastic_preset"),
            "nn_hidden": run_spec.get("fidelity_status", {}).get("nn_hidden"),
            "io_response_map_status": response_map_status,
            "io_response_map_blocker": blocker,
        },
        notes=(
            "GRU empirical/nonlinear standard row. Clean action behavior is "
            "evaluated against the C&S output-feedback LQR/Kalman reference; "
            "same-coordinate transition/value/Bellman rows are not applicable."
        ),
    )
    request = StandardCertificateRowRequest(
        spec=spec,
        architecture="gru",
        certificate_mode="empirical_nonlinear",
        status=(
            "partial_standard_certificate_blocked"
            if action_evidence_available
            else "standard_certificate_missing_action_evidence"
        ),
        component_kwargs={
            "candidate_actions": candidate if action_evidence_available else None,
            "reference_actions": reference if action_evidence_available else None,
            "action_weight": np.asarray(action_weight, dtype=np.float64),
            "optimizer_metadata": _optimizer_metadata(run_spec, training_summary),
            "recurrence_diagnostics": {
                "certificate_mode": "empirical_nonlinear",
                "controller_kind": "gru",
                "hidden_size": model_hps.get("hidden_size"),
                "n_replicates": model_hps.get("n_replicates"),
                "population_structure": model_hps.get("population_structure"),
                "io_response_map_status": response_map_status,
                "io_response_map_observation_dim": 4 if response_map_evidence_available else None,
                "io_response_map_blocker": blocker,
                "observation_history_covariance": covariance_metadata,
            },
            "candidate_observation_to_action_map": candidate_observation_to_action_map,
            "reference_observation_to_action_map": reference_observation_to_action_map,
            "observation_history_covariance": observation_history_covariance,
            "observation_history_covariance_metadata": covariance_metadata,
            "action_label": "command",
            "state_label": "feedbax_rollout_state_not_used",
        },
        metrics={
            "candidate_action_shape": [int(dim) for dim in candidate.shape],
            "reference_action_shape": [int(dim) for dim in reference.shape],
            "evaluation": evaluation_metadata or {},
            "reference": reference_metadata or {},
            "io_response_map_status": response_map_status,
            "io_response_map_blocker": blocker,
            "observation_history_covariance": covariance_metadata,
        },
        artifacts=_artifact_paths(
            run_spec_path=run_spec_path,
            model_path=model_path,
            training_summary_path=training_summary_path,
            repo_root=repo_root,
        ),
    )
    manifest = build_standard_certificate_manifest(request)
    components = tuple(
        _with_materializer_reasons(
            component,
            response_blocker=blocker,
            action_evidence_available=action_evidence_available,
        )
        for component in manifest.certificate_components
    )
    return replace(manifest, certificate_components=components)


def evaluate_gru_clean_actions(
    run_id: str,
    *,
    run_spec: dict[str, Any] | None = None,
    experiment: str = SOURCE_ISSUE_ID,
    use_validation_selected_checkpoints: bool = False,
    preferred_checkpoint_manifest_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a GRU pilot model and return clean action traces and I/O maps.

    Returns:
        Candidate actions with shape ``(replicate * trial, horizon, action)``,
        candidate feedback-to-action Jacobians with shape
        ``(replicate * trial, horizon, action, horizon * feedback)``, and
        JSON-compatible evaluation metadata.
    """

    run_spec = run_spec or _read_json(
        repo_root / "results" / experiment / "runs" / run_id / "run.json"
    )
    hps = dict_to_namespace(normalize_gru_hps(run_spec["hps"]), to_type=TreeNamespace)
    n_replicates = int(hps.model.n_replicates)
    pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run_spec.get("seed", 42))))
    if use_validation_selected_checkpoints:
        from rlrmp.analysis.pipelines.gru_checkpoint_selection import (
            load_validation_selected_checkpoint_model,
        )

        model, checkpoint_selection = load_validation_selected_checkpoint_model(
            experiment=experiment,
            run_id=run_id,
            run_spec=run_spec,
            preferred_manifest_path=preferred_checkpoint_manifest_path,
            checkpoint_selection_mode=(
                "fixed_bank_manifest"
                if preferred_checkpoint_manifest_path is not None
                else "sparse_history"
            ),
            repo_root=repo_root,
        )
    else:
        model, _hyperparameters = load_with_hyperparameters(
            _default_model_path(run_id, experiment=experiment, repo_root=repo_root),
            setup_func=lambda key, **_kwargs: setup_task_model_pair(hps, key=key).model,
        )
        checkpoint_selection = []
    clean_model = _disable_stochastic_runtime(model)
    trial_specs = pair.task.validation_trials
    n_trials = _trial_count(trial_specs)
    model_arrays, model_other = eqx.partition(
        clean_model,
        lambda leaf: _is_replicate_array(leaf, n_replicates),
    )
    stochastic_model_arrays, stochastic_model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_array(leaf, n_replicates),
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        states = pair.task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, n_trials),
        )
        return states.net.output, states.net.input

    outputs, net_inputs = eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(0), n_replicates),
    )
    actions = np.asarray(outputs, dtype=np.float64).reshape(
        n_replicates * n_trials,
        outputs.shape[-2],
        outputs.shape[-1],
    )
    response_map_net_inputs = _stochastic_response_map_net_inputs(
        model_arrays=stochastic_model_arrays,
        model_other=stochastic_model_other,
        pair=pair,
        n_replicates=n_replicates,
        n_rollout_trials=DEFAULT_RESPONSE_MAP_ROLLOUT_TRIALS,
    )
    response_maps = _gru_observation_to_action_maps(
        model_arrays=model_arrays,
        model_other=model_other,
        net_inputs=response_map_net_inputs,
        n_replicates=n_replicates,
        feedback_dim=_controller_feedback_dim(run_spec),
    )
    observation_history_covariance, observation_history_covariance_metadata = (
        observation_history_covariance_from_net_inputs(
            response_map_net_inputs,
            feedback_dim=_controller_feedback_dim(run_spec),
            source="empirical_validation_observation_history",
        )
    )
    return (
        actions,
        response_maps,
        {
            "status": "evaluated_clean_feedbax_rollout",
            "checkpoint_selection": (
                "validation_selected_per_replicate"
                if use_validation_selected_checkpoints
                else "final_checkpoint"
            ),
            "selected_checkpoints": [
                selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
            ],
            "n_replicates": n_replicates,
            "n_trials": n_trials,
            "noise": "feedbax Channel noise disabled; plant-process force noise disabled if present",
            "io_response_map": {
                "status": "evaluated_controller_local_jacobian_on_stochastic_histories",
                "input_channel": _controller_feedback_channel_label(run_spec),
                "n_rollout_trials_per_replicate": DEFAULT_RESPONSE_MAP_ROLLOUT_TRIALS,
                "map_shape": [int(dim) for dim in response_maps.shape],
            },
            "observation_history_covariance": observation_history_covariance_metadata,
            "_observation_history_covariance_array": observation_history_covariance,
        },
    )


def _stochastic_response_map_net_inputs(
    *,
    model_arrays: Any,
    model_other: Any,
    pair: Any,
    n_replicates: int,
    n_rollout_trials: int,
) -> Any:
    """Return stochastic network input histories for local I/O linearization."""

    trial_specs = _repeat_single_validation_trial(pair.task.validation_trials, n_rollout_trials)

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        states = pair.task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, n_rollout_trials),
        )
        return states.net.input

    return eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(1), n_replicates),
    )


def _gru_observation_to_action_maps(
    *,
    model_arrays: Any,
    model_other: Any,
    net_inputs: Any,
    n_replicates: int,
    feedback_dim: int,
) -> np.ndarray:
    """Return local GRU feedback-history to action-history Jacobians.

    The clean Feedbax rollout records the flattened vector consumed by the
    network at each step. For these additive-SISU pilots, the final
    ``feedback_dim`` entries are the delayed position/velocity feedback channel
    and the preceding entries are fixed task inputs. The Jacobian therefore
    holds task inputs fixed and differentiates commands with respect to the
    entire 4D feedback history.
    """

    def map_one_replicate(model_array_leaves: Any, replicate_net_inputs: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        net = replicate_model.nodes["net"]

        def map_one_trial(trial_net_inputs: Any) -> Any:
            task_inputs = trial_net_inputs[:, :-feedback_dim]
            feedback_seq = trial_net_inputs[:, -feedback_dim:]

            def rollout(feedback_flat: Any) -> Any:
                feedback_history = feedback_flat.reshape(feedback_seq.shape)
                state = init_state_from_component(net)
                outputs = []
                for t in range(task_inputs.shape[0]):
                    output, state = net(
                        {
                            "input": task_inputs[t],
                            "feedback": feedback_history[t],
                        },
                        state,
                        key=jr.PRNGKey(0),
                    )
                    outputs.append(output["output"])
                return jnp.stack(outputs, axis=0)

            return jax.jacfwd(rollout)(feedback_seq.reshape(-1))

        return jax.vmap(map_one_trial)(replicate_net_inputs)

    maps = eqx.filter_vmap(map_one_replicate, in_axes=(0, 0))(model_arrays, net_inputs)
    return np.asarray(maps, dtype=np.float64).reshape(
        n_replicates * maps.shape[1],
        maps.shape[-3],
        maps.shape[-2],
        maps.shape[-1],
    )


def observation_history_covariance_from_net_inputs(
    net_inputs: Any,
    *,
    feedback_dim: int,
    source: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate covariance for flattened delayed-feedback histories.

    Args:
        net_inputs: Network input histories with leading sample axes followed by
            ``(horizon, input)``. The final ``feedback_dim`` input channels are
            the observation/feedback basis used by the GRU response map.
        feedback_dim: Number of trailing delayed-feedback channels.
        source: Manifest label for the covariance lens.
    """

    values = np.asarray(net_inputs, dtype=np.float64)
    if values.ndim < 3:
        return None, _missing_observation_history_covariance_metadata(
            "net input histories must have leading sample axes plus horizon and input axes"
        )
    if values.shape[-1] < feedback_dim:
        return None, _missing_observation_history_covariance_metadata(
            "net input histories do not contain the declared delayed-feedback channels"
        )
    histories = values[..., -feedback_dim:]
    sample_count = int(np.prod(histories.shape[:-2], dtype=int))
    horizon = int(histories.shape[-2])
    history_dim = int(horizon * feedback_dim)
    if sample_count < 2:
        return None, _missing_observation_history_covariance_metadata(
            "at least two sampled histories are required for empirical covariance",
            sample_count=sample_count,
            history_dim=history_dim,
        )
    flattened = histories.reshape((sample_count, history_dim))
    if not np.all(np.isfinite(flattened)):
        return None, _missing_observation_history_covariance_metadata(
            "sampled histories contain non-finite values",
            sample_count=sample_count,
            history_dim=history_dim,
        )

    centered = flattened - np.mean(flattened, axis=0, keepdims=True)
    covariance = (centered.T @ centered) / float(sample_count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, {
        "status": "available",
        "source": source,
        "source_priority": 1,
        "basis": "flattened_delayed_position_velocity_observation_history",
        "sample_count": sample_count,
        "horizon": horizon,
        "observation_dim": int(feedback_dim),
        "history_dim": history_dim,
        "centering": "sample_mean_subtracted",
        "estimator": "unbiased_sample_covariance",
        "regularization": {
            "type": "none",
            "eigenvalue_floor": 0.0,
            "diagonal_jitter": 0.0,
            "ratio_denominator_floor": 1e-12,
        },
        "normalization": "expected_squared_output_energy_ratio",
        "audit_only": True,
        "future_lenses": {
            "sensory_noise_covariance": "missing_same_basis_contract",
            "perturbation_bank_covariance": "blocked_pending_issue_3992394",
        },
    }


def _missing_observation_history_covariance_metadata(
    reason: str,
    *,
    sample_count: int | None = None,
    history_dim: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "status": "missing",
        "reason": reason,
        "source": None,
        "sample_count": sample_count,
        "history_dim": history_dim,
        "centering": "not_applicable",
        "regularization": "not_applicable",
        "normalization": "not_applicable",
        "audit_only": True,
        "future_lenses": {
            "sensory_noise_covariance": "missing_same_basis_contract",
            "perturbation_bank_covariance": "blocked_pending_issue_3992394",
        },
    }
    return metadata


def cs_output_feedback_reference_actions(
    output_config: OutputFeedbackConfig = OutputFeedbackConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return clean C&S output-feedback LQR/Kalman reference actions."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    x0 = make_cs_output_feedback_initial_state(reference.plant, output_config)
    rollout = rollout_with_kalman_estimator(
        reference.plant,
        reference.lqr_solution.K,
        x0,
        config=output_config,
    )
    observation_dim = int(delayed_observation_matrix(reference.plant, output_config).shape[0])
    return np.asarray(rollout.u, dtype=np.float64), {
        "controller": "analytical_lqr_kalman_output_feedback",
        "gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "action_weight": np.asarray(reference.schedule.R, dtype=np.float64),
        "horizon": int(reference.schedule.T),
        "action_dim": int(reference.plant.m_u),
        "cs_observation_dim": observation_dim,
        "output_feedback_config": {
            "n_phys": int(output_config.n_phys),
            "delay_steps": int(output_config.delay_steps),
            "estimator_initial_covariance": float(output_config.estimator_initial_covariance),
        },
    }


def cs_output_feedback_observation_action_map(
    output_config: OutputFeedbackConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the stochastic extLQG 4D observation-history to command map."""

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    config = (
        position_velocity_observation_config(reference.plant)
        if output_config is None
        else output_config
    )
    covariances = default_cs_noise_covariances(reference.plant, config)
    comparator = build_extlqg_comparator_path(
        reference.plant,
        reference.lqr_solution.K,
        covariances,
        schedule=reference.schedule,
        config=config,
    )
    H = delayed_observation_matrix(reference.plant, config)
    horizon = int(reference.schedule.T)
    observation_dim = int(H.shape[0])
    action_dim = int(reference.plant.m_u)
    history_dim = horizon * observation_dim
    sensitivity = np.zeros((reference.plant.n, history_dim), dtype=np.float64)
    response = np.zeros((horizon, action_dim, history_dim), dtype=np.float64)
    A = np.asarray(reference.plant.A, dtype=np.float64)
    B = np.asarray(reference.plant.B, dtype=np.float64)
    K = np.asarray(comparator.controller_gains, dtype=np.float64)
    H_np = np.asarray(H, dtype=np.float64)
    G = np.asarray(comparator.estimator_gains, dtype=np.float64)
    for t in range(horizon):
        response[t] = -K[t] @ sensitivity
        sensitivity = (A - B @ K[t] - G[t] @ H_np) @ sensitivity
        sensitivity[:, t * observation_dim : (t + 1) * observation_dim] += G[t]
    return response, {
        "controller": "analytical_lqr_kalman_output_feedback",
        "controller_variant": "cs_released_extlqg_stochastic_fixed_point",
        "parity_status": comparator.parity_status,
        "extlqg_iterations": comparator.n_iterations,
        "extlqg_expected_cost": comparator.expected_cost,
        "gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "observation_channel": "delayed_position_velocity",
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "history_dim": history_dim,
        "map_shape": [int(dim) for dim in response.shape],
    }


def _repeat_single_validation_trial(trial_specs: Any, n_trials: int) -> Any:
    """Repeat the first validation trial along its leading trial axis.

    Older fixed-target rows have a one-trial validation spec, but target-relative
    multi-target rows expose a larger validation bank. The response-map
    linearization uses a fixed number of stochastic histories, so it needs a
    trial spec whose leading trial axis exactly matches ``n_trials``.
    """

    source_trials = _trial_count(trial_specs)

    def repeat_leaf(leaf: Any) -> Any:
        shape = getattr(leaf, "shape", None)
        if shape is not None and len(shape) >= 1 and shape[0] == source_trials:
            return jnp.repeat(leaf[:1], n_trials, axis=0)
        return leaf

    return jt.map(repeat_leaf, trial_specs)


def normalize_gru_hps(hps: dict[str, Any]) -> dict[str, Any]:
    """Normalize serialized GRU hparams into the training builder contract."""

    normalized = copy.deepcopy(hps)
    if normalized.get("hidden_type") == "equinox.nn._rnn.GRUCell":
        normalized["hidden_type"] = None
    return normalized


def _controller_feedback_dim(run_spec: Mapping[str, Any]) -> int:
    h0 = run_spec.get("model_summary", {}).get("initial_hidden_encoder", {})
    context_shape = h0.get("context_shape") if isinstance(h0, Mapping) else None
    if context_shape:
        return int(context_shape[0])
    model = run_spec.get("hps", {}).get("model", {})
    if isinstance(model, Mapping) and bool(model.get("force_filter_feedback")):
        return 6
    target_relative = run_spec.get("hps", {}).get("target_relative_multitarget", {})
    if isinstance(target_relative, Mapping) and bool(target_relative.get("force_filter_feedback")):
        return 6
    return 4


def _controller_feedback_channel_label(run_spec: Mapping[str, Any]) -> str:
    feedback_dim = _controller_feedback_dim(run_spec)
    if feedback_dim == 6:
        return "6D delayed position/velocity plus force-filter feedback"
    if feedback_dim == 4:
        return "4D delayed position/velocity feedback"
    return f"{feedback_dim}D delayed controller feedback"


def gru_io_response_map_blocker(
    run_spec: dict[str, Any],
    *,
    source_issue_id: str = SOURCE_ISSUE_ID,
) -> str:
    """Return the current reason GRU response maps cannot be compared honestly."""

    feedback_dim = _controller_feedback_dim(run_spec)
    reference_observation_dim = 8
    graph_meta = run_spec.get("feedbax_graph", {})
    graph_path = graph_meta.get("graph_spec_path", "model.graph.json")
    return (
        f"Response-map components are blocked: the {source_issue_id} Feedbax GraphSpec "
        f"({graph_path}) feeds the GRU {_controller_feedback_channel_label(run_spec)} "
        f"({feedback_dim}D), while the current C&S output-feedback reference "
        f"uses delayed_observation_matrix over the full physical block "
        f"({reference_observation_dim}D). No approved {feedback_dim}D-to-8D projection or "
        f"{feedback_dim}D analytical reference response-map contract is present."
    )


def write_gru_standard_result(
    result: dict[str, Any],
    *,
    note_path: Path = NOTE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    regeneration_spec_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Write the GRU standard-certificate note and manifest."""

    mkdir_p(note_path.parent)
    regeneration_spec_path = regeneration_spec_path or _regeneration_spec_path(manifest_path)
    result = copy.deepcopy(result)
    result["regeneration_spec"] = repo_relative(regeneration_spec_path, repo_root=repo_root)
    note_path.write_text(render_gru_standard_markdown(result), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_regeneration_spec(
        spec_path=regeneration_spec_path,
        diagnostic_name="gru_standard_certificate",
        materializer="rlrmp.analysis.pipelines.cs_gru_standard_materialization.write_gru_standard_result",
        command=None,
        parameters={
            "experiment": result.get("source_issue"),
            "run_ids": list(result.get("source_manifests", {}).keys()),
            "checkpoint_policy": result.get("checkpoint_policy"),
            "materializer_issue_id": result.get("issue"),
        },
        inputs=[
            {"role": "run_spec", "path": path}
            for path in result.get("source_manifests", {}).values()
        ]
        + [
            {"role": "model_or_checkpoint_artifact", "path": path}
            for path in result.get("source_artifacts", {}).values()
        ],
        outputs=[
            {"role": "standard_certificate_manifest", "path": manifest_path},
            {"role": "standard_certificate_note", "path": note_path},
        ],
        source_files=[
            "src/rlrmp/analysis/pipelines/cs_gru_standard_materialization.py",
            "src/rlrmp/analysis/pipelines/bridge_certificates.py",
            "src/rlrmp/analysis/pipelines/standard_certificate_materialization.py",
        ],
        notes=[
            "Standard certificate is in scope for regeneration specs.",
            "This spec records materialization provenance only; certificate values remain in the manifest.",
        ],
        repo_root=repo_root,
    )


def _regeneration_spec_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_regeneration_spec.json")


def render_gru_standard_markdown(result: dict[str, Any]) -> str:
    """Render a compact tracked note for the GRU materialization."""

    rows = result["rows"]
    failure_rows = result["failure_decomposition"]["rows"]
    blocker = "\n".join(f"- {item}" for item in result["summary"].get("blockers", ()))
    blocker_section = blocker if blocker else "_None for observation-to-action maps._"
    return f"""# GRU Standard Certificates

Issue: `{result["issue"]}`. Source run issue: `{result["source_issue"]}`.

This materialization applies the standard certificate umbrella contract to the
locally synced C&S stochastic GRU pilot rows. The rows use
`empirical_nonlinear` mode. Clean rollout action behavior is available;
same-coordinate transition, value, and Bellman components are explicitly
`not_applicable`. Observation-to-action response-map components are evaluated
under the shared 4D delayed position/velocity feedback contract; disturbance and
measurement-output response maps remain unavailable for these GRU rows.

## Observation-contract blocker

{blocker_section}

## Rows

| run | status | action mismatch | obs-action map | cov-weighted obs-action | transition | value | Bellman | class |
|---|---|---:|---:|---:|---|---|---|---|
{_markdown_row_table(rows, failure_rows)}
"""


def _markdown_row_table(rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> str:
    failure_by_id = {row["run_id"]: row for row in failure_rows}
    lines = []
    for row in rows:
        by_name = {component["name"]: component for component in row["certificate_components"]}
        run_id = row["spec"]["run_id"]
        lines.append(
            "| "
            f"{run_id} | "
            f"{row['status']} | "
            f"{_fmt(_summary(by_name, 'state_weighted_action_mismatch', 'aggregate_mismatch_ratio'))} | "
            f"{_fmt(_summary(by_name, 'observation_history_to_action_map_mismatch', 'aggregate_mismatch_ratio'))} | "
            f"{_fmt(_summary(by_name, 'observation_history_to_action_map_mismatch', 'covariance_weighted_aggregate_mismatch_ratio'))} | "
            f"{by_name[CLOSED_LOOP_TRANSITION_MISMATCH]['status']} | "
            f"{by_name[VALUE_POLICY_GAP]['status']} | "
            f"{by_name[BELLMAN_HESSIAN_RESIDUAL]['status']} | "
            f"{failure_by_id[run_id]['classification']['classification']} |"
        )
    return "\n".join(lines)


def _with_materializer_reasons(
    component: BridgeCertificateComponent,
    *,
    response_blocker: str,
    action_evidence_available: bool,
) -> BridgeCertificateComponent:
    if (
        component.name == STATE_WEIGHTED_ACTION_MISMATCH
        and component.status == "missing"
        and not action_evidence_available
    ):
        return BridgeCertificateComponent(
            name=component.name,
            status=component.status,
            reason=_NO_ACTION_EVIDENCE_REASON,
        )
    if component.name in _BLOCKED_RESPONSE_COMPONENTS and component.status == "missing":
        return BridgeCertificateComponent(
            name=component.name,
            status=component.status,
            summary=component.summary,
            reason=response_blocker,
        )
    return component


def _optimizer_metadata(
    run_spec: dict[str, Any],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "optimizer": "adamw",
        "learning_rate": run_spec.get("controller_lr"),
        "completed_batches": training_summary.get("completed_batches"),
        "n_train_batches": training_summary.get("n_train_batches")
        or run_spec.get("n_train_batches"),
        "training_duration_seconds": training_summary.get("training_duration_seconds"),
        "training_batches_per_second": training_summary.get("training_batches_per_second"),
        "full_training_launch": run_spec.get("full_training_launch"),
    }


def _artifact_paths(
    *,
    run_spec_path: Path | None,
    model_path: Path | None,
    training_summary_path: Path | None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    paths = {
        "run_spec": run_spec_path,
        "trained_model": model_path,
        "training_summary": training_summary_path,
    }
    return {
        key: repo_relative(path, repo_root=repo_root)
        for key, path in paths.items()
        if path is not None
    }


def _disable_stochastic_runtime(model: Any) -> Any:
    clean = jt.map(_disable_channel_noise, model, is_leaf=lambda leaf: isinstance(leaf, Channel))
    if PLANT_PROCESS_FORCE_NOISE_LABEL in getattr(clean, "nodes", {}):
        clean = add_plant_process_force_noise(clean, 0.0)
    return clean


def _disable_channel_noise(leaf: Any) -> Any:
    if not isinstance(leaf, Channel):
        return leaf
    if not leaf.add_noise or leaf.noise_func is None:
        return leaf
    return eqx.tree_at(lambda channel: channel.noise_func, leaf, _zero_channel_noise)


def _zero_channel_noise(_key: Any, output: Any) -> Any:
    return jnp.zeros_like(output)


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    return eqx.is_array(leaf) and leaf.ndim >= 1 and leaf.shape[0] == n_replicates


def _trial_count(trial_specs: Any) -> int:
    target = next(iter(trial_specs.targets.values())).value
    return int(target.shape[0])


def _classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = row["classification"]["classification"]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _summary(components: dict[str, dict[str, Any]], name: str, key: str) -> Any:
    return components.get(name, {}).get("summary", {}).get(key)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_model_path(
    run_id: str,
    *,
    experiment: str = SOURCE_ISSUE_ID,
    repo_root: Path = REPO_ROOT,
) -> Path:
    return _run_artifact_path(
        run_id,
        "trained_model.eqx",
        experiment=experiment,
        repo_root=repo_root,
    )


def _default_training_summary_path(
    run_id: str,
    *,
    experiment: str = SOURCE_ISSUE_ID,
    repo_root: Path = REPO_ROOT,
) -> Path:
    return _run_artifact_path(
        run_id,
        "training_summary.json",
        experiment=experiment,
        repo_root=repo_root,
    )


def _run_artifact_path(
    run_id: str,
    file_name: str,
    *,
    experiment: str = SOURCE_ISSUE_ID,
    repo_root: Path = REPO_ROOT,
) -> Path:
    artifact_run_root = repo_root / "_artifacts" / experiment / "runs"
    normalized = artifact_run_root / run_id / file_name
    if normalized.exists():
        return normalized
    return artifact_run_root / run_id / run_id / file_name


__all__ = [
    "MANIFEST_PATH",
    "MATERIALIZER_ISSUE_ID",
    "NOTE_PATH",
    "RUN_IDS",
    "SOURCE_ISSUE_ID",
    "build_gru_standard_manifest_from_actions",
    "cs_output_feedback_reference_actions",
    "evaluate_gru_clean_actions",
    "gru_io_response_map_blocker",
    "materialize_gru_standard_result",
    "materialize_gru_standard_row",
    "normalize_gru_hps",
    "observation_history_covariance_from_net_inputs",
    "render_gru_standard_markdown",
    "write_gru_standard_result",
]
