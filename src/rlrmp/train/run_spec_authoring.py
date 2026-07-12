"""C&S nominal-GRU run-spec authoring."""
# ruff: noqa: F401

from __future__ import annotations

from rlrmp.train.config_materialization import (
    CS_FEEDBAX_N_STEPS,
    CS_REGULARIZED_NN_HIDDEN,
    CS_STAGE_COUNT,
    DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON,
    DELAYED_REACH_TRAINING_MODE,
    _apply_smoke_overrides,
    _config_namespace,
    _initial_hidden_encoder_config,
    _training_diagnostics_enabled,
    build_hps,
    stochastic_preset,
)
import argparse
import subprocess
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
import jax.random as jr
from feedbax.config.namespace import TreeNamespace
from rlrmp.analysis.math.cs_game_card import (
    INIT_POS,
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID,
    TARGET_POS,
    build_canonical_game,
    build_no_integrator_game,
)
from rlrmp.model.feedback_descriptors import (
    DESCRIPTOR_PAYLOAD_KEY,
    controller_feedback_descriptor_payload,
)
from rlrmp.model.feedbax_graph import (
    EXECUTION_BACKEND,
    GRAPH_PLANT_INTERVENOR_NODE,
    RLRMPFeedbaxGraphBundle,
    build_runtime_rlrmp_feedbax_graph_bundle,
    build_point_mass_sensorimotor_graph_spec,
    write_graph_spec_bundle,
)
from rlrmp.loss import (
    CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
    CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
    CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
)
from rlrmp.io import compact_json_dumps
from rlrmp.paths import REPO_ROOT, mkdir_p, run_spec_path
from rlrmp.runtime.run_specs import validate_nominal_gru_run_spec
from rlrmp.runtime.training_run_specs import (
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    RLRMP_RUN_SPEC_PAYLOAD_KEY,
    attach_composed_training_specs,
    attach_post_run_provenance,
)
from rlrmp.model.stochastic_runtime import (
    graphspec_noise_contract,
    stochastic_runtime_config_from_model,
)
from rlrmp.train.broad_epsilon_training import (
    _batch_shape,
    run_broad_epsilon_pgd_inner_maximizer,
)
from rlrmp.train.closed_loop_finite_adversary import (
    FINITE_POLICY_BIAS_INPUT,
    FINITE_POLICY_GAINS_INPUT,
)
from rlrmp.train.fixed_target_perturbation_training import add_zero_graph_channel_inputs
from rlrmp.train.cs_perturbation_training import (
    consumed_calibration_budget_identities,
    make_broad_epsilon_pgd_pre_step,
    make_policy_adversary_pre_step,
    policy_adversary_objective,
    target_relative_validation_manifest,
    validation_bin_manifest,
)
from rlrmp.train.training_configs import (
    ADAPTIVE_EPSILON_TRAINING_MODE_LOSS_BLEND,
    BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE,
    BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM,
    BROAD_EPSILON_PGD_TRAINING_MODE,
    BROAD_EPSILON_TRAINING_MODE,
    LEGACY_PERTURBATION_TRAINING_MODE,
    PERTURBATION_TRAINING_MODE,
    POLICY_ADVERSARY_MEMORYLESS_MLP,
    POLICY_ADVERSARY_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
    TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
    PolicyFullStateEpsilonTrainingConfig,
    target_relative_target_support_config,
)
from rlrmp.train.task_model import (
    CS_LSS_PLANT_BACKEND,
    LEGACY_CAUSAL_PLANT_BACKEND,
    setup_task_model_pair,
)
from rlrmp.model.trainable import staged_network_trainable_paths
from rlrmp.train.training_configs import (
    CS_CONTROL_SCALE,
    CS_POSITION_SCALE,
    CS_VELOCITY_SCALE,
    CsNominalGruConfig,
    DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW,
)
from rlrmp.train.executor.checkpoints import (
    SCHEMA_VERSION,
    _plain,
    load_latest_checkpoint as load_latest_checkpoint,
)

TRAINING_DIAGNOSTICS_NPZ = "training_diagnostics.npz"

TRAINING_DIAGNOSTICS_MANIFEST = "training_diagnostics.json"

# Keep active tracked recipes under the generic results-JSON guard in
# tests/analysis/pipelines/test_tracked_diagnostic_payload_guards.py. Large
# composed records retain their full RLRMP extension for manifest custody, but
# avoid serializing that extension's fields a second time at the recipe root.
MAX_TRACKED_RUN_SPEC_BYTES = 500 * 1024
COMPACT_RUN_SPEC_KEY = "compact_run_spec"

def _config_default(field_name: str) -> Any:
    """Return the canonical model default for one config field."""

    return CsNominalGruConfig.model_fields[field_name].default


def derive_spec_dir(output_dir: Path) -> Path:
    """Return the tracked spec directory corresponding to an artifact directory."""

    out = Path(output_dir)
    artifact_root = REPO_ROOT / "_artifacts"
    spec_root = REPO_ROOT / "results"
    logical_out = out if out.is_absolute() else REPO_ROOT / out
    try:
        rel = logical_out.relative_to(artifact_root)
        return spec_root / rel
    except ValueError:
        pass
    resolved_out = out.resolve()
    try:
        rel = resolved_out.relative_to(artifact_root.resolve())
        return spec_root / rel
    except ValueError:
        return resolved_out.parent / f"{resolved_out.name}_spec"


def derive_spec_path(output_dir: Path) -> Path:
    """Return the canonical flat run-recipe file for an artifact directory.

    The recipe is written to ``results/<exp>/runs/<run>.json``. The sibling
    ``results/<exp>/runs/<run>/`` directory remains available for lightweight
    sidecars such as GraphSpec manifests.
    """

    sidecar_dir = derive_spec_dir(output_dir)
    spec_root = (REPO_ROOT / "results").resolve()
    try:
        rel = sidecar_dir.resolve().relative_to(spec_root)
    except ValueError:
        return sidecar_dir.parent / f"{sidecar_dir.name}.json"
    parts = rel.parts
    if len(parts) == 3 and parts[1] == "runs":
        return run_spec_path(parts[0], parts[2], for_write=True)
    return sidecar_dir.parent / f"{sidecar_dir.name}.json"


def _run_spec_path_for_write(*, output_dir: Path, spec_dir: Path, explicit_spec_dir: bool) -> Path:
    """Return the flat recipe path paired with ``spec_dir`` sidecars."""

    if explicit_spec_dir:
        return spec_dir.parent / f"{spec_dir.name}.json"
    return derive_spec_path(output_dir)


def _delayed_reach_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "delayed_reach", None), "enabled", False))


def build_game_card_provenance() -> dict[str, Any]:
    """Return lightweight C&S game-card provenance without solving Riccati systems."""

    plant, schedule = build_canonical_game()
    target = [float(x) for x in TARGET_POS.tolist()]
    init = [float(x) for x in INIT_POS.tolist()]
    return {
        "source_module": "rlrmp.analysis.math.cs_game_card",
        "canonical_builder": "build_canonical_game",
        "discretization": plant.discretization,
        "dt": float(plant.dt),
        "horizon_steps": int(schedule.T),
        "feedbax_task_n_steps": CS_FEEDBAX_N_STEPS,
        "feedbax_control_cost_stages": CS_STAGE_COUNT,
        "init_pos_m": init,
        "target_pos_m": target,
        "target_distance_m": float(TARGET_POS[0]),
        "hold_free": True,
        "single_reach": True,
        "plant": {
            "state_dim": int(plant.n),
            "control_dim": int(plant.m_u),
            "disturbance_dim": int(plant.m_w),
            "delay_steps": 5,
            "physical_state_dim": 8,
            "bw_shape": list(plant.Bw.shape),
            "bw_contract": "top physical 8x8 block is identity; lag rows are zero",
            "mass": 1.0,
            "damping": 0.1,
            "tau": 0.066,
        },
        "cost": {
            "schedule": "C&S Eq. 15 physical 8-state schedule with 5-step delay distribution",
            "position_weight": "fact_t * 1e6",
            "velocity_weight": "fact_t * 1e5",
            "force_and_integrator_weight": "1.0",
            "fact_t": "((t + 1) / T)^6, capped at 1",
            "R": "I_2",
            "terminal_Q_f": "diag([1e6, 1e6, 1e5, 1e5, 1, 1, 1, 1]) on physical state",
            "feedbax_force_filter_state_cost": "not_available",
            "feedbax_force_filter_state_cost_note": (
                "The GRU task metadata records the analytical force/integrator "
                "cost, but the current Feedbax loss only exposes clean effector "
                "position, velocity, and efferent-output terms here; no synthetic "
                "force/filter-state cost is added."
            ),
        },
        "output_feedback_certificate_gamma_factor": OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        "output_feedback_gamma_selection_issue": OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID,
        "gamma_usage": (
            "Recorded for C&S provenance only. This nominal run has no robust/minimax "
            "adversarial phase and does not claim a certificate pass."
        ),
    }


def build_loss_game_card_provenance(hps: TreeNamespace) -> dict[str, Any]:
    """Return game-card provenance with objective-specific loss notes."""

    card = build_game_card_provenance()
    no_integrator_state = bool(getattr(hps.model, "no_integrator_state", False))
    if no_integrator_state:
        card["canonical_builder"] = "build_no_integrator_game"
        card["comparator_variant"] = {
            "enabled": True,
            "name": "no_integrator_state",
            "canonical_cs2019_fidelity": False,
            "omitted_coordinates": ["eps_x_int", "eps_y_int"],
        }
        card["plant"] = {
            **card["plant"],
            "state_dim": int(getattr(hps.model, "state_dim", 36)),
            "disturbance_dim": int(getattr(hps.model, "physical_state_dim", 6)),
            "physical_state_dim": int(getattr(hps.model, "physical_state_dim", 6)),
            "bw_shape": [
                int(getattr(hps.model, "state_dim", 36)),
                int(getattr(hps.model, "physical_state_dim", 6)),
            ],
            "bw_contract": "top physical 6x6 block is identity; lag rows are zero",
        }
        card["cost"] = {
            **card["cost"],
            "schedule": "C&S Eq. 15 physical 6-state schedule with 5-step delay distribution",
            "force_and_integrator_weight": "force/filter entries only; integrator entries omitted",
            "terminal_Q_f": "diag([1e6, 1e6, 1e5, 1e5, 1, 1]) on physical state",
        }
    if _delayed_reach_enabled(hps):
        card["delayed_reach_projection"] = {
            "enabled": True,
            "rollout_control_stages": int(hps.task.n_steps),
            "canonical_cs_movement_horizon_steps": CS_STAGE_COUNT,
            "cost_indexing": "canonical Q/R/Q_f schedule starts at sampled go cue",
            "cost_tail_mode": str(hps.loss.delayed_movement_cost_tail_mode),
            "prep_epoch": "not part of canonical movement-stage C&S cost",
        }
    objective = str(getattr(hps.loss, "objective", CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE))
    if objective == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE:
        card["cost"] = {
            **card["cost"],
            "feedbax_force_filter_state_cost": "included_as_partial_ablation_running_term",
            "feedbax_force_filter_state_cost_note": (
                "This ablation preserves the historical partial Feedbax position/velocity "
                "terms, moves command cost to intended net.output, and adds a running "
                "force/filter state penalty over mechanics.vector force coordinates. It "
                "still omits disturbance-integrator state and terminal full-state Q_f costs."
            ),
        }
    elif objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE:
        card["cost"] = {
            **card["cost"],
            "feedbax_force_filter_state_cost": "included_via_full_qrf",
            "feedbax_force_filter_state_cost_note": (
                "Full analytical Q/R/Q_f loss scores force/filter state through the "
                "delay-augmented Q_t and Q_f matrices."
                if no_integrator_state
                else (
                    "Full analytical Q/R/Q_f loss scores force/filter and "
                    "disturbance-integrator state through the canonical delay-augmented "
                    "C&S Q_t and Q_f matrices."
                )
            ),
        }
    return card


def build_model_structure_summary(hps: TreeNamespace) -> dict[str, Any]:
    """Return the model/training summary embedded in ``run.json``."""

    pop = hps.model.population_structure
    stochastic_runtime = _stochastic_runtime_contract(hps)
    plant_backend = str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND))
    exact_lss = plant_backend == CS_LSS_PLANT_BACKEND
    h0 = _initial_hidden_encoder_metadata(hps)
    delayed_reach = _delayed_reach_enabled(hps)
    physical_state_dim = int(getattr(hps.model, "physical_state_dim", 8))
    state_dim = int(getattr(hps.model, "state_dim", 48))
    no_integrator_state = bool(getattr(hps.model, "no_integrator_state", False))
    go_cue_dim = 1 if delayed_reach else 0
    sisu_condition_input = _sisu_conditioned_pgd_input_key(hps)
    sisu_condition_dim = 1 if sisu_condition_input is not None else 0
    feedback_descriptors = _controller_feedback_descriptors(hps)
    return {
        "controller_kind": "gru",
        "plant_backend": plant_backend,
        "plant_backend_warning": (
            "legacy causal SimpleFeedback has a same-step force-filter-to-mechanics timing problem"
            if plant_backend == LEGACY_CAUSAL_PLANT_BACKEND
            else None
        ),
        "exact_cs_linear_state_space": exact_lss,
        "no_integrator_state": no_integrator_state,
        "state_dim": state_dim,
        "physical_state_dim": physical_state_dim,
        "fixed_plant_parameters": (
            ["nodes.mechanics.A", "nodes.mechanics.B", "nodes.mechanics.B_w"] if exact_lss else []
        ),
        "hidden_size": int(hps.model.hidden_size),
        "n_replicates": int(hps.model.n_replicates),
        "trainable": staged_network_trainable_paths(
            sisu_gating=str(getattr(hps, "sisu_gating", "additive")),
            initial_hidden_encoder=bool(h0["enabled"]),
        ),
        "initial_hidden_encoder": h0,
        "population_structure": {
            "n_input_only": int(pop.n_input_only),
            "n_readout_only": int(pop.n_readout_only),
            "n_recurrent_only": int(pop.n_recurrent_only),
            "n_input_readout": int(pop.n_input_readout),
        },
        "feedback": {
            "delay_steps": int(hps.model.feedback_delay_steps),
            "basis": _controller_feedback_basis(hps),
            "dimension": _controller_feedback_dim(hps),
            DESCRIPTOR_PAYLOAD_KEY: feedback_descriptors,
            "descriptor_basis_hash": feedback_descriptors["descriptor_basis_hash"],
            "noise_std": stochastic_runtime["sensory_noise_std"],
            "noise_role": "sensory_feedback",
            "noise_timing": (
                "Feedbax sensory Channel after delayed LSS feedback selector"
                if exact_lss
                else "Feedbax feedback Channel before controller"
            ),
            "delay_source": (
                f"C&S {state_dim}D LinearStateSpace delay-augmented state"
                if exact_lss
                else "Feedbax feedback Channel queue"
            ),
        },
        "go_cue": {
            "enabled": delayed_reach,
            "input_port": "input" if delayed_reach else None,
            "dimension": go_cue_dim,
            "sign": "0_during_prep_1_during_movement" if delayed_reach else None,
            "controller_input_index": 0 if delayed_reach else None,
        },
        "sisu_conditioning": {
            "enabled": sisu_condition_input is not None,
            "input_key": sisu_condition_input,
            "controller_input_port": "input" if sisu_condition_input is not None else None,
            "controller_input_index": 1 if delayed_reach and sisu_condition_input else 0,
            "budget_role": "pgd_energy_fraction" if sisu_condition_input is not None else None,
        },
        "controller_input_dimension": (
            _controller_feedback_dim(hps) + go_cue_dim + sisu_condition_dim
        ),
        "efferent": {
            "additive_motor_noise_std": stochastic_runtime["additive_motor_noise_std"],
            "signal_dependent_motor_noise_std": (
                stochastic_runtime["signal_dependent_motor_noise_std"]
            ),
            "noise_timing": (
                "Feedbax efferent Channel immediately before LinearStateSpace.force"
                if exact_lss
                else "pre_force_filter"
            ),
            "force_filter_tau_s": float(hps.model.tau_rise),
            "force_filter_role": (
                "coupled inside C&S LinearStateSpace dynamics"
                if exact_lss
                else "separate Feedbax FirstOrderFilter node"
            ),
        },
        "plant_process": {
            "force_noise_std": stochastic_runtime["plant_process_force_noise_std"],
            "noise_timing": (
                "mechanics.epsilon_sampled_task_input"
                if exact_lss
                else "post_force_filter_pre_mechanics"
            ),
            "state_diffusion": "mechanics.epsilon" if exact_lss else "not_used",
            "epsilon_bridge": (
                "sampled physical-process/load epsilon Task input bound to C&S "
                "LinearStateSpace mechanics.epsilon using the physical block of "
                "the released C&S process covariance"
                if exact_lss
                else "not_used"
            ),
        },
        "stochastic_runtime": stochastic_runtime,
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "nominal_only": _nominal_only(hps),
        "training_distribution": _training_distribution_metadata(hps),
        "delayed_reach": _plain(hps.delayed_reach),
        "adversarial_phase": _adversarial_phase(hps),
        "certificate_lens": "input_output_map_certificate",
        "certificate_coordinate_claim": "not_same_coordinate_gain",
        "analytical_delay_augmented_state_input": False,
        "certificate_claim": (
            "I/O map certificate framing only; the output-feedback GRU is not fed "
            "the 48D delay-augmented analytical state and is not claimed to share "
            "same-coordinate gains with the analytical controller."
        ),
    }


def build_training_run_graph_spec(hps: TreeNamespace, *, seed: int) -> Any:
    """Return the GraphSpec recorded in the composed Feedbax TrainingRunSpec."""

    if str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND)) != CS_LSS_PLANT_BACKEND:
        return build_graph_bundle(hps).graph_spec

    key_init = jr.split(jr.PRNGKey(int(seed)), 3)[0]
    pair = setup_task_model_pair(hps, key=key_init)
    return build_runtime_rlrmp_feedbax_graph_bundle(hps, pair.model).graph_spec


def build_graph_bundle(hps: TreeNamespace) -> RLRMPFeedbaxGraphBundle:
    """Build the GraphSpec bundle for the nominal GRU run."""

    graph_spec = build_point_mass_sensorimotor_graph_spec(
        hps,
        controller_kind="gru",
        intervention_type="FixedField",
    )
    task_spec = _task_spec(hps)
    loss_spec = _loss_spec(hps)
    training_spec = {
        "dt": float(hps.dt),
        "batch_size": int(hps.batch_size),
        "n_replicates": int(hps.model.n_replicates),
        "controller_kind": "gru",
        "plant_backend": str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND)),
        "trainable": staged_network_trainable_paths(
            sisu_gating=str(getattr(hps, "sisu_gating", "additive")),
            initial_hidden_encoder=_initial_hidden_encoder_enabled(hps),
        ),
        "method": str(hps.method),
        "nominal_only": _nominal_only(hps),
        "training_distribution": _training_distribution_metadata(hps),
        "adversarial_phase": _adversarial_phase(hps),
        "certificate_lens": "input_output_map_certificate",
        "analytical_delay_augmented_state_input": False,
        "stochastic_runtime": _stochastic_runtime_contract(hps),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "loss_objective": str(hps.loss.objective),
        "initial_hidden_encoder": _initial_hidden_encoder_metadata(hps),
        DESCRIPTOR_PAYLOAD_KEY: _controller_feedback_descriptors(hps),
    }
    model_structure = build_model_structure_summary(hps)
    feedback_descriptors = _controller_feedback_descriptors(hps)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "execution_backend": EXECUTION_BACKEND,
        "provenance_refs": {
            "delayed_reach": "$.delayed_reach",
            "loss_objective": "$.loss_objective",
            "model_structure.delayed_reach": "$.delayed_reach",
            "model_structure.stochastic_preset": "$.stochastic_preset",
            "model_structure.stochastic_runtime": "$.stochastic_runtime",
            "model_structure.training_distribution": "$.training_spec.training_distribution",
            "stochastic_preset": "$.stochastic_preset",
            "stochastic_runtime": "$.stochastic_runtime",
            "training_distribution": "$.training_spec.training_distribution",
        },
        "component_policy": {
            "rlrmp_component_types": [
                "FixedField",
            ],
            "feedbax_native_component_types": [
                "FeedbackChannels",
                "PointMass",
                "Channel",
            ],
            "nominal_intervention_policy": (
                f"{GRAPH_PLANT_INTERVENOR_NODE} is present only as an inactive legacy "
                "GraphSpec compatibility component; no robust/minimax adversary is scheduled."
            ),
        },
        "legacy_loader": {
            "setup_function": "rlrmp.train.task_model.setup_task_model_pair",
            "checkpoint_format": "jax_cookbook.save/load_with_hyperparameters",
        },
        "task_spec": task_spec,
        "loss_spec": loss_spec,
        "training_spec": training_spec,
        DESCRIPTOR_PAYLOAD_KEY: feedback_descriptors,
        "game_card_provenance": build_loss_game_card_provenance(hps),
        "model_structure": model_structure,
        "delayed_reach": _plain(hps.delayed_reach),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "stochastic_runtime": _stochastic_runtime_contract(hps),
        "loss_objective": str(hps.loss.objective),
    }
    return RLRMPFeedbaxGraphBundle(
        graph_spec=graph_spec,
        task_spec=task_spec,
        loss_spec=loss_spec,
        training_spec=training_spec,
        manifest=manifest,
    )


def _should_write_graph_spec(hps: TreeNamespace) -> bool:
    return str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND)) != CS_LSS_PLANT_BACKEND


def _write_graph_bundle_for_backend(
    hps: TreeNamespace,
    graph_bundle: RLRMPFeedbaxGraphBundle,
    spec_dir: Path,
) -> Path | None:
    manifest_path = spec_dir / "model.graph.manifest.json"
    if _should_write_graph_spec(hps):
        return write_graph_spec_bundle(graph_bundle, spec_dir)
    manifest = {
        **graph_bundle.manifest,
        "graph_export": {
            "status": "unavailable",
            "reason": (
                "C&S cs_lss runs use LinearStateSpace mechanics and delayed "
                "position/velocity feedback; the current compatibility GraphSpec "
                "builder serializes the legacy FirstOrderFilter -> PointMass path."
            ),
            "authoritative_sources": [
                "run.json.model_summary",
                "run.json.hps.model.plant_backend",
                "trained_model.eqx",
            ],
        },
    }
    manifest_path.write_text(_json_dumps(manifest), encoding="utf-8")
    return None


def _stochastic_runtime_contract(hps: TreeNamespace) -> dict[str, Any]:
    contract = graphspec_noise_contract(stochastic_runtime_config_from_model(hps.model))
    if str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND)) != CS_LSS_PLANT_BACKEND:
        return contract
    return {
        **contract,
        "sensory_runtime": (
            "Feedbax sensory Channel after the 4D delayed LSS feedback selector; "
            "the delay itself is represented by the C&S 48D LSS state"
        ),
        "command_runtime": (
            "Feedbax efferent Channel immediately before LinearStateSpace.force; "
            "additive and signal-dependent motor noise are both command-channel noise"
        ),
        "plant_process_runtime": (
            "Task-sampled physical-process/load epsilon bound to LinearStateSpace.epsilon"
        ),
        "state_diffusion": "mechanics.epsilon",
    }


def build_run_spec(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    spec_dir: Path,
    graph_bundle: RLRMPFeedbaxGraphBundle,
) -> dict[str, Any]:
    """Build the JSON payload for ``run.json``."""

    args = _config_namespace(args)
    hps = build_hps(args)
    training_distribution = _training_distribution_metadata(hps)
    validation_bins = _validation_bins_metadata(hps)
    calibration_consumed = _perturbation_training_enabled(hps) and bool(
        getattr(hps.perturbation_training, "calibrated_timing", False)
    )
    broad_epsilon_consumed = bool(getattr(hps.broad_epsilon_training, "enabled", False)) or bool(
        getattr(hps.broad_epsilon_pgd_training, "enabled", False)
    )
    consumed_data_identities = consumed_calibration_budget_identities(
        calibration_consumed=calibration_consumed,
        broad_epsilon_consumed=broad_epsilon_consumed,
    )
    delayed_reach = _plain(hps.delayed_reach)
    model_summary = build_model_structure_summary(hps)
    training_summary = {
        **graph_bundle.training_spec,
        "training_mode": _training_mode(hps),
        "n_train_batches": int(args.n_train_batches),
        "n_adversary_batches": 0,
        "n_policy_adversary_ascent_steps_per_controller_step": (
            int(
                PolicyFullStateEpsilonTrainingConfig.from_payload(
                    hps.policy_adversary_training
                ).n_steps
            )
            if _policy_adversary_training_enabled(hps)
            else 0
        ),
        "training_diagnostics": _training_diagnostics_metadata(args, output_dir),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": str(args.issue),
        "training_script": "scripts/launch_training.py",
        "mode": _run_mode(args),
        "artifact_output_dir": str(output_dir),
        "spec_dir": str(spec_dir),
        "nominal_only": _nominal_only(hps),
        "training_distribution": training_distribution,
        "delayed_reach": delayed_reach,
        "validation_bins": validation_bins,
        "provenance_refs": {
            "delayed_reach": "$.delayed_reach",
            "loss_objective": "$.loss_objective",
            "model_summary.training_distribution": "$.training_distribution",
            "stochastic_preset": "$.stochastic_preset",
            "training_summary.training_distribution": "$.training_distribution",
            "training_summary.validation_bins": "$.validation_bins",
            "validation_bins": "$.validation_bins",
        },
        "adversarial_phase": _adversarial_phase(hps),
        "modal_launch": "not_requested",
        "full_training_launch": "requested" if args.full_train else "not_requested",
        "seed": int(args.seed),
        "n_train_batches": int(args.n_train_batches),
        "batch_size": int(args.batch_size),
        "controller_lr": float(args.controller_lr),
        "optimizer": _optimizer_metadata(args),
        "checkpointing": _checkpoint_metadata(args, output_dir),
        "training_diagnostics": _training_diagnostics_metadata(args, output_dir),
        "loss_objective": str(hps.loss.objective),
        "fidelity_status": _fidelity_status(hps),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "game_card": build_loss_game_card_provenance(hps),
        "model_summary": model_summary,
        "task_timing": graph_bundle.task_spec,
        "loss_summary": graph_bundle.loss_spec,
        "training_summary": training_summary,
        "feedbax_graph": graph_bundle.to_run_metadata(),
        "consumed_data_identities": consumed_data_identities,
        "hps": _plain(hps),
        "provenance": {
            "git": _get_git_metadata(),
            "dependencies": _get_dependency_metadata(),
            "modal": {
                "launch": "not_requested",
                "app_name": "rlrmp-cs-stochastic-gru",
                "mode": "not_requested",
            },
            "gpu": _get_gpu_metadata(),
            "runtime": _get_runtime_metadata(),
        },
    }


def write_run_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Write, or dry-run, the stochastic C&S GRU spec artifacts."""

    compact_run_spec = bool(getattr(args, "compact_run_spec", False))
    args = _config_namespace(args)
    args = _apply_smoke_overrides(args)
    args = _config_namespace(args)
    output_dir = Path(args.output_dir)
    explicit_spec_dir = args.spec_dir is not None
    spec_dir = Path(args.spec_dir) if explicit_spec_dir else derive_spec_dir(output_dir)
    run_path = _run_spec_path_for_write(
        output_dir=output_dir,
        spec_dir=spec_dir,
        explicit_spec_dir=explicit_spec_dir,
    )
    hps = build_hps(args)
    graph_bundle = build_graph_bundle(hps)
    training_run_graph_spec = build_training_run_graph_spec(hps, seed=int(args.seed))
    payload = build_run_spec(
        args,
        output_dir=output_dir,
        spec_dir=spec_dir,
        graph_bundle=graph_bundle,
    )

    if args.dry_run:
        would_write = [str(run_path), str(spec_dir / "model.graph.manifest.json")]
        if _should_write_graph_spec(hps):
            would_write.append(str(spec_dir / "model.graph.json"))
        composed_payload = attach_composed_training_specs(
            payload,
            graph_spec=training_run_graph_spec,
            output_dir=output_dir,
            spec_dir=spec_dir,
        )
        return {
            "run_spec": compact_run_spec_if_needed(
                composed_payload,
                requested=compact_run_spec,
            ),
            "would_write": would_write,
        }

    mkdir_p(spec_dir)
    mkdir_p(run_path.parent)
    graph_path = _write_graph_bundle_for_backend(hps, graph_bundle, spec_dir)
    payload["feedbax_graph"] = graph_bundle.to_run_metadata(
        graph_spec_path=None if graph_path is None else graph_path.name,
    )
    payload = attach_composed_training_specs(
        payload,
        graph_spec=training_run_graph_spec,
        output_dir=output_dir,
        spec_dir=spec_dir,
    )
    payload = attach_post_run_provenance(
        payload,
        run_spec_path=run_path,
        artifact_dir=output_dir,
        manifest_root=REPO_ROOT / "_artifacts" / "feedbax_runs",
        graph_manifest_path=spec_dir / "model.graph.manifest.json",
        graph_spec_path=graph_path,
    )
    payload = attach_composed_training_specs(
        payload,
        graph_spec=training_run_graph_spec,
        output_dir=output_dir,
        spec_dir=spec_dir,
    )
    validate_nominal_gru_run_spec(payload, spec_dir=spec_dir)
    payload = compact_run_spec_if_needed(payload, requested=compact_run_spec)
    run_path.write_text(_json_dumps(payload), encoding="utf-8")
    return {
        "run_spec_path": str(run_path),
        "graph_spec_path": None if graph_path is None else str(graph_path),
        "graph_manifest_path": str(spec_dir / "model.graph.manifest.json"),
        "training_manifest_path": None,
    }


def _checkpoint_metadata(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "enabled": bool(args.full_train),
        "resume": bool(args.resume),
        "checkpoint_dir": str(Path(output_dir) / "checkpoints"),
        "latest_checkpoint": str(Path(output_dir) / "checkpoints" / "checkpoint_latest"),
        "numbered_pattern": "checkpoint_{completed_batches:07d}",
        "interval_batches": int(args.checkpoint_interval_batches),
        "contents": [
            "model.eqx",
            "optimizer_state.eqx",
            "history.eqx",
            "adversary_policy.eqx when --policy-adversary-training is active",
            "adversary_optimizer_state.eqx when --policy-adversary-training is active",
            "adaptive_epsilon_state in metadata.json when adaptive epsilon curriculum is active",
            "metadata.json",
        ],
    }


def _optimizer_metadata(args: argparse.Namespace) -> dict[str, Any]:
    schedule_name = "warmup_cosine" if int(args.lr_warmup_batches) > 0 else "delayed_cosine"
    return {
        "name": "adamw",
        "learning_rate_0": float(args.controller_lr),
        "schedule": schedule_name,
        "warmup_batches": int(args.lr_warmup_batches),
        "warmup_init_fraction": float(args.lr_warmup_init_fraction),
        "warmup_initial_learning_rate": float(args.controller_lr)
        * float(args.lr_warmup_init_fraction),
        "constant_lr_iterations": int(args.lr_warmup_batches),
        "cosine_annealing_alpha": float(args.lr_cosine_alpha),
        "final_learning_rate": float(args.controller_lr) * float(args.lr_cosine_alpha),
        "weight_decay": 0.0,
        "gradient_clip_norm": (
            None if args.gradient_clip_norm is None else float(args.gradient_clip_norm)
        ),
        "gradient_clip_kind": (None if args.gradient_clip_norm is None else "global_norm"),
        "training_diagnostics": _training_diagnostics_metadata(
            args,
            Path(args.output_dir),
        ),
    }


def _perturbation_training_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(hps.perturbation_training, "enabled", False))


def _target_relative_multitarget_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(hps.target_relative_multitarget, "enabled", False))


def _initial_hidden_encoder_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(hps.model, "initial_hidden_encoder", False))


def _initial_hidden_encoder_metadata(hps: TreeNamespace) -> dict[str, Any]:
    config = getattr(hps.model, "initial_hidden_encoder_config", None)
    if config is None:
        return _initial_hidden_encoder_config(
            enabled=_initial_hidden_encoder_enabled(hps),
            hidden_size=int(hps.model.hidden_size),
            context_dim=_controller_feedback_dim(hps),
            context_basis=_controller_feedback_basis(hps),
        )
    return _plain(config)


def _broad_epsilon_training_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "broad_epsilon_training", None), "enabled", False))


def _broad_epsilon_pgd_training_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "broad_epsilon_pgd_training", None), "enabled", False))


def _policy_adversary_training_enabled(hps: TreeNamespace) -> bool:
    return bool(getattr(getattr(hps, "policy_adversary_training", None), "enabled", False))


def _policy_adversary_policy_class(hps: TreeNamespace) -> str:
    if not _policy_adversary_training_enabled(hps):
        return "disabled"
    return PolicyFullStateEpsilonTrainingConfig.from_payload(
        hps.policy_adversary_training
    ).policy_class


def _broad_epsilon_pgd_mechanism(hps: TreeNamespace) -> str:
    pgd = getattr(hps, "broad_epsilon_pgd_training", None)
    return str(getattr(pgd, "adversary_mechanism", BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM))


def _broad_epsilon_pgd_finite_policy_inputs(hps: TreeNamespace) -> list[str]:
    if not _broad_epsilon_pgd_training_enabled(hps):
        return []
    mechanism = _broad_epsilon_pgd_mechanism(hps)
    if mechanism == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM:
        return []
    keys = [FINITE_POLICY_GAINS_INPUT]
    mechanism_payload = getattr(getattr(hps, "broad_epsilon_pgd_training", None), "mechanism", None)
    has_bias = bool(
        getattr(getattr(mechanism_payload, "required_policy_contract", None), "has_bias", False)
    )
    if has_bias:
        keys.append(FINITE_POLICY_BIAS_INPUT)
    return keys


def _adversarial_phase(hps: TreeNamespace) -> str:
    if _policy_adversary_training_enabled(hps):
        policy_class = _policy_adversary_policy_class(hps)
        if policy_class == POLICY_ADVERSARY_MEMORYLESS_MLP:
            return "learned_memoryless_policy_adversary"
        return f"learned_finite_{policy_class}_policy_adversary"
    if _broad_epsilon_pgd_training_enabled(hps):
        mechanism = _broad_epsilon_pgd_mechanism(hps)
        if mechanism == BROAD_EPSILON_PGD_DIRECT_EPSILON_MECHANISM:
            return "broad_epsilon_pgd_direct_epsilon"
        return f"broad_epsilon_pgd_live_finite_policy_{mechanism}"
    return "none"


def _nominal_only(hps: TreeNamespace) -> bool:
    return (
        not _perturbation_training_enabled(hps)
        and not _broad_epsilon_training_enabled(hps)
        and not _broad_epsilon_pgd_training_enabled(hps)
        and not _policy_adversary_training_enabled(hps)
        and not _target_relative_multitarget_enabled(hps)
        and not _initial_hidden_encoder_enabled(hps)
        and not _delayed_reach_enabled(hps)
    )


def _training_mode(hps: TreeNamespace) -> str:
    if _target_relative_multitarget_enabled(hps):
        parts = [
            (
                TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE
                if _initial_hidden_encoder_enabled(hps)
                else TARGET_RELATIVE_MULTITARGET_TRAINING_MODE
            )
        ]
        if _broad_epsilon_training_enabled(hps):
            parts.append(BROAD_EPSILON_TRAINING_MODE)
        if _broad_epsilon_pgd_training_enabled(hps):
            parts.append(BROAD_EPSILON_PGD_TRAINING_MODE)
        if _policy_adversary_training_enabled(hps):
            parts.append(POLICY_ADVERSARY_TRAINING_MODE)
        if _perturbation_training_enabled(hps):
            parts.append(PERTURBATION_TRAINING_MODE)
        if _delayed_reach_enabled(hps):
            parts.insert(0, DELAYED_REACH_TRAINING_MODE)
        return "+".join(parts)
    if _perturbation_training_enabled(hps):
        parts = [PERTURBATION_TRAINING_MODE]
    else:
        parts = []
    if _broad_epsilon_training_enabled(hps):
        parts.append(BROAD_EPSILON_TRAINING_MODE)
    if _broad_epsilon_pgd_training_enabled(hps):
        parts.append(BROAD_EPSILON_PGD_TRAINING_MODE)
    if _policy_adversary_training_enabled(hps):
        parts.append(POLICY_ADVERSARY_TRAINING_MODE)
    return "+".join(parts) if parts else "nominal"


def _controller_feedback_basis(hps: TreeNamespace) -> str:
    if _target_relative_multitarget_enabled(hps):
        if bool(getattr(hps.target_relative_multitarget, "force_filter_feedback", False)):
            return "target_relative_delayed_feedback_plus_force_filter"
        return "target_relative_delayed_feedback"
    return "raw_delayed_position_velocity"


def _controller_feedback_dim(hps: TreeNamespace) -> int:
    if _target_relative_multitarget_enabled(hps):
        return (
            6
            if bool(getattr(hps.target_relative_multitarget, "force_filter_feedback", False))
            else 4
        )
    return 4


def _controller_feedback_descriptors(hps: TreeNamespace) -> dict[str, Any]:
    return controller_feedback_descriptor_payload(
        feedback_dim=_controller_feedback_dim(hps),
        basis_id=_controller_feedback_basis(hps),
    )


def _validation_bins_metadata(hps: TreeNamespace) -> dict[str, Any]:
    if _target_relative_multitarget_enabled(hps):
        return target_relative_validation_manifest(hps.target_relative_multitarget)
    return validation_bin_manifest(hps.perturbation_training)


def _training_distribution_metadata(hps: TreeNamespace) -> dict[str, Any]:
    config = hps.perturbation_training
    target_config = hps.target_relative_multitarget
    if _target_relative_multitarget_enabled(hps):
        target_payload = target_config.target_distribution
        h0 = _initial_hidden_encoder_metadata(hps)
        return {
            "mode": _training_mode(hps),
            "training_axes": {
                "target_relative_multitarget": True,
                "delayed_reach": _delayed_reach_enabled(hps),
                "initial_hidden_encoder": bool(h0["enabled"]),
                "calibrated_perturbation_training": _perturbation_training_enabled(hps),
                "broad_full_state_epsilon_training": _broad_epsilon_training_enabled(hps),
                "broad_full_state_epsilon_pgd_training": (_broad_epsilon_pgd_training_enabled(hps)),
                "policy_adversary_training": _policy_adversary_training_enabled(hps),
                "force_filter_feedback": bool(
                    getattr(target_config, "force_filter_feedback", False)
                ),
            },
            "fixed_target_only": False,
            "target_stream": {
                "status": "consumed_as_static_target_relative_feedback",
                "input_port": "target",
                "contract": _plain(target_config.input_contract),
            },
            "go_cue_stream": (
                _plain(hps.delayed_reach.go_cue_input)
                if _delayed_reach_enabled(hps)
                else {"enabled": False}
            ),
            "delayed_reach": _plain(hps.delayed_reach),
            "initial_hidden_encoder": h0,
            "force_filter_feedback": _plain(target_config.force_filter_feedback),
            "broad_epsilon_training": (
                _plain(hps.broad_epsilon_training)
                if _broad_epsilon_training_enabled(hps)
                else {"enabled": False}
            ),
            "broad_epsilon_pgd_training": (
                _plain(hps.broad_epsilon_pgd_training)
                if _broad_epsilon_pgd_training_enabled(hps)
                else {"enabled": False}
            ),
            "policy_adversary_training": (
                _plain(hps.policy_adversary_training)
                if _policy_adversary_training_enabled(hps)
                else {"enabled": False}
            ),
            "perturbation_training": (
                _plain(hps.perturbation_training)
                if _perturbation_training_enabled(hps)
                else {"enabled": False}
            ),
            "target_distribution": _plain(target_payload),
            "original_target_anchor_m": _plain(target_payload.original_target_anchor_m),
            "seen_targets_m": _plain(target_payload.seen_targets_m),
            "held_out_targets_m": _plain(target_payload.held_out_targets_m),
            "validation_bins": _plain(target_config.validation_bins),
            "perturbation_mixture_emphasis": _plain(target_config.perturbation_mixture_emphasis),
            "checkpoint_selection_role": ("target_relative_multitarget_rollout_validation"),
            "nominal_quality_role": "original_anchor_and_seen_held_out_targets_reported",
            "controller_internal_mutation": False,
            "adversarial_phase": _adversarial_phase(hps),
        }
    if (
        not bool(getattr(config, "enabled", False))
        and not _broad_epsilon_training_enabled(hps)
        and not _broad_epsilon_pgd_training_enabled(hps)
        and not _policy_adversary_training_enabled(hps)
    ):
        return {
            "mode": "nominal",
            "fixed_target_only": True,
            "target_stream": "not_consumed",
        }
    return {
        "mode": str(getattr(config, "mode", PERTURBATION_TRAINING_MODE)),
        "legacy_mode": LEGACY_PERTURBATION_TRAINING_MODE,
        "fixed_target_only": True,
        "target_stream": {
            "status": "not_consumed",
            "reason": (
                "Current C&S GRU input is scalar external input plus delayed "
                "feedback; no target-position stream is supplied to the controller."
            ),
        },
        "mixture": {
            "nominal_fraction": float(config.nominal_fraction),
            "single_family_fraction": float(config.single_fraction),
            "mild_combined_fraction": float(config.combined_fraction),
            "combined_amplitude_scale": float(config.combined_amplitude_scale),
            "sampling": (
                "prng_driven_signed_random_axes_components_calibrated_timing_levels"
                if bool(getattr(config, "calibrated_timing", False))
                else "prng_driven_signed_random_axes_components_timings_levels"
            ),
            "calibrated_timing": bool(getattr(config, "calibrated_timing", False)),
            "movement_age_timing": bool(getattr(config, "movement_age_timing", False)),
            "physical_level": str(getattr(config, "physical_level", "moderate")),
            "physical_level_fraction_of_reach": float(
                getattr(config, "physical_level_fraction_of_reach", 0.10)
            ),
        },
        "mild_combined_families": ["initial_position", "command_input"],
        "single_family_bins": list(config.single_family_bins),
        "validation_bins": list(config.validation_bins),
        "timing_basis": _plain(config.timing_basis),
        "timing_bins": _plain(config.timing_bins),
        "calibrated_levels": _plain(config.mixture_semantics.calibrated_levels),
        "broad_epsilon_training": (
            _plain(hps.broad_epsilon_training)
            if _broad_epsilon_training_enabled(hps)
            else {"enabled": False}
        ),
        "broad_epsilon_pgd_training": (
            _plain(hps.broad_epsilon_pgd_training)
            if _broad_epsilon_pgd_training_enabled(hps)
            else {"enabled": False}
        ),
        "policy_adversary_training": (
            _plain(hps.policy_adversary_training)
            if _policy_adversary_training_enabled(hps)
            else {"enabled": False}
        ),
        "perturbation_training": (
            _plain(hps.perturbation_training)
            if _perturbation_training_enabled(hps)
            else {"enabled": False}
        ),
        "checkpoint_selection_role": "generalized_held_out_perturbation_validation",
        "nominal_quality_role": "reported_quality_sidecar_gate",
        "controller_internal_mutation": False,
        "adversarial_phase": _adversarial_phase(hps),
    }


def _training_diagnostics_metadata(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    enabled = _training_diagnostics_enabled(args)
    return {
        "enabled": enabled,
        "default_enabled": True,
        "opt_out_flag": "--no-training-diagnostics",
        "schema_version": f"{SCHEMA_VERSION}.training_diagnostics.v1",
        "format": "npz+json_manifest",
        "sidecar_path": str(Path(output_dir) / TRAINING_DIAGNOSTICS_NPZ) if enabled else None,
        "manifest_path": (
            str(Path(output_dir) / TRAINING_DIAGNOSTICS_MANIFEST) if enabled else None
        ),
        "source": (
            "optimizer_state plus RLRMP executor history; no raw gradients, "
            "batches, or activations are persisted"
        ),
        "scalar_groups": [
            "optimizer_gradient_norm_pre_clip",
            "optimizer_gradient_clipped",
            "optimizer_clipping_fraction",
            "optimizer_update_parameter_norm_ratio",
            "optimizer_learning_rate",
            "train_loss_terms",
            "validation_loss_terms",
            "pgd_broad_epsilon_inner_maximizer",
            "policy_adversary_inner_optimizer",
            "adaptive_epsilon_curriculum",
        ],
    }


def _run_mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry_run"
    if args.full_train:
        return "full_train"
    return "spec_write"


def _task_spec(hps: TreeNamespace) -> dict[str, Any]:
    plant_backend = str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND))
    target_relative = _target_relative_multitarget_enabled(hps)
    delayed_reach = _delayed_reach_enabled(hps)
    sisu_condition_input = _sisu_conditioned_pgd_input_key(hps)
    sisu_conditioned_pgd_budget = sisu_condition_input is not None
    rollout_steps = int(hps.task.n_steps) if delayed_reach else int(hps.task.n_steps) - 1
    if delayed_reach:
        movement_window = {
            "kind": "delayed_reach_movement_epoch",
            "start_transition": "sampled_go_cue_step",
            "go_cue_min_step": int(hps.delayed_reach.go_cue_sampling.min_step_inclusive),
            "go_cue_max_step": int(hps.delayed_reach.go_cue_sampling.max_step_inclusive),
            "cs_horizon_steps": CS_STAGE_COUNT,
            "cost_indexing": "movement_age_not_trial_age",
            "cost_tail_mode": str(hps.loss.delayed_movement_cost_tail_mode),
        }
        time_axis_contract = (
            "Delayed C&S task: target is visible from trial start, prep has no "
            "target-directed movement loss, and C&S stage costs are indexed by "
            "movement age from the sampled go cue."
        )
    else:
        movement_window = {
            "kind": "full_simple_reach_trial",
            "start_transition": 0,
            "end_transition": int(hps.task.n_steps) - 2,
        }
        time_axis_contract = (
            "Hold-free fixed nominal task: Feedbax n_steps=61 yields exactly 60 "
            "transition/control-cost stages and one position target per transition; "
            "delayed-reach epoch masks are not used."
        )
    if (
        plant_backend == CS_LSS_PLANT_BACKEND
        and target_relative
        and delayed_reach
        and sisu_conditioned_pgd_budget
    ):
        extra_inputs = ["input", "sisu", "target", "epsilon"]
    elif (
        plant_backend == CS_LSS_PLANT_BACKEND
        and target_relative
        and (delayed_reach or sisu_conditioned_pgd_budget)
    ):
        extra_inputs = ["input", "target", "epsilon"]
    elif plant_backend == CS_LSS_PLANT_BACKEND and target_relative:
        extra_inputs = ["target", "epsilon"]
    elif plant_backend == CS_LSS_PLANT_BACKEND:
        extra_inputs = ["input", "epsilon"]
    else:
        extra_inputs = ["sisu", f"intervene:{GRAPH_PLANT_INTERVENOR_NODE}"]
    extra_inputs = [*extra_inputs, *_broad_epsilon_pgd_finite_policy_inputs(hps)]
    return {
        "type": str(hps.task.type),
        "preset": _plain(getattr(hps.task, "preset", None)),
        "n_steps": int(hps.task.n_steps),
        "n_control_stages": _plain(getattr(hps.task, "n_control_stages", None)),
        "control_cost_stages": rollout_steps,
        "workspace": _plain(hps.task.workspace),
        "fixed_init_pos": _plain(hps.task.fixed_init_pos),
        "fixed_target_pos": _plain(hps.task.fixed_target_pos),
        "eval_grid_n": int(hps.task.eval_grid_n),
        "eval_n_directions": int(hps.task.eval_n_directions),
        "eval_reach_length": float(hps.task.eval_reach_length),
        "epoch_len_ranges": _plain(hps.task.epoch_len_ranges),
        "target_on_epochs": _plain(hps.task.target_on_epochs),
        "hold_epochs": _plain(hps.task.hold_epochs),
        "move_epochs": _plain(hps.task.move_epochs),
        "p_catch_trial": float(hps.task.p_catch_trial),
        "target_visible_from_start": _plain(getattr(hps.task, "target_visible_from_start", None)),
        "go_cue_event_name": _plain(getattr(hps.task, "go_cue_event_name", None)),
        "catch_metadata_policy": _plain(getattr(hps.task, "catch_metadata_policy", None)),
        "coordinate_contract": (
            "Feedbax SimpleReaches supplies mechanics.effector.pos targets in the same "
            "Cartesian metre coordinates as the point-mass effector state."
        ),
        "time_axis_contract": time_axis_contract,
        "movement_window": movement_window,
        "extra_inputs": extra_inputs,
        "delayed_reach": _plain(hps.delayed_reach),
        "target_relative_multitarget": (
            _plain(hps.target_relative_multitarget) if target_relative else {"enabled": False}
        ),
        "broad_epsilon_training": (
            _plain(hps.broad_epsilon_training)
            if _broad_epsilon_training_enabled(hps)
            else {"enabled": False}
        ),
        "broad_epsilon_pgd_training": (
            _plain(hps.broad_epsilon_pgd_training)
            if _broad_epsilon_pgd_training_enabled(hps)
            else {"enabled": False}
        ),
        "initial_hidden_encoder": _initial_hidden_encoder_metadata(hps),
    }


def _sisu_conditioned_pgd_input_key(hps: TreeNamespace) -> str | None:
    pgd = getattr(hps, "broad_epsilon_pgd_training", None)
    if not bool(getattr(pgd, "enabled", False)):
        return None
    pgd_schedule = getattr(pgd, "budget_schedule", None)
    pgd_schedule_mode = (
        getattr(pgd_schedule, "mode", None)
        if pgd_schedule is not None
        else getattr(pgd, "budget_schedule", "")
    )
    if str(pgd_schedule_mode) != BROAD_EPSILON_PGD_SISU_BUDGET_SCHEDULE:
        return None
    pgd_conditioning = getattr(pgd_schedule, "conditioning_scalar", None)
    pgd_condition_input = (
        getattr(pgd_conditioning, "input_key", None)
        if pgd_conditioning is not None
        else getattr(pgd, "sisu_condition_input", "auto")
    )
    if str(pgd_condition_input) == "auto":
        return "sisu" if _delayed_reach_enabled(hps) else "input"
    return str(pgd_condition_input)


def _delayed_pre_go_auxiliary_terms_metadata(hps: TreeNamespace) -> dict[str, Any]:
    weights = getattr(hps.loss, "weights", TreeNamespace())
    start_pos_norm = str(getattr(hps.loss, "delayed_pre_go_start_pos_hold_norm", "l2"))
    terms = {
        "delayed_pre_go_force_filter_hold": {
            "scale": float(getattr(weights, "delayed_pre_go_force_filter_hold", 0.0)),
            "state_key": "states.mechanics.vector delay blocks[..., 4:6]",
            "target": "zero_force_filter_state",
        },
        "delayed_pre_go_start_pos_hold": {
            "scale": float(getattr(weights, "delayed_pre_go_start_pos_hold", 0.0)),
            "state_key": "states.mechanics.effector.pos",
            "target": "trial_specs.inits['mechanics.vector'][..., :2]",
            "norm": start_pos_norm,
        },
        "delayed_pre_go_zero_vel_hold": {
            "scale": float(getattr(weights, "delayed_pre_go_zero_vel_hold", 0.0)),
            "state_key": "states.mechanics.effector.vel",
            "target": "zero_velocity",
        },
    }
    active = {name: meta for name, meta in terms.items() if meta["scale"] != 0.0}
    return {
        "scope": "prep_epoch_only" if _delayed_reach_enabled(hps) else "inactive",
        "epoch_indices": [0] if _delayed_reach_enabled(hps) else [],
        "movement_window_qrf_comparator": "unchanged",
        "terms": terms,
        "active_terms": active,
    }


def _loss_spec(hps: TreeNamespace) -> dict[str, Any]:
    objective = str(getattr(hps.loss, "objective", CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE))
    delayed_reach = _delayed_reach_enabled(hps)
    cs_time_indexing = (
        {
            "stage_schedule": "movement_age_from_go_cue",
            "movement_epoch_source": "trial_specs.timeline.epoch_bounds[-2:]",
            "prep_target_directed_movement_loss": "zero",
            "canonical_movement_horizon_steps": CS_STAGE_COUNT,
            "cost_tail_mode": str(hps.loss.delayed_movement_cost_tail_mode),
            "post_horizon_tail": (
                "zero_weight_after_canonical_horizon"
                if str(hps.loss.delayed_movement_cost_tail_mode)
                == DELAYED_MOVEMENT_COST_TAIL_CANONICAL_WINDOW
                else "hold_terminal_running_qr_weights_flat_to_trial_end"
            ),
        }
        if delayed_reach
        else {
            "stage_schedule": "trial_age_full_simple_reach",
            "canonical_movement_horizon_steps": CS_STAGE_COUNT,
        }
    )
    cs_fact_t = "((movement_age + 1) / 60)^6, capped at 1" if delayed_reach else "((t + 1) / T)^6"
    if objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE:
        no_integrator_state = bool(getattr(hps.model, "no_integrator_state", False))
        _plant, schedule = (
            build_no_integrator_game() if no_integrator_state else build_canonical_game()
        )
        physical_state_dim = 6 if no_integrator_state else 8
        q_diag = jnp.diag(schedule.Q[0])
        qf_diag = jnp.diag(schedule.Q_f)
        trial_type_normalization = _plain(
            getattr(hps.loss, "delayed_trial_type_normalization", {"enabled": False})
        )
        return {
            "weights": _plain(hps.loss.weights),
            "delayed_pre_go_auxiliary_terms": _delayed_pre_go_auxiliary_terms_metadata(hps),
            "delayed_trial_type_normalization": trial_type_normalization,
            "delayed_reach": _plain(hps.delayed_reach),
            "objective_profile": CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE,
            "objective_kind": "finite_horizon_quadratic",
            "grouped_reduction_implementation": (
                "rlrmp_bridge_pending_feedbax_69d8d76"
                if bool(trial_type_normalization.get("enabled", False))
                else "not_enabled"
            ),
            "source_module": (
                "rlrmp.analysis.math.cs_game_card.build_no_integrator_game"
                if no_integrator_state
                else "rlrmp.analysis.math.cs_game_card.build_canonical_game"
            ),
            "comparator_variant": "no_integrator_state" if no_integrator_state else None,
            "state_basis": {
                "state_key": "states.mechanics.vector",
                "dimension": int(schedule.Q.shape[-1]),
                "physical_block_size": physical_state_dim,
                "delay_blocks": int(schedule.Q.shape[-1] // physical_state_dim),
                "coordinate_transform": (
                    "absolute Feedbax position entries are converted to target-centred "
                    "analytical coordinates before applying Q_t and Q_f"
                ),
            },
            "time_indexing": {
                "running_state": (
                    "state before each movement command from sampled go cue"
                    if delayed_reach
                    else "trial init plus rollout states[:-1], paired with commands"
                ),
                "terminal_state": (
                    (
                        "final rollout state after the variable post-horizon tail"
                        if str(hps.loss.delayed_movement_cost_tail_mode)
                        == DELAYED_MOVEMENT_COST_TAIL_FLAT_AFTER_HORIZON
                        else "state after 60 movement commands from sampled go cue"
                    )
                    if delayed_reach
                    else "rollout states[-1]"
                ),
                "horizon_steps": int(schedule.T),
                **cs_time_indexing,
            },
            "matrix_shapes": {
                "Q": list(schedule.Q.shape),
                "R": list(schedule.R.shape),
                "Q_f": list(schedule.Q_f.shape),
            },
            "active_cs_terms": {
                "state_running_q": {
                    "term": "mechanics.vector^T Q_t mechanics.vector",
                    "source": "canonical delay-augmented C&S schedule.Q",
                    "initial_diag_first_block": [float(x) for x in q_diag[:8].tolist()],
                },
                "control_r": {
                    "term": "net.output^T R_t net.output",
                    "source": (
                        "canonical C&S schedule.R on intended controller command "
                        "before efferent/motor-channel noise"
                    ),
                    "diag": [float(x) for x in jnp.diag(schedule.R[0]).tolist()],
                },
                "terminal_q_f": {
                    "term": "mechanics.vector_T^T Q_f mechanics.vector_T",
                    "source": "canonical delay-augmented C&S schedule.Q_f",
                    "diag_first_block": [float(x) for x in qf_diag[:8].tolist()],
                },
            },
            "force_filter_state_cost": "included_via_Q_entries_4_5_each_delay_block",
            "disturbance_integrator_state_cost": "included_via_Q_entries_6_7_each_delay_block",
            "hidden_regularizer": {
                "term": "not_in_full_analytical_qrf_loss",
                "configured_weight": float(hps.loss.weights.nn_hidden),
            },
        }
    if objective == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE:
        return {
            "weights": _plain(hps.loss.weights),
            "delayed_pre_go_auxiliary_terms": _delayed_pre_go_auxiliary_terms_metadata(hps),
            "delayed_reach": _plain(hps.delayed_reach),
            "effector_pos_late": _plain(hps.loss.effector_pos_late),
            "effector_vel_late": _plain(hps.loss.effector_vel_late),
            "effector_pos_running_schedule": str(hps.loss.effector_pos_running_schedule),
            "objective_profile": CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE,
            "objective_kind": "partial_feedbax_ablation",
            "hypothesis": (
                "historical partial position/velocity terms plus intended-command "
                "control cost and LSS force/filter state cost"
            ),
            "active_cs_terms": {
                "stage_position": {
                    "term": "effector_pos_running",
                    "scale": float(hps.loss.weights.effector_pos_running),
                    "fact_t": cs_fact_t,
                },
                "stage_velocity": {
                    "term": "effector_vel_running",
                    "scale": float(hps.loss.weights.effector_vel_running),
                    "fact_t": cs_fact_t,
                },
                "control": {
                    "term": "nn_output",
                    "state_key": "states.net.output",
                    "scale": float(hps.loss.weights.nn_output),
                    "equivalent_R": "I_2 on intended controller command before noise",
                },
                "force_filter": {
                    "term": "mechanics_force_filter",
                    "state_key": "states.mechanics.vector delay blocks[..., 4:6]",
                    "scale": float(hps.loss.weights.mechanics_force_filter),
                    "basis": "force/filter coordinates from every 8D physical delay block",
                },
                "terminal_position": {
                    "term": "effector_terminal_pos",
                    "scale": float(hps.loss.weights.effector_terminal_pos),
                },
                "terminal_velocity": {
                    "term": "effector_terminal_vel",
                    "scale": float(hps.loss.weights.effector_terminal_vel),
                },
            },
            "force_filter_state_cost": "included_as_partial_ablation_running_term",
            "disturbance_integrator_state_cost": "omitted_in_this_ablation",
            "hidden_regularizer": {
                "term": "nn_hidden",
                "scale": float(hps.loss.weights.nn_hidden),
                "exact_fidelity_default": 0.0,
                "regularized_pair_scale": CS_REGULARIZED_NN_HIDDEN,
            },
            "simple_reach_position_loss_contract": (
                "effector_pos_running compares mechanics.effector.pos to the SimpleReaches "
                "same-coordinate target sequence over every transition, using the configured "
                "C&S Eq. 15 power-law discount when requested."
            ),
            "effector_hold_pos_schedule": str(hps.loss.effector_hold_pos_schedule),
            "position_powerlaw_power": float(hps.loss.position_powerlaw_power),
            "movement_ramp_shape": str(hps.loss.movement_ramp_shape),
            "movement_ramp_duration_steps": int(hps.loss.movement_ramp_duration_steps),
            "movement_ramp_power": float(hps.loss.movement_ramp_power),
            "time_indexing": cs_time_indexing,
        }

    return {
        "weights": _plain(hps.loss.weights),
        "delayed_pre_go_auxiliary_terms": _delayed_pre_go_auxiliary_terms_metadata(hps),
        "delayed_reach": _plain(hps.delayed_reach),
        "effector_pos_late": _plain(hps.loss.effector_pos_late),
        "effector_vel_late": _plain(hps.loss.effector_vel_late),
        "effector_pos_running_schedule": str(hps.loss.effector_pos_running_schedule),
        "objective_profile": CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE,
        "active_cs_terms": {
            "stage_position": {
                "term": "effector_pos_running",
                "scale": float(hps.loss.weights.effector_pos_running),
                "fact_t": cs_fact_t,
            },
            "stage_velocity": {
                "term": "effector_vel_running",
                "scale": float(hps.loss.weights.effector_vel_running),
                "fact_t": cs_fact_t,
            },
            "control": {
                "term": "nn_output",
                "scale": float(hps.loss.weights.nn_output),
                "equivalent_R": "I_2 on efferent output",
            },
            "terminal_position": {
                "term": "effector_terminal_pos",
                "scale": float(hps.loss.weights.effector_terminal_pos),
            },
            "terminal_velocity": {
                "term": "effector_terminal_vel",
                "scale": float(hps.loss.weights.effector_terminal_vel),
            },
        },
        "force_filter_state_cost": "not_available",
        "force_filter_state_cost_note": (
            "No force/filter-state quadratic term is synthesized because this "
            "nominal Feedbax loss path has no clean C&S physical force/integrator "
            "state target exposed through the task state contract."
        ),
        "hidden_regularizer": {
            "term": "nn_hidden",
            "scale": float(hps.loss.weights.nn_hidden),
            "exact_fidelity_default": 0.0,
            "regularized_pair_scale": CS_REGULARIZED_NN_HIDDEN,
        },
        "simple_reach_position_loss_contract": (
            "effector_pos_running compares mechanics.effector.pos to the SimpleReaches "
            "same-coordinate target sequence over every transition, using the configured "
            "C&S Eq. 15 power-law discount when requested."
        ),
        "effector_hold_pos_schedule": str(hps.loss.effector_hold_pos_schedule),
        "position_powerlaw_power": float(hps.loss.position_powerlaw_power),
        "movement_ramp_shape": str(hps.loss.movement_ramp_shape),
        "movement_ramp_duration_steps": int(hps.loss.movement_ramp_duration_steps),
        "movement_ramp_power": float(hps.loss.movement_ramp_power),
        "time_indexing": cs_time_indexing,
    }


def _fidelity_status(hps: TreeNamespace) -> dict[str, Any]:
    nn_hidden = float(hps.loss.weights.nn_hidden)
    no_extra_regularizer = nn_hidden == 0.0
    plant_backend = str(getattr(hps.model, "plant_backend", CS_LSS_PLANT_BACKEND))
    exact_lss = plant_backend == CS_LSS_PLANT_BACKEND
    objective = str(getattr(hps.loss, "objective", CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE))
    if objective == CS_FULL_ANALYTICAL_QRF_LOSS_OBJECTIVE:
        return {
            "objective": "cs_fidelity_stochastic_rollout",
            "loss_objective": objective,
            "exact_fidelity": False,
            "exact_objective_terms": exact_lss,
            "exact_objective_terms_scope": (
                "true for the implemented training scalar when plant_backend='cs_lss': "
                "the loss evaluates canonical C&S delay-augmented Q_t, R_t, and Q_f "
                "on the exposed LinearStateSpace state and command history"
            ),
            "objective_fidelity": {
                "implemented_terms": [
                    "delay_augmented_state_running_Q_t",
                    "command_running_R_t",
                    "delay_augmented_terminal_Q_f",
                ],
                "omitted_terms": [] if exact_lss else ["cs_lss_state_unavailable"],
                "extra_terms": [],
                "selection_policy": (
                    "rollout validation loss uses the same full analytical Q/R/Q_f "
                    "training scalar; analytical action and I/O metrics remain audit-only"
                ),
            },
            "exact_stochastic_rollout": False,
            "exact_stochastic_noise_sources": exact_lss,
            "exact_plant_matrices": exact_lss,
            "plant_backend": plant_backend,
            "temporary_stochastic_bridge": (
                "temporary RLRMP LSS wrapper implements sensory Channel, additive and "
                "signal-dependent motor Channel, and sampled physical-process mechanics.epsilon; "
                "future Feedbax acausal/ODE plant support should subsume this wrapper"
                if exact_lss
                else None
            ),
            "stochastic_preset": str(hps.model.stochastic_preset),
            "stochastic_projection": (
                "Feedbax GRU rollout uses C&S-shaped sensory, command, signal-dependent, "
                "and plant/load force noise channels without feeding the 48D "
                "delay-augmented analytical state to the GRU."
            ),
            "regularized_pair": False,
            "regularizer": "none",
            "nn_hidden": nn_hidden,
            "certificate_lens": "input_output_map_certificate",
            "same_coordinate_gain_certificate": False,
        }
    if objective == CS_PARTIAL_NET_FORCE_FILTER_LOSS_OBJECTIVE:
        extra_terms = (
            []
            if no_extra_regularizer
            else [
                {
                    "term": "nn_hidden",
                    "scale": nn_hidden,
                    "status": "auxiliary_regularizer_not_in_analytical_objective",
                }
            ]
        )
        return {
            "objective": "cs_fidelity_stochastic_rollout",
            "loss_objective": objective,
            "exact_fidelity": False,
            "exact_objective_terms": False,
            "exact_objective_terms_scope": (
                "ablation only: old partial position/velocity terms are kept, "
                "control is moved to intended net.output, and running force/filter "
                "state cost is added; this is not the full Q/R/Q_f objective"
            ),
            "objective_fidelity": {
                "implemented_terms": [
                    "running_position_cs_eq15_power6",
                    "terminal_position",
                    "running_velocity_cs_eq15_power6",
                    "terminal_velocity",
                    "intended_command_quadratic_net_output",
                    "running_force_filter_state_cost",
                ],
                "omitted_terms": [
                    {
                        "term": "disturbance_integrator_state_cost",
                        "analytical_role": (
                            "unit-weight disturbance-integrator state cost in the C&S 8D schedule"
                        ),
                        "status": "intentionally_omitted_for_force_filter_ablation",
                    },
                    {
                        "term": "terminal_force_filter_and_integrator_Q_f",
                        "analytical_role": "terminal full-state Q_f costs",
                        "status": "not_synthesized_in_partial_ablation",
                    },
                ],
                "extra_terms": extra_terms,
                "selection_policy": (
                    "rollout validation loss only; analytical action and I/O metrics are audit-only"
                ),
            },
            "exact_stochastic_rollout": False,
            "exact_stochastic_noise_sources": exact_lss,
            "exact_plant_matrices": exact_lss,
            "plant_backend": plant_backend,
            "temporary_stochastic_bridge": (
                "temporary RLRMP LSS wrapper implements sensory Channel, additive and "
                "signal-dependent motor Channel, and sampled physical-process mechanics.epsilon; "
                "future Feedbax acausal/ODE plant support should subsume this wrapper"
                if exact_lss
                else None
            ),
            "stochastic_preset": str(hps.model.stochastic_preset),
            "stochastic_projection": (
                "Feedbax GRU rollout uses C&S-shaped sensory, command, signal-dependent, "
                "and plant/load force noise channels without feeding the 48D "
                "delay-augmented analytical state to the GRU."
            ),
            "regularized_pair": not no_extra_regularizer,
            "regularizer": "none" if no_extra_regularizer else "nn_hidden",
            "nn_hidden": nn_hidden,
            "certificate_lens": "input_output_map_certificate",
            "same_coordinate_gain_certificate": False,
            "analytical_delay_augmented_state_input": False,
        }
    omitted_terms = [
        {
            "term": "force_filter_state_cost",
            "analytical_role": "unit-weight force/filter state cost in the C&S 8D schedule",
            "status": "not_synthesized_in_feedbax_gru_loss",
        },
        {
            "term": "disturbance_integrator_state_cost",
            "analytical_role": (
                "unit-weight disturbance-integrator state cost in the C&S 8D schedule"
            ),
            "status": "not_synthesized_in_feedbax_gru_loss",
        },
    ]
    extra_terms = (
        []
        if no_extra_regularizer
        else [
            {
                "term": "nn_hidden",
                "scale": nn_hidden,
                "status": "auxiliary_regularizer_not_in_analytical_objective",
            }
        ]
    )
    return {
        "objective": "cs_fidelity_stochastic_rollout",
        "loss_objective": objective,
        "exact_fidelity": False,
        "exact_objective_terms": False,
        "exact_objective_terms_scope": (
            "false because force/filter-state and disturbance-integrator state costs from "
            "the analytical C&S schedule are omitted from the current Feedbax GRU loss contract"
        ),
        "objective_fidelity": {
            "implemented_terms": [
                "running_position_cs_eq15_power6",
                "terminal_position",
                "running_velocity_cs_eq15_power6",
                "terminal_velocity",
                "command_quadratic_nn_output",
            ],
            "omitted_terms": omitted_terms,
            "extra_terms": extra_terms,
            "selection_policy": (
                "rollout validation loss only; analytical action and I/O metrics are audit-only"
            ),
        },
        "exact_stochastic_rollout": False,
        "exact_stochastic_noise_sources": exact_lss,
        "exact_plant_matrices": exact_lss,
        "plant_backend": plant_backend,
        "temporary_stochastic_bridge": (
            "temporary RLRMP LSS wrapper implements sensory Channel, additive and "
            "signal-dependent motor Channel, and sampled physical-process mechanics.epsilon; "
            "future Feedbax acausal/ODE plant support should subsume this wrapper"
            if exact_lss
            else None
        ),
        "stochastic_preset": str(hps.model.stochastic_preset),
        "stochastic_projection": (
            "Feedbax GRU rollout uses C&S-shaped sensory, command, signal-dependent, "
            "and plant/load force noise channels without feeding the 48D "
            "delay-augmented analytical state to the GRU."
        ),
        "regularized_pair": not no_extra_regularizer,
        "regularizer": "none" if no_extra_regularizer else "nn_hidden",
        "nn_hidden": nn_hidden,
        "certificate_lens": "input_output_map_certificate",
        "same_coordinate_gain_certificate": False,
        "analytical_delay_augmented_state_input": False,
    }


def _get_git_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for cmd, key in [
        (["git", "rev-parse", "HEAD"], "rlrmp_commit"),
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "rlrmp_branch"),
    ]:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=REPO_ROOT,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            metadata[key] = result.stdout.strip()
    return metadata


def _get_runtime_metadata() -> dict[str, Any]:
    metadata = {"jax_version": jax.__version__}
    try:
        import feedbax

        metadata["feedbax_version"] = getattr(feedbax, "__version__", "unknown")
    except ImportError:
        metadata["feedbax_version"] = "unavailable"
    return metadata


def _get_dependency_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "rlrmp": "local",
        "jax": jax.__version__,
    }
    for package in ("feedbax", "jax_cookbook", "modal"):
        try:
            module = __import__(package)
            metadata[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            metadata[package] = "unavailable"
    return metadata


def _get_gpu_metadata() -> dict[str, Any]:
    try:
        devices = jax.devices()
        return {
            "device_kinds": [device.device_kind for device in devices],
            "device_count": len(devices),
        }
    except Exception as exc:
        return {
            "device_kinds": None,
            "device_count": 0,
            "error": str(exc),
        }


def _json_dumps(payload: dict[str, Any]) -> str:
    return compact_json_dumps(payload)


def compact_run_spec_if_needed(
    payload: dict[str, Any],
    *,
    requested: bool,
) -> dict[str, Any]:
    """Return a compact envelope only when explicit authoring requests it.

    ``rlrmp_run_spec`` is the full, stamped RLRMP payload used for durable
    manifest custody. The compact envelope keeps it authoritative rather than
    duplicating its graph-heavy and governed-bank fields at the recipe root.
    The root mirrors only the task identity the Stage-2 fork gate needs before
    hydration, plus the immutable Feedbax training spec and graph pointers.
    """

    if not requested:
        return payload

    extension = payload.get(RLRMP_RUN_SPEC_PAYLOAD_KEY)
    if not isinstance(extension, dict):
        raise TypeError(f"{RLRMP_RUN_SPEC_PAYLOAD_KEY} must be an object before compaction")
    feedbax_training_spec = payload.get(FEEDBAX_TRAINING_RUN_SPEC_KEY)
    if not isinstance(feedbax_training_spec, dict):
        raise TypeError(
            f"{FEEDBAX_TRAINING_RUN_SPEC_KEY} must be an object before compaction"
        )
    training_distribution = payload.get("training_distribution")
    if not isinstance(training_distribution, dict):
        raise TypeError("training_distribution must be an object before compaction")
    if "perturbation_training" not in training_distribution:
        raise ValueError("training_distribution.perturbation_training is required before compaction")
    game_card = payload.get("game_card")
    if not isinstance(game_card, dict):
        raise TypeError("game_card must be an object before compaction")
    feedbax_graph = payload.get("feedbax_graph")
    if not isinstance(feedbax_graph, dict):
        raise TypeError("feedbax_graph must be an object before compaction")

    compact_payload = {
        COMPACT_RUN_SPEC_KEY: True,
        RLRMP_RUN_SPEC_PAYLOAD_KEY: extension,
        FEEDBAX_TRAINING_RUN_SPEC_KEY: feedbax_training_spec,
        "game_card": game_card,
        "training_distribution": {
            "perturbation_training": training_distribution["perturbation_training"],
        },
        "feedbax_graph": feedbax_graph,
    }
    compact_size = len(_json_dumps(compact_payload).encode("utf-8"))
    if compact_size > MAX_TRACKED_RUN_SPEC_BYTES:
        raise ValueError(
            "compact composed run spec remains above the tracked JSON budget: "
            f"{compact_size} > {MAX_TRACKED_RUN_SPEC_BYTES} bytes"
        )
    return compact_payload


__all__ = [
    "COMPACT_RUN_SPEC_KEY",
    "MAX_TRACKED_RUN_SPEC_BYTES",
    "TRAINING_DIAGNOSTICS_MANIFEST",
    "TRAINING_DIAGNOSTICS_NPZ",
    "_adversarial_phase",
    "_broad_epsilon_pgd_finite_policy_inputs",
    "_broad_epsilon_pgd_mechanism",
    "_broad_epsilon_pgd_training_enabled",
    "_broad_epsilon_training_enabled",
    "_checkpoint_metadata",
    "_config_default",
    "_controller_feedback_basis",
    "_controller_feedback_descriptors",
    "_controller_feedback_dim",
    "_delayed_pre_go_auxiliary_terms_metadata",
    "_delayed_reach_enabled",
    "_fidelity_status",
    "_get_dependency_metadata",
    "_get_git_metadata",
    "_get_gpu_metadata",
    "_get_runtime_metadata",
    "_initial_hidden_encoder_enabled",
    "_initial_hidden_encoder_metadata",
    "_json_dumps",
    "_loss_spec",
    "_nominal_only",
    "_optimizer_metadata",
    "_perturbation_training_enabled",
    "_policy_adversary_policy_class",
    "_policy_adversary_training_enabled",
    "_run_mode",
    "_run_spec_path_for_write",
    "_should_write_graph_spec",
    "_sisu_conditioned_pgd_input_key",
    "_stochastic_runtime_contract",
    "_target_relative_multitarget_enabled",
    "_task_spec",
    "_training_diagnostics_metadata",
    "_training_distribution_metadata",
    "_training_mode",
    "_validation_bins_metadata",
    "_write_graph_bundle_for_backend",
    "build_game_card_provenance",
    "build_graph_bundle",
    "build_loss_game_card_provenance",
    "build_model_structure_summary",
    "build_run_spec",
    "build_training_run_graph_spec",
    "compact_run_spec_if_needed",
    "derive_spec_dir",
    "derive_spec_path",
    "write_run_spec",
]
