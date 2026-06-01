"""Stochastic C&S-fidelity GRU run-spec construction and training.

This module prepares nominal, hold-free C&S-aligned GRU runs for issue
``30f2313``. The default CLI mode writes only the lightweight run spec and
GraphSpec; ``--full-train`` performs the explicitly launched training path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import optax
from feedbax._io import save as fbx_save
from feedbax.train import filter_spec_leaves, get_model_parameters
from feedbax.training.train import TaskTrainer, make_delayed_cosine_schedule
from feedbax.types import TreeNamespace, dict_to_namespace

from rlrmp.analysis.cs_game_card import (
    INIT_POS,
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    OUTPUT_FEEDBACK_GAMMA_SELECTION_ISSUE_ID,
    TARGET_POS,
    build_canonical_game,
)
from rlrmp.analysis.cs_released_simulation import (
    DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG,
    default_cs_noise_covariances,
)
from rlrmp.analysis.output_feedback import OutputFeedbackConfig
from rlrmp.feedbax_graph import (
    EXECUTION_BACKEND,
    PLANT_INTERVENOR_LABEL,
    RLRMPFeedbaxGraphBundle,
    build_point_mass_sensorimotor_graph_spec,
    write_graph_spec_bundle,
)
from rlrmp.modules.training.part2 import setup_task_model_pair
from rlrmp.paths import REPO_ROOT, mkdir_p
from rlrmp.run_specs import validate_nominal_gru_run_spec
from rlrmp.stochastic_runtime import (
    graphspec_noise_contract,
    stochastic_runtime_config_from_model,
)

ISSUE_ID = "30f2313"
SCHEMA_VERSION = "rlrmp.cs_stochastic_gru.v1"
DEFAULT_EXPERIMENT = ISSUE_ID
DEFAULT_RUN = "cs_stochastic_gru__no_hidden_penalty"
DEFAULT_OUTPUT_DIR = f"_artifacts/{DEFAULT_EXPERIMENT}/runs/{DEFAULT_RUN}"
DEFAULT_STOCHASTIC_PRESET = "cs2019-rollout"
DEFAULT_CHECKPOINT_INTERVAL_BATCHES = 500
CS_STAGE_COUNT = 60
CS_FEEDBAX_N_STEPS = CS_STAGE_COUNT + 1
CS_POSITION_SCALE = 1e6
CS_VELOCITY_SCALE = 1e5
CS_CONTROL_SCALE = 1.0
CS_REGULARIZED_NN_HIDDEN = 1e-5
VolumeCommit = Callable[[], None]


@dataclass(frozen=True)
class TrainingState:
    """Serializable state needed to resume the chunked C&S GRU training loop."""

    model: Any
    optimizer_state: Any
    completed_batches: int
    key: Any
    history: Any | None


@dataclass(frozen=True)
class StochasticPreset:
    """Named stochastic rollout preset for Feedbax-backed GRU runs."""

    name: str
    sensory_noise_std: float
    additive_motor_noise_std: float
    signal_dependent_motor_noise_std: float
    plant_process_force_noise_std: float
    source_contract: dict[str, Any]
    projection_notes: dict[str, str]

    def hps_fields(self) -> dict[str, float]:
        """Return model-hyperparameter fields controlled by the preset."""

        return {
            "sensory_noise_std": self.sensory_noise_std,
            "additive_motor_noise_std": self.additive_motor_noise_std,
            "signal_dependent_motor_noise_std": self.signal_dependent_motor_noise_std,
            "plant_process_force_noise_std": self.plant_process_force_noise_std,
        }

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable preset metadata."""

        return {
            "name": self.name,
            **self.hps_fields(),
            "source_contract": self.source_contract,
            "projection_notes": self.projection_notes,
        }


def stochastic_preset(name: str) -> StochasticPreset:
    """Return a named stochastic preset for the dedicated C&S GRU runner."""

    if name != DEFAULT_STOCHASTIC_PRESET:
        raise ValueError(
            f"Unknown stochastic preset {name!r}; expected {DEFAULT_STOCHASTIC_PRESET!r}"
        )
    plant, _schedule = build_canonical_game()
    output_config = OutputFeedbackConfig()
    noise_config = DEFAULT_CS_RELEASED_STOCHASTIC_NOISE_CONFIG
    covariances = default_cs_noise_covariances(
        plant,
        output_config,
        motor_covariance_scale=noise_config.motor_covariance_scale,
        process_covariance_scale=noise_config.process_covariance_scale,
        signal_dependent_scale=noise_config.signal_dependent_scale,
    )
    sensory_diag = jnp.diag(covariances.sensory)
    if not bool(jnp.allclose(sensory_diag, sensory_diag[0])):
        raise ValueError("C&S sensory covariance projection expects isotropic diagonal covariance")
    return StochasticPreset(
        name=name,
        sensory_noise_std=float(jnp.sqrt(sensory_diag[0])),
        additive_motor_noise_std=math.sqrt(noise_config.motor_covariance_scale),
        signal_dependent_motor_noise_std=noise_config.signal_dependent_scale,
        plant_process_force_noise_std=math.sqrt(
            output_config.process_covariance_scale * noise_config.process_covariance_scale
        ),
        source_contract={
            **noise_config.summary(),
            "output_feedback_process_covariance_scale": output_config.process_covariance_scale,
            "sensory_noise_scale": output_config.sensory_noise_scale,
            "sensory_covariance_diag": [float(x) for x in sensory_diag.tolist()],
            "motor_covariance_shape": list(covariances.motor.shape),
            "process_covariance_shape": list(covariances.process.shape),
            "signal_dependent_state_shape": list(covariances.signal_dependent_state.shape),
        },
        projection_notes={
            "sensory": (
                "Use the C&S sensory covariance diagonal standard deviation on the "
                "Feedbax delayed pos/vel feedback channel."
            ),
            "additive_motor": (
                "Project C&S input-image motor covariance to command-channel "
                "additive noise with std sqrt(motor_covariance_scale)."
            ),
            "signal_dependent_motor": (
                "Use the C&S Csdn scale as Feedbax pre-force-filter "
                "multiplicative command noise."
            ),
            "plant_process": (
                "Project C&S process/load covariance to independent force noise "
                "immediately upstream of mechanics, after the force filter."
            ),
            "state_diffusion": "No arbitrary full-state diffusion is used in Feedbax GRU rollout.",
        },
    )


def derive_spec_dir(output_dir: Path) -> Path:
    """Return the tracked spec directory corresponding to an artifact directory."""

    out = Path(output_dir).resolve()
    artifact_root = (REPO_ROOT / "_artifacts").resolve()
    spec_root = (REPO_ROOT / "results").resolve()
    try:
        rel = out.relative_to(artifact_root)
        return spec_root / rel
    except ValueError:
        return out.parent / f"{out.name}_spec"


def build_hps(args: argparse.Namespace) -> TreeNamespace:
    """Build nominal C&S-aligned GRU hyperparameters from CLI arguments."""

    args = _apply_smoke_overrides(args)
    plant, schedule = build_canonical_game()
    preset = stochastic_preset(args.stochastic_preset)
    if int(schedule.T) != CS_STAGE_COUNT:
        raise ValueError(f"Expected C&S stage count {CS_STAGE_COUNT}, got {schedule.T}")
    nn_hidden = CS_REGULARIZED_NN_HIDDEN if args.regularized_fidelity else 0.0
    n_input_readout = int(args.hidden_size) - (
        int(args.n_input_only) + int(args.n_readout_only) + int(args.n_recurrent_only)
    )
    if n_input_readout < 0:
        raise ValueError(
            "Population subgroups exceed hidden_size: "
            f"hidden_size={args.hidden_size}, "
            f"n_input_only={args.n_input_only}, "
            f"n_readout_only={args.n_readout_only}, "
            f"n_recurrent_only={args.n_recurrent_only}"
        )
    hps_dict = {
        "method": "nominal-cs-gru",
        "dt": float(plant.dt),
        "n_batches_condition": int(args.n_train_batches),
        "n_batches_baseline": 0,
        "batch_size": int(args.batch_size),
        "learning_rate_0": float(args.controller_lr),
        "n_scaleup_batches": 0,
        "constant_lr_iterations": 0,
        "cosine_annealing_alpha": 1.0,
        "weight_decay": 0.0,
        "state_reset_iterations": [],
        "intervention_scaleup_batches": [0, 0],
        "model": {
            "n_replicates": int(args.n_replicates),
            "effector_mass": 1.0,
            "hidden_size": int(args.hidden_size),
            "feedback_delay_steps": 5,
            "feedback_noise_std": 0.0,
            "motor_noise_std": 0.0,
            **preset.hps_fields(),
            "stochastic_preset": preset.name,
            "damping": 0.1,
            "tau_rise": 0.066,
            "population_structure": {
                "n_input_only": int(args.n_input_only),
                "n_readout_only": int(args.n_readout_only),
                "n_recurrent_only": int(args.n_recurrent_only),
                "n_input_readout": n_input_readout,
            },
        },
        "task": {
            "type": "fixed_simple_reach",
            "n_steps": CS_FEEDBAX_N_STEPS,
            "workspace": [[-0.02, -0.02], [float(TARGET_POS[0]) + 0.02, 0.02]],
            "fixed_init_pos": [float(x) for x in INIT_POS.tolist()],
            "fixed_target_pos": [float(x) for x in TARGET_POS.tolist()],
            "eval_grid_n": 1,
            "eval_n_directions": 1,
            "eval_reach_length": float(TARGET_POS[0]),
            "epoch_len_ranges": [[0, 1], [CS_STAGE_COUNT, CS_STAGE_COUNT + 1]],
            "target_on_epochs": [0],
            "hold_epochs": [],
            "move_epochs": [0],
            "p_catch_trial": 0.0,
        },
        "pert": {
            "type": "gusts",
            "std": 0.0,
            "duration_mean": 0,
            "n_expected": 0,
        },
        "loss": {
            "weights": {
                "goal_hit_in_window": 0.0,
                "effector_pos": 0.0,
                "effector_pos_running": float(args.effector_pos_running),
                "effector_vel_running": float(args.effector_vel_running),
                "effector_terminal_pos": float(args.effector_terminal_pos),
                "effector_terminal_vel": float(args.effector_terminal_vel),
                "effector_pos_mid": 0.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": 0.0,
                "effector_vel_late": 0.0,
                "effector_hold_pos": 0.0,
                "effector_hold_vel": 0.0,
                "effector_final_vel": float(args.effector_final_vel),
                "nn_output": float(args.nn_output),
                "nn_hidden": nn_hidden,
                "nn_hidden_derivative": 0.0,
                "nn_output_jerk": float(args.nn_output_jerk),
                "nn_output_pre_go": 0.0,
                "nn_hidden_derivative_pre_go": 0.0,
            },
            "effector_pos_late": {
                "start_step_after_go": int(schedule.T),
                "final_scale_factor": 1.0,
            },
            "effector_vel_late": {
                "start_step_after_go": int(schedule.T),
                "final_scale_factor": 1.0,
            },
            "effector_pos_running_schedule": "cs_eq15_power6",
            "effector_hold_pos_schedule": "disabled",
            "position_powerlaw_power": 6.0,
            "movement_ramp_shape": "none",
            "movement_ramp_duration_steps": 0,
            "movement_ramp_power": 1.0,
        },
        "loss_update": {
            "enabled": False,
            "target_ratio": 0.0,
            "alpha": 0.0,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_vel_running"],
            "start_iteration": 0,
        },
        "where": {
            0: ["nodes.net.hidden", "nodes.net.readout"],
        },
        "hidden_type": eqx.nn.GRUCell,
        "sisu_gating": "additive",
    }
    return dict_to_namespace(hps_dict, to_type=TreeNamespace)


def build_game_card_provenance() -> dict[str, Any]:
    """Return lightweight C&S game-card provenance without solving Riccati systems."""

    plant, schedule = build_canonical_game()
    target = [float(x) for x in TARGET_POS.tolist()]
    init = [float(x) for x in INIT_POS.tolist()]
    return {
        "source_module": "rlrmp.analysis.cs_game_card",
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


def build_model_structure_summary(hps: TreeNamespace) -> dict[str, Any]:
    """Return the model/training summary embedded in ``run.json``."""

    pop = hps.model.population_structure
    stochastic_runtime = graphspec_noise_contract(stochastic_runtime_config_from_model(hps.model))
    return {
        "controller_kind": "gru",
        "hidden_size": int(hps.model.hidden_size),
        "n_replicates": int(hps.model.n_replicates),
        "trainable": ["nodes.net.hidden", "nodes.net.readout"],
        "population_structure": {
            "n_input_only": int(pop.n_input_only),
            "n_readout_only": int(pop.n_readout_only),
            "n_recurrent_only": int(pop.n_recurrent_only),
            "n_input_readout": int(pop.n_input_readout),
        },
        "feedback": {
            "delay_steps": int(hps.model.feedback_delay_steps),
            "noise_std": stochastic_runtime["sensory_noise_std"],
            "noise_role": "sensory_feedback",
        },
        "efferent": {
            "additive_motor_noise_std": stochastic_runtime["additive_motor_noise_std"],
            "signal_dependent_motor_noise_std": (
                stochastic_runtime["signal_dependent_motor_noise_std"]
            ),
            "noise_timing": "pre_force_filter",
            "force_filter_tau_s": float(hps.model.tau_rise),
        },
        "plant_process": {
            "force_noise_std": stochastic_runtime["plant_process_force_noise_std"],
            "noise_timing": "post_force_filter_pre_mechanics",
            "state_diffusion": "not_used",
        },
        "stochastic_runtime": stochastic_runtime,
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "nominal_only": True,
        "adversarial_phase": "none",
        "certificate_lens": "input_output_map_certificate",
        "certificate_coordinate_claim": "not_same_coordinate_gain",
        "analytical_delay_augmented_state_input": False,
        "certificate_claim": (
            "I/O map certificate framing only; the output-feedback GRU is not fed "
            "the 48D delay-augmented analytical state and is not claimed to share "
            "same-coordinate gains with the analytical controller."
        ),
    }


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
        "trainable": ["nodes.net.hidden", "nodes.net.readout"],
        "method": str(hps.method),
        "nominal_only": True,
        "adversarial_phase": "none",
        "certificate_lens": "input_output_map_certificate",
        "analytical_delay_augmented_state_input": False,
        "stochastic_runtime": graphspec_noise_contract(
            stochastic_runtime_config_from_model(hps.model)
        ),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "execution_backend": EXECUTION_BACKEND,
        "component_policy": {
            "rlrmp_component_types": [
                "RLRMPFeedbackChannels",
                "RLRMPSimpleStagedNetwork",
                "FixedField",
            ],
            "nominal_intervention_policy": (
                f"{PLANT_INTERVENOR_LABEL} is present only as an inactive legacy "
                "GraphSpec compatibility component; no robust/minimax adversary is scheduled."
            ),
        },
        "legacy_loader": {
            "setup_function": "rlrmp.modules.training.part2.setup_task_model_pair",
            "checkpoint_format": "feedbax._io.save/load_with_hyperparameters",
        },
        "task_spec": task_spec,
        "loss_spec": loss_spec,
        "training_spec": training_spec,
        "game_card_provenance": build_game_card_provenance(),
        "model_structure": build_model_structure_summary(hps),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "stochastic_runtime": graphspec_noise_contract(
            stochastic_runtime_config_from_model(hps.model)
        ),
    }
    return RLRMPFeedbaxGraphBundle(
        graph_spec=graph_spec,
        task_spec=task_spec,
        loss_spec=loss_spec,
        training_spec=training_spec,
        manifest=manifest,
    )


def build_run_spec(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    spec_dir: Path,
    graph_bundle: RLRMPFeedbaxGraphBundle,
) -> dict[str, Any]:
    """Build the JSON payload for ``run.json``."""

    hps = build_hps(args)
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE_ID,
        "training_script": "scripts/train_cs_nominal_gru.py",
        "mode": _run_mode(args),
        "artifact_output_dir": str(output_dir),
        "spec_dir": str(spec_dir),
        "nominal_only": True,
        "adversarial_phase": "none",
        "modal_launch": "not_requested",
        "full_training_launch": "requested" if args.full_train else "not_requested",
        "seed": int(args.seed),
        "n_train_batches": int(args.n_train_batches),
        "batch_size": int(args.batch_size),
        "controller_lr": float(args.controller_lr),
        "checkpointing": _checkpoint_metadata(args, output_dir),
        "fidelity_status": _fidelity_status(hps),
        "stochastic_preset": stochastic_preset(str(hps.model.stochastic_preset)).summary(),
        "game_card": build_game_card_provenance(),
        "model_summary": build_model_structure_summary(hps),
        "task_timing": graph_bundle.task_spec,
        "loss_summary": graph_bundle.loss_spec,
        "training_summary": {
            **graph_bundle.training_spec,
            "training_mode": "nominal",
            "n_train_batches": int(args.n_train_batches),
            "n_adversary_batches": 0,
        },
        "feedbax_graph": graph_bundle.to_run_metadata(),
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

    args = _apply_smoke_overrides(args)
    output_dir = Path(args.output_dir)
    spec_dir = Path(args.spec_dir) if args.spec_dir is not None else derive_spec_dir(output_dir)
    hps = build_hps(args)
    graph_bundle = build_graph_bundle(hps)
    payload = build_run_spec(
        args,
        output_dir=output_dir,
        spec_dir=spec_dir,
        graph_bundle=graph_bundle,
    )

    if args.dry_run:
        return {
            "run_spec": payload,
            "would_write": [
                str(spec_dir / "run.json"),
                str(spec_dir / "model.graph.json"),
                str(spec_dir / "model.graph.manifest.json"),
            ],
        }

    mkdir_p(spec_dir)
    graph_path = write_graph_spec_bundle(graph_bundle, spec_dir)
    payload["feedbax_graph"] = graph_bundle.to_run_metadata(graph_spec_path=graph_path.name)
    validate_nominal_gru_run_spec(payload, spec_dir=spec_dir)
    run_path = spec_dir / "run.json"
    run_path.write_text(_json_dumps(payload), encoding="utf-8")
    return {
        "run_spec_path": str(run_path),
        "graph_spec_path": str(graph_path),
        "graph_manifest_path": str(spec_dir / "model.graph.manifest.json"),
    }


def run_full_training(
    args: argparse.Namespace,
    *,
    volume_commit: VolumeCommit | None = None,
) -> dict[str, Any]:
    """Run chunked stochastic C&S GRU training with durable checkpoints.

    Feedbax's trainer can accept an optimizer state, but its checkpoint restore
    path is model-centric. This wrapper owns the resume contract explicitly:
    checkpoints store model leaves, optimizer state, completed batch count,
    PRNG state, run/config metadata, and training history snapshots.
    """

    args = _apply_smoke_overrides(args)
    if int(args.n_train_batches) < 1:
        raise ValueError("--n-train-batches must be positive for --full-train")
    if int(args.checkpoint_interval_batches) < 1:
        raise ValueError("--checkpoint-interval-batches must be positive")

    spec_result = write_run_spec(args)
    output_dir = mkdir_p(Path(args.output_dir))
    run_spec_path = Path(spec_result["run_spec_path"])
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))

    hps = build_hps(args)
    key_init, key_train = jr.split(jr.PRNGKey(int(args.seed)), 2)
    pair = setup_task_model_pair(hps, key=key_init)
    trainer = _build_trainer(hps)
    where_train = _where_train()
    template_state = _initial_training_state(
        model=pair.model,
        trainer=trainer,
        where_train=where_train[0],
        key=key_train,
    )
    checkpoint_root = output_dir / "checkpoints"
    state = (
        load_latest_checkpoint(
            checkpoint_root,
            model_template=pair.model,
            optimizer_state_template=template_state.optimizer_state,
            history_template=None,
        )
        if args.resume and latest_checkpoint_path(checkpoint_root).exists()
        else template_state
    )

    chunks: list[dict[str, float | int | str]] = []
    training_started = time.perf_counter()
    while state.completed_batches < int(args.n_train_batches):
        remaining = int(args.n_train_batches) - state.completed_batches
        chunk_batches = min(int(args.checkpoint_interval_batches), remaining)
        key_chunk, key_next = jr.split(state.key, 2)
        chunk_started = time.perf_counter()
        model, history_chunk, optimizer_state = trainer(
            pair.task,
            state.model,
            n_batches=chunk_batches,
            # Keep each Feedbax trainer call local to the chunk. The optimizer
            # state carries schedule progress across chunks; passing a nonzero
            # idx_start here would make Feedbax index past the chunk-local PRNG
            # key array.
            idx_start=0,
            opt_state=state.optimizer_state,
            key=key_chunk,
            ensembled=True,
            loss_func=pair.task.loss_func,
            where_train=where_train,
            batch_size=int(hps.batch_size),
            log_step=max(1, int(args.log_step)),
            disable_progress=bool(args.disable_progress),
            verbose_progress=not bool(args.quiet_progress),
        )
        chunk_duration_seconds = time.perf_counter() - chunk_started
        history = _append_history(state.history, history_chunk)
        completed = state.completed_batches + chunk_batches
        history_chunk_path = output_dir / "history_chunks" / f"history_{completed:07d}.eqx"
        history_chunk_path.parent.mkdir(parents=True, exist_ok=True)
        fbx_save(history_chunk_path, history_chunk)
        state = TrainingState(
            model=model,
            optimizer_state=optimizer_state,
            completed_batches=completed,
            key=key_next,
            history=history,
        )
        checkpoint_path = save_training_checkpoint(
            checkpoint_root,
            state,
            args=args,
            run_spec=run_spec,
        )
        _commit_volume(volume_commit)
        chunks.append(
            {
                "completed_batches": completed,
                "checkpoint": str(checkpoint_path),
                "history_chunk": str(history_chunk_path),
                "chunk_batches": chunk_batches,
                "duration_seconds": chunk_duration_seconds,
                "batches_per_second": chunk_batches / chunk_duration_seconds,
            }
        )
    training_duration_seconds = time.perf_counter() - training_started

    final_model_path = output_dir / "trained_model.eqx"
    final_history_path = output_dir / "training_history.eqx"
    final_summary_path = output_dir / "training_summary.json"
    fbx_save(final_model_path, state.model, hyperparameters=run_spec)
    if state.history is not None:
        fbx_save(final_history_path, state.history)
    final_summary = {
        "schema_version": f"{SCHEMA_VERSION}.training.v1",
        "issue": ISSUE_ID,
        "completed_batches": state.completed_batches,
        "n_train_batches": int(args.n_train_batches),
        "training_duration_seconds": training_duration_seconds,
        "training_batches_per_second": (
            state.completed_batches / training_duration_seconds
            if training_duration_seconds > 0
            else None
        ),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "final_model_path": str(final_model_path),
        "training_history_path": str(final_history_path),
        "run_spec_path": str(run_spec_path),
        "graph_spec_path": spec_result["graph_spec_path"],
        "chunks": chunks,
    }
    _atomic_write_json(final_summary_path, final_summary)
    _commit_volume(volume_commit)
    return {
        **spec_result,
        "final_model_path": str(final_model_path),
        "training_history_path": str(final_history_path),
        "training_summary_path": str(final_summary_path),
        "latest_checkpoint": str(latest_checkpoint_path(checkpoint_root)),
        "completed_batches": state.completed_batches,
    }


def save_training_checkpoint(
    checkpoint_root: Path,
    state: TrainingState,
    *,
    args: argparse.Namespace,
    run_spec: dict[str, Any],
) -> Path:
    """Write a numbered checkpoint and atomically repoint ``checkpoint_latest``."""

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"checkpoint_{state.completed_batches:07d}"
    target = checkpoint_root / checkpoint_name
    tmp = checkpoint_root / f".{checkpoint_name}.tmp"
    if tmp.exists():
        _remove_tree(tmp)
    tmp.mkdir(parents=True)

    eqx.tree_serialise_leaves(tmp / "model.eqx", state.model)
    eqx.tree_serialise_leaves(tmp / "optimizer_state.eqx", state.optimizer_state)
    if state.history is not None:
        fbx_save(tmp / "history.eqx", state.history)
    metadata = {
        "schema_version": f"{SCHEMA_VERSION}.checkpoint.v1",
        "issue": ISSUE_ID,
        "completed_batches": state.completed_batches,
        "n_train_batches": int(args.n_train_batches),
        "checkpoint_interval_batches": int(args.checkpoint_interval_batches),
        "seed": int(args.seed),
        "next_prng_key": _plain(state.key),
        "stochastic_preset": str(args.stochastic_preset),
        "run_spec": run_spec,
    }
    _atomic_write_json(tmp / "metadata.json", metadata)
    if target.exists():
        _remove_tree(target)
    os.replace(tmp, target)
    _atomic_latest_link(checkpoint_root, checkpoint_name)
    _atomic_write_json(
        checkpoint_root / "checkpoint_index.json",
        {
            "latest": checkpoint_name,
            "latest_path": str(latest_checkpoint_path(checkpoint_root)),
            "completed_batches": state.completed_batches,
        },
    )
    return target


def load_latest_checkpoint(
    checkpoint_root: Path,
    *,
    model_template: Any,
    optimizer_state_template: Any,
    history_template: Any | None = None,
) -> TrainingState:
    """Load ``checkpoint_latest`` using explicit model and optimizer templates."""

    checkpoint_path = latest_checkpoint_path(checkpoint_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint_latest found under {checkpoint_root}")
    metadata = json.loads((checkpoint_path / "metadata.json").read_text(encoding="utf-8"))
    model = eqx.tree_deserialise_leaves(checkpoint_path / "model.eqx", model_template)
    optimizer_state = eqx.tree_deserialise_leaves(
        checkpoint_path / "optimizer_state.eqx",
        optimizer_state_template,
    )
    history_path = checkpoint_path / "history.eqx"
    history = (
        eqx.tree_deserialise_leaves(history_path, history_template)
        if history_template is not None and history_path.exists()
        else None
    )
    return TrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=int(metadata["completed_batches"]),
        key=jnp.asarray(metadata["next_prng_key"], dtype=jnp.uint32),
        history=history,
    )


def latest_checkpoint_path(checkpoint_root: Path) -> Path:
    """Return the path used by the durable latest-checkpoint contract."""

    return checkpoint_root / "checkpoint_latest"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Prepare a stochastic C&S-fidelity GRU run spec.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spec-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train-batches", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--controller-lr", type=float, default=1e-2)
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=180)
    parser.add_argument(
        "--stochastic-preset",
        choices=[DEFAULT_STOCHASTIC_PRESET],
        default=DEFAULT_STOCHASTIC_PRESET,
        help=(
            "Named stochastic rollout contract. The preset fixes sensory, command, "
            "signal-dependent, and plant/load force noise and records concrete "
            "values in run.json."
        ),
    )
    parser.add_argument(
        "--target-m",
        type=float,
        default=float(TARGET_POS[0]),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--n-input-only", type=int, default=0)
    parser.add_argument("--n-readout-only", type=int, default=0)
    parser.add_argument("--n-recurrent-only", type=int, default=0)
    parser.add_argument("--effector-pos-running", type=float, default=CS_POSITION_SCALE)
    parser.add_argument("--effector-vel-running", type=float, default=CS_VELOCITY_SCALE)
    parser.add_argument("--effector-terminal-pos", type=float, default=CS_POSITION_SCALE)
    parser.add_argument("--effector-terminal-vel", type=float, default=CS_VELOCITY_SCALE)
    parser.add_argument("--effector-final-vel", type=float, default=0.0)
    parser.add_argument("--nn-output", type=float, default=CS_CONTROL_SCALE)
    parser.add_argument("--nn-output-jerk", type=float, default=0.0)
    parser.add_argument(
        "--regularized-fidelity",
        action="store_true",
        help="Mark the paired non-exact run and use nn_hidden=1e-5.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny local values; with --full-train this runs a one-batch smoke.",
    )
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint-interval-batches",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL_BATCHES,
    )
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--disable-progress", action="store_true", default=True)
    parser.add_argument("--quiet-progress", action="store_true", default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the would-write payload without creating files.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    volume_commit: VolumeCommit | None = None,
) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    result = run_full_training(args, volume_commit=volume_commit) if args.full_train else write_run_spec(args)
    print(_json_dumps(result), end="")
    return 0


def _apply_smoke_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoke:
        return args
    values = vars(args).copy()
    values.update(
        {
            "n_train_batches": 1,
            "batch_size": 2,
            "n_replicates": 1,
            "hidden_size": 4,
            "n_input_only": 0,
            "n_readout_only": 0,
            "n_recurrent_only": 0,
            "checkpoint_interval_batches": 1,
            "log_step": 1,
        }
    )
    return argparse.Namespace(**values)


def _build_trainer(hps: TreeNamespace) -> TaskTrainer:
    schedule = make_delayed_cosine_schedule(
        float(hps.learning_rate_0),
        constant_steps=int(hps.constant_lr_iterations),
        total_steps=int(hps.n_batches_condition),
        alpha=float(hps.cosine_annealing_alpha),
    )
    optimizer = optax.inject_hyperparams(
        partial(optax.adamw, weight_decay=float(hps.weight_decay))
    )(learning_rate=schedule)
    return TaskTrainer(optimizer=optimizer, checkpointing=False)


def _where_train() -> dict[int, Callable[[Any], tuple[Any, Any]]]:
    def where_train_fn(model):
        net = model.nodes["net"]
        return (net.hidden, net.readout)

    return {0: where_train_fn}


def _initial_training_state(
    *,
    model: Any,
    trainer: TaskTrainer,
    where_train: Callable[[Any], Any],
    key: Any,
) -> TrainingState:
    where_train_spec = filter_spec_leaves(model, where_train)
    model_parameters = get_model_parameters(model, where_train_spec)
    optimizer_state = eqx.filter_vmap(trainer.optimizer.init)(model_parameters)
    return TrainingState(
        model=model,
        optimizer_state=optimizer_state,
        completed_batches=0,
        key=key,
        history=None,
    )


def _append_history(history: Any | None, chunk: Any) -> Any:
    if history is None:
        return chunk
    return jt.map(_append_history_leaf, history, chunk, is_leaf=lambda x: x is None)


def _append_history_leaf(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if eqx.is_array(left) and eqx.is_array(right):
        if left.ndim == 0 or right.ndim == 0:
            return right
        return jnp.concatenate([left, right], axis=0)
    return right


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
            "metadata.json",
        ],
    }


def _run_mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry_run"
    if args.full_train:
        return "full_train"
    return "spec_write"


def _commit_volume(volume_commit: VolumeCommit | None) -> None:
    if volume_commit is not None:
        volume_commit()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(_json_dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_latest_link(checkpoint_root: Path, checkpoint_name: str) -> None:
    latest = checkpoint_root / "checkpoint_latest"
    tmp_link = checkpoint_root / ".checkpoint_latest.tmp"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(checkpoint_name, tmp_link)
    os.replace(tmp_link, latest)


def _remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def _task_spec(hps: TreeNamespace) -> dict[str, Any]:
    return {
        "type": str(hps.task.type),
        "n_steps": int(hps.task.n_steps),
        "control_cost_stages": int(hps.task.n_steps) - 1,
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
        "coordinate_contract": (
            "Feedbax SimpleReaches supplies mechanics.effector.pos targets in the same "
            "Cartesian metre coordinates as the point-mass effector state."
        ),
        "time_axis_contract": (
            "Hold-free fixed nominal task: Feedbax n_steps=61 yields exactly 60 "
            "transition/control-cost stages and one position target per transition; "
            "delayed-reach epoch masks are not used."
        ),
        "movement_window": {
            "kind": "full_simple_reach_trial",
            "start_transition": 0,
            "end_transition": int(hps.task.n_steps) - 2,
        },
        "extra_inputs": ["sisu", f"intervene:{PLANT_INTERVENOR_LABEL}"],
    }


def _loss_spec(hps: TreeNamespace) -> dict[str, Any]:
    return {
        "weights": _plain(hps.loss.weights),
        "effector_pos_late": _plain(hps.loss.effector_pos_late),
        "effector_vel_late": _plain(hps.loss.effector_vel_late),
        "effector_pos_running_schedule": str(hps.loss.effector_pos_running_schedule),
        "objective_profile": "cs_fidelity",
        "active_cs_terms": {
            "stage_position": {
                "term": "effector_pos_running",
                "scale": float(hps.loss.weights.effector_pos_running),
                "fact_t": "((t + 1) / T)^6",
            },
            "stage_velocity": {
                "term": "effector_vel_running",
                "scale": float(hps.loss.weights.effector_vel_running),
                "fact_t": "((t + 1) / T)^6",
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
    }


def _fidelity_status(hps: TreeNamespace) -> dict[str, Any]:
    nn_hidden = float(hps.loss.weights.nn_hidden)
    exact = nn_hidden == 0.0
    return {
        "objective": "cs_fidelity_stochastic_rollout",
        "exact_fidelity": False,
        "exact_objective_terms": exact,
        "exact_stochastic_rollout": False,
        "stochastic_preset": str(hps.model.stochastic_preset),
        "stochastic_projection": (
            "Feedbax GRU rollout uses C&S-shaped sensory, command, signal-dependent, "
            "and plant/load force noise channels without feeding the 48D "
            "delay-augmented analytical state to the GRU."
        ),
        "regularized_pair": not exact,
        "regularizer": "none" if exact else "nn_hidden",
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


def _plain(value: Any) -> Any:
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
