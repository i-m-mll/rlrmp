"""Objective-comparator sidecars for GRU/full-QRF analytical comparisons."""

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
from rlrmp.paths import REPO_ROOT


SCHEMA_VERSION = "rlrmp.objective_comparator_sidecar.v6"
FULL_ANALYTICAL_QRF_OBJECTIVE = "full_analytical_qrf"
FULL_QRF_TERM_NAMES = (
    "running_state_q",
    "terminal_state_q_f",
    "command_r",
    "force_filter_state",
    "disturbance_integrator_state",
)
DEFAULT_MONTE_CARLO_STATUS = {
    "status": "not_implemented",
    "lens": "same_noise_bank_monte_carlo_full_qrf",
    "reason": (
        "same-noise-bank extLQG-vs-GRU realized comparison was not materialized; "
        "the available tracked source only contains validation-selected GRU "
        "realized full-QRF scalars and the analytical extLQG expected-cost "
        "decomposition"
    ),
    "required_for_available": [
        "shared initial-condition bank",
        "shared process/sensory/motor noise draws",
        "extLQG realized full-Q/R/Q_f scorer on that bank",
        "GRU realized full-Q/R/Q_f scorer on the same bank",
    ],
}
DEFAULT_PER_TERM_STATUS = {
    "status": "not_implemented",
    "lens": "realized_full_qrf_per_term_validation",
    "terms": list(FULL_QRF_TERM_NAMES),
    "reason": (
        "validation checkpoint manifests currently expose scalar full-QRF objectives, "
        "not running-state, terminal-state, command, force/filter, and "
        "disturbance-integrator contributions"
    ),
}
DEFAULT_SHARED_ROLLOUT_STATUS = {
    "status": "not_implemented",
    "lens": "shared_rollout_full_qrf",
    "reason": "shared-rollout comparator was not supplied to the sidecar builder",
    "selection_role": "audit_only_not_used_for_checkpoint_selection",
}
STATE_COMPONENT_SLICES = {
    "position": (0, 2),
    "velocity": (2, 4),
    "force_filter": (4, 6),
    "disturbance_integrator": (6, 8),
}
_STANDARD_SPLIT_BANK_LENS_SPECS = {
    "deterministic_nominal": {"x0": (), "epsilon": ()},
    "x0_position_only": {"x0": ("position",), "epsilon": ()},
    "x0_velocity_only": {"x0": ("velocity",), "epsilon": ()},
    "x0_force_filter_only": {"x0": ("force_filter",), "epsilon": ()},
    "x0_disturbance_integrator_only": {"x0": ("disturbance_integrator",), "epsilon": ()},
    "process_epsilon_position_only": {"x0": (), "epsilon": ("position",)},
    "process_epsilon_velocity_only": {"x0": (), "epsilon": ("velocity",)},
    "process_epsilon_force_filter_only": {"x0": (), "epsilon": ("force_filter",)},
    "process_epsilon_integrator_only": {"x0": (), "epsilon": ("disturbance_integrator",)},
    "x0_position_velocity": {"x0": ("position", "velocity"), "epsilon": ()},
    "x0_plus_epsilon": {
        "x0": ("position", "velocity", "force_filter", "disturbance_integrator"),
        "epsilon": ("position", "velocity", "force_filter", "disturbance_integrator"),
    },
}
_STANDARD_SPLIT_BANK_LENSES = tuple(_STANDARD_SPLIT_BANK_LENS_SPECS)
DEFAULT_SPLIT_BANK_STATUS = {
    "status": "not_available",
    "lens": "standard_split_rollout_bank_full_qrf",
    "reason": "standard split-bank comparator was not supplied to the sidecar builder",
    "selection_role": "audit_only_not_used_for_checkpoint_selection",
    "lenses": {
        lens: {
            "status": "not_available",
            "shared_initial_state_components": list(spec["x0"]),
            "shared_process_load_epsilon_components": list(spec["epsilon"]),
        }
        for lens, spec in _STANDARD_SPLIT_BANK_LENS_SPECS.items()
    },
}


@dataclass(frozen=True)
class SharedRolloutBank:
    """Shared C&S rollout bank for extLQG-vs-GRU objective comparison."""

    bank_id: str
    seed: int
    initial_states: np.ndarray
    process_epsilon: np.ndarray
    initial_covariance: float

    @property
    def n_trials(self) -> int:
        """Return the number of rollout trials in the shared bank."""

        return int(self.initial_states.shape[0])

    def to_json(self) -> dict[str, Any]:
        """Return public bank provenance without large sampled arrays."""

        return {
            "bank_id": self.bank_id,
            "seed": self.seed,
            "n_trials": self.n_trials,
            "initial_state": {
                "status": "shared",
                "source": "analytical_output_feedback_initial_state_covariance",
                "covariance": self.initial_covariance,
                "shape": list(self.initial_states.shape),
            },
            "process_load_epsilon": {
                "status": "shared",
                "source": "rlrmp.train.task_model._sample_cs_lss_process_epsilon",
                "channel": "TaskTrialSpec.inputs['epsilon'] -> mechanics.epsilon",
                "shape": list(self.process_epsilon.shape),
            },
            "sensory_noise": {
                "status": "not_shared",
                "reason": (
                    "GRU sensory noise is sampled inside the Feedbax graph during "
                    "eval_trials, while extLQG uses explicit sensory draw arrays; "
                    "the current graph contract does not expose an equivalent "
                    "sampled sensory-noise input channel for both arms."
                ),
            },
            "command_or_motor_noise": {
                "status": "not_shared",
                "reason": (
                    "GRU motor/command noise is sampled by internal graph channels; "
                    "only the external process/load epsilon input is shared in this "
                    "materialization."
                ),
            },
        }


def build_shared_rollout_bank(
    *,
    seed: int = 20260603,
    n_trials: int = 32,
    bank_id: str = "cs_lss_shared_x0_epsilon_v1",
) -> SharedRolloutBank:
    """Sample the standard shared initial-state/process-epsilon bank."""

    from rlrmp.analysis.math.cs_game_card import build_canonical_game
    from rlrmp.analysis.math.cs_released_simulation import _default_output_feedback_initial_state
    from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
    from rlrmp.train.task_model import _cs_lss_process_epsilon_factor

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    plant, schedule = build_canonical_game()
    config = OutputFeedbackConfig()
    x0 = _default_output_feedback_initial_state(plant, config)
    key_x0, key_epsilon = jr.split(jr.PRNGKey(seed))
    initial_covariance = float(config.estimator_initial_covariance)
    initial_states = x0[None, :] + jnp.sqrt(initial_covariance) * jr.normal(
        key_x0,
        (n_trials, plant.n),
        dtype=jnp.float64,
    )
    epsilon_factor = _cs_lss_process_epsilon_factor()
    epsilon_standard = jr.normal(
        key_epsilon,
        (n_trials, schedule.T, epsilon_factor.shape[0]),
        dtype=jnp.float64,
    )
    process_epsilon = jnp.einsum("btd,ed->bte", epsilon_standard, epsilon_factor)
    return SharedRolloutBank(
        bank_id=bank_id,
        seed=seed,
        initial_states=np.asarray(initial_states, dtype=np.float64),
        process_epsilon=np.asarray(process_epsilon, dtype=np.float64),
        initial_covariance=initial_covariance,
    )


@dataclass(frozen=True)
class ExtLQGCostDecomposition:
    """C&S extLQG expected-cost components under the full-QRF objective lens."""

    deterministic_initial_state: float
    initial_covariance_trace: float
    accumulated_noise_scalar: float
    provenance: str
    total_expected_cost: float | None = None

    @property
    def component_sum(self) -> float:
        """Return the sum of deterministic, covariance, and noise terms."""

        return (
            self.deterministic_initial_state
            + self.initial_covariance_trace
            + self.accumulated_noise_scalar
        )

    @property
    def expected_cost(self) -> float:
        """Return the declared total expected cost, or the component sum."""

        if self.total_expected_cost is None:
            return self.component_sum
        return self.total_expected_cost

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable decomposition record."""

        return {
            "lens": "extlqg_covariance_inclusive_expected_cost",
            "deterministic_initial_state": self.deterministic_initial_state,
            "initial_covariance_trace": self.initial_covariance_trace,
            "accumulated_noise_scalar": self.accumulated_noise_scalar,
            "component_sum": self.component_sum,
            "total_expected_cost": self.expected_cost,
            "component_sum_delta": self.expected_cost - self.component_sum,
            "comparable_scalar": self.deterministic_initial_state,
            "comparable_scalar_lens": "extlqg_deterministic_initial_state_full_qrf",
            "provenance": self.provenance,
        }


def build_objective_comparator_sidecar(
    *,
    issue: str,
    source_manifest: str,
    checkpoint_selection: Mapping[str, Any],
    extlqg: ExtLQGCostDecomposition,
    scope: str,
    generated_by: str,
    same_noise_bank_monte_carlo: Mapping[str, Any] | None = None,
    run_metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    per_term_realized_scoring: Mapping[str, Any] | None = None,
    shared_rollout_comparator: Mapping[str, Any] | None = None,
    split_bank_comparator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON sidecar from validation-selected checkpoint records."""

    runs = checkpoint_selection.get("runs")
    if not isinstance(runs, Mapping):
        raise ValueError("checkpoint_selection must contain a mapping at key 'runs'")

    metadata_by_id = run_metadata_by_id or {}
    shared_rollout = dict(shared_rollout_comparator or DEFAULT_SHARED_ROLLOUT_STATUS)
    split_bank = dict(split_bank_comparator or _split_bank_from_shared_rollout(shared_rollout))
    checkpoint_policy = str(
        checkpoint_selection.get("checkpoint_policy") or "validation_selected_per_replicate"
    )
    sidecar_rows = [
        _build_run_row(
            run_id=str(run_id),
            selections=_expect_sequence(selections),
            checkpoint_policy=checkpoint_policy,
            extlqg=extlqg,
            run_metadata=metadata_by_id.get(str(run_id)),
            shared_rollout_run=_shared_rollout_run(shared_rollout, str(run_id)),
            split_bank_run=_split_bank_run(split_bank, str(run_id)),
        )
        for run_id, selections in sorted(runs.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": issue,
        "scope": scope,
        "source_manifest": source_manifest,
        "generated_by": generated_by,
        "checkpoint_policy": {
            "label": checkpoint_policy,
            "source": checkpoint_selection.get("schema_version"),
            "selection_source": checkpoint_selection.get("selection_source"),
            "selection_policy": checkpoint_selection.get("selection_policy"),
            "caveat": (
                "Checkpoint selection is inherited from the supplied checkpoint "
                "manifest. Analytical action, I/O, and extLQG comparator metrics "
                "are audit-only and are not used for checkpoint selection."
            ),
        },
        "objective_lenses": {
            "gru_validation_selected_realized_full_qrf": {
                "kind": "realized_validation_objective",
                "definition": (
                    "sum_t x_t^T Q_t x_t + u_t^T R_t u_t + x_T^T Q_f x_T "
                    "using states.mechanics.vector for x and states.net.output for u"
                ),
                "same_initial_conditions_as_extlqg": "deterministic_single_reach_contract",
                "noise_bank": "validation_rollout_bank_not_exported",
            },
            "extlqg_deterministic_initial_state_full_qrf": {
                "kind": "deterministic_analytical_term",
                "definition": "x0^T Sx0 x0 from the extLQG/computeOFC recursion",
                "noise_bank": "none_deterministic_term",
            },
            "extlqg_covariance_inclusive_expected_cost": {
                "kind": "expected_cost",
                "definition": (
                    "deterministic initial-state term plus initial covariance trace "
                    "plus accumulated process/sensory/motor noise scalar"
                ),
                "noise_bank": "analytical_covariance_expectation_not_realized_validation_bank",
            },
            "same_noise_bank_monte_carlo_full_qrf": {
                "kind": "realized_same_noise_bank_monte_carlo",
                "definition": (
                    "GRU and extLQG rescored with the same initial-condition and "
                    "process/sensory/motor noise draws under the full-Q/R/Q_f lens"
                ),
            },
            "realized_full_qrf_per_term_validation": {
                "kind": "realized_validation_objective_decomposition",
                "definition": (
                    "total full-Q/R/Q_f cost decomposed into running state, terminal "
                    "state, command, force/filter, and disturbance-integrator terms"
                ),
            },
            "shared_rollout_full_qrf": {
                "kind": "realized_shared_rollout_comparison",
                "definition": (
                    "validation-selected GRU checkpoints and extLQG evaluated on the "
                    "same sampled initial-state/process-epsilon bank, then scored by "
                    "the same full-Q/R/Q_f cost and per-term decomposition"
                ),
                "interpretation": (
                    "stress-test-only unless the extLQG x0-only sanity check passes; "
                    "the split-bank block separates deterministic, component-specific "
                    "x0, component-specific process-epsilon, x0 position+velocity, "
                    "and x0+epsilon lenses"
                ),
            },
            "standard_split_rollout_bank_full_qrf": {
                "kind": "realized_shared_rollout_split_bank",
                "definition": (
                    "audit-only deterministic nominal, component-specific x0, "
                    "component-specific process-epsilon, x0 position+velocity, and "
                    "x0+epsilon realized full-QRF rescoring on standardized banks"
                ),
                "checkpoint_selection_role": "audit_only_not_used_for_checkpoint_selection",
            },
        },
        "extlqg_decomposition": extlqg.to_json(),
        "same_noise_bank_monte_carlo": dict(
            same_noise_bank_monte_carlo or _same_noise_status_from_shared_rollout(shared_rollout)
        ),
        "per_term_realized_scoring": dict(per_term_realized_scoring or DEFAULT_PER_TERM_STATUS),
        "shared_rollout_comparator": shared_rollout,
        "standard_split_bank_comparator": split_bank,
        "rows": sidecar_rows,
        "caveats": [
            (
                "The apples-to-apples scalar for the available GRU validation "
                "records is restricted to rows whose run spec declares the "
                "full analytical Q/R/Q_f objective; the deterministic extLQG "
                "term is not interchangeable with the covariance-inclusive "
                "expected cost."
            ),
            "This sidecar is diagnostic only and is not a standard-certificate gate.",
            (
                "GRU values are validation-selected realized full-QRF scalars; "
                "the shared-rollout and split-bank blocks are audit-only post-hoc "
                "rescores and are not used for checkpoint selection."
            ),
            (
                "The x0+epsilon shared-rollout block is stress-test-only unless "
                "the extLQG x0-only sanity check supports expected-cost wording."
            ),
            (
                "Split-bank GRU hidden states are initialized from the checkpoint "
                "model default rather than conditioned on the perturbed x0, so x0 "
                "lenses are recovery stress tests rather than expected-cost "
                "comparisons."
            ),
        ],
    }


def shared_full_qrf_cost_summary(
    *,
    states: Any,
    commands: Any,
    initial_states: Any,
    state_basis: str = "absolute_workspace",
) -> dict[str, Any]:
    """Score realized full-Q/R/Q_f rollout costs with standard sidecar terms.

    Args:
        states: Rollout mechanics vectors with shape ``(..., T, 48)``.
        commands: Controller commands with shape ``(..., T, 2)``.
        initial_states: Initial mechanics vectors broadcastable to ``(..., 48)``.
        state_basis: Coordinate basis for ``states`` and ``initial_states``.
            ``"absolute_workspace"`` means the x/y position channels are absolute
            Feedbax mechanics coordinates and must be target-centered before
            scoring. ``"target_centered"`` means analytical extLQG-style states
            are already expressed relative to the target and must not be shifted
            again.
    """

    from rlrmp.analysis.math.cs_game_card import TARGET_POS, build_canonical_game

    if state_basis not in {"absolute_workspace", "target_centered"}:
        raise ValueError(
            "state_basis must be 'absolute_workspace' or 'target_centered', "
            f"got {state_basis!r}."
        )
    _plant, schedule = build_canonical_game()
    state_array = np.asarray(states, dtype=np.float64)
    command_array = np.asarray(commands, dtype=np.float64)
    initial_array = np.asarray(initial_states, dtype=np.float64)
    if state_array.shape[-1] != schedule.Q.shape[-1]:
        raise ValueError(
            f"Full-Q/R/Q_f scorer expected state dim {schedule.Q.shape[-1]}, "
            f"got {state_array.shape[-1]}."
        )
    if command_array.shape[-1] != schedule.R.shape[-1]:
        raise ValueError(
            f"Full-Q/R/Q_f scorer expected command dim {schedule.R.shape[-1]}, "
            f"got {command_array.shape[-1]}."
        )
    horizon = int(schedule.T)
    if state_array.shape[-2] != horizon:
        raise ValueError(f"Full-Q/R/Q_f scorer expected {horizon} states.")
    if command_array.shape[-2] != horizon:
        raise ValueError(f"Full-Q/R/Q_f scorer expected {horizon} commands.")
    initial_array = np.broadcast_to(initial_array, (*state_array.shape[:-2], state_array.shape[-1]))
    x_pre = np.concatenate([initial_array[..., None, :], state_array[..., :-1, :]], axis=-2)
    if state_basis == "absolute_workspace":
        x_pre = _goal_centered_vectors(x_pre, target_pos=TARGET_POS)
        x_terminal = _goal_centered_vectors(state_array[..., -1, :], target_pos=TARGET_POS)
        state_transform = "subtract TARGET_POS from each physical delay block x/y"
    else:
        x_terminal = np.asarray(state_array[..., -1, :], dtype=np.float64)
        state_transform = "none; states are already target-centered"
    q = np.asarray(schedule.Q, dtype=np.float64)
    r = np.asarray(schedule.R, dtype=np.float64)
    q_f = np.asarray(schedule.Q_f, dtype=np.float64)
    groups = _state_term_groups(state_array.shape[-1])
    running_state = _state_quadratic_group(x_pre, q, groups["running_state"])
    force_filter = _state_quadratic_group(x_pre, q, groups["force_filter_state"])
    disturbance_integrator = _state_quadratic_group(
        x_pre,
        q,
        groups["disturbance_integrator_state"],
    )
    terminal_state = _terminal_quadratic_group(x_terminal, q_f, groups["running_state"])
    terminal_force = _terminal_quadratic_group(x_terminal, q_f, groups["force_filter_state"])
    terminal_integrator = _terminal_quadratic_group(
        x_terminal,
        q_f,
        groups["disturbance_integrator_state"],
    )
    command_control = np.sum(
        np.einsum("...ti,tij,...tj->...t", command_array, r, command_array),
        axis=-1,
    )
    force_filter = force_filter + terminal_force
    disturbance_integrator = disturbance_integrator + terminal_integrator
    total = (
        running_state
        + terminal_state
        + command_control
        + force_filter
        + disturbance_integrator
    )
    return {
        "status": "available",
        "lens": "shared_rollout_realized_full_qrf",
        "basis": {
            "state_key": "states.mechanics.vector",
            "command_key": "states.net.output or extLQG u_command",
            "state_basis": state_basis,
            "state_transform": state_transform,
            "schedule_source": "rlrmp.analysis.math.cs_game_card.build_canonical_game",
            "term_split": "coordinate masks over each 8D delay block",
        },
        "total": _summary_with_values(total),
        "running_state": _summary_with_values(running_state),
        "terminal_state": _summary_with_values(terminal_state),
        "command_control": _summary_with_values(command_control),
        "force_filter_state": _summary_with_values(force_filter),
        "disturbance_integrator_state": _summary_with_values(disturbance_integrator),
        "term_sum_delta": _summary_stats(
            total
            - (
                running_state
                + terminal_state
                + command_control
                + force_filter
                + disturbance_integrator
            )
        ),
    }


def materialize_shared_rollout_comparator(
    *,
    experiment: str,
    run_ids: Sequence[str],
    checkpoint_manifest: Mapping[str, Any],
    bank: SharedRolloutBank | None = None,
    n_trials: int = 32,
    seed: int = 20260603,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Evaluate extLQG and selected GRUs on one shared bank."""

    from feedbax.types import TreeNamespace, dict_to_namespace
    from rlrmp.analysis.math.cs_game_card import build_canonical_game
    from rlrmp.analysis.pipelines.cs_gru_standard_materialization import normalize_gru_hps
    from rlrmp.analysis.math.cs_released_simulation import (
        build_extlqg_comparator_path,
        default_cs_noise_covariances,
        zero_forward_noise_draws,
    )
    from rlrmp.analysis.pipelines.gru_checkpoint_selection import load_validation_selected_checkpoint_model
    from rlrmp.analysis.pipelines.gru_pilot_figures import repeat_single_validation_trial, resolve_run_inputs
    from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
    from rlrmp.train.task_model import setup_task_model_pair

    bank = bank or build_shared_rollout_bank(seed=seed, n_trials=n_trials)
    plant, schedule = build_canonical_game()
    config = OutputFeedbackConfig()
    covariances = default_cs_noise_covariances(plant, config)
    extlqg_path = build_extlqg_comparator_path(
        plant,
        jnp.zeros((schedule.T, plant.m_u, plant.n), dtype=jnp.float64),
        covariances,
        schedule=schedule,
        config=config,
    )
    zero_draws = zero_forward_noise_draws(T=schedule.T, plant=plant, config=config)
    extlqg_cost_by_lens = _extlqg_split_bank_costs(
        bank=bank,
        extlqg_path=extlqg_path,
        plant=plant,
        schedule=schedule,
        config=config,
        covariances=covariances,
        zero_draws=zero_draws,
    )
    extlqg_cost = extlqg_cost_by_lens["x0_plus_epsilon"]
    checkpoint_policy = str(
        checkpoint_manifest.get("checkpoint_policy") or "validation_selected_per_replicate"
    )
    x0_sanity = {
        "status": "not_applicable",
        "lens": "extlqg_x0_only_realized_vs_expected_trace",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "reason": (
            "The v6 split-bank lens set intentionally separates x0 components and "
            "does not materialize an all-components x0-only lens; use the per-lens "
            "component ratios rather than expected-cost wording."
        ),
        "expected_cost_wording_allowed": False,
    }

    runs = resolve_run_inputs(experiment=experiment, run_ids=run_ids, labels=None, repo_root=repo_root)
    run_results = {}
    for run in runs:
        hps = dict_to_namespace(normalize_gru_hps(run.run_spec["hps"]), to_type=TreeNamespace)
        n_replicates = int(hps.model.n_replicates)
        pair = setup_task_model_pair(hps, key=jr.PRNGKey(int(run.run_spec.get("seed", 42))))
        model, checkpoint_selection = load_validation_selected_checkpoint_model(
            experiment=experiment,
            run_id=run.run_id,
            run_spec=run.run_spec,
            preferred_manifest=checkpoint_manifest,
            checkpoint_selection_mode=(
                "fixed_bank_manifest"
                if checkpoint_policy == "fixed_bank_rescored_per_replicate"
                else "sparse_history"
            ),
            repo_root=repo_root,
        )
        base_trial_specs = repeat_single_validation_trial(pair.task.validation_trials, bank.n_trials)
        gru_cost_by_lens = _gru_split_bank_costs(
            model=model,
            task=pair.task,
            base_trial_specs=base_trial_specs,
            bank=bank,
            n_replicates=n_replicates,
            seed=seed,
        )
        gru_cost = gru_cost_by_lens["x0_plus_epsilon"]
        run_results[run.run_id] = {
            "status": "available",
            "checkpoint_policy": checkpoint_policy,
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "checkpoint_selection": [
                selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
            ],
            "n_replicates": n_replicates,
            "gru_cost": _public_cost_summary(gru_cost),
            "extlqg_cost": _public_cost_summary(extlqg_cost),
            "gru_vs_extlqg": _cost_comparison(gru_cost, extlqg_cost),
            "standard_split_bank": {
                lens: {
                    "status": "available",
                    "selection_role": "audit_only_not_used_for_checkpoint_selection",
                    "gru_cost": _public_cost_summary(gru_cost_by_lens[lens]),
                    "extlqg_cost": _public_cost_summary(extlqg_cost_by_lens[lens]),
                    "gru_vs_extlqg": _cost_comparison(
                        gru_cost_by_lens[lens],
                        extlqg_cost_by_lens[lens],
                    ),
                }
                for lens in _STANDARD_SPLIT_BANK_LENSES
            },
        }
    return {
        "status": "available",
        "lens": "shared_rollout_full_qrf",
        "checkpoint_policy": checkpoint_policy,
        "interpretation": "stress_test_only",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "bank": bank.to_json(),
        "extlqg": {
            "status": "available",
            "parity_status": extlqg_path.parity_status,
            "n_iterations": int(extlqg_path.n_iterations),
            "expected_cost": extlqg_path.expected_cost,
            "cost": _public_cost_summary(extlqg_cost),
            "standard_split_bank": {
                lens: _public_cost_summary(cost)
                for lens, cost in extlqg_cost_by_lens.items()
            },
            "x0_only_sanity_check": x0_sanity,
        },
        "noise_comparability": {
            "shared_channels": ["initial_state", "process_load_epsilon"],
            "not_shared_channels": ["sensory_noise", "command_or_motor_noise"],
            "limitation": (
                "This is a shared initial-state plus process/load epsilon comparator. "
                "Sensory and command/motor noise are explicitly not claimed as shared."
            ),
        },
        "fairness_residuals": _split_bank_fairness_residuals(),
        "runs": run_results,
        "standard_split_bank_comparator": _standard_split_bank_public(
            bank=bank,
            extlqg_path=extlqg_path,
            extlqg_cost_by_lens=extlqg_cost_by_lens,
            x0_sanity=x0_sanity,
            run_results=run_results,
        ),
        "source_checkpoint_manifest_schema": checkpoint_manifest.get("schema_version"),
    }


def extlqg_x0_only_sanity_check(
    *,
    x0_only_cost: Mapping[str, Any],
    extlqg: ExtLQGCostDecomposition,
    relative_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare realized extLQG x0-only cost with deterministic+trace expectation."""

    observed = _summary_mean(x0_only_cost, "total")
    expected = extlqg.deterministic_initial_state + extlqg.initial_covariance_trace
    if observed is None:
        return {
            "status": "blocked",
            "reason": "x0-only total cost mean is unavailable",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }
    absolute_delta = float(observed - expected)
    relative_delta = abs(absolute_delta) / max(abs(expected), 1e-12)
    status = "pass" if relative_delta <= relative_tolerance else "warning"
    return {
        "status": status,
        "lens": "extlqg_x0_only_realized_vs_expected_trace",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "realized_x0_only_cost_mean": float(observed),
        "expected_deterministic_plus_initial_covariance_trace": float(expected),
        "deterministic_initial_state": float(extlqg.deterministic_initial_state),
        "initial_covariance_trace": float(extlqg.initial_covariance_trace),
        "absolute_delta": absolute_delta,
        "relative_delta": float(relative_delta),
        "relative_tolerance": float(relative_tolerance),
        "expected_cost_wording_allowed": status == "pass",
    }


def render_objective_comparator_markdown(sidecar: Mapping[str, Any]) -> str:
    """Render a compact Markdown companion for an objective-comparator sidecar."""

    decomposition = _expect_mapping(sidecar["extlqg_decomposition"])
    rows = _expect_sequence(sidecar["rows"])
    same_noise = _expect_mapping(sidecar["same_noise_bank_monte_carlo"])
    per_term = _expect_mapping(sidecar["per_term_realized_scoring"])
    shared_rollout = _expect_mapping(sidecar.get("shared_rollout_comparator", {}))
    split_bank = _expect_mapping(sidecar.get("standard_split_bank_comparator", {}))
    sanity = _expect_mapping(split_bank.get("extlqg_x0_only_sanity_check", {}))
    lines = [
        "# Full-QRF objective comparator sidecar",
        "",
        f"Schema: `{sidecar['schema_version']}`.",
        "",
        f"Scope: {sidecar['scope']}.",
        "",
        "This is an objective-lens diagnostic, not a standard-certificate gate.",
        "",
        "## Objective lenses",
        "",
        "| lens | status | comparability |",
        "|---|---|---|",
        (
            "| deterministic extLQG | available | deterministic full-Q/R/Q_f initial-state "
            "term; comparable only to full-Q/R/Q_f realized scalars |"
        ),
        (
            "| covariance-inclusive extLQG expected cost | available | not directly "
            "comparable to realized GRU validation scalars |"
        ),
        (
            "| realized GRU validation | available for full-Q/R/Q_f scalar rows | "
            "validation-selected audit metric, not checkpoint selection input |"
        ),
        (
            "| full same-noise-bank Monte Carlo | not_implemented | full shared "
            "sensory/command/motor noise is not exposed for both arms; see the "
            "partial shared-rollout comparator below |"
        ),
        (
            "| realized per-term full-Q/R/Q_f scoring | "
            f"{per_term['status']} | requires scorer output for running state, terminal, "
            "command, force/filter, and disturbance-integrator terms |"
        ),
        (
            "| shared-rollout comparator | "
            f"{shared_rollout.get('status', 'not_available')} | shared initial-state and "
            "process/load epsilon bank; sensory/command noise limits declared |"
        ),
        (
            "| standard split-bank comparator | "
            f"{split_bank.get('status', 'not_available')} | deterministic nominal, "
            "component-specific x0/process-epsilon, x0 position+velocity, and "
            "x0+epsilon audit-only lenses |"
        ),
        "",
        "## extLQG decomposition",
        "",
        "| component | value | lens |",
        "|---|---:|---|",
        (
            "| deterministic initial-state term | "
            f"{_fmt(decomposition['deterministic_initial_state'])} | comparable to "
            "realized/validation full-QRF values |"
        ),
        (
            "| initial covariance trace term | "
            f"{_fmt(decomposition['initial_covariance_trace'])} | expected-cost sidecar only |"
        ),
        (
            "| accumulated noise scalar | "
            f"{_fmt(decomposition['accumulated_noise_scalar'])} | expected-cost sidecar only |"
        ),
        (
            "| total expected cost | "
            f"{_fmt(decomposition['total_expected_cost'])} | not directly comparable to GRU "
            "validation values |"
        ),
        (
            "| x0-only realized sanity | "
            f"{sanity.get('status', 'not_available')} | realized extLQG x0-only cost "
            "vs deterministic + initial-covariance-trace expectation |"
        ),
        "",
        "## GRU comparison",
        "",
        (
            "| run | row comparability | mean selected validation | deterministic extLQG | "
            "selected/deterministic | total expected cost | selected/total | per-term scoring |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        row_map = _expect_mapping(row)
        comparability = _expect_mapping(row_map["comparability"])
        per_term_status = _expect_mapping(row_map["per_term_realized_scoring"])
        lines.append(
            "| "
            f"`{row_map['run_id']}` | "
            f"{comparability['status']} | "
            f"{_fmt(row_map['gru_mean_selected_validation_full_qrf'])} | "
            f"{_fmt(row_map['extlqg_deterministic_full_qrf'])} | "
            f"{_fmt(row_map['selected_to_extlqg_deterministic_ratio'])} | "
            f"{_fmt(row_map['extlqg_total_expected_cost'])} | "
            f"{_fmt(row_map['selected_to_extlqg_total_ratio_not_apples_to_apples'])} | "
            f"{per_term_status['status']} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            (
                "- `selected/total` is retained only as a labeled non-apples-to-apples "
                "diagnostic for continuity with the provisional sidecar."
            ),
            (
                "- The partial x0+epsilon shared-rollout comparator is stress-test-only; "
                "expected-cost wording is allowed only when an extLQG x0-only sanity "
                f"check passes. Current status: `{sanity.get('status', 'not_available')}`."
            ),
        ]
    )
    for caveat in _expect_sequence(sidecar["caveats"]):
        lines.append(f"- {caveat}")
    lines.extend(
        [
            "",
            "Full same-noise-bank Monte Carlo: `not_implemented` - full shared "
            "sensory/command/motor noise is not exposed for both arms. Partial "
            f"shared-rollout replacement: `{same_noise['status']}` - {same_noise['reason']}",
            "",
            "Per-term realized scoring: "
            f"`{per_term['status']}` - {per_term['reason']}",
            "",
        ]
    )
    lines.extend(_render_shared_rollout_markdown(shared_rollout))
    lines.extend(_render_split_bank_markdown(split_bank))
    return "\n".join(lines)


def write_objective_comparator_sidecar(
    sidecar: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write JSON and Markdown sidecar artifacts."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_objective_comparator_markdown(sidecar), encoding="utf-8")


def compute_default_extlqg_cost_decomposition() -> ExtLQGCostDecomposition:
    """Compute the canonical C&S extLQG expected-cost decomposition."""

    import jax.numpy as jnp

    from rlrmp.analysis.math.cs_game_card import (
        OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        materialize_reference,
    )
    from rlrmp.analysis.math.cs_released_simulation import (
        _compute_ext_kalman,
        _compute_ofc,
        _default_output_feedback_initial_state,
        default_cs_noise_covariances,
    )
    from rlrmp.analysis.math.output_feedback import (
        delayed_observation_matrix,
        position_velocity_observation_config,
    )

    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    plant = reference.plant
    schedule = reference.schedule
    config = position_velocity_observation_config(plant)
    covariances = default_cs_noise_covariances(plant, config)
    h_matrix = delayed_observation_matrix(plant, config)
    state_noise = covariances.motor + covariances.process
    initial_covariance = jnp.eye(plant.n, dtype=jnp.float64) * jnp.asarray(
        config.estimator_initial_covariance,
        dtype=jnp.float64,
    )
    estimator_gains = jnp.zeros((schedule.T, plant.n, h_matrix.shape[0]), dtype=jnp.float64)
    current = 1.0e6
    deterministic = 0.0
    initial_trace = 0.0
    scalar = 0.0
    expected = current
    iteration = 0
    for iteration in range(1, 101):
        controller_gains, sx0, se0, scalar_cost = _compute_ofc(
            plant,
            schedule,
            estimator_gains,
            h_matrix,
            covariances.signal_dependent_state,
            state_noise,
            covariances.sensory,
        )
        estimator_gains, _state_covariances = _compute_ext_kalman(
            plant,
            h_matrix,
            controller_gains,
            covariances.signal_dependent_state,
            state_noise,
            covariances.sensory,
            initial_covariance,
            initial_covariance,
        )
        x0 = _default_output_feedback_initial_state(plant, config)
        deterministic = float(x0 @ sx0 @ x0)
        initial_trace = float(jnp.trace((sx0 + se0) @ initial_covariance))
        scalar = float(scalar_cost)
        expected = deterministic + initial_trace + scalar
        relative_change = abs(current - expected) / max(abs(expected), 1e-300)
        current = expected
        if relative_change <= 1e-14:
            break

    return ExtLQGCostDecomposition(
        deterministic_initial_state=deterministic,
        initial_covariance_trace=initial_trace,
        accumulated_noise_scalar=scalar,
        total_expected_cost=expected,
        provenance=(
            "canonical C&S extLQG fixed-point decomposition from "
            "materialize_reference(output_feedback_certificate_gamma_factor), "
            f"{iteration} iterations"
        ),
    )


def materialize_gru_objective_comparator_sidecar(
    *,
    experiment: str,
    run_ids: Sequence[str],
    labels: Sequence[str] | None = None,
    checkpoint_policy: str,
    use_validation_selected_checkpoints: bool,
    checkpoint_manifest: Mapping[str, Any] | None,
    checkpoint_manifest_path: Path | None,
    standard_manifest_path: Path,
    output_path: Path,
    note_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize the GRU objective-comparator sidecar for a post-run bundle."""

    del labels
    if not use_validation_selected_checkpoints:
        return {
            "status": "skipped",
            "reason": "objective_comparator_requires_validation_selected_checkpoints",
        }
    if checkpoint_policy not in {
        "validation_selected_per_replicate",
        "fixed_bank_rescored_per_replicate",
    }:
        return {
            "status": "skipped",
            "reason": "unsupported_checkpoint_policy",
            "checkpoint_policy": checkpoint_policy,
        }
    if checkpoint_manifest is None:
        if checkpoint_manifest_path is None:
            raise ValueError("checkpoint_manifest or checkpoint_manifest_path is required")
        checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))

    extlqg = compute_default_extlqg_cost_decomposition()
    run_metadata_by_id = {
        str(run_id): load_run_objective_metadata(
            repo_root / "results" / experiment / "runs" / str(run_id) / "run.json",
            repo_root=repo_root,
        )
        for run_id in run_ids
    }
    sidecar = build_objective_comparator_sidecar(
        issue=experiment,
        source_manifest=_repo_relative(standard_manifest_path, repo_root=repo_root),
        checkpoint_selection=checkpoint_manifest,
        extlqg=extlqg,
        scope=(
            f"{checkpoint_policy} checkpoints for C&S GRU runs: "
            + ", ".join(str(run_id) for run_id in run_ids)
        ),
        generated_by="rlrmp.analysis.pipelines.objective_comparator.materialize_gru_objective_comparator_sidecar",
        run_metadata_by_id=run_metadata_by_id,
        shared_rollout_comparator=_try_materialize_shared_rollout_comparator(
            experiment=experiment,
            run_ids=run_ids,
            checkpoint_manifest=checkpoint_manifest,
            repo_root=repo_root,
        ),
    )
    write_objective_comparator_sidecar(
        sidecar,
        json_path=output_path,
        markdown_path=note_path,
    )
    split_bank = sidecar.get("standard_split_bank_comparator", {})
    return {
        "status": "materialized",
        "schema_version": sidecar["schema_version"],
        "n_rows": len(sidecar["rows"]),
        "extlqg_deterministic_full_qrf": extlqg.deterministic_initial_state,
        "extlqg_total_expected_cost": extlqg.expected_cost,
        "standard_split_bank_comparator_status": (
            split_bank.get("status") if isinstance(split_bank, dict) else None
        ),
    }


def _build_run_row(
    *,
    run_id: str,
    selections: Sequence[Any],
    checkpoint_policy: str,
    extlqg: ExtLQGCostDecomposition,
    run_metadata: Mapping[str, Any] | None = None,
    shared_rollout_run: Mapping[str, Any] | None = None,
    split_bank_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = [_float_from_selection(item, "scoring_validation_objective") for item in selections]
    best_logged = [
        _float_from_selection(item, "best_logged_validation_objective") for item in selections
    ]
    mean_selected = sum(selected) / len(selected)
    mean_best_logged = sum(best_logged) / len(best_logged)
    metadata = dict(run_metadata or _missing_run_metadata())
    comparability = _row_comparability(metadata)
    is_comparable = comparability["status"] == "comparable_deterministic_full_qrf"
    return {
        "run_id": run_id,
        "checkpoint_policy": checkpoint_policy,
        "n_replicates": len(selections),
        "training_objective": metadata,
        "comparability": comparability,
        "gru_realized_lens": "gru_selected_checkpoint_realized_full_qrf",
        "extlqg_comparable_lens": "extlqg_deterministic_initial_state_full_qrf",
        "gru_mean_selected_validation_full_qrf": mean_selected,
        "gru_mean_best_logged_validation_full_qrf": mean_best_logged,
        "extlqg_deterministic_full_qrf": extlqg.deterministic_initial_state,
        "selected_to_extlqg_deterministic_ratio": (
            mean_selected / extlqg.deterministic_initial_state if is_comparable else None
        ),
        "best_logged_to_extlqg_deterministic_ratio": (
            mean_best_logged / extlqg.deterministic_initial_state if is_comparable else None
        ),
        "extlqg_total_expected_cost": extlqg.expected_cost,
        "selected_to_extlqg_total_ratio_not_apples_to_apples": (
            mean_selected / extlqg.expected_cost if is_comparable else None
        ),
        "per_term_realized_scoring": dict(DEFAULT_PER_TERM_STATUS),
        "same_noise_bank_monte_carlo": dict(DEFAULT_MONTE_CARLO_STATUS),
        "shared_rollout_comparator": dict(
            shared_rollout_run
            or {
                "status": "not_available",
                "reason": "shared-rollout comparator has no row for this run",
            }
        ),
        "standard_split_bank_comparator": dict(
            split_bank_run
            or {
                "status": "not_available",
                "reason": "standard split-bank comparator has no row for this run",
            }
        ),
        "selected_checkpoints": list(selections),
    }


def load_run_objective_metadata(
    run_spec_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load the objective metadata needed to judge sidecar comparability."""

    if not run_spec_path.exists():
        return _missing_run_metadata(path=run_spec_path)
    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    loss_summary = _expect_mapping(run_spec.get("loss_summary", {}))
    return {
        "status": "available",
        "run_spec_path": _repo_relative(run_spec_path, repo_root=repo_root or REPO_ROOT),
        "loss_objective": run_spec.get("loss_objective"),
        "objective_profile": loss_summary.get("objective_profile"),
        "full_qrf_lens": _full_qrf_lens_from_loss_summary(loss_summary),
    }


def _full_qrf_lens_from_loss_summary(loss_summary: Mapping[str, Any]) -> dict[str, Any]:
    active_terms = _expect_mapping(loss_summary.get("active_cs_terms", {}))
    return {
        "status": (
            "available"
            if {"state_running_q", "terminal_q_f", "control_r"}.issubset(active_terms)
            else "not_comparable"
        ),
        "state_basis": loss_summary.get("state_basis"),
        "time_indexing": loss_summary.get("time_indexing"),
        "matrix_shapes": loss_summary.get("matrix_shapes"),
        "active_terms": sorted(str(term) for term in active_terms),
        "force_filter_state_cost": loss_summary.get("force_filter_state_cost"),
        "disturbance_integrator_state_cost": loss_summary.get(
            "disturbance_integrator_state_cost"
        ),
    }


def _missing_run_metadata(path: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "missing",
        "loss_objective": None,
        "objective_profile": None,
        "full_qrf_lens": {
            "status": "not_comparable",
            "reason": "run objective metadata was not supplied",
        },
    }
    if path is not None:
        payload["run_spec_path"] = str(path)
    return payload


def _row_comparability(run_metadata: Mapping[str, Any]) -> dict[str, Any]:
    loss_objective = run_metadata.get("loss_objective")
    objective_profile = run_metadata.get("objective_profile")
    if (
        run_metadata.get("status") == "available"
        and loss_objective == FULL_ANALYTICAL_QRF_OBJECTIVE
        and objective_profile == FULL_ANALYTICAL_QRF_OBJECTIVE
    ):
        return {
            "status": "comparable_deterministic_full_qrf",
            "reason": (
                "run spec declares the full analytical Q/R/Q_f objective; the "
                "available scalar comparison is against the deterministic "
                "extLQG initial-state term only"
            ),
            "not_a_checkpoint_selection_input": True,
        }
    return {
        "status": "not_comparable",
        "reason": (
            "row was not explicitly rescored under the full analytical Q/R/Q_f "
            "lens, so scalar objective superiority must not be inferred"
        ),
        "loss_objective": loss_objective,
        "objective_profile": objective_profile,
    }


def _try_materialize_shared_rollout_comparator(
    *,
    experiment: str,
    run_ids: Sequence[str],
    checkpoint_manifest: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Materialize the shared rollout comparator, returning a structured blocker."""

    try:
        return materialize_shared_rollout_comparator(
            experiment=experiment,
            run_ids=run_ids,
            checkpoint_manifest=checkpoint_manifest,
            repo_root=repo_root,
        )
    except (FileNotFoundError, ValueError, KeyError, AttributeError) as exc:
        return {
            "status": "blocked",
            "lens": "shared_rollout_full_qrf",
            "reason": str(exc),
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        }


def _shared_rollout_run(
    shared_rollout: Mapping[str, Any],
    run_id: str,
) -> Mapping[str, Any] | None:
    runs = shared_rollout.get("runs")
    if not isinstance(runs, Mapping):
        return None
    run = runs.get(run_id)
    return _expect_mapping(run) if isinstance(run, Mapping) else None


def _split_bank_run(
    split_bank: Mapping[str, Any],
    run_id: str,
) -> Mapping[str, Any] | None:
    runs = split_bank.get("runs")
    if not isinstance(runs, Mapping):
        return None
    run = runs.get(run_id)
    return _expect_mapping(run) if isinstance(run, Mapping) else None


def _split_bank_from_shared_rollout(shared_rollout: Mapping[str, Any]) -> dict[str, Any]:
    split_bank = shared_rollout.get("standard_split_bank_comparator")
    if isinstance(split_bank, Mapping):
        return dict(split_bank)
    return dict(DEFAULT_SPLIT_BANK_STATUS)


def _same_noise_status_from_shared_rollout(shared_rollout: Mapping[str, Any]) -> dict[str, Any]:
    if shared_rollout.get("status") != "available":
        return dict(DEFAULT_MONTE_CARLO_STATUS)
    return {
        "status": "available_with_limitations",
        "lens": "shared_rollout_full_qrf",
        "reason": (
            "shared-rollout comparator materialized common random inputs for initial "
            "state and process/load epsilon; sensory and command/motor noise are "
            "explicitly not shared under the current GRU graph contract"
        ),
        "shared_channels": ["initial_state", "process_load_epsilon"],
        "not_shared_channels": ["sensory_noise", "command_or_motor_noise"],
        "replacement_block": "shared_rollout_comparator",
        "interpretation": "stress_test_only_partial_x0_plus_epsilon",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
    }


def _extlqg_split_bank_costs(
    *,
    bank: SharedRolloutBank,
    extlqg_path: Any,
    plant: Any,
    schedule: Any,
    config: Any,
    covariances: Any,
    zero_draws: Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate extLQG costs for the standard split-bank rollout lenses."""

    from rlrmp.analysis.math.cs_released_simulation import (
        _default_output_feedback_initial_state,
        simulate_lqg_released_forward,
    )

    default_initial = np.asarray(
        _default_output_feedback_initial_state(plant, config),
        dtype=np.float64,
    )
    lens_inputs = _split_bank_inputs(
        bank=bank,
        default_initial=np.broadcast_to(default_initial, bank.initial_states.shape),
    )
    costs: dict[str, dict[str, Any]] = {}
    for lens in _STANDARD_SPLIT_BANK_LENSES:
        states = []
        commands = []
        initial_states = lens_inputs[lens]["initial_states"]
        process_epsilon = lens_inputs[lens]["process_epsilon"]
        for initial_state, epsilon in zip(
            initial_states,
            process_epsilon,
            strict=True,
        ):
            rollout = simulate_lqg_released_forward(
                plant,
                extlqg_path.controller_gains,
                jnp.asarray(initial_state, dtype=jnp.float64),
                draws=zero_draws,
                covariances=covariances,
                estimator_gains=extlqg_path.estimator_gains,
                adversary_epsilon=jnp.asarray(epsilon, dtype=jnp.float64),
                config=config,
            )
            states.append(np.asarray(rollout.x[1:], dtype=np.float64))
            commands.append(np.asarray(rollout.u_command, dtype=np.float64))
        costs[lens] = shared_full_qrf_cost_summary(
            states=np.stack(states, axis=0),
            commands=np.stack(commands, axis=0),
            initial_states=initial_states,
            state_basis="target_centered",
        )
    return costs


def _gru_split_bank_costs(
    *,
    model: Any,
    task: Any,
    base_trial_specs: Any,
    bank: SharedRolloutBank,
    n_replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate GRU costs for the standard split-bank rollout lenses."""

    default_initial = np.asarray(base_trial_specs.inits["mechanics.vector"], dtype=np.float64)
    if default_initial.ndim == 1:
        default_initial = np.broadcast_to(default_initial, bank.initial_states.shape)
    else:
        default_initial = np.broadcast_to(default_initial[:1], bank.initial_states.shape)
    lens_inputs = _split_bank_inputs(bank=bank, default_initial=default_initial)
    costs: dict[str, dict[str, Any]] = {}
    for lens in _STANDARD_SPLIT_BANK_LENSES:
        initial_states = lens_inputs[lens]["initial_states"]
        process_epsilon = lens_inputs[lens]["process_epsilon"]
        lens_bank = SharedRolloutBank(
            bank_id=f"{bank.bank_id}:{lens}",
            seed=bank.seed,
            initial_states=np.asarray(initial_states, dtype=np.float64),
            process_epsilon=np.asarray(process_epsilon, dtype=np.float64),
            initial_covariance=bank.initial_covariance,
        )
        trial_specs = _trial_specs_with_shared_bank(base_trial_specs, lens_bank)
        states = _evaluate_replicate_model_states(
            model=model,
            task=task,
            trial_specs=trial_specs,
            n_replicates=n_replicates,
            seed=seed,
        )
        costs[lens] = shared_full_qrf_cost_summary(
            states=np.asarray(states.mechanics.vector, dtype=np.float64),
            commands=np.asarray(states.net.output, dtype=np.float64),
            initial_states=lens_bank.initial_states,
            state_basis="absolute_workspace",
        )
    return costs


def _split_bank_inputs(
    *,
    bank: SharedRolloutBank,
    default_initial: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Return component-masked initial states and process epsilons for each lens."""

    default_initial = np.asarray(default_initial, dtype=np.float64)
    initial_states = np.asarray(bank.initial_states, dtype=np.float64)
    process_epsilon = np.asarray(bank.process_epsilon, dtype=np.float64)
    default_initial = np.broadcast_to(default_initial, initial_states.shape)
    initial_delta = initial_states - default_initial
    return {
        lens: {
            "initial_states": default_initial
            + _mask_state_components(initial_delta, spec["x0"]),
            "process_epsilon": _mask_physical_components(process_epsilon, spec["epsilon"]),
        }
        for lens, spec in _STANDARD_SPLIT_BANK_LENS_SPECS.items()
    }


def _mask_state_components(values: np.ndarray, components: Sequence[str]) -> np.ndarray:
    mask = np.zeros(values.shape[-1], dtype=bool)
    if values.shape[-1] % 8 != 0:
        raise ValueError(f"state dimension {values.shape[-1]} is not divisible by 8")
    for start in range(0, values.shape[-1], 8):
        for component in components:
            component_start, component_stop = STATE_COMPONENT_SLICES[component]
            mask[start + component_start : start + component_stop] = True
    return np.where(mask, values, 0.0)


def _mask_physical_components(values: np.ndarray, components: Sequence[str]) -> np.ndarray:
    mask = np.zeros(values.shape[-1], dtype=bool)
    for component in components:
        component_start, component_stop = STATE_COMPONENT_SLICES[component]
        if component_stop > values.shape[-1]:
            raise ValueError(
                f"component {component!r} exceeds epsilon dimension {values.shape[-1]}"
            )
        mask[component_start:component_stop] = True
    return np.where(mask, values, 0.0)


def _split_bank_lens_metadata(lens: str) -> dict[str, Any]:
    spec = _STANDARD_SPLIT_BANK_LENS_SPECS[lens]
    x0_components = list(spec["x0"])
    epsilon_components = list(spec["epsilon"])
    return {
        "status": "available",
        "shared_initial_state": bool(x0_components),
        "shared_process_load_epsilon": bool(epsilon_components),
        "shared_initial_state_components": x0_components,
        "shared_process_load_epsilon_components": epsilon_components,
        "interpretation": (
            "stress_test_only"
            if x0_components
            else "apples_to_apples_split_bank_sidecar"
        ),
    }


def _split_bank_fairness_residuals() -> dict[str, Any]:
    return {
        "initial_observation_history": {
            "status": "partially_consistent",
            "note": (
                "The comparator replaces trial_specs.inits['mechanics.vector'] for "
                "x0 lenses. It does not separately rewrite any pre-existing "
                "observation-history buffers; the C&S LSS graph observes the "
                "perturbed initial mechanics state from the rollout start."
            ),
        },
        "gru_hidden_state_initialization": {
            "status": "stress_test_only",
            "note": (
                "GRU recurrent hidden state starts from the checkpoint/model default "
                "during eval_trials and is not conditioned on the perturbed x0."
            ),
        },
        "noise_channels": {
            "shared": ["initial_state", "process_load_epsilon"],
            "not_shared": ["sensory_noise", "command_or_motor_noise"],
            "note": (
                "Process/load epsilon is injected through TaskTrialSpec.inputs['epsilon']; "
                "sensory and command/motor noise remain graph-internal for the GRU arm "
                "and explicit zero draws for the extLQG arm in this materialization."
            ),
        },
    }


def _standard_split_bank_public(
    *,
    bank: SharedRolloutBank,
    extlqg_path: Any,
    extlqg_cost_by_lens: Mapping[str, Mapping[str, Any]],
    x0_sanity: Mapping[str, Any],
    run_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the public split-bank comparator block."""

    return {
        "status": "available",
        "lens": "standard_split_rollout_bank_full_qrf",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "bank": bank.to_json(),
        "lenses": {
            lens: _split_bank_lens_metadata(lens)
            for lens in _STANDARD_SPLIT_BANK_LENSES
        },
        "fairness_residuals": _split_bank_fairness_residuals(),
        "extlqg": {
            "status": "available",
            "parity_status": extlqg_path.parity_status,
            "n_iterations": int(extlqg_path.n_iterations),
            "cost_by_lens": {
                lens: _public_cost_summary(cost)
                for lens, cost in extlqg_cost_by_lens.items()
            },
        },
        "extlqg_x0_only_sanity_check": dict(x0_sanity),
        "runs": {
            run_id: {
                "status": run.get("status"),
                "selection_role": "audit_only_not_used_for_checkpoint_selection",
                "lenses": dict(run.get("standard_split_bank", {})),
            }
            for run_id, run in run_results.items()
        },
    }


def _summary_mean(summary: Mapping[str, Any], key: str) -> float | None:
    item = summary.get(key)
    if not isinstance(item, Mapping):
        return None
    value = item.get("mean")
    return None if value is None else float(value)


def _trial_specs_with_shared_bank(trial_specs: Any, bank: SharedRolloutBank) -> Any:
    if "mechanics.vector" not in trial_specs.inits:
        raise ValueError("shared rollout requires trial_specs.inits['mechanics.vector']")
    if "epsilon" not in trial_specs.inputs:
        raise ValueError("shared rollout requires trial_specs.inputs['epsilon']")
    updated = eqx.tree_at(
        lambda ts: ts.inits["mechanics.vector"],
        trial_specs,
        jnp.asarray(bank.initial_states, dtype=jnp.float64),
    )
    return eqx.tree_at(
        lambda ts: ts.inputs["epsilon"],
        updated,
        jnp.asarray(bank.process_epsilon, dtype=jnp.float64),
    )


def _evaluate_replicate_model_states(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> Any:
    model_arrays, model_other = eqx.partition(
        model,
        lambda leaf: _is_replicate_array(leaf, n_replicates),
    )

    def eval_one_replicate(model_array_leaves: Any, key: Any) -> Any:
        replicate_model = eqx.combine(model_array_leaves, model_other)
        return task.eval_trials(
            replicate_model,
            trial_specs,
            jr.split(key, bank_batch_size(trial_specs)),
        )

    return eqx.filter_vmap(eval_one_replicate, in_axes=(0, 0))(
        model_arrays,
        jr.split(jr.PRNGKey(seed), n_replicates),
    )


def bank_batch_size(trial_specs: Any) -> int:
    for initial_state in trial_specs.inits.values():
        shape = getattr(initial_state, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[0])
    target = trial_specs.inputs.get("effector_target")
    if target is not None and hasattr(target, "pos"):
        return int(target.pos.shape[0])
    raise ValueError("could not infer shared rollout bank size")


def _is_replicate_array(leaf: Any, n_replicates: int) -> bool:
    shape = getattr(leaf, "shape", None)
    return bool(shape) and int(shape[0]) == n_replicates


def _state_term_groups(state_dim: int) -> dict[str, list[int]]:
    if state_dim % 8 != 0:
        raise ValueError(f"state dimension {state_dim} is not divisible by 8")
    groups = {
        "running_state": [],
        "force_filter_state": [],
        "disturbance_integrator_state": [],
    }
    for start in range(0, state_dim, 8):
        groups["running_state"].extend(range(start, start + 4))
        groups["force_filter_state"].extend(range(start + 4, start + 6))
        groups["disturbance_integrator_state"].extend(range(start + 6, start + 8))
    return groups


def _state_quadratic_group(values: np.ndarray, matrices: np.ndarray, indices: Sequence[int]) -> Any:
    idx = np.asarray(indices, dtype=np.int64)
    selected = values[..., idx]
    selected_matrices = matrices[:, idx[:, None], idx]
    return np.sum(np.einsum("...ti,tij,...tj->...t", selected, selected_matrices, selected), axis=-1)


def _terminal_quadratic_group(values: np.ndarray, matrix: np.ndarray, indices: Sequence[int]) -> Any:
    idx = np.asarray(indices, dtype=np.int64)
    selected = values[..., idx]
    selected_matrix = matrix[idx[:, None], idx]
    return np.einsum("...i,ij,...j->...", selected, selected_matrix, selected)


def _goal_centered_vectors(values: Any, *, target_pos: Any) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    target = np.asarray(target_pos, dtype=np.float64)
    if result.shape[-1] % 8 != 0:
        raise ValueError(f"state dimension {result.shape[-1]} is not divisible by 8")
    for start in range(0, result.shape[-1], 8):
        result[..., start : start + 2] -= target
    return result


def _summary_with_values(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        **_summary_stats(array),
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def _summary_stats(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    flat = array.reshape(-1)
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
    }


def _public_cost_summary(cost: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "status": cost.get("status"),
        "lens": cost.get("lens"),
        "basis": dict(cost.get("basis", {})),
    }
    for key in FULL_QRF_TERM_NAMES:
        source_key = _cost_source_key(key)
        result[key] = _without_values(_expect_mapping(cost[source_key]))
    result["total"] = _without_values(_expect_mapping(cost["total"]))
    result["term_sum_delta"] = dict(_expect_mapping(cost["term_sum_delta"]))
    return result


def _without_values(summary: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(summary)
    payload.pop("values", None)
    return payload


def _cost_source_key(term_name: str) -> str:
    return {
        "running_state_q": "running_state",
        "terminal_state_q_f": "terminal_state",
        "command_r": "command_control",
        "force_filter_state": "force_filter_state",
        "disturbance_integrator_state": "disturbance_integrator_state",
    }[term_name]


def _cost_comparison(gru_cost: Mapping[str, Any], extlqg_cost: Mapping[str, Any]) -> dict[str, Any]:
    comparison = {"status": "available", "terms": {}}
    terms = _expect_mapping(comparison["terms"])
    for term in ("total", *FULL_QRF_TERM_NAMES):
        source_key = "total" if term == "total" else _cost_source_key(term)
        gru_mean = float(_expect_mapping(gru_cost[source_key])["mean"])
        extlqg_mean = float(_expect_mapping(extlqg_cost[source_key])["mean"])
        terms[term] = {
            "gru_mean": gru_mean,
            "extlqg_mean": extlqg_mean,
            "delta_mean": gru_mean - extlqg_mean,
            "ratio_to_extlqg": (
                None if abs(extlqg_mean) <= 1e-300 else gru_mean / extlqg_mean
            ),
        }
    return comparison


def _render_shared_rollout_markdown(shared_rollout: Mapping[str, Any]) -> list[str]:
    status = shared_rollout.get("status", "not_available")
    lines = ["## Shared-rollout comparator", ""]
    if status != "available":
        lines.extend([f"Status: `{status}` - {shared_rollout.get('reason', 'not available')}", ""])
        return lines
    bank = _expect_mapping(shared_rollout["bank"])
    noise = _expect_mapping(shared_rollout["noise_comparability"])
    lines.extend(
        [
            (
                f"Bank `{bank['bank_id']}` uses {bank['n_trials']} trials, seed "
                f"`{bank['seed']}`, shared initial states, and shared process/load epsilon."
            ),
            "",
            f"Limitation: {noise['limitation']}",
            "",
            "| run | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_id, run in sorted(_expect_mapping(shared_rollout["runs"]).items()):
        comparison = _expect_mapping(_expect_mapping(run)["gru_vs_extlqg"])
        terms = _expect_mapping(comparison["terms"])
        total = _expect_mapping(terms["total"])
        lines.append(
            "| "
            f"`{run_id}` | "
            f"{_fmt(total['gru_mean'])} | "
            f"{_fmt(total['extlqg_mean'])} | "
            f"{_fmt(total['ratio_to_extlqg'])} | "
            f"{_fmt(_expect_mapping(terms['running_state_q'])['ratio_to_extlqg'])} | "
            f"{_fmt(_expect_mapping(terms['terminal_state_q_f'])['ratio_to_extlqg'])} | "
            f"{_fmt(_expect_mapping(terms['command_r'])['ratio_to_extlqg'])} | "
            f"{_fmt(_expect_mapping(terms['force_filter_state'])['ratio_to_extlqg'])} | "
            f"{_fmt(_expect_mapping(terms['disturbance_integrator_state'])['ratio_to_extlqg'])} |"
        )
    lines.append("")
    return lines


def _render_split_bank_markdown(split_bank: Mapping[str, Any]) -> list[str]:
    status = split_bank.get("status", "not_available")
    lines = ["## Standard split-bank comparator", ""]
    if status != "available":
        lines.extend([f"Status: `{status}` - {split_bank.get('reason', 'not available')}", ""])
        return lines
    lines.extend(
        [
            "| run | lens | GRU total | extLQG total | GRU/extLQG | running | terminal | command | force/filter | integrator |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_id, run in sorted(_expect_mapping(split_bank["runs"]).items()):
        lenses = _expect_mapping(_expect_mapping(run)["lenses"])
        for lens in _STANDARD_SPLIT_BANK_LENSES:
            lens_result = _expect_mapping(lenses[lens])
            comparison = _expect_mapping(lens_result["gru_vs_extlqg"])
            terms = _expect_mapping(comparison["terms"])
            total = _expect_mapping(terms["total"])
            lines.append(
                "| "
                f"`{run_id}` | "
                f"`{lens}` | "
                f"{_fmt(total['gru_mean'])} | "
                f"{_fmt(total['extlqg_mean'])} | "
                f"{_fmt(total['ratio_to_extlqg'])} | "
                f"{_fmt(_expect_mapping(terms['running_state_q'])['ratio_to_extlqg'])} | "
                f"{_fmt(_expect_mapping(terms['terminal_state_q_f'])['ratio_to_extlqg'])} | "
                f"{_fmt(_expect_mapping(terms['command_r'])['ratio_to_extlqg'])} | "
                f"{_fmt(_expect_mapping(terms['force_filter_state'])['ratio_to_extlqg'])} | "
                f"{_fmt(_expect_mapping(terms['disturbance_integrator_state'])['ratio_to_extlqg'])} |"
            )
    residuals = _expect_mapping(split_bank.get("fairness_residuals", {}))
    if residuals:
        lines.extend(["", "Fairness/residual notes:", ""])
        for key in (
            "initial_observation_history",
            "gru_hidden_state_initialization",
            "noise_channels",
        ):
            item = _expect_mapping(residuals.get(key, {}))
            lines.append(f"- `{key}`: {item.get('status', 'declared')} - {item.get('note', '')}")
    lines.append("")
    return lines


def _float_from_selection(selection: Any, key: str) -> float:
    selection_map = _expect_mapping(selection)
    return float(selection_map[key])


def _expect_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping, found {type(value).__name__}")
    return value


def _expect_sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected sequence, found {type(value).__name__}")
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "not_comparable"
    return f"{float(value):.8g}"


def _load_checkpoint_selection(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    checkpoint_selection = manifest.get("checkpoint_selection", manifest)
    return _expect_mapping(checkpoint_selection)


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for materializing a sidecar from tracked manifests."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--generated-by", default="python -m rlrmp.analysis.pipelines.objective_comparator")
    parser.add_argument("--extlqg-deterministic", required=True, type=float)
    parser.add_argument("--extlqg-initial-covariance", required=True, type=float)
    parser.add_argument("--extlqg-accumulated-noise", required=True, type=float)
    parser.add_argument("--extlqg-total", required=True, type=float)
    parser.add_argument("--extlqg-provenance", required=True)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sidecar = build_objective_comparator_sidecar(
        issue=args.issue,
        source_manifest=str(args.manifest),
        checkpoint_selection=_load_checkpoint_selection(manifest),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=args.extlqg_deterministic,
            initial_covariance_trace=args.extlqg_initial_covariance,
            accumulated_noise_scalar=args.extlqg_accumulated_noise,
            total_expected_cost=args.extlqg_total,
            provenance=args.extlqg_provenance,
        ),
        scope=args.scope,
        generated_by=args.generated_by,
    )
    write_objective_comparator_sidecar(
        sidecar,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )


if __name__ == "__main__":
    main()
