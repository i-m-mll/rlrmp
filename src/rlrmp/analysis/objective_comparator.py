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


SCHEMA_VERSION = "rlrmp.objective_comparator_sidecar.v4"
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
                "source": "rlrmp.modules.training.part2._sample_cs_lss_process_epsilon",
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

    from rlrmp.analysis.cs_game_card import build_canonical_game
    from rlrmp.analysis.cs_released_simulation import _default_output_feedback_initial_state
    from rlrmp.analysis.output_feedback import OutputFeedbackConfig
    from rlrmp.modules.training.part2 import _cs_lss_process_epsilon_factor

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
) -> dict[str, Any]:
    """Build a JSON sidecar from validation-selected checkpoint records."""

    runs = checkpoint_selection.get("runs")
    if not isinstance(runs, Mapping):
        raise ValueError("checkpoint_selection must contain a mapping at key 'runs'")

    metadata_by_id = run_metadata_by_id or {}
    shared_rollout = dict(shared_rollout_comparator or DEFAULT_SHARED_ROLLOUT_STATUS)
    sidecar_rows = [
        _build_run_row(
            run_id=str(run_id),
            selections=_expect_sequence(selections),
            extlqg=extlqg,
            run_metadata=metadata_by_id.get(str(run_id)),
            shared_rollout_run=_shared_rollout_run(shared_rollout, str(run_id)),
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
            "label": "validation_selected_per_replicate",
            "source": checkpoint_selection.get("schema_version"),
            "selection_policy": checkpoint_selection.get("selection_policy"),
            "caveat": (
                "Checkpoint selection is inherited from the validation-selected "
                "GRU manifest. Analytical action, I/O, and extLQG comparator "
                "metrics are audit-only and are not used for checkpoint selection."
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
            },
        },
        "extlqg_decomposition": extlqg.to_json(),
        "same_noise_bank_monte_carlo": dict(
            same_noise_bank_monte_carlo or _same_noise_status_from_shared_rollout(shared_rollout)
        ),
        "per_term_realized_scoring": dict(per_term_realized_scoring or DEFAULT_PER_TERM_STATUS),
        "shared_rollout_comparator": shared_rollout,
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
                "the shared-rollout block is an audit-only post-hoc rescore and "
                "is not used for checkpoint selection."
            ),
        ],
    }


def shared_full_qrf_cost_summary(
    *,
    states: Any,
    commands: Any,
    initial_states: Any,
) -> dict[str, Any]:
    """Score realized full-Q/R/Q_f rollout costs with standard sidecar terms."""

    from rlrmp.analysis.cs_game_card import TARGET_POS, build_canonical_game

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
    x_pre = _goal_centered_vectors(x_pre, target_pos=TARGET_POS)
    x_terminal = _goal_centered_vectors(state_array[..., -1, :], target_pos=TARGET_POS)
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
            "state_transform": "subtract TARGET_POS from each physical delay block x/y",
            "schedule_source": "rlrmp.analysis.cs_game_card.build_canonical_game",
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
    """Evaluate extLQG and validation-selected GRUs on one shared bank."""

    from feedbax.types import TreeNamespace, dict_to_namespace
    from rlrmp.analysis.cs_game_card import build_canonical_game
    from rlrmp.analysis.cs_gru_standard_materialization import normalize_gru_hps
    from rlrmp.analysis.cs_released_simulation import (
        build_extlqg_comparator_path,
        default_cs_noise_covariances,
        simulate_lqg_released_forward,
        zero_forward_noise_draws,
    )
    from rlrmp.analysis.gru_checkpoint_selection import load_validation_selected_checkpoint_model
    from rlrmp.analysis.gru_pilot_figures import repeat_single_validation_trial, resolve_run_inputs
    from rlrmp.analysis.output_feedback import OutputFeedbackConfig
    from rlrmp.modules.training.part2 import setup_task_model_pair

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
    ext_states = []
    ext_commands = []
    for initial_state, epsilon in zip(bank.initial_states, bank.process_epsilon, strict=True):
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
        ext_states.append(np.asarray(rollout.x[1:], dtype=np.float64))
        ext_commands.append(np.asarray(rollout.u_command, dtype=np.float64))
    extlqg_cost = shared_full_qrf_cost_summary(
        states=np.stack(ext_states, axis=0),
        commands=np.stack(ext_commands, axis=0),
        initial_states=bank.initial_states,
    )

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
            repo_root=repo_root,
        )
        trial_specs = _trial_specs_with_shared_bank(
            repeat_single_validation_trial(pair.task.validation_trials, bank.n_trials),
            bank,
        )
        states = _evaluate_replicate_model_states(
            model=model,
            task=pair.task,
            trial_specs=trial_specs,
            n_replicates=n_replicates,
            seed=seed,
        )
        gru_cost = shared_full_qrf_cost_summary(
            states=np.asarray(states.mechanics.vector, dtype=np.float64),
            commands=np.asarray(states.net.output, dtype=np.float64),
            initial_states=bank.initial_states,
        )
        run_results[run.run_id] = {
            "status": "available",
            "checkpoint_policy": "validation_selected_per_replicate",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "checkpoint_selection": [
                selection.to_json(repo_root=repo_root) for selection in checkpoint_selection
            ],
            "n_replicates": n_replicates,
            "gru_cost": _public_cost_summary(gru_cost),
            "extlqg_cost": _public_cost_summary(extlqg_cost),
            "gru_vs_extlqg": _cost_comparison(gru_cost, extlqg_cost),
        }
    return {
        "status": "available",
        "lens": "shared_rollout_full_qrf",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "bank": bank.to_json(),
        "extlqg": {
            "status": "available",
            "parity_status": extlqg_path.parity_status,
            "n_iterations": int(extlqg_path.n_iterations),
            "expected_cost": extlqg_path.expected_cost,
            "cost": _public_cost_summary(extlqg_cost),
        },
        "noise_comparability": {
            "shared_channels": ["initial_state", "process_load_epsilon"],
            "not_shared_channels": ["sensory_noise", "command_or_motor_noise"],
            "limitation": (
                "This is a shared initial-state plus process/load epsilon comparator. "
                "Sensory and command/motor noise are explicitly not claimed as shared."
            ),
        },
        "runs": run_results,
        "source_checkpoint_manifest_schema": checkpoint_manifest.get("schema_version"),
    }


def render_objective_comparator_markdown(sidecar: Mapping[str, Any]) -> str:
    """Render a compact Markdown companion for an objective-comparator sidecar."""

    decomposition = _expect_mapping(sidecar["extlqg_decomposition"])
    rows = _expect_sequence(sidecar["rows"])
    same_noise = _expect_mapping(sidecar["same_noise_bank_monte_carlo"])
    per_term = _expect_mapping(sidecar["per_term_realized_scoring"])
    shared_rollout = _expect_mapping(sidecar.get("shared_rollout_comparator", {}))
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

    from rlrmp.analysis.cs_game_card import (
        OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
        materialize_reference,
    )
    from rlrmp.analysis.cs_released_simulation import (
        _compute_ext_kalman,
        _compute_ofc,
        _default_output_feedback_initial_state,
        default_cs_noise_covariances,
    )
    from rlrmp.analysis.output_feedback import (
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
    if checkpoint_policy != "validation_selected_per_replicate":
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
            "validation-selected checkpoints for C&S GRU runs: "
            + ", ".join(str(run_id) for run_id in run_ids)
        ),
        generated_by="rlrmp.analysis.objective_comparator.materialize_gru_objective_comparator_sidecar",
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
    return {
        "status": "materialized",
        "schema_version": sidecar["schema_version"],
        "n_rows": len(sidecar["rows"]),
        "extlqg_deterministic_full_qrf": extlqg.deterministic_initial_state,
        "extlqg_total_expected_cost": extlqg.expected_cost,
    }


def _build_run_row(
    *,
    run_id: str,
    selections: Sequence[Any],
    extlqg: ExtLQGCostDecomposition,
    run_metadata: Mapping[str, Any] | None = None,
    shared_rollout_run: Mapping[str, Any] | None = None,
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
        "checkpoint_policy": "validation_selected_per_replicate",
        "n_replicates": len(selections),
        "training_objective": metadata,
        "comparability": comparability,
        "gru_realized_lens": "gru_validation_selected_realized_full_qrf",
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
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
    }


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
    parser.add_argument("--generated-by", default="python -m rlrmp.analysis.objective_comparator")
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
