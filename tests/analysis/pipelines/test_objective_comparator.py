"""Tests for objective-comparator sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import rlrmp.analysis.pipelines.objective_comparator as objective_comparator
from rlrmp.disturbance import PLANT_INTERVENOR_LABEL
from rlrmp.eval.ensemble import eval_ensemble_on_trials
from rlrmp.analysis.pipelines.objective_comparator import (
    SCHEMA_VERSION,
    ExtLQGCostDecomposition,
    SharedRolloutBank,
    build_objective_comparator_sidecar,
    extlqg_x0_only_sanity_check,
    load_run_objective_metadata,
    materialize_gru_objective_comparator_sidecar,
    render_objective_comparator_markdown,
    shared_full_qrf_cost_summary,
    write_objective_comparator_sidecar,
)


class _FakeNamespace(eqx.Module):
    vector: Any | None = None
    output: Any | None = None


class _FakeStates(eqx.Module):
    mechanics: _FakeNamespace
    net: _FakeNamespace


class _FakeModel(eqx.Module):
    gain: Any


class _FakeTrialSpecs(eqx.Module):
    inits: dict[str, Any]
    inputs: dict[str, Any]


class _FakeTask:
    def eval_trials(self, model: _FakeModel, trial_specs: _FakeTrialSpecs, keys: Any) -> Any:
        n_trials = int(trial_specs.inits["mechanics.vector"].shape[0])
        state_dim = int(trial_specs.inits["mechanics.vector"].shape[-1])
        horizon = 60
        states = jnp.broadcast_to(
            trial_specs.inits["mechanics.vector"][:, None, :],
            (n_trials, horizon, state_dim),
        )
        key_signal = jnp.mod(jnp.sum(keys, axis=-1), 997).astype(jnp.float32)
        epsilon_signal = jnp.sum(trial_specs.inputs["epsilon"], axis=(1, 2))
        command_signal = model.gain + 0.0001 * key_signal + 0.001 * epsilon_signal
        commands = jnp.broadcast_to(command_signal[:, None, None], (n_trials, horizon, 2))
        return _FakeStates(
            mechanics=_FakeNamespace(vector=states),
            net=_FakeNamespace(output=commands),
        )


def _legacy_extlqg_cost_decomposition_for_test() -> ExtLQGCostDecomposition:
    """Mirror the pre-dedup objective-comparator loop for equivalence testing."""

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
        provenance=f"legacy objective-comparator inline loop, {iteration} iterations",
    )


def _checkpoint_selection() -> dict[str, object]:
    return {
        "schema_version": "feedbax.manifest.v1",
        "selection_policy": "validation-selected test policy",
        "runs": {
            "run_b": [
                {
                    "replicate": 0,
                    "scoring_validation_objective": 44.0,
                    "best_logged_validation_objective": 43.0,
                }
            ],
            "run_a": [
                {
                    "replicate": 0,
                    "scoring_validation_objective": 10.0,
                    "best_logged_validation_objective": 9.0,
                },
                {
                    "replicate": 1,
                    "scoring_validation_objective": 14.0,
                    "best_logged_validation_objective": 11.0,
                },
            ],
        },
    }


def _full_qrf_run_metadata() -> dict[str, object]:
    return {
        "status": "available",
        "loss_objective": "full_analytical_qrf",
        "objective_profile": "full_analytical_qrf",
        "full_qrf_lens": {
            "status": "available",
            "active_terms": ["control_r", "state_running_q", "terminal_q_f"],
        },
    }


def _unit_split_lenses() -> dict[str, object]:
    return {
        lens: {
            "status": "available",
            "gru_vs_extlqg": {"terms": _unit_comparator_terms()},
        }
        for lens in objective_comparator._STANDARD_SPLIT_BANK_LENSES
    }


def _unit_comparator_terms() -> dict[str, object]:
    return {
        term: {
            "gru_mean": 2.0,
            "extlqg_mean": 1.0,
            "delta_mean": 1.0,
            "ratio_to_extlqg": 2.0,
        }
        for term in ("total", *objective_comparator.FULL_QRF_TERM_NAMES)
    }


def test_extlqg_decomposition_reports_component_sum_and_declared_total() -> None:
    decomposition = ExtLQGCostDecomposition(
        deterministic_initial_state=4.0,
        initial_covariance_trace=3.0,
        accumulated_noise_scalar=2.0,
        total_expected_cost=9.5,
        provenance="unit-test",
    )

    payload = decomposition.to_json()

    assert payload["component_sum"] == 9.0
    assert payload["total_expected_cost"] == 9.5
    assert payload["component_sum_delta"] == 0.5
    assert payload["comparable_scalar"] == 4.0
    assert payload["comparable_scalar_lens"] == "extlqg_deterministic_initial_state_full_qrf"


def test_build_objective_comparator_sidecar_uses_deterministic_comparator_lens() -> None:
    shared_rollout = {
        "status": "available",
        "lens": "shared_rollout_full_qrf",
        "interpretation": "stress_test_only",
        "selection_role": "audit_only_not_used_for_checkpoint_selection",
        "bank": {
            "bank_id": "unit-bank",
            "seed": 7,
            "n_trials": 2,
        },
        "noise_comparability": {
            "limitation": "unit limitation",
        },
        "runs": {
            "run_a": {
                "status": "available",
                "gru_vs_extlqg": {"terms": {"total": {"ratio_to_extlqg": 1.25}}},
            }
        },
        "standard_split_bank_comparator": {
            "status": "available",
            "lens": "standard_split_rollout_bank_full_qrf",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
            "lenses": {
                "deterministic_nominal": {"status": "available"},
                "x0_position_only": {"status": "available"},
                "x0_velocity_only": {"status": "available"},
                "x0_force_filter_only": {"status": "available"},
                "x0_disturbance_integrator_only": {"status": "available"},
                "process_epsilon_position_only": {"status": "available"},
                "process_epsilon_velocity_only": {"status": "available"},
                "process_epsilon_force_filter_only": {"status": "available"},
                "process_epsilon_integrator_only": {"status": "available"},
                "x0_position_velocity": {"status": "available"},
                "x0_plus_epsilon": {
                    "status": "available",
                    "interpretation": "stress_test_only",
                },
            },
            "extlqg_x0_only_sanity_check": {
                "status": "pass",
                "expected_cost_wording_allowed": True,
            },
            "runs": {
                "run_a": {
                    "status": "available",
                    "lenses": _unit_split_lenses(),
                },
            },
        },
    }
    sidecar = build_objective_comparator_sidecar(
        issue="abc1234",
        source_manifest="source.json",
        checkpoint_selection=_checkpoint_selection(),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
        scope="unit scope",
        generated_by="unit",
        run_metadata_by_id={
            "run_a": _full_qrf_run_metadata(),
            "run_b": _full_qrf_run_metadata(),
        },
        shared_rollout_comparator=shared_rollout,
    )

    assert sidecar["schema_version"] == SCHEMA_VERSION
    assert sidecar["extlqg_decomposition"]["total_expected_cost"] == 44.0
    assert sidecar["same_noise_bank_monte_carlo"]["status"] == "available_with_limitations"
    assert sidecar["same_noise_bank_monte_carlo"]["lens"] == "shared_rollout_full_qrf"
    assert sidecar["per_term_realized_scoring"]["status"] == "not_implemented"
    assert (
        sidecar["objective_lenses"]["extlqg_covariance_inclusive_expected_cost"]["noise_bank"]
        == "analytical_covariance_expectation_not_realized_validation_bank"
    )

    first_row = sidecar["rows"][0]
    assert first_row["run_id"] == "run_a"
    assert first_row["comparability"]["status"] == "comparable_deterministic_full_qrf"
    assert first_row["gru_mean_selected_validation_full_qrf"] == 12.0
    assert first_row["selected_to_extlqg_deterministic_ratio"] == 1.0
    assert first_row["selected_to_extlqg_total_ratio_not_apples_to_apples"] == 12.0 / 44.0
    assert first_row["extlqg_comparable_lens"] == "extlqg_deterministic_initial_state_full_qrf"
    assert first_row["per_term_realized_scoring"]["status"] == "not_implemented"
    assert sidecar["shared_rollout_comparator"]["status"] == "available"
    assert sidecar["shared_rollout_comparator"]["interpretation"] == "stress_test_only"
    assert sidecar["standard_split_bank_comparator"]["status"] == "available"
    assert tuple(sidecar["standard_split_bank_comparator"]["lenses"]) == (
        "deterministic_nominal",
        "x0_position_only",
        "x0_velocity_only",
        "x0_force_filter_only",
        "x0_disturbance_integrator_only",
        "process_epsilon_position_only",
        "process_epsilon_velocity_only",
        "process_epsilon_force_filter_only",
        "process_epsilon_integrator_only",
        "x0_position_velocity",
        "x0_plus_epsilon",
    )
    assert (
        sidecar["standard_split_bank_comparator"]["lenses"]["x0_plus_epsilon"][
            "interpretation"
        ]
        == "stress_test_only"
    )
    assert (
        sidecar["standard_split_bank_comparator"]["extlqg_x0_only_sanity_check"][
            "expected_cost_wording_allowed"
        ]
        is True
    )
    assert first_row["shared_rollout_comparator"]["status"] == "available"
    assert first_row["standard_split_bank_comparator"]["status"] == "available"
    assert sidecar["rows"][1]["shared_rollout_comparator"]["status"] == "not_available"


def test_shared_rollout_bank_serializes_declared_shared_channels() -> None:
    bank = SharedRolloutBank(
        bank_id="unit-bank",
        seed=123,
        initial_states=np.zeros((3, 48), dtype=np.float64),
        process_epsilon=np.zeros((3, 60, 8), dtype=np.float64),
        initial_covariance=0.01,
    )

    payload = bank.to_json()

    assert payload["n_trials"] == 3
    assert payload["initial_state"]["status"] == "shared"
    assert payload["process_load_epsilon"]["status"] == "shared"
    assert payload["sensory_noise"]["status"] == "not_shared"
    assert payload["command_or_motor_noise"]["status"] == "not_shared"


def test_split_bank_inputs_mask_x0_and_process_components() -> None:
    bank = SharedRolloutBank(
        bank_id="unit-bank",
        seed=123,
        initial_states=np.arange(16, dtype=np.float64).reshape(1, 16),
        process_epsilon=np.arange(8, dtype=np.float64).reshape(1, 1, 8),
        initial_covariance=0.01,
    )
    default_initial = np.zeros((1, 16), dtype=np.float64)

    lens_inputs = objective_comparator._split_bank_inputs(
        bank=bank,
        default_initial=default_initial,
    )

    np.testing.assert_allclose(
        lens_inputs["x0_position_only"]["initial_states"],
        np.array([[0, 1, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0, 0, 0, 0, 0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        lens_inputs["x0_velocity_only"]["initial_states"],
        np.array([[0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 10, 11, 0, 0, 0, 0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        lens_inputs["process_epsilon_force_filter_only"]["process_epsilon"],
        np.array([[[0, 0, 0, 0, 4, 5, 0, 0]]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        lens_inputs["x0_plus_epsilon"]["initial_states"],
        bank.initial_states,
    )
    np.testing.assert_allclose(
        lens_inputs["x0_plus_epsilon"]["process_epsilon"],
        bank.process_epsilon,
    )


def test_gru_split_bank_costs_match_legacy_per_lens_evaluation() -> None:
    bank = SharedRolloutBank(
        bank_id="unit-bank",
        seed=123,
        initial_states=np.arange(48, dtype=np.float64).reshape(1, 48) * 0.01,
        process_epsilon=np.arange(480, dtype=np.float64).reshape(1, 60, 8) * 0.001,
        initial_covariance=0.01,
    )
    model = _FakeModel(gain=jnp.array([0.25, 0.75], dtype=jnp.float64))
    task = _FakeTask()
    base_trial_specs = _FakeTrialSpecs(
        inits={"mechanics.vector": jnp.zeros((1, 48), dtype=jnp.float64)},
        inputs={"epsilon": jnp.zeros((1, 60, 8), dtype=jnp.float64)},
    )

    # Warm the mocked JAX path so the parity assertion is about serial semantics,
    # not first-trace behavior in the fake task.
    _legacy_gru_split_bank_costs(
        model=model,
        task=task,
        base_trial_specs=base_trial_specs,
        bank=bank,
        n_replicates=2,
        seed=17,
    )
    objective_comparator._gru_split_bank_costs(
        model=model,
        task=task,
        base_trial_specs=base_trial_specs,
        bank=bank,
        n_replicates=2,
        seed=17,
    )

    expected = _legacy_gru_split_bank_costs(
        model=model,
        task=task,
        base_trial_specs=base_trial_specs,
        bank=bank,
        n_replicates=2,
        seed=17,
    )
    actual = objective_comparator._gru_split_bank_costs(
        model=model,
        task=task,
        base_trial_specs=base_trial_specs,
        bank=bank,
        n_replicates=2,
        seed=17,
    )

    assert tuple(actual) == tuple(objective_comparator._STANDARD_SPLIT_BANK_LENSES)
    for lens in objective_comparator._STANDARD_SPLIT_BANK_LENSES:
        for term in ("total", "command_control", "running_state"):
            assert actual[lens][term]["shape"] == expected[lens][term]["shape"]
            np.testing.assert_allclose(
                actual[lens][term]["values"],
                expected[lens][term]["values"],
            )


def test_shared_full_qrf_cost_summary_decomposes_zero_rollout() -> None:
    states = np.zeros((2, 60, 48), dtype=np.float64)
    commands = np.zeros((2, 60, 2), dtype=np.float64)
    initial_states = np.zeros((2, 48), dtype=np.float64)

    summary = shared_full_qrf_cost_summary(
        states=states,
        commands=commands,
        initial_states=initial_states,
    )

    total = summary["total"]["mean"]
    term_sum = sum(
        summary[key]["mean"]
        for key in (
            "running_state",
            "terminal_state",
            "command_control",
            "force_filter_state",
            "disturbance_integrator_state",
        )
    )
    assert summary["status"] == "available"
    assert summary["term_sum_delta"]["min"] == 0.0
    assert summary["term_sum_delta"]["max"] == 0.0
    np.testing.assert_allclose(
        total,
        term_sum,
        rtol=np.finfo(np.float32).eps,
        atol=0.0,
    )
    assert summary["command_control"]["mean"] == 0.0
    assert summary["total"]["shape"] == [2]


def test_shared_full_qrf_cost_summary_matches_numpy_and_jax_inputs() -> None:
    states = np.zeros((1, 60, 48), dtype=np.float64)
    states[..., :, 0] = 0.01
    commands = np.ones((1, 60, 2), dtype=np.float64)
    initial_states = np.zeros((1, 48), dtype=np.float64)

    numpy_summary = shared_full_qrf_cost_summary(
        states=states,
        commands=commands,
        initial_states=initial_states,
        state_basis="target_centered",
    )
    jax_summary = shared_full_qrf_cost_summary(
        states=jnp.asarray(states),
        commands=jnp.asarray(commands),
        initial_states=jnp.asarray(initial_states),
        state_basis="target_centered",
    )

    for key in (
        "total",
        "running_state",
        "terminal_state",
        "command_control",
        "force_filter_state",
        "disturbance_integrator_state",
    ):
        assert jax_summary[key]["shape"] == numpy_summary[key]["shape"]
        np.testing.assert_allclose(jax_summary[key]["values"], numpy_summary[key]["values"])
        np.testing.assert_allclose(jax_summary[key]["mean"], numpy_summary[key]["mean"])


def test_shared_full_qrf_cost_summary_declares_state_basis_contract() -> None:
    from rlrmp.analysis.math.cs_game_card import TARGET_POS

    states = np.zeros((1, 60, 48), dtype=np.float64)
    commands = np.zeros((1, 60, 2), dtype=np.float64)
    initial_states = np.zeros((1, 48), dtype=np.float64)

    target_centered = shared_full_qrf_cost_summary(
        states=states,
        commands=commands,
        initial_states=initial_states,
        state_basis="target_centered",
    )
    absolute_workspace = shared_full_qrf_cost_summary(
        states=states,
        commands=commands,
        initial_states=initial_states,
        state_basis="absolute_workspace",
    )
    absolute_at_target = shared_full_qrf_cost_summary(
        states=_states_at_target(states, TARGET_POS),
        commands=commands,
        initial_states=_states_at_target(initial_states, TARGET_POS),
        state_basis="absolute_workspace",
    )

    assert target_centered["basis"]["state_transform"] == "none; states are already target-centered"
    assert target_centered["total"]["mean"] == 0.0
    assert absolute_workspace["basis"]["state_basis"] == "absolute_workspace"
    assert absolute_workspace["total"]["mean"] > 0.0
    assert absolute_at_target["total"]["mean"] == 0.0


def test_extlqg_deterministic_nominal_realized_cost_uses_target_centered_basis() -> None:
    import jax.numpy as jnp

    from rlrmp.analysis.math.cs_game_card import build_canonical_game
    from rlrmp.analysis.math.cs_released_simulation import (
        _default_output_feedback_initial_state,
        build_extlqg_comparator_path,
        default_cs_noise_covariances,
        simulate_lqg_released_forward,
        zero_forward_noise_draws,
    )
    from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig

    plant, schedule = build_canonical_game()
    config = OutputFeedbackConfig()
    covariances = default_cs_noise_covariances(plant, config)
    comparator_path = build_extlqg_comparator_path(
        plant,
        jnp.zeros((schedule.T, plant.m_u, plant.n), dtype=jnp.float64),
        covariances,
        schedule=schedule,
        config=config,
    )
    x0 = _default_output_feedback_initial_state(plant, config)
    rollout = simulate_lqg_released_forward(
        plant,
        comparator_path.controller_gains,
        x0,
        draws=zero_forward_noise_draws(T=schedule.T, plant=plant, config=config),
        covariances=covariances,
        estimator_gains=comparator_path.estimator_gains,
        adversary_epsilon=jnp.zeros((schedule.T, plant.m_w), dtype=jnp.float64),
        config=config,
    )
    summary = shared_full_qrf_cost_summary(
        states=np.asarray(rollout.x[1:], dtype=np.float64)[None, ...],
        commands=np.asarray(rollout.u_command, dtype=np.float64)[None, ...],
        initial_states=np.asarray(x0, dtype=np.float64)[None, :],
        state_basis="target_centered",
    )

    realized = summary["total"]["mean"]

    assert summary["basis"]["state_basis"] == "target_centered"
    assert np.isclose(realized, 4368.510655149214, rtol=2e-3)
    assert realized < 10_000.0


def _states_at_target(values: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    for start in range(0, result.shape[-1], 8):
        result[..., start : start + 2] = target_pos
    return result


def _legacy_gru_split_bank_costs(
    *,
    model: Any,
    task: Any,
    base_trial_specs: Any,
    bank: SharedRolloutBank,
    n_replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    default_initial = np.asarray(base_trial_specs.inits["mechanics.vector"], dtype=np.float64)
    if default_initial.ndim == 1:
        default_initial = np.broadcast_to(default_initial, bank.initial_states.shape)
    else:
        default_initial = np.broadcast_to(default_initial[:1], bank.initial_states.shape)
    lens_inputs = objective_comparator._split_bank_inputs(
        bank=bank,
        default_initial=default_initial,
    )
    costs: dict[str, dict[str, Any]] = {}
    for lens in objective_comparator._STANDARD_SPLIT_BANK_LENSES:
        lens_bank = SharedRolloutBank(
            bank_id=f"{bank.bank_id}:{lens}",
            seed=bank.seed,
            initial_states=np.asarray(lens_inputs[lens]["initial_states"], dtype=np.float64),
            process_epsilon=np.asarray(lens_inputs[lens]["process_epsilon"], dtype=np.float64),
            initial_covariance=bank.initial_covariance,
        )
        trial_specs = objective_comparator._trial_specs_with_shared_bank(
            base_trial_specs,
            lens_bank,
        )
        states = _legacy_evaluate_replicate_model_states(
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


def _legacy_evaluate_replicate_model_states(
    *,
    model: Any,
    task: Any,
    trial_specs: Any,
    n_replicates: int,
    seed: int,
) -> Any:
    n_trials = objective_comparator.bank_batch_size(trial_specs)
    adapted_specs = SimpleNamespace(
        inits=trial_specs.inits,
        inputs=trial_specs.inputs,
        intervene={
            PLANT_INTERVENOR_LABEL: SimpleNamespace(scale=jnp.zeros((n_trials,)))
        },
    )
    return eval_ensemble_on_trials(
        task,
        model,
        adapted_specs,
        key=jr.PRNGKey(seed),
        n_replicates=n_replicates,
    )


def test_extlqg_x0_only_sanity_check_reports_pass_and_warning() -> None:
    extlqg = ExtLQGCostDecomposition(
        deterministic_initial_state=10.0,
        initial_covariance_trace=2.0,
        accumulated_noise_scalar=7.0,
        provenance="unit-test",
    )
    passing = extlqg_x0_only_sanity_check(
        x0_only_cost={"total": {"mean": 12.1}},
        extlqg=extlqg,
        relative_tolerance=0.02,
    )
    warning = extlqg_x0_only_sanity_check(
        x0_only_cost={"total": {"mean": 15.0}},
        extlqg=extlqg,
        relative_tolerance=0.02,
    )

    assert passing["status"] == "pass"
    assert passing["expected_deterministic_plus_initial_covariance_trace"] == 12.0
    assert passing["expected_cost_wording_allowed"] is True
    assert warning["status"] == "warning"
    assert warning["expected_cost_wording_allowed"] is False


def test_build_objective_comparator_sidecar_marks_partial_rows_not_comparable() -> None:
    sidecar = build_objective_comparator_sidecar(
        issue="abc1234",
        source_manifest="source.json",
        checkpoint_selection=_checkpoint_selection(),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
        scope="unit scope",
        generated_by="unit",
        run_metadata_by_id={
            "run_a": {
                "status": "available",
                "loss_objective": "partial_net_output_force_filter",
                "objective_profile": "partial_net_output_force_filter",
            },
            "run_b": _full_qrf_run_metadata(),
        },
    )

    first_row = sidecar["rows"][0]

    assert first_row["comparability"]["status"] == "not_comparable"
    assert first_row["selected_to_extlqg_deterministic_ratio"] is None
    assert first_row["selected_to_extlqg_total_ratio_not_apples_to_apples"] is None
    assert "must not be inferred" in first_row["comparability"]["reason"]


def test_write_objective_comparator_sidecar_serializes_json_and_markdown(tmp_path) -> None:
    sidecar = build_objective_comparator_sidecar(
        issue="abc1234",
        source_manifest="source.json",
        checkpoint_selection=_checkpoint_selection(),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
        scope="unit scope",
        generated_by="unit",
        run_metadata_by_id={
            "run_a": _full_qrf_run_metadata(),
            "run_b": _full_qrf_run_metadata(),
        },
        shared_rollout_comparator={
            "status": "available",
            "bank": {"bank_id": "unit-bank", "seed": 7, "n_trials": 2},
            "noise_comparability": {"limitation": "unit limitation"},
            "runs": {
                "run_a": {
                    "status": "available",
                    "gru_vs_extlqg": {"terms": _unit_comparator_terms()},
                },
                "run_b": {
                    "status": "available",
                    "gru_vs_extlqg": {"terms": _unit_comparator_terms()},
                },
            },
            "standard_split_bank_comparator": {
                "status": "available",
                "runs": {
                    "run_a": {"status": "available", "lenses": _unit_split_lenses()},
                    "run_b": {"status": "available", "lenses": _unit_split_lenses()},
                },
            },
        },
    )
    json_path = tmp_path / "sidecar.json"
    markdown_path = tmp_path / "sidecar.md"

    write_objective_comparator_sidecar(
        sidecar,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert reloaded["schema_version"] == SCHEMA_VERSION
    assert "shared_rollout_comparator" not in reloaded["rows"][0]
    assert "standard_split_bank_comparator" not in reloaded["rows"][0]
    assert (
        reloaded["rows"][0]["shared_rollout_comparator_ref"]
        == "/shared_rollout_comparator/runs/run_a"
    )
    assert "\n  " not in json_path.read_text(encoding="utf-8")
    expected_markdown = render_objective_comparator_markdown(sidecar)
    assert markdown == (
        "<!-- AUTO-GENERATED: objective_comparator -->\n"
        f"{expected_markdown}"
        "<!-- /AUTO-GENERATED -->\n"
    )
    assert "Scope: unit scope." in markdown
    assert "not directly comparable to GRU validation values" in markdown
    assert "selected/total" in markdown
    assert "same-noise-bank Monte Carlo" in markdown
    assert "Per-term realized scoring" in markdown


def test_load_run_objective_metadata_extracts_full_qrf_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "results" / "abc1234" / "runs").mkdir(parents=True)
    (tmp_path / "results" / "abc1234" / "runs" / "run_a.json").write_text(
        "{}",
        encoding="utf-8",
    )

    def fake_resolve(experiment: str, run_id: str, *, repo_root: Path):
        assert (experiment, run_id, repo_root) == ("abc1234", "run_a", tmp_path)
        return {
            "loss_objective": "full_analytical_qrf",
            "loss_summary": {
                "objective_profile": "full_analytical_qrf",
                "active_cs_terms": {
                    "state_running_q": {},
                    "terminal_q_f": {},
                    "control_r": {},
                },
                "force_filter_state_cost": "included_via_Q_entries_4_5_each_delay_block",
                "disturbance_integrator_state_cost": (
                    "included_via_Q_entries_6_7_each_delay_block"
                ),
            },
        }

    monkeypatch.setattr(objective_comparator, "resolve_run_record", fake_resolve)

    metadata = load_run_objective_metadata("abc1234", "run_a", repo_root=tmp_path)

    assert metadata["status"] == "available"
    assert metadata["run_spec_path"] == "results/abc1234/runs/run_a.json"
    assert metadata["loss_objective"] == "full_analytical_qrf"
    assert metadata["full_qrf_lens"]["status"] == "available"
    assert metadata["full_qrf_lens"]["active_terms"] == [
        "control_r",
        "state_running_q",
        "terminal_q_f",
    ]


def test_default_extlqg_cost_decomposition_matches_legacy_inline_iteration() -> None:
    legacy = _legacy_extlqg_cost_decomposition_for_test()
    deduped = objective_comparator.compute_default_extlqg_cost_decomposition()

    np.testing.assert_allclose(
        [
            deduped.deterministic_initial_state,
            deduped.initial_covariance_trace,
            deduped.accumulated_noise_scalar,
            deduped.total_expected_cost,
        ],
        [
            legacy.deterministic_initial_state,
            legacy.initial_covariance_trace,
            legacy.accumulated_noise_scalar,
            legacy.total_expected_cost,
        ],
        rtol=0.0,
        atol=1e-9,
    )
    assert deduped.provenance.endswith(", 6 iterations")


def test_materialize_gru_objective_comparator_sidecar_uses_validation_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = tmp_path / "results" / "abc1234" / "notes" / "standard_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    output_json = tmp_path / "results" / "abc1234" / "notes" / "objective.json"
    output_md = tmp_path / "results" / "abc1234" / "notes" / "objective.md"

    monkeypatch.setattr(
        objective_comparator,
        "compute_default_extlqg_cost_decomposition",
        lambda: ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
    )

    result = materialize_gru_objective_comparator_sidecar(
        experiment="abc1234",
        run_ids=("run_a", "run_b"),
        checkpoint_policy="validation_selected_per_replicate",
        use_validation_selected_checkpoints=True,
        checkpoint_manifest=_checkpoint_selection(),
        checkpoint_manifest_path=None,
        standard_manifest_path=manifest_path,
        output_path=output_json,
        note_path=output_md,
        repo_root=tmp_path,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))

    assert result["status"] == "materialized"
    assert result["n_rows"] == 2
    assert payload["source_manifest"] == "results/abc1234/notes/standard_manifest.json"
    assert payload["rows"][0]["comparability"]["status"] == "not_comparable"
    assert payload["rows"][0]["selected_to_extlqg_deterministic_ratio"] is None
    assert output_md.exists()
