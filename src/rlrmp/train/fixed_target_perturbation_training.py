"""Fixed-target perturbation mixture sampling and trial adaptation."""
# ruff: noqa: F401, F403, F405

from __future__ import annotations

from rlrmp.train.training_configs import *  # noqa: F403

from rlrmp.train.broad_epsilon_training import (
    _PgdAscentResult,
    _batch_shape,
    _broad_epsilon_l2_radius,
    _broad_epsilon_pgd_trust_radius,
    _broadcast_finite_policy_params_to_batch,
    _ensure_broad_epsilon_input,
    _epsilon_energy_per_trial,
    _epsilon_time_mask,
    _expand_radius,
    _finite_policy_epsilon_from_rollout,
    _finite_policy_tree_norm,
    _flattened_per_trial_norm,
    _flattened_per_trial_safe_norm,
    _mask_finite_policy_params,
    _normalize_flattened_per_trial,
    _project_flattened_per_trial_l2_ball,
    _resolve_sisu_condition_input,
    _run_broad_epsilon_pgd_ascent,
    _run_finite_broad_epsilon_pgd_inner_maximizer,
    _set_input,
    _shared_policy_time_mask,
    _sisu_condition_values,
    _trial_reach_length_m,
    _trial_target_position_m,
    _zero_finite_policy_params,
    run_broad_epsilon_pgd_inner_maximizer,
)
from functools import lru_cache
from typing import Any, Callable, Literal, Mapping
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax import AbstractTask, TaskTrialSpec, WhereDict
from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    AdditiveGraphChannelTargetSpec,
)
from jaxtyping import PRNGKeyArray
from rlrmp.data_products.calibration import (
    CALIBRATION_PRODUCT_RELPATH,
    CALIBRATION_PRODUCT_ROLE,
    CALIBRATION_PRODUCT_SCHEMA_VERSION,
    load_open_loop_calibration,
    load_perturbation_calibration_defaults,
)
from rlrmp.model.feedbax_channel_adapters import (
    additive_channel_payload_dim,
    additive_channel_provenance,
    materialize_additive_channel_adapters_on_graph,
)


def apply_training_perturbation_mixture(
    trial_specs: TaskTrialSpec,
    config: Any,
    key: PRNGKeyArray,
    batch_info=None,
) -> TaskTrialSpec:
    """Apply one PRNG-driven fixed-target perturbation-training batch."""

    cfg = FixedTargetPerturbationTrainingConfig.from_payload(config)
    specs = graph_adapter_specs(force_filter_feedback=cfg.force_filter_feedback)
    trial_specs = add_zero_graph_channel_inputs(
        trial_specs,
        force_filter_feedback=cfg.force_filter_feedback,
    )
    batch_shape = _batch_shape(trial_specs)
    (
        key_mix,
        key_family,
        key_pos,
        key_vel,
        key_process,
        key_command,
        key_sensory,
        key_lateral,
    ) = jr.split(key, 8)
    mixture = jr.uniform(key_mix, batch_shape)
    single_mask = (
        (mixture >= float(cfg.nominal_fraction))
        & (mixture < float(cfg.nominal_fraction + cfg.single_fraction))
    ).astype(jnp.float32)
    combined_mask = (mixture >= float(cfg.nominal_fraction + cfg.single_fraction)).astype(
        jnp.float32
    )
    active_single_family_bins = _active_single_family_bins(cfg)
    family_index = jr.randint(key_family, batch_shape, 0, len(active_single_family_bins))

    if cfg.calibrated_timing:
        if cfg.movement_age_timing:
            trial_specs = _add_movement_onset_state_offset_random_components(
                trial_specs,
                base_amount=_calibrated_initial_amount(
                    trial_specs,
                    cfg,
                    "initial_position",
                ),
                component_offset=0,
                n_components=2,
                active_mask=(
                    single_mask
                    * _family_mask(family_index, "initial_position", active_single_family_bins)
                    + combined_mask * float(cfg.combined_amplitude_scale)
                ),
                key=key_pos,
            )
            trial_specs = _add_movement_onset_state_offset_random_components(
                trial_specs,
                base_amount=_calibrated_initial_amount(
                    trial_specs,
                    cfg,
                    "initial_velocity",
                ),
                component_offset=2,
                n_components=2,
                active_mask=single_mask
                * _family_mask(family_index, "initial_velocity", active_single_family_bins),
                key=key_vel,
            )
        else:
            trial_specs = _offset_initial_random_components(
                trial_specs,
                base_amount=_calibrated_initial_amount(
                    trial_specs,
                    cfg,
                    "initial_position",
                ),
                component_offset=0,
                n_components=2,
                active_mask=(
                    single_mask
                    * _family_mask(family_index, "initial_position", active_single_family_bins)
                    + combined_mask * float(cfg.combined_amplitude_scale)
                ),
                randomize_amplitude_level=False,
                key=key_pos,
            )
            trial_specs = _offset_initial_random_components(
                trial_specs,
                base_amount=_calibrated_initial_amount(
                    trial_specs,
                    cfg,
                    "initial_velocity",
                ),
                component_offset=2,
                n_components=2,
                active_mask=single_mask
                * _family_mask(family_index, "initial_velocity", active_single_family_bins),
                randomize_amplitude_level=False,
                key=key_vel,
            )
        trial_specs = _add_process_epsilon_calibrated_random_pulse(
            trial_specs,
            cfg,
            active_mask=single_mask
            * _family_mask(family_index, "process_epsilon", active_single_family_bins),
            key=key_process,
        )
        trial_specs = _add_graph_channel_calibrated_random_pulse(
            trial_specs,
            cfg,
            specs["command_input"],
            active_mask=(
                single_mask * _family_mask(family_index, "command_input", active_single_family_bins)
                + combined_mask * float(cfg.combined_amplitude_scale)
            ),
            key=key_command,
        )
        trial_specs = _add_target_aligned_lateral_calibrated_random_pulse(
            trial_specs,
            cfg,
            specs["command_input"],
            active_mask=single_mask
            * _family_mask(
                family_index,
                TARGET_ALIGNED_LATERAL_LOAD_BIN,
                active_single_family_bins,
            ),
            key=key_lateral,
        )
        trial_specs = _add_graph_channel_calibrated_random_pulse(
            trial_specs,
            cfg,
            specs["sensory_feedback"],
            active_mask=single_mask
            * _family_mask(family_index, "sensory_feedback", active_single_family_bins),
            key=key_sensory,
        )
    else:
        trial_specs = _offset_initial_random_components(
            trial_specs,
            base_amount=cfg.initial_position_offset_m,
            component_offset=0,
            n_components=2,
            active_mask=(
                single_mask
                * _family_mask(family_index, "initial_position", active_single_family_bins)
                + combined_mask * float(cfg.combined_amplitude_scale)
            ),
            key=key_pos,
        )
        trial_specs = _offset_initial_random_components(
            trial_specs,
            base_amount=cfg.initial_velocity_offset_m_s,
            component_offset=2,
            n_components=2,
            active_mask=single_mask
            * _family_mask(family_index, "initial_velocity", active_single_family_bins),
            key=key_vel,
        )
        trial_specs = _add_process_epsilon_random_pulse(
            trial_specs,
            base_amount=cfg.process_epsilon_scale,
            active_mask=single_mask
            * _family_mask(family_index, "process_epsilon", active_single_family_bins),
            duration=cfg.pulse_duration_steps,
            key=key_process,
        )
        trial_specs = _add_graph_channel_random_pulse(
            trial_specs,
            specs["command_input"],
            base_amount=cfg.command_input_pulse_n,
            active_mask=(
                single_mask * _family_mask(family_index, "command_input", active_single_family_bins)
                + combined_mask * float(cfg.combined_amplitude_scale)
            ),
            duration=cfg.pulse_duration_steps,
            key=key_command,
        )
        trial_specs = _add_graph_channel_random_pulse(
            trial_specs,
            specs["sensory_feedback"],
            base_amount=cfg.sensory_feedback_offset_m,
            active_mask=single_mask
            * _family_mask(family_index, "sensory_feedback", active_single_family_bins),
            duration=trial_specs.timeline.n_steps,
            key=key_sensory,
        )
    # Train trials are produced inside Feedbax's vmap'd training step, so their
    # PyTree leaves must be JAX values. Keep string/list provenance in run specs
    # and validation sidecars rather than returning it through this dynamic path.
    return trial_specs


def _family_mask(
    family_index: jnp.ndarray,
    bin_name: PerturbationBin,
    active_bins: tuple[PerturbationBin, ...] = SINGLE_FAMILY_BINS,
) -> jnp.ndarray:
    if bin_name not in active_bins:
        return jnp.zeros_like(family_index, dtype=jnp.float32)
    return (family_index == active_bins.index(bin_name)).astype(jnp.float32)


def _calibrated_initial_amount(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: Literal["initial_position", "initial_velocity"],
) -> jnp.ndarray:
    target_peak_delta_x = _target_peak_delta_x_m(trial_specs, config)
    if bin_name == "initial_position":
        return target_peak_delta_x
    sensitivity = load_open_loop_calibration()["initial_velocity_offset"]["initial_condition"]
    return target_peak_delta_x / jnp.asarray(sensitivity, dtype=jnp.float32)


def _target_peak_delta_x_m(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
) -> jnp.ndarray:
    reach_length = _trial_reach_length_m(trial_specs)
    return reach_length * jnp.asarray(
        REACH_RELATIVE_LEVELS[config.physical_level],
        dtype=jnp.float32,
    )


@lru_cache(maxsize=8)
def _closed_loop_table_reach_m(config: FixedTargetPerturbationTrainingConfig) -> float:
    table = _closed_loop_table(config)
    target_rule = table.get("target_rule", {})
    reach = float(target_rule.get("reach_length_m", TARGET_SUPPORT_CONST_REACH_M))
    if reach <= 0.0:
        raise ValueError("Closed-loop calibration table reach_length_m must be positive.")
    return reach


def _closed_loop_table(config: FixedTargetPerturbationTrainingConfig) -> dict[str, Any]:
    if not config.closed_loop_calibration_table_path:
        raise ValueError("Closed-loop calibration table path is required for this regime.")
    return _load_closed_loop_calibration_table(config.closed_loop_calibration_table_path)


def _closed_loop_amplitudes_by_timing(
    config: FixedTargetPerturbationTrainingConfig,
    *,
    family: str,
    timing_labels: tuple[str, ...],
    component: str | None = None,
    axis: str | None = None,
    reducer: Literal["single", "mean"] = "single",
) -> jnp.ndarray:
    rows = _closed_loop_table(config)["rows"]
    values: list[float] = []
    for timing_label in timing_labels:
        matches = [
            row
            for row in rows
            if row.get("family") == family
            and row.get("physical_level") == config.physical_level
            and row.get("timing_bin") == timing_label
            and (component is None or row.get("component") == component)
            and (axis is None or row.get("axis") == axis)
        ]
        if not matches:
            raise ValueError(
                "Closed-loop calibration table has no row for "
                f"family={family!r}, physical_level={config.physical_level!r}, "
                f"timing_bin={timing_label!r}, component={component!r}, axis={axis!r}."
            )
        if reducer == "single" and len(matches) != 1:
            raise ValueError(
                "Closed-loop calibration table lookup expected one row for "
                f"family={family!r}, timing_bin={timing_label!r}, got {len(matches)}."
            )
        values.append(float(np.mean([float(row["amplitude"]) for row in matches])))
    return jnp.asarray(values, dtype=jnp.float32)


def _calibrated_timing_indexed_amounts(
    *,
    config: FixedTargetPerturbationTrainingConfig,
    family: str,
    timing_labels: tuple[str, ...],
    target_peak_delta_x: jnp.ndarray,
    dtype: Any,
) -> jnp.ndarray:
    if _calibration_uses_closed_loop(config, family):
        reach_scale = _trial_reach_length_m_from_peak(target_peak_delta_x, config) / jnp.asarray(
            _closed_loop_table_reach_m(config),
            dtype=dtype,
        )
        amplitudes = _closed_loop_amplitudes_by_timing(
            config,
            family=family,
            timing_labels=timing_labels,
            component=(
                "random_force_pulse_cardinal_basis" if family == "command_input_pulse" else None
            ),
            reducer="mean" if family == "command_input_pulse" else "single",
        ).astype(dtype)
        return jnp.expand_dims(jnp.asarray(reach_scale, dtype=dtype), -1) * amplitudes
    sensitivities = jnp.asarray(
        [load_open_loop_calibration()[family][timing_label] for timing_label in timing_labels],
        dtype=dtype,
    )
    return jnp.expand_dims(jnp.asarray(target_peak_delta_x, dtype=dtype), -1) / sensitivities


def _trial_reach_length_m_from_peak(
    target_peak_delta_x: jnp.ndarray,
    config: FixedTargetPerturbationTrainingConfig,
) -> jnp.ndarray:
    return jnp.asarray(target_peak_delta_x, dtype=jnp.float32) / jnp.asarray(
        REACH_RELATIVE_LEVELS[config.physical_level],
        dtype=jnp.float32,
    )


def _process_epsilon_sensitivity_table(dtype: Any) -> jnp.ndarray:
    rows = []
    for family in PROCESS_EPSILON_COMPONENT_FAMILIES:
        rows.append(
            [
                load_open_loop_calibration()[family][timing_label]
                for timing_label in TIMING_LABELS_PLANT
            ]
        )
    return jnp.asarray(rows, dtype=dtype)


def _offset_initial_random_components(
    trial_specs: TaskTrialSpec,
    *,
    base_amount: float | jnp.ndarray,
    component_offset: int,
    n_components: int,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
    randomize_amplitude_level: bool = True,
) -> TaskTrialSpec:
    key_component, key_sign, key_level = jr.split(key, 3)
    vector = jnp.asarray(trial_specs.inits["mechanics.vector"])
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, n_components) + int(component_offset)
    sign = _random_sign(key_sign, batch_shape)
    level = (
        _random_amplitude_level(key_level, batch_shape)
        if randomize_amplitude_level
        else jnp.ones(batch_shape, dtype=jnp.float32)
    )
    amount = jnp.asarray(base_amount, dtype=vector.dtype) * sign * level * active_mask
    component_mask = jax.nn.one_hot(component, vector.shape[-1], dtype=vector.dtype)
    updated = vector + _expand_to_rank(amount, vector.ndim) * component_mask
    return eqx.tree_at(lambda ts: ts.inits["mechanics.vector"], trial_specs, updated)


def _add_movement_onset_state_offset_random_components(
    trial_specs: TaskTrialSpec,
    *,
    base_amount: float | jnp.ndarray,
    component_offset: int,
    n_components: int,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_sign = jr.split(key, 2)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, n_components) + int(component_offset)
    amount = (
        jnp.asarray(base_amount, dtype=epsilon.dtype)
        * _random_sign(
            key_sign,
            batch_shape,
        )
        * active_mask
    )
    pulse = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component,
        amount=amount,
        duration=1,
        start=_movement_start_index(trial_specs, batch_shape=batch_shape),
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, epsilon + pulse)


def _add_process_epsilon_random_pulse(
    trial_specs: TaskTrialSpec,
    *,
    base_amount: float,
    active_mask: jnp.ndarray,
    duration: int,
    starts: tuple[int, ...] | None = None,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_start, key_sign, key_level = jr.split(key, 4)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, epsilon.shape[-1])
    amount = (
        jnp.asarray(base_amount, dtype=epsilon.dtype)
        * _random_sign(key_sign, batch_shape)
        * _random_amplitude_level(key_level, batch_shape)
        * active_mask
    )
    pulse = _random_pulse_tensor(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component,
        amount=amount,
        duration=duration,
        starts=starts,
        key=key_start,
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, epsilon + pulse)


def _add_process_epsilon_calibrated_random_pulse(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    *,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_start, key_sign = jr.split(key, 3)
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, epsilon.shape[-1])
    start, start_index = _sample_pulse_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        duration=config.pulse_duration_steps,
        starts=_plant_timing_starts(),
        timing_basis=_calibrated_timing_basis(trial_specs, config, batch_shape=batch_shape),
        key=key_start,
    )
    target_peak_delta_x = _target_peak_delta_x_m(trial_specs, config)
    sensitivity = _process_epsilon_sensitivity_table(epsilon.dtype)[component, start_index]
    amount = (
        jnp.asarray(target_peak_delta_x, dtype=epsilon.dtype)
        / sensitivity
        * _random_sign(key_sign, batch_shape)
        * active_mask
    )
    pulse = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=epsilon.shape[-2],
        width=epsilon.shape[-1],
        component=component,
        amount=amount,
        duration=config.pulse_duration_steps,
        start=start,
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, epsilon + pulse)


def _add_graph_channel_random_pulse(
    trial_specs: TaskTrialSpec,
    spec: AdditiveGraphChannelAdapterSpec,
    *,
    base_amount: float,
    active_mask: jnp.ndarray,
    duration: int,
    starts: tuple[int, ...] | None = None,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    if spec.label == GRAPH_ADAPTER_SPECS["command_input"].label:
        key_start, key_direction, key_level = jr.split(key, 3)
        payload = _zero_graph_payload(trial_specs, spec)
        batch_shape = _batch_shape(trial_specs)
        amount = (
            jnp.asarray(base_amount, dtype=payload.dtype)
            * _random_amplitude_level(key_level, batch_shape)
            * active_mask
        )
        start, _start_index = _sample_pulse_start(
            batch_shape=batch_shape,
            n_steps=payload.shape[-2],
            duration=duration,
            starts=starts,
            key=key_start,
        )
        updated = payload + _command_input_direction_pulse(
            batch_shape=batch_shape,
            n_steps=payload.shape[-2],
            width=payload.shape[-1],
            amount=amount,
            duration=duration,
            start=start,
            key=key_direction,
            dtype=payload.dtype,
        )
        return _set_input(trial_specs, spec.input_key, updated)

    key_component, key_start, key_sign, key_level = jr.split(key, 4)
    payload = _zero_graph_payload(trial_specs, spec)
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, payload.shape[-1])
    amount = (
        jnp.asarray(base_amount, dtype=payload.dtype)
        * _random_sign(key_sign, batch_shape)
        * _random_amplitude_level(key_level, batch_shape)
        * active_mask
    )
    updated = payload + _random_pulse_tensor(
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        width=payload.shape[-1],
        component=component,
        amount=amount,
        duration=duration,
        starts=starts,
        key=key_start,
        dtype=payload.dtype,
    )
    return _set_input(trial_specs, spec.input_key, updated)


def _add_graph_channel_calibrated_random_pulse(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    spec: AdditiveGraphChannelAdapterSpec,
    *,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_start, key_sign, key_direction = jr.split(key, 4)
    payload = _zero_graph_payload(trial_specs, spec)
    batch_shape = _batch_shape(trial_specs)
    starts = (
        _plant_timing_starts()
        if spec.label == GRAPH_ADAPTER_SPECS["command_input"].label
        else _controller_visible_timing_starts()
    )
    start, start_index = _sample_pulse_start(
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        duration=config.pulse_duration_steps,
        starts=starts,
        timing_basis=_calibrated_timing_basis(trial_specs, config, batch_shape=batch_shape),
        key=key_start,
    )
    if spec.label == GRAPH_ADAPTER_SPECS["command_input"].label:
        target_peak_delta_x = _target_peak_delta_x_m(trial_specs, config)
        amount_by_timing = _calibrated_timing_indexed_amounts(
            config=config,
            family="command_input_pulse",
            timing_labels=TIMING_LABELS_PLANT,
            target_peak_delta_x=target_peak_delta_x,
            dtype=payload.dtype,
        )
        amount = jnp.take_along_axis(
            amount_by_timing,
            jnp.expand_dims(start_index, axis=-1),
            axis=-1,
        )[..., 0]
        amount = amount * active_mask
        updated = payload + _command_input_direction_pulse(
            batch_shape=batch_shape,
            n_steps=payload.shape[-2],
            width=payload.shape[-1],
            amount=amount,
            duration=config.pulse_duration_steps,
            start=start,
            key=key_direction,
            dtype=payload.dtype,
        )
        return _set_input(trial_specs, spec.input_key, updated)
    else:
        component = jr.randint(
            key_component,
            batch_shape,
            0,
            _randomized_payload_width(spec, config, payload.shape[-1]),
        )
        amount_by_component = _controller_visible_component_amounts(
            trial_specs,
            config,
            timing_index=start_index,
            dtype=payload.dtype,
        )
        amount = jnp.take_along_axis(
            amount_by_component,
            jnp.expand_dims(component, axis=-1),
            axis=-1,
        )[..., 0]
    amount = amount * _random_sign(key_sign, batch_shape) * active_mask
    updated = payload + _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        width=payload.shape[-1],
        component=component,
        amount=amount,
        duration=config.pulse_duration_steps,
        start=start,
        dtype=payload.dtype,
    )
    return _set_input(trial_specs, spec.input_key, updated)


def _command_input_direction_pulse(
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    width: int,
    amount: jnp.ndarray,
    duration: int,
    start: int | jnp.ndarray,
    key: PRNGKeyArray,
    dtype: Any,
) -> jnp.ndarray:
    if int(width) < 2:
        raise ValueError("command-input random-direction pulses require at least two components.")
    direction = jr.normal(key, (*batch_shape, 2), dtype=dtype)
    direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction / jnp.maximum(direction_norm, jnp.asarray(1e-12, dtype=dtype))
    amount = jnp.asarray(amount, dtype=dtype)
    pulse_x = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=n_steps,
        width=width,
        component=jnp.zeros(batch_shape, dtype=jnp.int32),
        amount=amount * direction[..., 0],
        duration=duration,
        start=start,
        dtype=dtype,
    )
    pulse_y = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=n_steps,
        width=width,
        component=jnp.ones(batch_shape, dtype=jnp.int32),
        amount=amount * direction[..., 1],
        duration=duration,
        start=start,
        dtype=dtype,
    )
    return pulse_x + pulse_y


def _add_target_aligned_lateral_calibrated_random_pulse(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    spec: AdditiveGraphChannelAdapterSpec,
    *,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    if not _calibration_uses_closed_loop(
        config,
        "target_aligned_lateral_command_load_pulse",
    ):
        return trial_specs
    key_start, key_sign = jr.split(key)
    payload = _zero_graph_payload(trial_specs, spec)
    batch_shape = _batch_shape(trial_specs)
    start, start_index = _sample_pulse_start(
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        duration=config.pulse_duration_steps,
        starts=_plant_timing_starts(),
        timing_basis=_calibrated_timing_basis(trial_specs, config, batch_shape=batch_shape),
        key=key_start,
    )
    target_peak_delta_x = _target_peak_delta_x_m(trial_specs, config)
    reach_scale = _trial_reach_length_m_from_peak(target_peak_delta_x, config) / jnp.asarray(
        _closed_loop_table_reach_m(config),
        dtype=payload.dtype,
    )
    amount_by_timing = _closed_loop_amplitudes_by_timing(
        config,
        family="target_aligned_lateral_command_load_pulse",
        timing_labels=TIMING_LABELS_PLANT,
        component="target_aligned_lateral_load",
        axis="y",
    ).astype(payload.dtype)
    amount = jnp.take_along_axis(
        jnp.broadcast_to(amount_by_timing, (*batch_shape, len(amount_by_timing))),
        jnp.expand_dims(start_index, axis=-1),
        axis=-1,
    )[..., 0]
    amount = amount * reach_scale.astype(payload.dtype) * _random_sign(key_sign, batch_shape)
    amount = amount * active_mask
    updated = payload + _target_aligned_lateral_direction_pulse(
        trial_specs,
        batch_shape=batch_shape,
        n_steps=payload.shape[-2],
        width=payload.shape[-1],
        amount=amount,
        duration=config.pulse_duration_steps,
        start=start,
        dtype=payload.dtype,
    )
    return _set_input(trial_specs, spec.input_key, updated)


def _target_aligned_lateral_direction_pulse(
    trial_specs: TaskTrialSpec,
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    width: int,
    amount: jnp.ndarray,
    duration: int,
    start: int | jnp.ndarray,
    dtype: Any,
) -> jnp.ndarray:
    if int(width) < 2:
        raise ValueError("target-aligned lateral command loads require at least two components.")
    target = _trial_target_position_m(trial_specs)
    init_pos = jnp.asarray(trial_specs.inits["mechanics.vector"])[..., :2]
    reach = target - init_pos
    reach_norm = jnp.linalg.norm(reach, axis=-1, keepdims=True)
    safe_reach = reach / jnp.maximum(reach_norm, jnp.asarray(1e-12, dtype=reach.dtype))
    fallback = jnp.broadcast_to(jnp.asarray([0.0, 1.0], dtype=dtype), (*batch_shape, 2))
    lateral = jnp.stack([-safe_reach[..., 1], safe_reach[..., 0]], axis=-1)
    lateral = jnp.where(reach_norm > 1e-12, lateral, fallback)
    amount = jnp.asarray(amount, dtype=dtype)
    pulse_x = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=n_steps,
        width=width,
        component=jnp.zeros(batch_shape, dtype=jnp.int32),
        amount=amount * lateral[..., 0],
        duration=duration,
        start=start,
        dtype=dtype,
    )
    pulse_y = _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=n_steps,
        width=width,
        component=jnp.ones(batch_shape, dtype=jnp.int32),
        amount=amount * lateral[..., 1],
        duration=duration,
        start=start,
        dtype=dtype,
    )
    return pulse_x + pulse_y


def _controller_visible_component_amounts(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    *,
    timing_index: jnp.ndarray | None = None,
    dtype: Any,
) -> jnp.ndarray:
    position_amount = _target_peak_delta_x_m(trial_specs, config)
    if _calibration_uses_closed_loop(config, "sensory_feedback_offset"):
        if timing_index is None:
            raise ValueError("Closed-loop sensory calibration requires a timing index.")
        component_values = []
        for component, axis in (
            ("position", "x"),
            ("position", "y"),
            ("velocity", "vx"),
            ("velocity", "vy"),
        ):
            values = _closed_loop_amplitudes_by_timing(
                config,
                family="sensory_feedback_offset",
                timing_labels=TIMING_LABELS_CONTROLLER_VISIBLE,
                component=component,
                axis=axis,
            )
            selected = jnp.take_along_axis(
                jnp.broadcast_to(values.astype(dtype), (*jnp.shape(position_amount), len(values))),
                jnp.expand_dims(timing_index, axis=-1),
                axis=-1,
            )[..., 0]
            component_values.append(selected)
        if config.force_filter_feedback:
            zero = jnp.zeros_like(jnp.asarray(position_amount, dtype=dtype))
            component_values.extend([zero, zero])
        reach_scale = _trial_reach_length_m(trial_specs) / jnp.asarray(
            _closed_loop_table_reach_m(config),
            dtype=dtype,
        )
        return jnp.stack(component_values, axis=-1) * jnp.expand_dims(
            jnp.asarray(reach_scale, dtype=dtype),
            axis=-1,
        )
    velocity_amount = jnp.asarray(
        load_open_loop_calibration().controller_visible_velocity_scale_m_s
        * REACH_RELATIVE_LEVELS[config.physical_level],
        dtype=dtype,
    )
    components = [
        jnp.asarray(position_amount, dtype=dtype),
        jnp.asarray(position_amount, dtype=dtype),
        jnp.broadcast_to(velocity_amount, jnp.shape(position_amount)),
        jnp.broadcast_to(velocity_amount, jnp.shape(position_amount)),
    ]
    if config.force_filter_feedback:
        zero = jnp.zeros_like(jnp.asarray(position_amount, dtype=dtype))
        components.extend([zero, zero])
    return jnp.stack(components, axis=-1)


def _randomized_payload_width(
    spec: AdditiveGraphChannelAdapterSpec,
    config: FixedTargetPerturbationTrainingConfig,
    payload_width: int,
) -> int:
    if config.force_filter_feedback and spec.label in {
        GRAPH_ADAPTER_SPECS["sensory_feedback"].label,
        GRAPH_ADAPTER_SPECS["delayed_observation"].label,
    }:
        return min(4, int(payload_width))
    return int(payload_width)


def _random_pulse_tensor(
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    width: int,
    component: jnp.ndarray,
    amount: jnp.ndarray,
    duration: int,
    starts: tuple[int, ...] | None,
    key: PRNGKeyArray,
    dtype: Any,
) -> jnp.ndarray:
    start, _ = _sample_pulse_start(
        batch_shape=batch_shape,
        n_steps=n_steps,
        duration=duration,
        starts=starts,
        key=key,
    )
    return _pulse_tensor_from_start(
        batch_shape=batch_shape,
        n_steps=n_steps,
        width=width,
        component=component,
        amount=amount,
        duration=duration,
        start=start,
        dtype=dtype,
    )


def _sample_pulse_start(
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    duration: int,
    starts: tuple[int, ...] | None,
    key: PRNGKeyArray,
    timing_basis: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if starts is None:
        max_start = max(1, int(n_steps) - int(duration) + 1)
        start_index = jr.randint(key, batch_shape, 0, max_start)
        return start_index, start_index
    valid_starts = tuple(
        int(start)
        for start in starts
        if 0 <= int(start) < int(n_steps) and int(start) + int(duration) <= int(n_steps)
    )
    if not valid_starts:
        raise ValueError("At least one calibrated timing-bin start must fit the trial.")
    if valid_starts != starts:
        raise ValueError(
            "Calibrated timing mode requires all declared timing bins to fit the trial."
        )
    start_values = jnp.asarray(valid_starts, dtype=jnp.int32)
    start_index = jr.randint(key, batch_shape, 0, len(valid_starts))
    start = start_values[start_index]
    if timing_basis is not None:
        start = start + jnp.asarray(timing_basis, dtype=jnp.int32)
    return start, start_index


def _pulse_tensor_from_start(
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    width: int,
    component: jnp.ndarray,
    amount: jnp.ndarray,
    duration: int,
    start: jnp.ndarray,
    dtype: Any,
) -> jnp.ndarray:
    time = jnp.arange(int(n_steps))
    time_mask = (
        (time >= jnp.expand_dims(start, axis=-1))
        & (time < jnp.expand_dims(start, axis=-1) + int(duration))
    ).astype(dtype)
    component_mask = jax.nn.one_hot(component, int(width), dtype=dtype)
    return (
        _expand_to_rank(amount, len(batch_shape) + 2)
        * jnp.expand_dims(time_mask, axis=-1)
        * jnp.expand_dims(component_mask, axis=-2)
    )


def _plant_timing_starts() -> tuple[int, ...]:
    return tuple(
        int(bin_.start_time_index)
        for bin_ in load_perturbation_calibration_defaults().plant_timing_bins
    )


def _controller_visible_timing_starts() -> tuple[int, ...]:
    return tuple(
        int(bin_.start_time_index)
        for bin_ in load_perturbation_calibration_defaults().controller_visible_timing_bins
    )


def _calibrated_timing_basis(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    *,
    batch_shape: tuple[int, ...],
) -> jnp.ndarray | None:
    if not config.movement_age_timing:
        return None
    return _movement_start_index(trial_specs, batch_shape=batch_shape)


def _movement_start_index(
    trial_specs: TaskTrialSpec,
    *,
    batch_shape: tuple[int, ...] | None = None,
) -> jnp.ndarray:
    bounds = trial_specs.timeline.epoch_bounds
    if batch_shape is None:
        batch_shape = _batch_shape(trial_specs)
    if bounds is None:
        return jnp.zeros(batch_shape, dtype=jnp.int32)
    bounds = jnp.asarray(bounds)
    if bounds.ndim == 1:
        start = jnp.asarray(bounds[-2], dtype=jnp.int32)
        return jnp.broadcast_to(start, batch_shape)
    start = jnp.asarray(bounds[..., -2], dtype=jnp.int32)
    if batch_shape and start.shape != batch_shape:
        start = jnp.broadcast_to(start, batch_shape)
    return start


def _random_sign(key: PRNGKeyArray, shape: tuple[int, ...]) -> jnp.ndarray:
    return jnp.where(jr.bernoulli(key, 0.5, shape), 1.0, -1.0).astype(jnp.float32)


def _random_amplitude_level(key: PRNGKeyArray, shape: tuple[int, ...]) -> jnp.ndarray:
    index = jr.randint(key, shape, 0, len(AMPLITUDE_LEVELS))
    return jnp.asarray(AMPLITUDE_LEVELS, dtype=jnp.float32)[index]


def _expand_to_rank(value: jnp.ndarray, rank: int) -> jnp.ndarray:
    expanded = jnp.asarray(value)
    while expanded.ndim < rank:
        expanded = jnp.expand_dims(expanded, axis=-1)
    return expanded


def _zero_graph_payload(
    trial_specs: TaskTrialSpec,
    spec: AdditiveGraphChannelAdapterSpec,
) -> jnp.ndarray:
    batch_shape = _batch_shape(trial_specs)
    n_steps = int(trial_specs.timeline.n_steps)
    return jnp.zeros(
        (*batch_shape, n_steps, additive_channel_payload_dim(spec)),
        dtype=jnp.float32,
    )


def add_zero_graph_channel_inputs(
    trial_specs: TaskTrialSpec,
    *,
    force_filter_feedback: bool = False,
) -> TaskTrialSpec:
    """Ensure all graph adapter payload inputs exist with zero values."""

    for spec in graph_adapter_specs(force_filter_feedback=force_filter_feedback).values():
        if spec.input_key not in trial_specs.inputs:
            trial_specs = _set_input(
                trial_specs,
                spec.input_key,
                _zero_graph_payload(trial_specs, spec),
            )
    return trial_specs


__all__ = [
    "_add_graph_channel_calibrated_random_pulse",
    "_add_graph_channel_random_pulse",
    "_add_movement_onset_state_offset_random_components",
    "_add_process_epsilon_calibrated_random_pulse",
    "_add_process_epsilon_random_pulse",
    "_add_target_aligned_lateral_calibrated_random_pulse",
    "_calibrated_initial_amount",
    "_calibrated_timing_basis",
    "_calibrated_timing_indexed_amounts",
    "_closed_loop_amplitudes_by_timing",
    "_closed_loop_table",
    "_closed_loop_table_reach_m",
    "_command_input_direction_pulse",
    "_controller_visible_component_amounts",
    "_controller_visible_timing_starts",
    "_expand_to_rank",
    "_family_mask",
    "_movement_start_index",
    "_offset_initial_random_components",
    "_plant_timing_starts",
    "_process_epsilon_sensitivity_table",
    "_pulse_tensor_from_start",
    "_random_amplitude_level",
    "_random_pulse_tensor",
    "_random_sign",
    "_randomized_payload_width",
    "_sample_pulse_start",
    "_target_aligned_lateral_direction_pulse",
    "_target_peak_delta_x_m",
    "_trial_reach_length_m_from_peak",
    "_zero_graph_payload",
    "add_zero_graph_channel_inputs",
    "apply_training_perturbation_mixture",
]
