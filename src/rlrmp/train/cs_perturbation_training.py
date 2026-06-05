"""Fixed-target perturbation-generalized training config for C&S GRU runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax._mapping import WhereDict
from feedbax.graph import Component, Wire
from feedbax.task import AbstractTask, TaskTrialSpec
from jaxtyping import PRNGKeyArray, PyTree


PERTURBATION_TRAINING_MODE = "fixed_target_perturbation_randomized"
LEGACY_PERTURBATION_TRAINING_MODE = "fixed_target_perturbation_generalized"
TARGET_RELATIVE_MULTITARGET_TRAINING_MODE = "target_relative_multitarget_static"
TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE = "target_relative_multitarget_static_h0"
MILD_COMBINED_FAMILIES: tuple["PerturbationBin", ...] = (
    "initial_position",
    "command_input",
)
AMPLITUDE_LEVELS: tuple[float, ...] = (0.5, 1.0)
ORIGINAL_TARGET_ANCHOR_M: tuple[float, float] = (0.15, 0.0)
DEFAULT_SEEN_TARGET_DIRECTIONS_DEG: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
DEFAULT_HELD_OUT_TARGET_DIRECTIONS_DEG: tuple[float, ...] = (30.0, 150.0, 210.0, 330.0)
DEFAULT_SEEN_TARGET_AMPLITUDES_M: tuple[float, ...] = (0.10, 0.15)
DEFAULT_HELD_OUT_TARGET_AMPLITUDES_M: tuple[float, ...] = (0.12, 0.18)

PerturbationBin = Literal[
    "nominal",
    "initial_position",
    "initial_velocity",
    "process_epsilon",
    "command_input",
    "sensory_feedback",
    "delayed_observation",
    "mild_combined",
]

VALIDATION_BINS: tuple[PerturbationBin, ...] = (
    "nominal",
    "initial_position",
    "initial_velocity",
    "process_epsilon",
    "command_input",
    "sensory_feedback",
    "delayed_observation",
    "mild_combined",
)

SINGLE_FAMILY_BINS: tuple[PerturbationBin, ...] = (
    "initial_position",
    "initial_velocity",
    "process_epsilon",
    "command_input",
    "sensory_feedback",
    "delayed_observation",
)

GRAPH_CHANNEL_BINS: tuple[PerturbationBin, ...] = (
    "command_input",
    "sensory_feedback",
    "delayed_observation",
)


@dataclass(frozen=True)
class GraphAdapterSpec:
    """External additive graph-channel adapter contract for training."""

    label: str
    input_key: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    input_port: str
    output_port: str
    future_graphspec_mapping: str

    @property
    def insertion_point(self) -> str:
        """Return the source-to-target graph edge represented by this adapter."""

        return (
            f"{self.source_node}.{self.source_port} -> "
            f"{self.target_node}.{self.target_port}"
        )

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable adapter provenance."""

        return {
            "adapter": "fixed_target_perturbation_training_additive_graph_channel",
            "label": self.label,
            "input_key": self.input_key,
            "insertion_point": self.insertion_point,
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
            "temporary_pre_graphspec": True,
            "future_graphspec_mapping": self.future_graphspec_mapping,
            "controller_input_mutated": False,
            "controller_internal_state_mutated": False,
        }


class AdditiveTrainingGraphChannelAdapter(Component):
    """Add an external time-varying offset to a graph edge payload."""

    input_ports = ("signal", "offset")
    output_ports = ("signal",)

    label: str = eqx.field(static=True)

    def __init__(self, *, label: str):
        self.label = str(label)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], eqx.nn.State]:
        del key
        return {"signal": inputs["signal"] + inputs["offset"]}, state


GRAPH_ADAPTER_SPECS: dict[PerturbationBin, GraphAdapterSpec] = {
    "command_input": GraphAdapterSpec(
        label="perturbation_training_command_input",
        input_key="perturbation_training:command_input",
        source_node="efferent",
        source_port="output",
        target_node="mechanics",
        target_port="force",
        input_port="signal",
        output_port="signal",
        future_graphspec_mapping=(
            "named additive command_input channel on efferent.output -> mechanics.force"
        ),
    ),
    "sensory_feedback": GraphAdapterSpec(
        label="perturbation_training_sensory_feedback",
        input_key="perturbation_training:sensory_feedback",
        source_node="sensory",
        source_port="output",
        target_node="net",
        target_port="feedback",
        input_port="signal",
        output_port="signal",
        future_graphspec_mapping=(
            "named additive sensory_feedback channel after sensory noise before net.feedback"
        ),
    ),
    "delayed_observation": GraphAdapterSpec(
        label="perturbation_training_delayed_observation",
        input_key="perturbation_training:delayed_observation",
        source_node="feedback",
        source_port="feedback",
        target_node="sensory",
        target_port="input",
        input_port="signal",
        output_port="signal",
        future_graphspec_mapping=(
            "named additive delayed_observation channel before sensory.input noise"
        ),
    ),
}


@dataclass(frozen=True)
class FixedTargetPerturbationTrainingConfig:
    """Mixture and amplitudes for fixed-target C&S GRU perturbation training."""

    enabled: bool = False
    nominal_fraction: float = 0.45
    single_fraction: float = 0.45
    combined_fraction: float = 0.10
    combined_amplitude_scale: float = 0.5
    initial_position_offset_m: float = 0.01
    initial_velocity_offset_m_s: float = 0.05
    process_epsilon_scale: float = 0.01
    command_input_pulse_n: float = 1.0
    sensory_feedback_offset_m: float = 0.01
    delayed_observation_offset_m: float = 0.01
    pulse_start_step: int = 20
    pulse_duration_steps: int = 5

    def __post_init__(self) -> None:
        total = self.nominal_fraction + self.single_fraction + self.combined_fraction
        if not np.isclose(total, 1.0):
            raise ValueError(
                "Perturbation-training fractions must sum to 1.0; "
                f"got {total:.6g}."
            )
        if not 0.40 <= self.nominal_fraction <= 0.50:
            raise ValueError("Nominal perturbation-training fraction must be 40-50%.")
        if not 0.40 <= self.single_fraction <= 0.50:
            raise ValueError("Single-family perturbation-training fraction must be 40-50%.")
        if not 0.05 <= self.combined_fraction <= 0.15:
            raise ValueError("Mild-combined perturbation-training fraction must be 5-15%.")
        if self.combined_amplitude_scale <= 0.0 or self.combined_amplitude_scale > 1.0:
            raise ValueError("Combined perturbation amplitude scale must be in (0, 1].")

    def to_hps_dict(self) -> dict[str, Any]:
        """Return the TreeNamespace-compatible config payload."""

        return {
            "enabled": self.enabled,
            "mode": PERTURBATION_TRAINING_MODE if self.enabled else "nominal",
            "legacy_mode": (
                LEGACY_PERTURBATION_TRAINING_MODE if self.enabled else None
            ),
            "sampling": {
                "kind": "prng_driven_fixed_target",
                "uses_supplied_key": True,
                "randomized_fields": [
                    "mixture_membership",
                    "single_family",
                    "sign",
                    "axis_or_component",
                    "pulse_start",
                    "amplitude_level",
                ],
                "amplitude_levels": list(AMPLITUDE_LEVELS),
                "mild_combined_families": list(MILD_COMBINED_FAMILIES),
            },
            "mixture_semantics": perturbation_training_mixture_semantics(self),
            "nominal_fraction": self.nominal_fraction,
            "single_fraction": self.single_fraction,
            "combined_fraction": self.combined_fraction,
            "combined_amplitude_scale": self.combined_amplitude_scale,
            "single_family_bins": list(SINGLE_FAMILY_BINS),
            "validation_bins": list(VALIDATION_BINS),
            "families": {
                "initial_position": {
                    "channel": "initial_state",
                    "family": "initial_position_offset",
                    "amplitude": self.initial_position_offset_m,
                    "units": "m",
                },
                "initial_velocity": {
                    "channel": "initial_state",
                    "family": "initial_velocity_offset",
                    "amplitude": self.initial_velocity_offset_m_s,
                    "units": "m/s",
                },
                "process_epsilon": {
                    "channel": "process_epsilon",
                    "family": "process_epsilon_pulse",
                    "amplitude": self.process_epsilon_scale,
                    "units": "epsilon",
                },
                "command_input": {
                    "channel": "command_input",
                    "family": "command_input_pulse",
                    "amplitude": self.command_input_pulse_n,
                    "units": "N",
                },
                "sensory_feedback": {
                    "channel": "sensory_feedback",
                    "family": "sensory_feedback_offset",
                    "amplitude": self.sensory_feedback_offset_m,
                    "units": "m_or_m_s_channel_units",
                },
                "delayed_observation": {
                    "channel": "delayed_observation",
                    "family": "delayed_observation_offset",
                    "amplitude": self.delayed_observation_offset_m,
                    "units": "m_or_m_s_channel_units",
                },
            },
            "pulse": {
                "start_step": self.pulse_start_step,
                "duration_steps": self.pulse_duration_steps,
            },
            "target_stream": {
                "status": "not_applicable",
                "reason": "fixed-target C&S GRU does not consume a target-position stream",
            },
            "controller_internal_mutation": False,
        }

    def to_json(self) -> dict[str, Any]:
        """Return run-spec metadata."""

        payload = self.to_hps_dict()
        payload["graph_adapter_inputs"] = {
            bin_name: GRAPH_ADAPTER_SPECS[bin_name].to_json()
            for bin_name in GRAPH_CHANNEL_BINS
        }
        return payload


@dataclass(frozen=True)
class TargetRelativeMultiTargetTrainingConfig:
    """Structured static-target distribution for target-relative GRU training."""

    enabled: bool = False
    seen_directions_deg: tuple[float, ...] = DEFAULT_SEEN_TARGET_DIRECTIONS_DEG
    held_out_directions_deg: tuple[float, ...] = DEFAULT_HELD_OUT_TARGET_DIRECTIONS_DEG
    seen_amplitudes_m: tuple[float, ...] = DEFAULT_SEEN_TARGET_AMPLITUDES_M
    held_out_amplitudes_m: tuple[float, ...] = DEFAULT_HELD_OUT_TARGET_AMPLITUDES_M
    original_target_anchor_m: tuple[float, float] = ORIGINAL_TARGET_ANCHOR_M

    def __post_init__(self) -> None:
        if len(self.original_target_anchor_m) != 2:
            raise ValueError("original_target_anchor_m must be a 2D target.")
        if not self.seen_directions_deg:
            raise ValueError("At least one seen target direction is required.")
        if not self.held_out_directions_deg:
            raise ValueError("At least one held-out target direction is required.")
        if not self.seen_amplitudes_m:
            raise ValueError("At least one seen target amplitude is required.")
        if not self.held_out_amplitudes_m:
            raise ValueError("At least one held-out target amplitude is required.")
        if any(float(value) <= 0.0 for value in self.seen_amplitudes_m):
            raise ValueError("Seen target amplitudes must be positive.")
        if any(float(value) <= 0.0 for value in self.held_out_amplitudes_m):
            raise ValueError("Held-out target amplitudes must be positive.")
        seen = _target_tuples(self.seen_directions_deg, self.seen_amplitudes_m)
        if tuple(float(x) for x in self.original_target_anchor_m) not in seen:
            raise ValueError(
                "The structured seen target distribution must include the original "
                "15 cm forward target anchor."
            )

    @property
    def seen_targets_m(self) -> tuple[tuple[float, float], ...]:
        """Return train/seen target positions in metres."""

        return _target_tuples(self.seen_directions_deg, self.seen_amplitudes_m)

    @property
    def held_out_targets_m(self) -> tuple[tuple[float, float], ...]:
        """Return held-out validation target positions in metres."""

        return _target_tuples(self.held_out_directions_deg, self.held_out_amplitudes_m)

    @property
    def validation_targets_m(self) -> tuple[tuple[float, float], ...]:
        """Return validation targets, including the original anchor."""

        return _dedupe_targets(
            (
                tuple(float(x) for x in self.original_target_anchor_m),
                *self.seen_targets_m,
                *self.held_out_targets_m,
            )
        )

    def to_hps_dict(self) -> dict[str, Any]:
        """Return the TreeNamespace-compatible config payload."""

        return {
            "enabled": self.enabled,
            "mode": (
                TARGET_RELATIVE_MULTITARGET_TRAINING_MODE if self.enabled else "disabled"
            ),
            "input_contract": target_relative_input_contract(),
            "target_distribution": {
                "kind": "structured_static_targets",
                "original_target_anchor_m": list(self.original_target_anchor_m),
                "seen_directions_deg": list(self.seen_directions_deg),
                "seen_amplitudes_m": list(self.seen_amplitudes_m),
                "held_out_directions_deg": list(self.held_out_directions_deg),
                "held_out_amplitudes_m": list(self.held_out_amplitudes_m),
                "seen_targets_m": [list(row) for row in self.seen_targets_m],
                "held_out_targets_m": [list(row) for row in self.held_out_targets_m],
                "validation_targets_m": [list(row) for row in self.validation_targets_m],
            },
            "validation_bins": target_relative_validation_bins(self),
            "perturbation_mixture_emphasis": target_relative_perturbation_emphasis(),
            "moving_targets": False,
            "teacher_or_jacobian_supervision": False,
            "command_port_comparator": "diagnostic_only",
        }

    def to_json(self) -> dict[str, Any]:
        """Return run-spec metadata."""

        return self.to_hps_dict()


class TargetRelativeMultiTargetTrainingTaskAdapter(AbstractTask):
    """Rewrite static target trials and expose controller-visible target input."""

    task: object
    config: Any

    def __getattr__(self, name: str):
        return getattr(self.task, name)

    @property
    def loss_func(self):
        return self.task.loss_func

    @property
    def n_steps(self) -> int:
        return int(self.task.n_steps)

    @property
    def seed_validation(self) -> int:
        return int(self.task.seed_validation)

    @property
    def intervention_specs(self):
        return self.task.intervention_specs

    @property
    def input_dependencies(self):
        return self.task.input_dependencies

    def add_input(self, name: str, input_fn, exist_ok: bool = True):
        return eqx.tree_at(
            lambda adapter: adapter.task,
            self,
            self.task.add_input(name, input_fn, exist_ok=exist_ok),
        )

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
        return self.get_train_trial_with_intervenor_params(key, batch_info)

    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
        batch_info=None,
    ) -> TaskTrialSpec:
        base = self.task.get_train_trial_with_intervenor_params(key, batch_info)
        return apply_training_target_distribution(base, self.config, key)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        del key
        return apply_validation_target_distribution(self.task.validation_trials, self.config)

    @property
    def n_validation_trials(self) -> int:
        return len(config_from_target_hps(self.config).validation_targets_m)

    def validation_plots(self, states, trial_specs=None):
        return self.task.validation_plots(states, trial_specs=trial_specs)

    @property
    def validation_trials(self) -> TaskTrialSpec:
        return apply_validation_target_distribution(self.task.validation_trials, self.config)


class FixedTargetPerturbationTrainingTaskAdapter(AbstractTask):
    """Apply fixed-target perturbation mixture and validation bins to a task."""

    task: object
    config: Any
    validation_bin: str | None = None

    def __getattr__(self, name: str):
        return getattr(self.task, name)

    @property
    def loss_func(self):
        return self.task.loss_func

    @property
    def n_steps(self) -> int:
        return int(self.task.n_steps)

    @property
    def seed_validation(self) -> int:
        return int(self.task.seed_validation)

    @property
    def intervention_specs(self):
        return self.task.intervention_specs

    @property
    def input_dependencies(self):
        return self.task.input_dependencies

    def add_input(self, name: str, input_fn, exist_ok: bool = True):
        return eqx.tree_at(
            lambda adapter: adapter.task,
            self,
            self.task.add_input(name, input_fn, exist_ok=exist_ok),
        )

    def get_train_trial(self, key: PRNGKeyArray, batch_info=None) -> TaskTrialSpec:
        return self.task.get_train_trial(key, batch_info)

    def get_train_trial_with_intervenor_params(
        self,
        key: PRNGKeyArray,
        batch_info=None,
    ) -> TaskTrialSpec:
        key_trial, key_pert = jr.split(key)
        base = self.task.get_train_trial_with_intervenor_params(key_trial, batch_info)
        return apply_training_perturbation_mixture(base, self.config, key_pert, batch_info)

    def get_validation_trials(self, key: PRNGKeyArray) -> TaskTrialSpec:
        base = self.task.get_validation_trials(key)
        return apply_validation_bin(base, self.config, self.validation_bin or "nominal")

    @property
    def n_validation_trials(self) -> int:
        return int(self.task.n_validation_trials)

    def validation_plots(self, states, trial_specs=None):
        return self.task.validation_plots(states, trial_specs=trial_specs)

    @property
    def validation_trials(self) -> TaskTrialSpec:
        return apply_validation_bin(
            self.task.validation_trials,
            self.config,
            self.validation_bin or "nominal",
        )


def config_from_hps(config: Any) -> FixedTargetPerturbationTrainingConfig:
    """Normalize an hps training-distribution payload to a dataclass."""

    return FixedTargetPerturbationTrainingConfig(
        enabled=bool(getattr(config, "enabled", False)),
        nominal_fraction=float(getattr(config, "nominal_fraction", 0.45)),
        single_fraction=float(getattr(config, "single_fraction", 0.45)),
        combined_fraction=float(getattr(config, "combined_fraction", 0.10)),
        combined_amplitude_scale=float(getattr(config, "combined_amplitude_scale", 0.5)),
        initial_position_offset_m=float(
            getattr(config, "initial_position_offset_m", 0.01)
        ),
        initial_velocity_offset_m_s=float(
            getattr(config, "initial_velocity_offset_m_s", 0.05)
        ),
        process_epsilon_scale=float(getattr(config, "process_epsilon_scale", 0.01)),
        command_input_pulse_n=float(getattr(config, "command_input_pulse_n", 1.0)),
        sensory_feedback_offset_m=float(getattr(config, "sensory_feedback_offset_m", 0.01)),
        delayed_observation_offset_m=float(
            getattr(config, "delayed_observation_offset_m", 0.01)
        ),
        pulse_start_step=int(getattr(config, "pulse_start_step", 20)),
        pulse_duration_steps=int(getattr(config, "pulse_duration_steps", 5)),
    )


def perturbation_training_mixture_semantics(
    config: FixedTargetPerturbationTrainingConfig,
) -> dict[str, Any]:
    """Return explicit fixed-target perturbation-training sampling semantics."""

    return {
        "schema_version": "rlrmp.cs_perturbation_training_mixture_semantics.v1",
        "experimental_factor_note": (
            "Perturbation uncertainty level is an experimental factor distinct from "
            "physical perturbation amplitude. Broader randomized families, signs, "
            "components, timings, or mixtures can induce robustness rather than "
            "only testing ordinary feedback control."
        ),
        "calibration_note": (
            "Raw amplitudes are not physical-effect calibrated. Calibration should "
            "bin each family by nominal open-loop command-replay peak delta x, then "
            "report closed-loop extLQG and GRU responses at the same calibrated "
            "amplitudes."
        ),
        "membership": {
            "unit": "per training trial",
            "nominal_fraction": float(config.nominal_fraction),
            "single_family_fraction": float(config.single_fraction),
            "mild_combined_fraction": float(config.combined_fraction),
            "nominal": "no explicit perturbation beyond ordinary stochastic runtime",
            "single_family": (
                "one family sampled uniformly from single_family_bins, then "
                "family-specific component/sign/timing/level randomization is applied"
            ),
            "mild_combined": (
                "initial_position and command_input are both active, scaled by "
                "combined_amplitude_scale; the other families are inactive"
            ),
        },
        "single_family_bins": list(SINGLE_FAMILY_BINS),
        "mild_combined_families": list(MILD_COMBINED_FAMILIES),
        "amplitude_levels": list(AMPLITUDE_LEVELS),
        "families": {
            "initial_position": {
                "base_amplitude": float(config.initial_position_offset_m),
                "units": "m",
                "emission": (
                    "offset one random mechanics.vector position component among x/y "
                    "at t=0 by sign * amplitude_level * base_amplitude"
                ),
                "randomized": ["axis", "sign", "amplitude_level"],
            },
            "initial_velocity": {
                "base_amplitude": float(config.initial_velocity_offset_m_s),
                "units": "m/s",
                "emission": (
                    "offset one random mechanics.vector velocity component among x/y "
                    "at t=0 by sign * amplitude_level * base_amplitude"
                ),
                "randomized": ["axis", "sign", "amplitude_level"],
            },
            "process_epsilon": {
                "base_amplitude": float(config.process_epsilon_scale),
                "units": "epsilon",
                "emission": (
                    "add a duration-limited pulse to one random epsilon component "
                    "over a random start time"
                ),
                "randomized": ["epsilon_component", "start_time", "sign", "amplitude_level"],
                "duration_steps": int(config.pulse_duration_steps),
            },
            "command_input": {
                "base_amplitude": float(config.command_input_pulse_n),
                "units": "N",
                "emission": (
                    "add a duration-limited pulse to one random command-channel "
                    "component over a random start time"
                ),
                "randomized": ["axis", "start_time", "sign", "amplitude_level"],
                "duration_steps": int(config.pulse_duration_steps),
            },
            "sensory_feedback": {
                "base_amplitude": float(config.sensory_feedback_offset_m),
                "units": "m_or_m_s_channel_units",
                "emission": (
                    "add an offset pulse on one random 4D sensory-feedback component; "
                    "current training uses full-trial duration"
                ),
                "randomized": ["feedback_component", "start_time", "sign", "amplitude_level"],
            },
            "delayed_observation": {
                "base_amplitude": float(config.delayed_observation_offset_m),
                "units": "m_or_m_s_channel_units",
                "emission": (
                    "add an offset pulse on one random 4D delayed-observation component; "
                    "current training uses full-trial duration"
                ),
                "randomized": ["observation_component", "start_time", "sign", "amplitude_level"],
            },
        },
        "validation_difference": (
            "Validation bins are deterministic family-separated probes, not a replay "
            "of the full training mixture. They expose separate nominal, "
            "single-family, and mild-combined bins for checkpoint selection/reporting."
        ),
    }


def config_from_target_hps(config: Any) -> TargetRelativeMultiTargetTrainingConfig:
    """Normalize an hps target-distribution payload to a dataclass."""

    return TargetRelativeMultiTargetTrainingConfig(
        enabled=bool(getattr(config, "enabled", False)),
        seen_directions_deg=tuple(
            float(x)
            for x in getattr(
                config,
                "seen_directions_deg",
                DEFAULT_SEEN_TARGET_DIRECTIONS_DEG,
            )
        ),
        held_out_directions_deg=tuple(
            float(x)
            for x in getattr(
                config,
                "held_out_directions_deg",
                DEFAULT_HELD_OUT_TARGET_DIRECTIONS_DEG,
            )
        ),
        seen_amplitudes_m=tuple(
            float(x)
            for x in getattr(
                config,
                "seen_amplitudes_m",
                DEFAULT_SEEN_TARGET_AMPLITUDES_M,
            )
        ),
        held_out_amplitudes_m=tuple(
            float(x)
            for x in getattr(
                config,
                "held_out_amplitudes_m",
                DEFAULT_HELD_OUT_TARGET_AMPLITUDES_M,
            )
        ),
        original_target_anchor_m=tuple(
            float(x)
            for x in getattr(
                config,
                "original_target_anchor_m",
                ORIGINAL_TARGET_ANCHOR_M,
            )
        ),
    )


def install_perturbation_training_graph_adapters(model: Any) -> Any:
    """Install the fixed external additive channel adapters on a C&S GRU graph."""

    for spec in GRAPH_ADAPTER_SPECS.values():
        model = _insert_additive_graph_channel_adapter(model, spec)
    return model


def apply_training_target_distribution(
    trial_specs: TaskTrialSpec,
    config: Any,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    """Apply one PRNG-driven static target draw from the seen target set."""

    cfg = config_from_target_hps(config)
    targets = jnp.asarray(cfg.seen_targets_m, dtype=jnp.float32)
    batch_shape = _batch_shape(trial_specs)
    index = jr.randint(key, batch_shape, 0, targets.shape[0])
    target = targets[index]
    return _with_static_target(trial_specs, target, metadata=None)


def apply_validation_target_distribution(
    trial_specs: TaskTrialSpec,
    config: Any,
) -> TaskTrialSpec:
    """Return validation trials covering original, seen, and held-out targets."""

    cfg = config_from_target_hps(config)
    targets = jnp.asarray(cfg.validation_targets_m, dtype=jnp.float32)
    trial_specs = _with_static_target(trial_specs, targets, metadata=None)
    extra = dict(trial_specs.extra or {})
    extra["target_relative_multitarget_bins"] = target_relative_validation_bins(cfg)
    extra["target_relative_input_contract"] = target_relative_input_contract()
    return TaskTrialSpec(
        inits=WhereDict(trial_specs.inits),
        inputs=trial_specs.inputs,
        targets=trial_specs.targets,
        intervene=trial_specs.intervene,
        timeline=trial_specs.timeline,
        extra=extra,
    )


def apply_training_perturbation_mixture(
    trial_specs: TaskTrialSpec,
    config: Any,
    key: PRNGKeyArray,
    batch_info=None,
) -> TaskTrialSpec:
    """Apply one PRNG-driven fixed-target perturbation-training batch."""

    cfg = config_from_hps(config)
    trial_specs = add_zero_graph_channel_inputs(trial_specs)
    batch_shape = _batch_shape(trial_specs)
    (
        key_mix,
        key_family,
        key_pos,
        key_vel,
        key_process,
        key_command,
        key_sensory,
        key_delayed,
    ) = jr.split(key, 8)
    mixture = jr.uniform(key_mix, batch_shape)
    single_mask = (
        (mixture >= float(cfg.nominal_fraction))
        & (mixture < float(cfg.nominal_fraction + cfg.single_fraction))
    ).astype(jnp.float32)
    combined_mask = (
        mixture >= float(cfg.nominal_fraction + cfg.single_fraction)
    ).astype(jnp.float32)
    family_index = jr.randint(key_family, batch_shape, 0, len(SINGLE_FAMILY_BINS))

    trial_specs = _offset_initial_random_components(
        trial_specs,
        base_amount=cfg.initial_position_offset_m,
        component_offset=0,
        n_components=2,
        active_mask=(
            single_mask * _family_mask(family_index, "initial_position")
            + combined_mask * float(cfg.combined_amplitude_scale)
        ),
        key=key_pos,
    )
    trial_specs = _offset_initial_random_components(
        trial_specs,
        base_amount=cfg.initial_velocity_offset_m_s,
        component_offset=2,
        n_components=2,
        active_mask=single_mask * _family_mask(family_index, "initial_velocity"),
        key=key_vel,
    )
    trial_specs = _add_process_epsilon_random_pulse(
        trial_specs,
        base_amount=cfg.process_epsilon_scale,
        active_mask=single_mask * _family_mask(family_index, "process_epsilon"),
        duration=cfg.pulse_duration_steps,
        key=key_process,
    )
    trial_specs = _add_graph_channel_random_pulse(
        trial_specs,
        GRAPH_ADAPTER_SPECS["command_input"],
        base_amount=cfg.command_input_pulse_n,
        active_mask=(
            single_mask * _family_mask(family_index, "command_input")
            + combined_mask * float(cfg.combined_amplitude_scale)
        ),
        duration=cfg.pulse_duration_steps,
        key=key_command,
    )
    trial_specs = _add_graph_channel_random_pulse(
        trial_specs,
        GRAPH_ADAPTER_SPECS["sensory_feedback"],
        base_amount=cfg.sensory_feedback_offset_m,
        active_mask=single_mask * _family_mask(family_index, "sensory_feedback"),
        duration=trial_specs.timeline.n_steps,
        key=key_sensory,
    )
    trial_specs = _add_graph_channel_random_pulse(
        trial_specs,
        GRAPH_ADAPTER_SPECS["delayed_observation"],
        base_amount=cfg.delayed_observation_offset_m,
        active_mask=single_mask * _family_mask(family_index, "delayed_observation"),
        duration=trial_specs.timeline.n_steps,
        key=key_delayed,
    )
    # Train trials are produced inside Feedbax's vmap'd training step, so their
    # PyTree leaves must be JAX values. Keep string/list provenance in run specs
    # and validation sidecars rather than returning it through this dynamic path.
    return trial_specs


def apply_validation_bin(
    trial_specs: TaskTrialSpec,
    config: Any,
    bin_name: str,
) -> TaskTrialSpec:
    """Apply one deterministic validation perturbation bin."""

    cfg = (
        config_from_hps(config)
        if not isinstance(config, FixedTargetPerturbationTrainingConfig)
        else config
    )
    if bin_name == "nominal":
        return _with_perturbation_metadata(trial_specs, "nominal")
    if bin_name == "mild_combined":
        trial_specs = _apply_single_bin(
            trial_specs,
            cfg,
            "initial_position",
            cfg.combined_amplitude_scale,
        )
        trial_specs = _apply_single_bin(
            trial_specs,
            cfg,
            "command_input",
            cfg.combined_amplitude_scale,
        )
        return _with_perturbation_metadata(
            trial_specs,
            "mild_combined",
            families=("initial_position", "command_input"),
        )
    if bin_name not in SINGLE_FAMILY_BINS:
        raise ValueError(f"Unknown perturbation validation bin {bin_name!r}.")
    return _apply_single_bin(trial_specs, cfg, bin_name, 1.0)


def validation_bin_manifest(config: Any) -> dict[str, Any]:
    """Return validation-bin metadata for run specs and sidecars."""

    cfg = (
        config_from_hps(config)
        if not isinstance(config, FixedTargetPerturbationTrainingConfig)
        else config
    )
    selection_role = (
        "aggregate rollout loss over predeclared held-out perturbation bins selects "
        "checkpoints; analytical action, I/O, and Jacobian metrics are audit-only"
        if cfg.enabled
        else "nominal rollout validation loss selects checkpoints"
    )
    validation_role = (
        "generalized_held_out_perturbation_rollout_loss"
        if cfg.enabled
        else "nominal_rollout_validation_loss"
    )
    return {
        "schema_version": "rlrmp.cs_fixed_target_perturbation_validation_bins.v1",
        "validation_role": validation_role,
        "selection_role": selection_role,
        "nominal_quality_role": (
            "nominal bin remains a reported quality sidecar/gate and is not an "
            "analytical-fidelity selector"
        ),
        "bins": [
            {
                "bin": bin_name,
                "families": _bin_families(bin_name),
                "target_stream_mutated": False,
            }
            for bin_name in VALIDATION_BINS
        ],
        "config": cfg.to_json(),
    }


def target_relative_validation_manifest(config: Any) -> dict[str, Any]:
    """Return target-relative validation-bin metadata for run specs."""

    cfg = config_from_target_hps(config)
    return {
        "schema_version": "rlrmp.cs_target_relative_multitarget_validation_bins.v1",
        "validation_role": "target_relative_multitarget_rollout_loss",
        "selection_role": (
            "rollout loss over original-anchor, seen-target, held-out-target, and "
            "perturbation-emphasis bins selects checkpoints; analytical action and "
            "I/O metrics remain audit-only"
        ),
        "target_centered_scoring": "trial_static_target",
        "bins": target_relative_validation_bins(cfg),
        "input_contract": target_relative_input_contract(),
        "config": cfg.to_json(),
    }


def target_relative_validation_bins(config: TargetRelativeMultiTargetTrainingConfig) -> list[dict[str, Any]]:
    """Return target validation-bin rows."""

    anchor = tuple(float(x) for x in config.original_target_anchor_m)
    seen = config.seen_targets_m
    held_out = config.held_out_targets_m
    validation_targets = config.validation_targets_m
    seen_held_out_targets = _dedupe_targets((*seen, *held_out))
    return [
        {
            "bin": "original_target_nominal",
            "target_role": "anchor",
            "targets_m": [list(anchor)],
        },
        {
            "bin": "seen_multitarget_nominal",
            "target_role": "seen_training_support",
            "targets_m": [list(row) for row in seen],
        },
        {
            "bin": "held_out_multitarget_nominal",
            "target_role": "held_out_validation_support",
            "targets_m": [list(row) for row in held_out],
        },
        {
            "bin": "initial_position_offsets",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["initial_position"],
        },
        {
            "bin": "initial_velocity_offsets",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["initial_velocity"],
        },
        {
            "bin": "sensory_feedback_offsets",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["sensory_feedback"],
        },
        {
            "bin": "delayed_observation_offsets",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["delayed_observation"],
        },
        {
            "bin": "process_load_epsilon",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["process_epsilon"],
        },
        {
            "bin": "mild_combined",
            "target_role": "seen_and_held_out_static_targets",
            "targets_m": [list(row) for row in seen_held_out_targets],
            "families": ["initial_position", "sensory_feedback", "process_epsilon"],
        },
        {
            "bin": "command_input_diagnostic",
            "target_role": "diagnostic_only",
            "targets_m": [list(row) for row in validation_targets],
            "families": ["command_input"],
            "checkpoint_selection": "excluded_unless_comparator_defined",
        },
    ]


def target_relative_input_contract() -> dict[str, Any]:
    """Return the documented controller-visible target-relative sign contract."""

    return {
        "controller_feedback_basis": "target_relative_delayed_feedback",
        "static_target_input": "known_immediately_not_visually_delayed",
        "sign_convention": [
            "target_x - delayed_x",
            "target_y - delayed_y",
            "-delayed_vx",
            "-delayed_vy",
        ],
        "shape": [4],
        "moving_targets": "out_of_scope",
    }


def target_relative_perturbation_emphasis() -> dict[str, Any]:
    """Return the target-relative perturbation-mixture design intent."""

    return {
        "nominal_multitarget_fraction": [0.50, 0.70],
        "initial_position_velocity_fraction": [0.10, 0.20],
        "sensory_or_delayed_observation_fraction": [0.10, 0.20],
        "process_or_load_fraction": [0.05, 0.15],
        "command_input": "optional_diagnostic_only",
    }


def _apply_single_bin(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
    amplitude_scale: float,
) -> TaskTrialSpec:
    return _with_perturbation_metadata(
        _apply_single_bin_raw(trial_specs, config, bin_name, amplitude_scale),
        bin_name,
    )


def _apply_single_bin_raw(
    trial_specs: TaskTrialSpec,
    config: FixedTargetPerturbationTrainingConfig,
    bin_name: PerturbationBin,
    amplitude_scale: float,
) -> TaskTrialSpec:
    if bin_name == "initial_position":
        return _offset_initial_vector(
            trial_specs,
            axis=1,
            amount=config.initial_position_offset_m * amplitude_scale,
        )
    if bin_name == "initial_velocity":
        return _offset_initial_vector(
            trial_specs,
            axis=3,
            amount=config.initial_velocity_offset_m_s * amplitude_scale,
        )
    if bin_name == "process_epsilon":
        return _add_process_epsilon_pulse(
            trial_specs,
            amount=config.process_epsilon_scale * amplitude_scale,
            start=config.pulse_start_step,
            duration=config.pulse_duration_steps,
        )
    if bin_name == "command_input":
        return _add_graph_channel_pulse(
            trial_specs,
            GRAPH_ADAPTER_SPECS["command_input"],
            amount=config.command_input_pulse_n * amplitude_scale,
            start=config.pulse_start_step,
            duration=config.pulse_duration_steps,
        )
    if bin_name == "sensory_feedback":
        return _add_graph_channel_pulse(
            trial_specs,
            GRAPH_ADAPTER_SPECS["sensory_feedback"],
            amount=config.sensory_feedback_offset_m * amplitude_scale,
            start=0,
            duration=trial_specs.timeline.n_steps,
        )
    if bin_name == "delayed_observation":
        return _add_graph_channel_pulse(
            trial_specs,
            GRAPH_ADAPTER_SPECS["delayed_observation"],
            amount=config.delayed_observation_offset_m * amplitude_scale,
            start=0,
            duration=trial_specs.timeline.n_steps,
        )
    raise ValueError(f"Unsupported perturbation bin {bin_name!r}.")


def _cycle_amplitude(
    index: jnp.ndarray,
    *,
    single_indices: tuple[int, ...],
    combined_indices: tuple[int, ...],
    cfg: FixedTargetPerturbationTrainingConfig,
) -> jnp.ndarray:
    single = jnp.zeros_like(index, dtype=jnp.float32)
    for value in single_indices:
        single = single + (index == value).astype(jnp.float32)
    combined = jnp.zeros_like(index, dtype=jnp.float32)
    for value in combined_indices:
        combined = combined + (index == value).astype(jnp.float32)
    return single.astype(jnp.float32) + combined.astype(jnp.float32) * float(
        cfg.combined_amplitude_scale
    )


def _family_mask(family_index: jnp.ndarray, bin_name: PerturbationBin) -> jnp.ndarray:
    return (family_index == SINGLE_FAMILY_BINS.index(bin_name)).astype(jnp.float32)


def _insert_additive_graph_channel_adapter(model: Any, spec: GraphAdapterSpec) -> Any:
    if spec.label in getattr(model, "nodes", {}):
        return model
    old_wire = Wire(spec.source_node, spec.source_port, spec.target_node, spec.target_port)
    graph = model.remove_wire(old_wire)
    graph = graph.add_node(
        spec.label,
        AdditiveTrainingGraphChannelAdapter(label=spec.label),
    )
    graph = graph.add_wire(Wire(spec.source_node, spec.source_port, spec.label, spec.input_port))
    graph = graph.add_wire(Wire(spec.label, spec.output_port, spec.target_node, spec.target_port))
    graph = eqx.tree_at(lambda g: g.input_ports, graph, (*graph.input_ports, spec.input_key))
    return eqx.tree_at(
        lambda g: g.input_bindings,
        graph,
        {**graph.input_bindings, spec.input_key: (spec.label, "offset")},
    )


def _offset_initial_vector(
    trial_specs: TaskTrialSpec,
    *,
    axis: int,
    amount: float,
) -> TaskTrialSpec:
    vector = jnp.asarray(trial_specs.inits["mechanics.vector"])
    updated = vector.at[..., axis].add(jnp.asarray(amount, dtype=vector.dtype))
    return eqx.tree_at(lambda ts: ts.inits["mechanics.vector"], trial_specs, updated)


def _offset_initial_random_components(
    trial_specs: TaskTrialSpec,
    *,
    base_amount: float,
    component_offset: int,
    n_components: int,
    active_mask: jnp.ndarray,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
    key_component, key_sign, key_level = jr.split(key, 3)
    vector = jnp.asarray(trial_specs.inits["mechanics.vector"])
    batch_shape = _batch_shape(trial_specs)
    component = jr.randint(key_component, batch_shape, 0, n_components) + int(component_offset)
    sign = _random_sign(key_sign, batch_shape)
    level = _random_amplitude_level(key_level, batch_shape)
    amount = jnp.asarray(base_amount, dtype=vector.dtype) * sign * level * active_mask
    component_mask = jax.nn.one_hot(component, vector.shape[-1], dtype=vector.dtype)
    updated = vector + _expand_to_rank(amount, vector.ndim) * component_mask
    return eqx.tree_at(lambda ts: ts.inits["mechanics.vector"], trial_specs, updated)


def _add_process_epsilon_pulse(
    trial_specs: TaskTrialSpec,
    *,
    amount: float,
    start: int,
    duration: int,
) -> TaskTrialSpec:
    epsilon = jnp.asarray(trial_specs.inputs["epsilon"])
    end = min(int(start) + int(duration), epsilon.shape[-2])
    updated = epsilon.at[..., int(start) : end, 5].add(jnp.asarray(amount, dtype=epsilon.dtype))
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, updated)


def _add_process_epsilon_random_pulse(
    trial_specs: TaskTrialSpec,
    *,
    base_amount: float,
    active_mask: jnp.ndarray,
    duration: int,
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
        key=key_start,
        dtype=epsilon.dtype,
    )
    return eqx.tree_at(lambda ts: ts.inputs["epsilon"], trial_specs, epsilon + pulse)


def _add_graph_channel_pulse(
    trial_specs: TaskTrialSpec,
    spec: GraphAdapterSpec,
    *,
    amount: float,
    start: int,
    duration: int,
) -> TaskTrialSpec:
    payload = _zero_graph_payload(trial_specs, spec)
    end = min(int(start) + int(duration), payload.shape[-2])
    updated = payload.at[..., int(start) : end, 0].add(jnp.asarray(amount, dtype=payload.dtype))
    return _set_input(trial_specs, spec.input_key, updated)


def _add_graph_channel_random_pulse(
    trial_specs: TaskTrialSpec,
    spec: GraphAdapterSpec,
    *,
    base_amount: float,
    active_mask: jnp.ndarray,
    duration: int,
    key: PRNGKeyArray,
) -> TaskTrialSpec:
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
        key=key_start,
        dtype=payload.dtype,
    )
    return _set_input(trial_specs, spec.input_key, updated)


def _random_pulse_tensor(
    *,
    batch_shape: tuple[int, ...],
    n_steps: int,
    width: int,
    component: jnp.ndarray,
    amount: jnp.ndarray,
    duration: int,
    key: PRNGKeyArray,
    dtype: Any,
) -> jnp.ndarray:
    max_start = max(1, int(n_steps) - int(duration) + 1)
    start = jr.randint(key, batch_shape, 0, max_start)
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


def _zero_graph_payload(trial_specs: TaskTrialSpec, spec: GraphAdapterSpec) -> jnp.ndarray:
    batch_shape = _batch_shape(trial_specs)
    n_steps = int(trial_specs.timeline.n_steps)
    width = 2 if spec.target_node == "mechanics" else 4
    return jnp.zeros((*batch_shape, n_steps, width), dtype=jnp.float32)


def add_zero_graph_channel_inputs(trial_specs: TaskTrialSpec) -> TaskTrialSpec:
    """Ensure all graph adapter payload inputs exist with zero values."""

    for spec in GRAPH_ADAPTER_SPECS.values():
        if spec.input_key not in trial_specs.inputs:
            trial_specs = _set_input(
                trial_specs,
                spec.input_key,
                _zero_graph_payload(trial_specs, spec),
            )
    return trial_specs


def _set_input(trial_specs: TaskTrialSpec, key: str, value: Any) -> TaskTrialSpec:
    inputs = dict(trial_specs.inputs)
    inputs[key] = value
    return eqx.tree_at(lambda ts: ts.inputs, trial_specs, inputs)


def _with_static_target(
    trial_specs: TaskTrialSpec,
    target: jnp.ndarray,
    *,
    metadata: dict[str, Any] | None,
) -> TaskTrialSpec:
    target_array = jnp.asarray(target)
    n_steps = int(trial_specs.timeline.n_steps)
    batch_shape = target_array.shape[:-1]
    target_sequence = jnp.broadcast_to(
        jnp.expand_dims(target_array, axis=-2),
        (*batch_shape, n_steps, 2),
    )
    target_spec = trial_specs.targets["mechanics.effector.pos"]
    updated_target_spec = eqx.tree_at(lambda spec: spec.value, target_spec, target_sequence)
    updated_target_spec = jax.tree.map(
        lambda leaf: _broadcast_trial_array(leaf, batch_shape),
        updated_target_spec,
    )
    targets = dict(trial_specs.targets)
    targets["mechanics.effector.pos"] = updated_target_spec
    inits = {
        key: _broadcast_trial_array(value, batch_shape)
        for key, value in dict(trial_specs.inits).items()
    }
    inputs = dict(trial_specs.inputs)
    if "effector_target" in inputs and hasattr(inputs["effector_target"], "pos"):
        inputs["effector_target"] = eqx.tree_at(
            lambda state: state.pos,
            inputs["effector_target"],
            target_sequence,
        )
        inputs["effector_target"] = jax.tree.map(
            lambda leaf: _broadcast_trial_array(leaf, batch_shape),
            inputs["effector_target"],
        )
    inputs["target"] = target_sequence
    inputs = {
        key: (
            value
            if key in {"target", "effector_target"}
            else _broadcast_trial_array(value, batch_shape)
        )
        for key, value in inputs.items()
    }
    timeline = jax.tree.map(
        lambda leaf: _broadcast_trial_array(leaf, batch_shape),
        trial_specs.timeline,
    )
    intervene = jax.tree.map(
        lambda leaf: _broadcast_trial_array(leaf, batch_shape),
        trial_specs.intervene,
    )
    extra = trial_specs.extra if metadata is None else {**dict(trial_specs.extra or {}), **metadata}
    return TaskTrialSpec(
        inits=WhereDict(inits),
        inputs=inputs,
        targets=WhereDict(targets),
        intervene=intervene,
        timeline=timeline,
        extra=extra,
    )


def _broadcast_trial_array(value: Any, batch_shape: tuple[int, ...]) -> Any:
    if not batch_shape:
        return value
    if not eqx.is_array(value):
        return value
    array = jnp.asarray(value)
    if array.ndim == 0:
        return value
    if array.shape[: len(batch_shape)] == batch_shape:
        return value
    tail = array.shape[-1:] if array.ndim <= 2 else array.shape[-2:]
    try:
        return jnp.broadcast_to(array, (*batch_shape, *tail))
    except ValueError:
        return value


def _with_perturbation_metadata(
    trial_specs: TaskTrialSpec,
    bin_name: str,
    *,
    families: tuple[str, ...] | None = None,
) -> TaskTrialSpec:
    trial_specs = add_zero_graph_channel_inputs(trial_specs)
    extra = dict(trial_specs.extra or {})
    extra["perturbation_training_bin"] = bin_name
    extra["perturbation_training_families"] = list(families or _bin_families(bin_name))
    return TaskTrialSpec(
        inits=WhereDict(trial_specs.inits),
        inputs=trial_specs.inputs,
        targets=trial_specs.targets,
        intervene=trial_specs.intervene,
        timeline=trial_specs.timeline,
        extra=extra,
    )


def _batch_shape(trial_specs: TaskTrialSpec) -> tuple[int, ...]:
    target = trial_specs.targets["mechanics.effector.pos"].value
    return tuple(target.shape[:-2]) if target.ndim >= 3 else ()


def _bin_families(bin_name: str) -> tuple[str, ...]:
    if bin_name == "nominal":
        return ()
    if bin_name == "mild_combined":
        return MILD_COMBINED_FAMILIES
    return (bin_name,)


def _target_tuples(
    directions_deg: tuple[float, ...],
    amplitudes_m: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    rows: list[tuple[float, float]] = []
    for amplitude in amplitudes_m:
        for direction in directions_deg:
            radians = np.deg2rad(float(direction))
            rows.append(
                (
                    round(float(amplitude) * float(np.cos(radians)), 12),
                    round(float(amplitude) * float(np.sin(radians)), 12),
                )
            )
    return _dedupe_targets(tuple(rows))


def _dedupe_targets(targets: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    seen: set[tuple[float, float]] = set()
    rows: list[tuple[float, float]] = []
    for target in targets:
        key = (round(float(target[0]), 12), round(float(target[1]), 12))
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    return tuple(rows)


def planned_fixed_target_perturbation_rows(
    *,
    experiment: str = "aacb9ed",
) -> list[dict[str, Any]]:
    """Return the first two planned local perturbation-generalized run rows."""

    rows = []
    for lr_label, lr in (("lr1e-3", 1e-3), ("lr3e-3", 3e-3)):
        run = f"fixed_target_random_perturb_fullqrf_warmcos__{lr_label}_clip5_b64"
        rows.append(
            {
                "experiment": experiment,
                "run": run,
                "controller_lr": lr,
                "batch_size": 64,
                "gradient_clip_norm": 5.0,
                "n_replicates": 5,
                "loss_objective": "full_analytical_qrf",
                "lr_schedule": "warmup_cosine",
                "perturbation_training": PERTURBATION_TRAINING_MODE,
                "checkpoint_selection": "generalized_held_out_perturbation_validation",
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_cs_nominal_gru.py",
                    "--issue",
                    "aacb9ed",
                    "--output-dir",
                    f"_artifacts/{experiment}/runs/{run}",
                    "--n-train-batches",
                    "12000",
                    "--batch-size",
                    "64",
                    "--controller-lr",
                    str(lr),
                    "--gradient-clip-norm",
                    "5",
                    "--lr-warmup-batches",
                    "500",
                    "--lr-warmup-init-fraction",
                    "0.1",
                    "--lr-cosine-alpha",
                    "0.01",
                    "--n-replicates",
                    "5",
                    "--loss-objective",
                    "full_analytical_qrf",
                    "--perturbation-training",
                    "--full-train",
                    "--resume",
                ],
            }
        )
    return rows


def planned_target_relative_multitarget_rows(
    *,
    experiment: str = "ba82f3d",
) -> list[dict[str, Any]]:
    """Return the planned target-relative smoke/main run rows."""

    rows = [
        {
            "experiment": experiment,
            "run": "target_relative_multitarget_fullqrf_smoke",
            "controller_lr": 1e-3,
            "batch_size": 2,
            "gradient_clip_norm": 5.0,
            "n_replicates": 1,
            "loss_objective": "full_analytical_qrf",
            "row_kind": "smoke",
            "training": TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
            "command": [
                "uv",
                "run",
                "python",
                "scripts/train_cs_nominal_gru.py",
                "--issue",
                "ba82f3d",
                "--output-dir",
                f"/tmp/{experiment}_target_relative_smoke",
                "--target-relative-multitarget",
                "--perturbation-training",
                "--loss-objective",
                "full_analytical_qrf",
                "--controller-lr",
                "0.001",
                "--gradient-clip-norm",
                "5",
                "--smoke",
                "--full-train",
                "--resume",
            ],
        }
    ]
    for lr_label, lr in (("lr1e-3", 1e-3), ("lr3e-3", 3e-3)):
        run = f"target_relative_multitarget_fullqrf_warmcos__{lr_label}_clip5_b64"
        rows.append(
            {
                "experiment": experiment,
                "run": run,
                "controller_lr": lr,
                "batch_size": 64,
                "gradient_clip_norm": 5.0,
                "n_replicates": 5,
                "loss_objective": "full_analytical_qrf",
                "lr_schedule": "warmup_cosine",
                "row_kind": "main",
                "training": TARGET_RELATIVE_MULTITARGET_TRAINING_MODE,
                "checkpoint_selection": "target_relative_multitarget_rollout_validation",
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_cs_nominal_gru.py",
                    "--issue",
                    "ba82f3d",
                    "--output-dir",
                    f"_artifacts/{experiment}/runs/{run}",
                    "--n-train-batches",
                    "12000",
                    "--batch-size",
                    "64",
                    "--controller-lr",
                    str(lr),
                    "--gradient-clip-norm",
                    "5",
                    "--lr-warmup-batches",
                    "500",
                    "--lr-warmup-init-fraction",
                    "0.1",
                    "--lr-cosine-alpha",
                    "0.01",
                    "--n-replicates",
                    "5",
                    "--loss-objective",
                    "full_analytical_qrf",
                    "--target-relative-multitarget",
                    "--perturbation-training",
                    "--full-train",
                    "--resume",
                ],
            }
        )
    return rows


def planned_target_relative_multitarget_h0_rows(
    *,
    experiment: str = "643f101",
) -> list[dict[str, Any]]:
    """Return the planned H0 target-relative smoke/main run rows."""

    rows = [
        {
            "experiment": experiment,
            "run": "target_relative_multitarget_h0_fullqrf_smoke",
            "controller_lr": 1e-3,
            "batch_size": 2,
            "gradient_clip_norm": 5.0,
            "n_replicates": 1,
            "loss_objective": "full_analytical_qrf",
            "row_kind": "smoke",
            "training": TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
            "initial_hidden_encoder": "zero_affine_target_relative_feedback",
            "command": [
                "uv",
                "run",
                "python",
                "scripts/train_cs_nominal_gru.py",
                "--issue",
                "643f101",
                "--output-dir",
                f"/tmp/{experiment}_target_relative_h0_smoke",
                "--target-relative-multitarget",
                "--initial-hidden-encoder",
                "--perturbation-training",
                "--loss-objective",
                "full_analytical_qrf",
                "--controller-lr",
                "0.001",
                "--gradient-clip-norm",
                "5",
                "--smoke",
                "--full-train",
                "--resume",
            ],
        }
    ]
    for lr_label, lr in (("lr1e-3", 1e-3), ("lr3e-3", 3e-3)):
        run = f"target_relative_multitarget_h0_fullqrf_warmcos__{lr_label}_clip5_b64"
        rows.append(
            {
                "experiment": experiment,
                "run": run,
                "controller_lr": lr,
                "batch_size": 64,
                "gradient_clip_norm": 5.0,
                "n_replicates": 5,
                "n_train_batches": 12000,
                "loss_objective": "full_analytical_qrf",
                "lr_schedule": "warmup_cosine",
                "row_kind": "main",
                "training": TARGET_RELATIVE_MULTITARGET_H0_TRAINING_MODE,
                "initial_hidden_encoder": "zero_affine_target_relative_feedback",
                "training_diagnostics": "default_enabled",
                "checkpoint_selection": "target_relative_multitarget_rollout_validation",
                "comparison_rows": [
                    (
                        "results/ba82f3d/runs/"
                        f"target_relative_multitarget_fullqrf_warmcos__{lr_label}_clip5_b64"
                    )
                ],
                "command": [
                    "uv",
                    "run",
                    "python",
                    "scripts/train_cs_nominal_gru.py",
                    "--issue",
                    "643f101",
                    "--output-dir",
                    f"_artifacts/{experiment}/runs/{run}",
                    "--n-train-batches",
                    "12000",
                    "--batch-size",
                    "64",
                    "--controller-lr",
                    str(lr),
                    "--gradient-clip-norm",
                    "5",
                    "--lr-warmup-batches",
                    "500",
                    "--lr-warmup-init-fraction",
                    "0.1",
                    "--lr-cosine-alpha",
                    "0.01",
                    "--n-replicates",
                    "5",
                    "--loss-objective",
                    "full_analytical_qrf",
                    "--target-relative-multitarget",
                    "--initial-hidden-encoder",
                    "--perturbation-training",
                    "--full-train",
                    "--resume",
                ],
            }
        )
    return rows
