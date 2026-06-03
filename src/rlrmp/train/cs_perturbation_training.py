"""Fixed-target perturbation-generalized training config for C&S GRU runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from feedbax._mapping import WhereDict
from feedbax.graph import Component, Wire
from feedbax.task import AbstractTask, TaskTrialSpec
from jaxtyping import PRNGKeyArray, PyTree


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
            "mode": "fixed_target_perturbation_generalized" if self.enabled else "nominal",
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


def install_perturbation_training_graph_adapters(model: Any) -> Any:
    """Install the fixed external additive channel adapters on a C&S GRU graph."""

    for spec in GRAPH_ADAPTER_SPECS.values():
        model = _insert_additive_graph_channel_adapter(model, spec)
    return model


def apply_training_perturbation_mixture(
    trial_specs: TaskTrialSpec,
    config: Any,
    key: PRNGKeyArray,
    batch_info=None,
) -> TaskTrialSpec:
    """Apply one configured training-mixture row.

    The first C&S GRU training screen uses a deterministic 20-batch cycle:
    9 nominal, 9 single-family, and 2 mild-combined batches. This realizes the
    requested 45% / 45% / 10% mixture without Python control flow on traced PRNG
    values inside Feedbax's JIT-compiled training step.
    """

    del key
    cfg = config_from_hps(config)
    trial_specs = add_zero_graph_channel_inputs(trial_specs)
    if batch_info is None:
        return _with_perturbation_metadata(trial_specs, "nominal")

    index = jnp.mod(jnp.asarray(batch_info.current), 20)
    trial_specs = _offset_initial_vector(
        trial_specs,
        axis=1,
        amount=cfg.initial_position_offset_m
        * _cycle_amplitude(index, single_indices=(9, 15), combined_indices=(18,), cfg=cfg),
    )
    trial_specs = _offset_initial_vector(
        trial_specs,
        axis=3,
        amount=cfg.initial_velocity_offset_m_s
        * _cycle_amplitude(index, single_indices=(10,), combined_indices=(), cfg=cfg),
    )
    trial_specs = _add_process_epsilon_pulse(
        trial_specs,
        amount=cfg.process_epsilon_scale
        * _cycle_amplitude(index, single_indices=(11, 16), combined_indices=(18,), cfg=cfg),
        start=cfg.pulse_start_step,
        duration=cfg.pulse_duration_steps,
    )
    trial_specs = _add_graph_channel_pulse(
        trial_specs,
        GRAPH_ADAPTER_SPECS["command_input"],
        amount=cfg.command_input_pulse_n
        * _cycle_amplitude(index, single_indices=(12, 17), combined_indices=(), cfg=cfg),
        start=cfg.pulse_start_step,
        duration=cfg.pulse_duration_steps,
    )
    trial_specs = _add_graph_channel_pulse(
        trial_specs,
        GRAPH_ADAPTER_SPECS["sensory_feedback"],
        amount=cfg.sensory_feedback_offset_m
        * _cycle_amplitude(index, single_indices=(13,), combined_indices=(19,), cfg=cfg),
        start=0,
        duration=trial_specs.timeline.n_steps,
    )
    return _add_graph_channel_pulse(
        trial_specs,
        GRAPH_ADAPTER_SPECS["delayed_observation"],
        amount=cfg.delayed_observation_offset_m
        * _cycle_amplitude(index, single_indices=(14,), combined_indices=(19,), cfg=cfg),
        start=0,
        duration=trial_specs.timeline.n_steps,
    )


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
    return {
        "schema_version": "rlrmp.cs_fixed_target_perturbation_validation_bins.v1",
        "selection_role": (
            "scalar validation loss may select checkpoints; bin metrics are sidecar/audit"
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
        return ("initial_position", "command_input")
    return (bin_name,)


def planned_fixed_target_perturbation_rows(
    *,
    experiment: str = "aacb9ed",
) -> list[dict[str, Any]]:
    """Return the first two planned local perturbation-generalized run rows."""

    rows = []
    for lr_label, lr in (("lr1e-3", 1e-3), ("lr3e-3", 3e-3)):
        run = f"fixed_target_perturbation_fullqrf_warmcos__{lr_label}_clip5_b64"
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
                "perturbation_training": "fixed_target_perturbation_generalized",
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
