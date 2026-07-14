"""Governed same-basis evidence for linear-recurrent certificates.

Feedbax owns manifest resolution and checkpoint custody decoding. This module's
public construction boundary accepts only a strict adapter populated from those
authoritative surfaces, verifies exact raw bytes before numeric coercion, and
then delegates to a private numerical kernel.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
from feedbax.analysis import ResolvedEvaluationInput, resolve_evaluation_inputs
from feedbax.contracts.manifest import EvaluationRunSpec, ParentRef
from feedbax.training import ResolvedCheckpointTransaction, resolve_checkpoint_custody_ref
from feedbax.runtime import Channel
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rlrmp.eval.model_slots import ModelSlotProjection, project_training_model_slot
from rlrmp.eval.replicates import is_replicate_array
from rlrmp.eval.trial_inputs import trial_effector_target_position
from rlrmp.loss import CsAnalyticalQrfLoss
from rlrmp.runtime.spec_migrations import (
    LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVAL_PARAMS_SCHEMA_ID,
    LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVAL_PARAMS_SCHEMA_VERSION,
    LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_SCHEMA_ID,
    LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_SCHEMA_VERSION,
)


LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE = (
    "rlrmp.eval.linear_recurrent_augmented_reference"
)
LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_ROLE = "linear_recurrent_augmented_reference_evidence"
REFERENCE_FAMILY = "cs_a1_standard_certificate"
REFERENCE_VERSION = "v1"
REFERENCE_SOLVER = "finite_horizon_discrete_lqr_v1"
CONTROL_CONVENTION = "action_equals_k_times_state"
A1_FROZEN_MATRIX_SHA256 = "78108ca2286af701583e5c4eb87a92736820b5c9260129637722c61831a9e52f"
EVALUATION_BANK_ID = "cs_training_task_validation_bank"
EVALUATION_BANK_VERSION = "v1"
EVALUATION_BANK_SEED = 0
NON_NOMINAL_REASON_CODE = "exact_augmented_closure_requires_nominal_clean"
NON_NOMINAL_OUTCOME_SCHEMA = "rlrmp.eval.linear_recurrent_augmented_reference.outcome.v1"
NON_NOMINAL_COMPONENTS = (
    "candidate_transition",
    "reference_transition",
    "candidate_value_matrices",
    "reference_value_matrices",
    "bellman_hessian",
)
_ZERO_BANK_INPUT_KEYS = (
    "epsilon",
    "perturbation_training.command_input",
    "perturbation_training.delayed_observation",
    "perturbation_training.sensory_feedback",
)
_NOMINAL_CLEAN_TRANSFORMATION = {
    "schema": "rlrmp.eval.nominal_clean_transformation.v1",
    "sensory_channel_noise": "disabled_out_of_place",
    "efferent_channel_noise": "disabled_out_of_place",
    "required_channel_delay_steps": 0,
    "zero_trial_inputs": _ZERO_BANK_INPUT_KEYS,
}
_A1_STATE_DIM = 36
_A1_TRIAL_COUNT = 72
_A1_HORIZON = 60
_A1_FEEDBACK_SPEC = {
    "state_slices": {
        "delayed_position": {"indices": [30, 31]},
        "delayed_velocity": {"indices": [32, 33]},
        "delayed_force": {"indices": [34, 35]},
    },
    "channels": [
        {"slice": "delayed_position", "transform": "target_minus", "target_slice": [0, 2]},
        {"slice": "delayed_velocity", "transform": "negate"},
        {"slice": "delayed_force", "transform": "identity"},
    ],
    "output_size": 6,
    "expected_state_dim": 36,
}
_BASIS_TIMING = (
    "feedbax_history_k_z_equals_target_relative_x_k_plus_1_and_h_k_"
    "predicts_raw_u_k_plus_1_and_x_k_plus_2"
)
_STATE_FRAME = "controller_visible_target_relative_post_step_mechanics_at_history_k"
_HIDDEN_FRAME = "post_net_hidden_at_history_k_previous_for_next_controller_action"
_HASH_CONTRACT = "rlrmp-exact-array-identity-v1"
_ALPHA_RTOL = 1e-12
_DIAGNOSTIC_NAMES = frozenset(
    {
        "linear_recurrence",
        "basis_identity",
        "reference_identity",
        "evidence_identity",
        "timing",
        "state_frame",
        "hidden_frame",
        "dt",
        "tau",
        "alpha",
        "state_dim",
        "hidden_dim",
        "observation_dim",
        "action_dim",
        "horizon",
        "recurrent_spectral_radius",
    }
)

EvaluationLens: TypeAlias = Literal[
    "nominal_clean",
    "riccati_epsilon",
    "process_noise",
    "coverage_induced",
    "held_out_validation",
]
_EVALUATION_LENSES = frozenset(
    {
        "nominal_clean",
        "riccati_epsilon",
        "process_noise",
        "coverage_induced",
        "held_out_validation",
    }
)
_COMPONENT_NAMES = (
    "augmented_states",
    "candidate_augmented_action_sensitivity",
    "reference_augmented_action_sensitivity",
    "candidate_transition",
    "reference_transition",
    "candidate_value_matrices",
    "reference_value_matrices",
    "bellman_hessian",
    "action_weight",
)


class LinearRecurrentAugmentedReferenceParams(BaseModel):
    """Governed scientific choices plus a machine-local custody binding."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["rlrmp.spec.evaluation.linear_recurrent_augmented_reference"] = (
        LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVAL_PARAMS_SCHEMA_ID
    )
    schema_version: Literal["rlrmp.spec.evaluation.linear_recurrent_augmented_reference.v1"] = (
        LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVAL_PARAMS_SCHEMA_VERSION
    )
    states_custody: Literal["durable"] = "durable"
    reference_family: Literal["cs_a1_standard_certificate"] = REFERENCE_FAMILY
    reference_version: Literal["v1"] = REFERENCE_VERSION
    checkpoint_custody_root: Path = Field(exclude=True)
    evaluation_bank: Literal["cs_training_task_validation_bank"] = EVALUATION_BANK_ID
    evaluation_bank_version: Literal["v1"] = EVALUATION_BANK_VERSION
    evaluation_bank_seed: Literal[0] = EVALUATION_BANK_SEED
    matrix_identity_sha256: Literal[
        "78108ca2286af701583e5c4eb87a92736820b5c9260129637722c61831a9e52f"
    ] = A1_FROZEN_MATRIX_SHA256
    evaluation_lens: EvaluationLens = "nominal_clean"

    @field_validator("checkpoint_custody_root")
    @classmethod
    def _absolute_checkpoint_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("checkpoint_custody_root must be absolute")
        return value


@dataclass(frozen=True)
class AugmentedBasisDescriptor:
    """Ordered state-coordinate contract from the authoritative task binding."""

    state_coordinates: tuple[str, ...]
    target_relative: tuple[bool, ...]
    hidden_coordinates: tuple[str, ...]
    timing: str = _BASIS_TIMING
    state_frame: str = _STATE_FRAME
    hidden_frame: str = _HIDDEN_FRAME


@dataclass(frozen=True)
class GovernedSourceLineage:
    """Complete authoritative source lineage for one evidence product."""

    graph_spec_identity: str
    task_spec_identity: str
    task_binding_identity: str
    training_intent_identity: str
    resolved_training_identity: str
    execution_identity: str
    evaluation_manifest_identity: str
    checkpoint_transaction_identity: str
    checkpoint_manifest_identity: str
    checkpoint_root_identity: str
    model_slot_content_digest: str
    model_slot_structural_abi: str


@dataclass(frozen=True)
class GovernedArrayIdentities:
    """Declared exact-byte identities checked before numeric coercion."""

    checkpoint_operators: str
    evaluation_arrays: str
    plant_operators: str
    observation_operator: str
    cost_schedule: str
    reference_policy: str


@dataclass(frozen=True)
class GovernedReferenceDescriptor:
    """Fixed reference contract; alternatives require a new schema version."""

    family: str
    version: str
    solver: str
    control_convention: str
    source_identity: str


@dataclass(frozen=True)
class GovernedLinearRecurrentReferenceInput:
    """Strict future adapter target for ca2f937 plus d81a868 public outputs."""

    basis: AugmentedBasisDescriptor
    lineage: GovernedSourceLineage
    array_identities: GovernedArrayIdentities
    reference: GovernedReferenceDescriptor
    evaluation_lens: EvaluationLens
    coupled_states: ArrayLike
    target_states: ArrayLike
    hidden_states: ArrayLike
    input_weight: ArrayLike
    recurrent_weight: ArrayLike
    readout_weight: ArrayLike
    state_transition: ArrayLike
    action_input: ArrayLike
    observation_map: ArrayLike
    state_cost: ArrayLike
    action_cost: ArrayLike
    terminal_state_cost: ArrayLike
    reference_state_action: ArrayLike
    dt: float
    tau: float
    architecture: str
    component_type: str
    activation: str
    use_bias: bool
    readout_use_bias: bool
    use_noise: bool


@dataclass(frozen=True)
class ResolvedGovernedReferenceSources:
    """Verified public Feedbax sources selected without ambient state."""

    training: ResolvedEvaluationInput
    checkpoint: ResolvedCheckpointTransaction


class MissingGovernedReferenceAuthorityError(ValueError):
    """Raised when manifests lack authority required by the scientific producer."""


class NonNominalAugmentedClosureError(ValueError):
    """Raised when a lens cannot claim an exact closed augmented operator."""

    reason_code = NON_NOMINAL_REASON_CODE


def non_nominal_augmented_reference_outcome(
    params: LinearRecurrentAugmentedReferenceParams,
) -> dict[str, Any]:
    """Return the durable reason-coded outcome for a non-clean manifest row."""

    if params.evaluation_lens == "nominal_clean":
        raise ValueError("nominal_clean requires evidence production, not not_applicable output")
    return {
        "schema": NON_NOMINAL_OUTCOME_SCHEMA,
        "status": "not_applicable",
        "reason_code": NON_NOMINAL_REASON_CODE,
        "evaluation_lens": params.evaluation_lens,
        "not_applicable_components": list(NON_NOMINAL_COMPONENTS),
        "explanation": (
            "Exact augmented transition/value/Bellman closure requires the identity-bound "
            "nominal-clean bank; this lens must not reuse its operator."
        ),
    }


def scientific_evaluation_identity(run_spec: EvaluationRunSpec) -> str:
    """Hash evaluation authority without the machine-local custody-root literal."""

    payload = run_spec.model_dump(mode="json", exclude_none=True)
    params = dict(payload.get("params", {}))
    params.pop("checkpoint_custody_root", None)
    payload["params"] = params
    return _payload_sha256("augmented-reference-evaluation-authority-v1", payload)


def resolve_governed_reference_sources(
    run_spec: EvaluationRunSpec,
    *,
    manifest_root: str | Path,
    checkpoint_root: str | Path,
) -> ResolvedGovernedReferenceSources:
    """Resolve the exact training manifest and its terminal model checkpoint.

    Both roots are explicit caller authorities. Selection uses only ParentRefs
    carried by the resolved ``TrainingRunManifest`` and the manifest's terminal
    completed-batch count; it never consults ``latest.json``, params, or
    ``training_run_ids``.
    """

    resolved_inputs = resolve_evaluation_inputs(run_spec, manifest_root=manifest_root)
    if len(resolved_inputs) != 1:  # Defensive against a future resolver contract drift.
        raise ValueError("public evaluation-input resolver did not return exactly one input")
    training = resolved_inputs[0]
    manifest = training.manifest
    if manifest.completed_batches is None:
        raise MissingGovernedReferenceAuthorityError(
            "TrainingRunManifest.completed_batches is required for terminal checkpoint selection"
        )
    checkpoint_refs = tuple(manifest.checkpoint_custody)
    if not checkpoint_refs:
        raise MissingGovernedReferenceAuthorityError(
            "TrainingRunManifest.checkpoint_custody declares no checkpoint ParentRef"
        )
    if any(not isinstance(ref, ParentRef) for ref in checkpoint_refs):
        raise MissingGovernedReferenceAuthorityError(
            "TrainingRunManifest.checkpoint_custody must contain only checkpoint ParentRefs"
        )

    terminal: list[ResolvedCheckpointTransaction] = []
    for declared_ref in checkpoint_refs:
        resolved = resolve_checkpoint_custody_ref(
            declared_ref,
            allowed_root=checkpoint_root,
            slot_names=("model",),
        )
        if resolved.parent_ref.model_dump(mode="json") != declared_ref.model_dump(mode="json"):
            raise ValueError("resolved checkpoint ParentRef differs from manifest declaration")
        if resolved.manifest.transaction_id != declared_ref.id:
            raise ValueError("resolved checkpoint transaction identity differs from ParentRef")
        if manifest.job_id is None or resolved.manifest.run_id != manifest.job_id:
            raise ValueError("checkpoint run lineage differs from TrainingRunManifest.job_id")
        if "model" not in resolved.slots or len(resolved.slots) != 1:
            raise ValueError("checkpoint resolver did not decode exactly the required model slot")
        if resolved.manifest.completed_training_batches == manifest.completed_batches:
            terminal.append(resolved)
    if len(terminal) != 1:
        raise MissingGovernedReferenceAuthorityError(
            "TrainingRunManifest checkpoint declarations must identify exactly one transaction "
            f"at completed_batches={manifest.completed_batches}; got {len(terminal)}"
        )
    return ResolvedGovernedReferenceSources(training=training, checkpoint=terminal[0])


def require_complete_governed_reference_authority(
    sources: ResolvedGovernedReferenceSources,
) -> None:
    """Fail closed until public manifests govern every scientific input.

    Current native training manifests authenticate the training contract and
    model slot, but do not carry the evaluated post-step coupled/target/hidden
    state histories required to form ``z_t``. Those arrays cannot be recovered
    from a checkpoint without executing a separately governed evaluation.
    """

    manifest = sources.training.manifest
    missing: list[str] = []
    for field_name in (
        "graph_spec",
        "training_spec",
        "task_spec",
        "task_binding_spec",
        "intent_hash",
        "resolved_semantics_root_hash",
        "execution_hash",
    ):
        if getattr(manifest, field_name) is None:
            missing.append(f"TrainingRunManifest.{field_name}")
    missing.append(
        "public evaluation contract carrying controller-visible target-relative post-step "
        "coupled states, matching target states, and previous hidden states"
    )
    if missing:
        raise MissingGovernedReferenceAuthorityError(
            "governed augmented-reference lineage is incomplete: " + ", ".join(missing)
        )


def build_governed_reference_input_from_projection(
    projection: ModelSlotProjection,
    params: LinearRecurrentAugmentedReferenceParams,
    *,
    evaluation_manifest_identity: str,
) -> GovernedLinearRecurrentReferenceInput:
    """Execute the governed bank and derive all same-basis scientific inputs.

    This boundary intentionally accepts the authenticated 639e30f projection,
    not a model, task, rollout array, operator, dimension, or checkpoint id.
    The v1 evidence schema is one-controller-per-product, so ensemble checkpoints
    fail closed instead of silently selecting a replicate.
    """

    if not isinstance(projection, ModelSlotProjection):
        raise TypeError("governed runtime derivation requires a ModelSlotProjection")
    if not isinstance(params, LinearRecurrentAugmentedReferenceParams):
        raise TypeError("governed runtime derivation requires validated governed params")
    if params.evaluation_lens != "nominal_clean":
        raise NonNominalAugmentedClosureError(
            f"{NON_NOMINAL_REASON_CODE}: evaluation_lens={params.evaluation_lens!r}; "
            "v1 exact transition/value/Bellman evidence is defined only for the "
            "identity-bound nominal-clean bank"
        )
    if projection.provenance.architecture != "linear_recurrence":
        raise ValueError("augmented reference evidence requires linear_recurrence architecture")
    if projection.n_replicates != 1:
        raise ValueError(
            "v1 augmented reference evidence requires exactly one governed model replicate"
        )

    checkpoint_model = _single_projected_model(projection)
    model, transformation_identity = _nominal_clean_model(checkpoint_model)
    trial_specs, trial_payload_identity = _nominal_clean_trials(projection.task.validation_trials)
    _validate_a1_projection_authority(projection, model, trial_specs)
    batch_size = _trial_batch_size(trial_specs)
    states = projection.task.eval_trials(
        model,
        trial_specs,
        jr.split(jr.PRNGKey(params.evaluation_bank_seed), batch_size),
    )
    mechanics_vector = np.asarray(states.mechanics.vector)
    hidden_history = np.asarray(states.net.hidden)
    net_output = np.asarray(states.net.output)
    controller_input = np.asarray(states.sensory.output) + np.asarray(
        trial_specs.inputs["perturbation_training.sensory_feedback"]
    )
    efferent_output = np.asarray(states.efferent.output)
    if mechanics_vector.ndim != 3 or hidden_history.ndim != 3 or net_output.ndim != 3:
        raise ValueError("governed eval_trials histories must have trial/time/feature axes")
    if mechanics_vector.shape[:2] != hidden_history.shape[:2]:
        raise ValueError("mechanics and hidden histories do not share trial/time axes")
    if net_output.shape[:2] != mechanics_vector.shape[:2]:
        raise ValueError("action and mechanics histories do not share trial/time axes")

    graph_nodes = model.nodes
    if set(("mechanics", "feedback", "net", "efferent")) - set(graph_nodes):
        raise ValueError("projected graph lacks governed mechanics/feedback/net/efferent nodes")
    mechanics = graph_nodes["mechanics"]
    selector = graph_nodes["feedback"]
    net = graph_nodes["net"]
    if set(("cell", "readout")) - set(net.nodes):
        raise ValueError("projected linear controller lacks governed cell/readout nodes")
    recurrent = net.nodes["cell"]
    readout = net.nodes["readout"]
    efferent = graph_nodes["efferent"]
    sensory = graph_nodes["sensory"]
    if not isinstance(sensory, Channel) or not isinstance(efferent, Channel):
        raise ValueError("nominal-clean v1 requires public Feedbax Channel sensory/efferent nodes")
    if sensory.add_noise or efferent.add_noise:
        raise ValueError("nominal-clean evaluation model still has enabled channel noise")
    if sensory.delay != 0:
        raise ValueError("v1 exact observation algebra requires zero sensory delay")
    if getattr(efferent, "delay", None) != 0:
        raise ValueError("v1 augmented reference evidence requires zero efferent delay")
    if not np.allclose(net_output, efferent_output, rtol=0.0, atol=0.0):
        raise ValueError("runtime state hook no longer aliases net.output to efferent.output")
    if states.sensory.noise is not None or states.efferent.noise is not None:
        raise ValueError("nominal-clean channel histories unexpectedly record noise")
    raw_action_history = efferent_output
    if recurrent.activation_name != "identity":
        raise ValueError("projected VanillaRNN activation is not identity")
    cell = recurrent.cell
    if cell.use_bias or cell.use_noise or getattr(readout, "use_bias", False):
        raise ValueError("v1 linear recurrence requires bias-free, noise-free controller nodes")

    state_transition = _constant_operator(mechanics.A, "mechanics.A")
    action_input = _constant_operator(mechanics.B, "mechanics.B")
    observation_map, basis, target_indices = _observation_basis(
        selector,
        state_dim=mechanics_vector.shape[-1],
        hidden_dim=hidden_history.shape[-1],
    )
    target_states = np.zeros_like(mechanics_vector)
    target_position = np.asarray(trial_effector_target_position(trial_specs))
    if target_position.shape != (
        batch_size,
        mechanics_vector.shape[1],
        len(target_indices),
    ):
        raise ValueError("governed validation target shape conflicts with target-relative basis")
    target_states[..., target_indices] = target_position

    input_weight = np.asarray(cell.weight_ih)
    recurrent_weight = np.asarray(cell.weight_hh)
    readout_weight = np.asarray(readout.layer.weight)
    _validate_recorded_action_alignment(
        mechanics_vector=mechanics_vector,
        target_states=target_states,
        hidden_history=hidden_history,
        raw_actions=raw_action_history,
        controller_inputs=controller_input,
        observation_map=observation_map,
        state_transition=state_transition,
        action_input=action_input,
        input_weight=input_weight,
        recurrent_weight=recurrent_weight,
        readout_weight=readout_weight,
        dt=float(cell.dt),
        tau=float(cell.tau),
    )

    loss = _analytical_qrf_loss(projection.task.loss_func)
    state_cost = np.asarray(loss.Q)
    action_cost = np.asarray(loss.R)
    terminal_state_cost = np.asarray(loss.Q_f)
    history_length = mechanics_vector.shape[1]
    if state_cost.shape[0] == history_length and action_cost.shape[0] == history_length:
        # History index zero is already post-step. The closed z[k] -> z[k+1]
        # operator therefore carries the action/state cost at history k+1.
        state_cost = state_cost[1:]
        action_cost = action_cost[1:]
    if state_cost.shape[0] != history_length - 1:
        raise ValueError("governed bank history length conflicts with analytical cost horizon")
    if action_cost.shape[0] != history_length - 1:
        raise ValueError("governed bank history length conflicts with action-cost horizon")
    reference_state_action = _finite_horizon_lqr_action_operator(
        state_transition,
        action_input,
        state_cost,
        action_cost,
        terminal_state_cost,
    )

    run_spec = projection.run_spec
    manifest = projection.training_manifest
    provenance = projection.provenance
    graph_identity = _payload_sha256("graph-spec-v1", run_spec.graph)
    task_identity = _payload_sha256("task-spec-v1", run_spec.task)
    task_binding_identity = _payload_sha256(
        "task-binding-v1",
        {
            "feedback_node": run_spec.graph.inline["nodes"]["feedback"],
            "evaluation_bank": params.evaluation_bank,
            "evaluation_bank_version": params.evaluation_bank_version,
            "evaluation_bank_seed": params.evaluation_bank_seed,
            "evaluation_lens": params.evaluation_lens,
            "nominal_clean_transformation_identity": transformation_identity,
            "trial_payload_identity": trial_payload_identity,
        },
    )
    lineage = GovernedSourceLineage(
        graph_spec_identity=graph_identity,
        task_spec_identity=task_identity,
        task_binding_identity=task_binding_identity,
        training_intent_identity=_payload_sha256("training-intent-v1", run_spec),
        resolved_training_identity=(f"{manifest.id}@sha256:{projection.training_manifest_sha256}"),
        execution_identity=f"sha256:{provenance.transaction_root_sha256}",
        evaluation_manifest_identity=evaluation_manifest_identity,
        checkpoint_transaction_identity=provenance.checkpoint_transaction_id,
        checkpoint_manifest_identity=f"sha256:{provenance.checkpoint_manifest_sha256}",
        checkpoint_root_identity=f"sha256:{provenance.transaction_root_sha256}",
        model_slot_content_digest=f"sha256:{provenance.slot_root_sha256}",
        model_slot_structural_abi=f"sha256:{provenance.structural_abi_sha256}",
    )
    identities = GovernedArrayIdentities(
        checkpoint_operators=checkpoint_operator_sha256(
            input_weight, recurrent_weight, readout_weight
        ),
        evaluation_arrays=checkpoint_array_sha256(mechanics_vector, target_states, hidden_history),
        plant_operators=plant_operator_sha256(state_transition, action_input),
        observation_operator=observation_operator_sha256(observation_map),
        cost_schedule=cost_schedule_sha256(state_cost, action_cost, terminal_state_cost),
        reference_policy=reference_action_operator_sha256(reference_state_action),
    )
    return GovernedLinearRecurrentReferenceInput(
        basis=basis,
        lineage=lineage,
        array_identities=identities,
        reference=GovernedReferenceDescriptor(
            family=params.reference_family,
            version=params.reference_version,
            solver=REFERENCE_SOLVER,
            control_convention=CONTROL_CONVENTION,
            source_identity=f"sha256:{params.matrix_identity_sha256}",
        ),
        evaluation_lens=params.evaluation_lens,
        coupled_states=mechanics_vector,
        target_states=target_states,
        hidden_states=hidden_history,
        input_weight=input_weight,
        recurrent_weight=recurrent_weight,
        readout_weight=readout_weight,
        state_transition=state_transition,
        action_input=action_input,
        observation_map=observation_map,
        state_cost=state_cost,
        action_cost=action_cost,
        terminal_state_cost=terminal_state_cost,
        reference_state_action=reference_state_action,
        dt=float(cell.dt),
        tau=float(cell.tau),
        architecture=projection.provenance.architecture,
        component_type=type(recurrent).__name__,
        activation=recurrent.activation_name,
        use_bias=bool(cell.use_bias),
        readout_use_bias=bool(getattr(readout, "use_bias", False)),
        use_noise=bool(cell.use_noise),
    )


def produce_governed_linear_recurrent_augmented_reference_evidence(
    run_spec: EvaluationRunSpec,
    params: LinearRecurrentAugmentedReferenceParams,
    *,
    manifest_root: str | Path,
    checkpoint_root: str | Path,
    evaluation_manifest_identity: str,
) -> LinearRecurrentAugmentedReferenceEvidence:
    """Resolve, project, evaluate, derive, and build one durable evidence product."""

    sources = resolve_governed_reference_sources(
        run_spec,
        manifest_root=manifest_root,
        checkpoint_root=checkpoint_root,
    )
    projection = project_training_model_slot(sources.training, sources.checkpoint)
    governed = build_governed_reference_input_from_projection(
        projection,
        params,
        evaluation_manifest_identity=evaluation_manifest_identity,
    )
    return build_governed_linear_recurrent_augmented_reference_evidence(governed)


def _nominal_clean_model(model: Any) -> tuple[Any, str]:
    """Return an immutable zero-channel-noise evaluation copy and its identity."""

    nodes = getattr(model, "nodes", None)
    if not isinstance(nodes, Mapping) or not {"sensory", "efferent"} <= set(nodes):
        raise ValueError("nominal-clean v1 requires governed sensory and efferent graph nodes")
    sensory, efferent = nodes["sensory"], nodes["efferent"]
    if not isinstance(sensory, Channel) or not isinstance(efferent, Channel):
        raise ValueError("nominal-clean v1 requires public Feedbax Channel nodes")
    if sensory.delay != 0 or efferent.delay != 0:
        raise ValueError(
            "nominal-clean exact augmented closure requires zero sensory and efferent delays"
        )

    def without_noise(channel: Channel) -> Channel:
        return Channel(
            delay=channel.delay,
            noise_func=channel.noise_func,
            add_noise=False,
            input_proto=channel.input_proto,
            init_value=channel.init_value,
            noise_model=channel.noise_model,
            noise_role=channel.noise_role,
            noise_timing=channel.noise_timing,
        )

    clean = eqx.tree_at(
        lambda graph: (graph.nodes["sensory"], graph.nodes["efferent"]),
        model,
        (without_noise(sensory), without_noise(efferent)),
    )
    identity = _payload_sha256(
        "nominal-clean-model-transformation-v1",
        {
            **_NOMINAL_CLEAN_TRANSFORMATION,
            "source_channel_contract": {
                "sensory": _channel_contract(sensory),
                "efferent": _channel_contract(efferent),
            },
        },
    )
    return clean, identity


def _validate_a1_projection_authority(
    projection: ModelSlotProjection,
    model: Any,
    trial_specs: Any,
) -> None:
    """Require the exact governed C&S task/graph/bank before binding A1 identity."""

    run_spec = projection.run_spec
    if projection.provenance.method_ref != "rlrmp/cs_supervised/v1":
        raise ValueError("A1 augmented reference requires rlrmp/cs_supervised/v1")
    if projection.provenance.architecture != "linear_recurrence":
        raise ValueError("A1 augmented reference requires linear_recurrence architecture")
    if run_spec.task.type != "fixed_simple_reach":
        raise ValueError("A1 augmented reference requires fixed_simple_reach task authority")
    task_params = run_spec.task.params
    game_card = task_params.get("game_card", {})
    plant = game_card.get("plant", {}) if isinstance(game_card, Mapping) else {}
    if (
        plant.get("state_dim") != _A1_STATE_DIM
        or plant.get("physical_state_dim") != 6
        or plant.get("delay_steps") != 5
        or task_params.get("control_cost_stages") != _A1_HORIZON
    ):
        raise ValueError("A1 task plant/horizon authority drifted from the frozen 36D basis")
    target_bank = task_params.get("target_relative_multitarget", {})
    bins = target_bank.get("validation_bins") if isinstance(target_bank, Mapping) else None
    roles = {item.get("bin") for item in bins or [] if isinstance(item, Mapping)}
    if (
        not isinstance(bins, list)
        or {
            "original_target_nominal",
            "seen_multitarget_nominal",
            "held_out_multitarget_nominal",
        }
        - roles
    ):
        raise ValueError("A1 task lacks the governed nominal validation-bank roles")
    graph = run_spec.graph.inline
    feedback = graph.get("nodes", {}).get("feedback", {})
    if feedback.get("type") != "StateFeedbackSelector" or feedback.get("params") != (
        _A1_FEEDBACK_SPEC
    ):
        raise ValueError("A1 graph feedback selector authority drifted")
    mechanics = model.nodes.get("mechanics")
    if np.asarray(getattr(mechanics, "A", None)).shape != (_A1_STATE_DIM, _A1_STATE_DIM):
        raise ValueError("A1 mechanics operator is not exactly 36D")
    targets = np.asarray(trial_effector_target_position(trial_specs))
    if targets.shape != (_A1_TRIAL_COUNT, _A1_HORIZON, 2):
        raise ValueError("A1 governed validation bank must be exactly 72 x 60 x 2 targets")
    inputs = trial_specs.inputs
    expected_shapes = {
        "epsilon": (_A1_TRIAL_COUNT, _A1_HORIZON, 6),
        "perturbation_training.command_input": (_A1_TRIAL_COUNT, _A1_HORIZON, 2),
        "perturbation_training.delayed_observation": (_A1_TRIAL_COUNT, _A1_HORIZON, 6),
        "perturbation_training.sensory_feedback": (_A1_TRIAL_COUNT, _A1_HORIZON, 6),
        "target": (_A1_TRIAL_COUNT, _A1_HORIZON, 2),
    }
    for key, shape in expected_shapes.items():
        if key not in inputs or np.asarray(inputs[key]).shape != shape:
            raise ValueError(f"A1 governed validation-bank input {key!r} shape/role drifted")


def _channel_contract(channel: Channel) -> dict[str, Any]:
    return {
        "type": f"{type(channel).__module__}.{type(channel).__qualname__}",
        "delay": channel.delay,
        "add_noise": channel.add_noise,
        "noise_model": channel.noise_model,
        "noise_role": channel.noise_role,
        "noise_timing": channel.noise_timing,
        "input_proto": exact_array_sha256(channel.input_proto),
    }


def _nominal_clean_trials(trial_specs: Any) -> tuple[Any, str]:
    """Zero every governed disturbance input and hash the exact bank payload."""

    inputs = getattr(trial_specs, "inputs", None)
    if not isinstance(inputs, Mapping):
        raise ValueError("governed validation bank inputs must be a named mapping")
    missing = set(_ZERO_BANK_INPUT_KEYS) - set(inputs)
    if missing:
        raise ValueError(f"nominal-clean bank lacks governed zero inputs: {sorted(missing)}")
    clean_inputs = dict(inputs)
    for key in _ZERO_BANK_INPUT_KEYS:
        value = jnp.asarray(inputs[key])
        if value.ndim < 3 or not np.issubdtype(np.asarray(value).dtype, np.number):
            raise ValueError(f"governed bank input {key!r} must be trial/time/numeric")
        clean_inputs[key] = jnp.zeros_like(value)
    clean = eqx.tree_at(lambda value: value.inputs, trial_specs, clean_inputs)
    identity = _named_array_tree_sha256("nominal-clean-evaluation-bank-v1", clean)
    return clean, identity


def _named_array_tree_sha256(role: str, tree: Any) -> str:
    digest = hashlib.sha256(role.encode() + b"\0")
    array_count = 0
    for path, leaf in jt.flatten_with_path(tree)[0]:
        if not eqx.is_array(leaf):
            continue
        array_count += 1
        path_text = "/".join(str(entry) for entry in path).encode()
        digest.update(len(path_text).to_bytes(8, "big"))
        digest.update(path_text)
        digest.update(exact_array_sha256(leaf).encode())
    if array_count == 0:
        raise ValueError("governed evaluation bank has no exact array payload")
    return f"sha256:{digest.hexdigest()}"


def _single_projected_model(projection: ModelSlotProjection) -> Any:
    arrays, other = eqx.partition(
        projection.model,
        lambda leaf: is_replicate_array(leaf, projection.n_replicates),
    )
    selected = jt.map(lambda leaf: leaf[0], arrays)
    return eqx.combine(selected, other)


def _trial_batch_size(trial_specs: Any) -> int:
    target = np.asarray(trial_effector_target_position(trial_specs))
    if target.ndim != 3 or target.shape[-1] != 2 or target.shape[0] < 1:
        raise ValueError("governed validation bank must expose trial/time/2 effector targets")
    return int(target.shape[0])


def _constant_operator(value: ArrayLike, name: str) -> NDArray[Any]:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be one exact constant matrix")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite numeric data")
    return array


def _observation_basis(
    selector: Any,
    *,
    state_dim: int,
    hidden_dim: int,
) -> tuple[NDArray[np.float64], AugmentedBasisDescriptor, tuple[int, ...]]:
    if type(selector).__name__ != "StateFeedbackSelector":
        raise ValueError("governed feedback node must be StateFeedbackSelector")
    if state_dim != _A1_STATE_DIM:
        raise ValueError("A1 augmented reference requires exactly 36 coupled coordinates")
    if selector.expected_state_dim != _A1_STATE_DIM:
        raise ValueError("A1 feedback selector must declare exactly 36 coupled coordinates")
    if selector.state_slices_param != _A1_FEEDBACK_SPEC["state_slices"] or (
        selector.channels_param != _A1_FEEDBACK_SPEC["channels"]
    ):
        raise ValueError("A1 feedback selector coordinates/transforms drifted")
    if selector.expected_state_dim != state_dim:
        raise ValueError("feedback selector state dimension conflicts with rollout mechanics")
    slice_map = dict(selector.state_slices)
    observation = np.zeros((selector.output_size, state_dim), dtype=np.float64)
    target_relative = [False] * state_dim
    coordinate_names = [f"mechanics.vector[{index}]" for index in range(state_dim)]
    target_indices: list[int] = []
    cursor = 0
    for channel in selector.channels:
        indices = tuple(int(index) for index in slice_map[channel.state_slice])
        transform = channel.transform
        if transform == "target_minus":
            sign = -1.0
            if channel.target_slice is None or len(channel.target_slice) != len(indices):
                raise ValueError("target-minus feedback channel lacks exact target coordinates")
            target_indices.extend(indices)
            for index in indices:
                target_relative[index] = True
        elif transform == "negate":
            sign = -1.0
        elif transform == "identity":
            sign = 1.0
        else:
            raise ValueError(f"unsupported governed feedback transform {transform!r}")
        for local, index in enumerate(indices):
            observation[cursor + local, index] = sign
            coordinate_names[index] = f"mechanics.vector[{index}]:{channel.state_slice}"
        cursor += len(indices)
    if cursor != selector.output_size or len(target_indices) != 2:
        raise ValueError("feedback selector does not define the exact C&S target-relative basis")
    basis = AugmentedBasisDescriptor(
        state_coordinates=tuple(coordinate_names),
        target_relative=tuple(target_relative),
        hidden_coordinates=tuple(f"net.hidden[{index}]" for index in range(hidden_dim)),
    )
    return observation, basis, tuple(target_indices)


def _analytical_qrf_loss(loss_func: Any) -> CsAnalyticalQrfLoss:
    terms = getattr(loss_func, "terms", None)
    if not isinstance(terms, Mapping) or set(terms) != {"full_analytical_qrf"}:
        raise ValueError("governed task must expose exactly the full_analytical_qrf objective")
    loss = terms["full_analytical_qrf"]
    if not isinstance(loss, CsAnalyticalQrfLoss):
        raise ValueError("full_analytical_qrf term has an unsupported implementation")
    return loss


def _validate_recorded_action_alignment(
    *,
    mechanics_vector: NDArray[Any],
    target_states: NDArray[Any],
    hidden_history: NDArray[Any],
    raw_actions: NDArray[Any],
    controller_inputs: NDArray[Any],
    observation_map: NDArray[Any],
    state_transition: NDArray[Any],
    action_input: NDArray[Any],
    input_weight: NDArray[Any],
    recurrent_weight: NDArray[Any],
    readout_weight: NDArray[Any],
    dt: float,
    tau: float,
) -> None:
    """Prove Feedbax's post-step history timing over the complete bank.

    History index ``k`` records post-mechanics ``x_(k+1)``, VanillaRNN hidden
    ``h_k``, and the post-efferent noisy command. The raw controller command is
    reconstructed before this function as ``efferent.output - efferent.noise``.
    Therefore ``z_k = [x_(k+1); h_k]`` predicts ``h_(k+1)``, raw ``u_(k+1)``,
    and ``x_(k+2)`` through ``z[:, :-1] -> z[:, 1:]``. No synthetic initial
    hidden state participates in the certificate basis.
    """

    if mechanics_vector.shape[1] < 2:
        raise ValueError("two-step action timing validation requires at least two states")
    alpha = _positive(dt, "dt") / _positive(tau, "tau")
    effective_recurrence = (1.0 - alpha) * np.eye(hidden_history.shape[-1]) + (
        alpha * recurrent_weight
    )
    direct_actions = np.einsum("bth,ah->bta", hidden_history, readout_weight)
    if not np.allclose(direct_actions, raw_actions, rtol=2e-5, atol=2e-6):
        raise ValueError("de-noised action history is not the readout of stored hidden history")

    relative = mechanics_vector[:, :-1] - target_states[:, :-1]
    feedback = np.einsum("os,bts->bto", observation_map, relative)
    if controller_inputs.shape != raw_actions.shape[:-1] + (input_weight.shape[1],):
        raise ValueError("reconstructed controller input history has incompatible axes")
    if not np.allclose(controller_inputs[:, 1:], feedback, rtol=2e-5, atol=2e-6):
        raise ValueError(
            "nominal-clean controller input is not C times the preceding target-relative state"
        )
    expected_hidden = np.einsum("bto,ho->bth", controller_inputs[:, 1:], alpha * input_weight)
    expected_hidden += np.einsum(
        "bth,kh->btk",
        hidden_history[:, :-1],
        effective_recurrence,
    )
    expected_actions = np.einsum("bth,ah->bta", expected_hidden, readout_weight)
    if not np.allclose(expected_hidden, hidden_history[:, 1:], rtol=2e-5, atol=2e-6):
        absolute_delta = np.abs(expected_hidden - hidden_history[:, 1:])
        max_index = tuple(
            int(index)
            for index in np.unravel_index(np.argmax(absolute_delta), absolute_delta.shape)
        )
        time_maxima = np.max(absolute_delta, axis=(0, 2))
        raise ValueError(
            "stored hidden history is not aligned with the next controller update; "
            f"max_abs_delta={float(absolute_delta[max_index]):.9g}, "
            f"argmax={max_index}, first_three_time_maxima={time_maxima[:3].tolist()}"
        )
    if not np.allclose(expected_actions, raw_actions[:, 1:], rtol=2e-5, atol=2e-6):
        raise ValueError(
            "next raw actions are not aligned with post-step mechanics and stored hidden"
        )
    expected_mechanics = np.einsum(
        "ij,btj->bti", state_transition, mechanics_vector[:, :-1]
    ) + np.einsum("ij,btj->bti", action_input, raw_actions[:, 1:])
    if not np.allclose(expected_mechanics, mechanics_vector[:, 1:], rtol=2e-5, atol=2e-6):
        raise ValueError(
            "next post-step mechanics are not aligned with the next de-noised controller action"
        )


def _finite_horizon_lqr_action_operator(
    state_transition: NDArray[Any],
    action_input: NDArray[Any],
    state_cost: NDArray[Any],
    action_cost: NDArray[Any],
    terminal_state_cost: NDArray[Any],
) -> NDArray[np.float64]:
    """Return u=Kx under this module's sign convention from governed Q/R/Qf."""

    q = np.asarray(state_cost, dtype=np.float64)
    r = np.asarray(action_cost, dtype=np.float64)
    horizon = q.shape[0]
    a = np.broadcast_to(np.asarray(state_transition, dtype=np.float64), (horizon,) + q.shape[1:])
    b0 = np.asarray(action_input, dtype=np.float64)
    b = np.broadcast_to(b0, (horizon,) + b0.shape)
    p_next = np.asarray(terminal_state_cost, dtype=np.float64)
    operators: list[NDArray[np.float64]] = []
    for time_index in range(horizon - 1, -1, -1):
        lhs = r[time_index] + b[time_index].T @ p_next @ b[time_index]
        positive_gain = np.linalg.solve(lhs, b[time_index].T @ p_next @ a[time_index])
        operators.append(-positive_gain)
        p_next = q[time_index] + a[time_index].T @ p_next @ (
            a[time_index] - b[time_index] @ positive_gain
        )
        p_next = 0.5 * (p_next + p_next.T)
    return np.stack(tuple(reversed(operators)), axis=0)


def _payload_sha256(domain: str, value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(domain.encode() + b"\0" + payload).hexdigest()
    return f"sha256:{digest}"


class EvidenceComponentDescriptor(BaseModel):
    """Exact identity and storage shape for one durable evidence component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    sha256: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class LinearRecurrentAugmentedReferenceEvidence:
    """Typed output whose arrays and identities are immutable and re-verifiable."""

    schema_id: str
    schema_version: str
    product_role: str
    evaluation_manifest_identity: str
    evaluation_lens: str
    basis_descriptor: AugmentedBasisDescriptor
    source_lineage: GovernedSourceLineage
    source_array_identities: GovernedArrayIdentities
    reference_descriptor: GovernedReferenceDescriptor
    basis_identity: str
    reference_identity: str
    evidence_identity: str
    component_descriptors: Mapping[str, EvidenceComponentDescriptor]
    augmented_states: NDArray[np.float64]
    candidate_augmented_action_sensitivity: NDArray[np.float64]
    reference_augmented_action_sensitivity: NDArray[np.float64]
    candidate_transition: NDArray[np.float64]
    reference_transition: NDArray[np.float64]
    candidate_value_matrices: NDArray[np.float64]
    reference_value_matrices: NDArray[np.float64]
    bellman_hessian: NDArray[np.float64]
    action_weight: NDArray[np.float64]
    recurrence_diagnostics: Mapping[str, Any]


class _StrictPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _BasisPayload(_StrictPayloadModel):
    state_coordinates: tuple[str, ...]
    target_relative: tuple[bool, ...]
    hidden_coordinates: tuple[str, ...]
    timing: str
    state_frame: str
    hidden_frame: str


class _LineagePayload(_StrictPayloadModel):
    graph_spec_identity: str
    task_spec_identity: str
    task_binding_identity: str
    training_intent_identity: str
    resolved_training_identity: str
    execution_identity: str
    evaluation_manifest_identity: str
    checkpoint_transaction_identity: str
    checkpoint_manifest_identity: str
    checkpoint_root_identity: str
    model_slot_content_digest: str
    model_slot_structural_abi: str


class _ReferencePayload(_StrictPayloadModel):
    family: str
    version: str
    solver: str
    control_convention: str
    source_identity: str


class _ArrayIdentitiesPayload(_StrictPayloadModel):
    checkpoint_operators: str
    evaluation_arrays: str
    plant_operators: str
    observation_operator: str
    cost_schedule: str
    reference_policy: str


class _SerializedComponent(_StrictPayloadModel):
    descriptor: EvidenceComponentDescriptor
    values: Any


class _RecurrencePayload(_StrictPayloadModel):
    linear_recurrence: Literal[True]
    basis_identity: str
    reference_identity: str
    evidence_identity: str
    timing: str
    state_frame: str
    hidden_frame: str
    dt: float
    tau: float
    alpha: float
    state_dim: int = Field(ge=1)
    hidden_dim: int = Field(ge=1)
    observation_dim: int = Field(ge=1)
    action_dim: int = Field(ge=1)
    horizon: int = Field(ge=1)
    recurrent_spectral_radius: float


class LinearRecurrentAugmentedReferenceEvidencePayload(_StrictPayloadModel):
    """Strict JSON-compatible durable state payload boundary."""

    schema_id: Literal["rlrmp.linear_recurrent_augmented_reference_evidence"]
    schema_version: Literal["rlrmp.linear_recurrent_augmented_reference_evidence.v1"]
    product_role: Literal["linear_recurrent_augmented_reference_evidence"]
    evaluation_manifest_identity: str
    evaluation_lens: EvaluationLens
    basis_descriptor: _BasisPayload
    source_lineage: _LineagePayload
    source_array_identities: _ArrayIdentitiesPayload
    reference_descriptor: _ReferencePayload
    basis_identity: str
    reference_identity: str
    evidence_identity: str
    components: dict[str, _SerializedComponent]
    recurrence_diagnostics: _RecurrencePayload

    @model_validator(mode="after")
    def _exact_components(self) -> LinearRecurrentAugmentedReferenceEvidencePayload:
        if set(self.components) != set(_COMPONENT_NAMES):
            raise ValueError(f"components must be exactly {sorted(_COMPONENT_NAMES)}")
        for name, component in self.components.items():
            if component.descriptor.name != name:
                raise ValueError(f"component descriptor name mismatch for {name!r}")
        return self


def build_governed_linear_recurrent_augmented_reference_evidence(
    governed: GovernedLinearRecurrentReferenceInput,
) -> LinearRecurrentAugmentedReferenceEvidence:
    """Verify authoritative identities, then construct same-basis evidence."""

    if not isinstance(governed, GovernedLinearRecurrentReferenceInput):
        raise TypeError("governed construction requires the strict resolved-input adapter")
    raw = _raw_arrays(governed)
    _validate_governed_contract(governed, raw)
    return _build_numeric_evidence(governed, raw)


def governed_evidence_component_kwargs(
    evidence: LinearRecurrentAugmentedReferenceEvidence,
) -> dict[str, Any]:
    """Expose standard-certificate inputs after re-verifying typed evidence."""

    _verify_typed_evidence(evidence)
    return {name: getattr(evidence, name) for name in _COMPONENT_NAMES} | {
        "recurrence_diagnostics": dict(evidence.recurrence_diagnostics),
        "state_label": "target_relative_post_step_coupled_state_and_previous_hidden",
    }


def serialize_governed_evidence(
    evidence: LinearRecurrentAugmentedReferenceEvidence,
) -> dict[str, Any]:
    """Serialize verified evidence to a strict EvaluationRecipeResult state mapping."""

    _verify_typed_evidence(evidence)
    components = {
        name: {
            "descriptor": evidence.component_descriptors[name].model_dump(mode="json"),
            "values": getattr(evidence, name).tolist(),
        }
        for name in _COMPONENT_NAMES
    }
    payload = {
        "schema_id": evidence.schema_id,
        "schema_version": evidence.schema_version,
        "product_role": evidence.product_role,
        "evaluation_manifest_identity": evidence.evaluation_manifest_identity,
        "evaluation_lens": evidence.evaluation_lens,
        "basis_descriptor": _dataclass_payload(evidence.basis_descriptor),
        "source_lineage": _dataclass_payload(evidence.source_lineage),
        "source_array_identities": _dataclass_payload(evidence.source_array_identities),
        "reference_descriptor": _dataclass_payload(evidence.reference_descriptor),
        "basis_identity": evidence.basis_identity,
        "reference_identity": evidence.reference_identity,
        "evidence_identity": evidence.evidence_identity,
        "components": components,
        "recurrence_diagnostics": dict(evidence.recurrence_diagnostics),
    }
    return LinearRecurrentAugmentedReferenceEvidencePayload.model_validate(payload).model_dump(
        mode="json"
    )


def deserialize_governed_evidence(
    payload: Mapping[str, Any],
) -> LinearRecurrentAugmentedReferenceEvidence:
    """Validate and materialize a durable evidence state mapping."""

    validated = LinearRecurrentAugmentedReferenceEvidencePayload.model_validate(payload)
    arrays: dict[str, NDArray[Any]] = {}
    descriptors: dict[str, EvidenceComponentDescriptor] = {}
    for name, component in validated.components.items():
        descriptor = component.descriptor
        try:
            dtype = np.dtype(descriptor.dtype)
        except TypeError as exc:
            raise ValueError(f"invalid dtype for component {name!r}") from exc
        array = np.asarray(component.values, dtype=dtype)
        if array.shape != descriptor.shape:
            raise ValueError(f"component {name!r} shape conflicts with descriptor")
        if exact_array_sha256(array) != descriptor.sha256:
            raise ValueError(f"component {name!r} bytes conflict with descriptor")
        arrays[name] = _immutable(array)
        descriptors[name] = descriptor
    evidence = LinearRecurrentAugmentedReferenceEvidence(
        schema_id=validated.schema_id,
        schema_version=validated.schema_version,
        product_role=validated.product_role,
        evaluation_manifest_identity=validated.evaluation_manifest_identity,
        evaluation_lens=validated.evaluation_lens,
        basis_descriptor=AugmentedBasisDescriptor(**validated.basis_descriptor.model_dump()),
        source_lineage=GovernedSourceLineage(**validated.source_lineage.model_dump()),
        source_array_identities=GovernedArrayIdentities(
            **validated.source_array_identities.model_dump()
        ),
        reference_descriptor=GovernedReferenceDescriptor(
            **validated.reference_descriptor.model_dump()
        ),
        basis_identity=validated.basis_identity,
        reference_identity=validated.reference_identity,
        evidence_identity=validated.evidence_identity,
        component_descriptors=MappingProxyType(descriptors),
        recurrence_diagnostics=MappingProxyType(validated.recurrence_diagnostics.model_dump()),
        **arrays,
    )
    _verify_typed_evidence(evidence)
    return evidence


def exact_array_sha256(value: ArrayLike) -> str:
    """Hash exact dtype, shape, and C-order bytes under the versioned contract."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(_HASH_CONTRACT.encode())
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def checkpoint_operator_sha256(*values: ArrayLike) -> str:
    return _group_hash("checkpoint-operators-v1", *values)


def checkpoint_array_sha256(*values: ArrayLike) -> str:
    return _group_hash("evaluation-arrays-v1", *values)


def plant_operator_sha256(*values: ArrayLike) -> str:
    return _group_hash("plant-operators-v1", *values)


def observation_operator_sha256(value: ArrayLike) -> str:
    return _group_hash("observation-operator-v1", value)


def cost_schedule_sha256(*values: ArrayLike) -> str:
    return _group_hash("cost-schedule-v1", *values)


def reference_action_operator_sha256(value: ArrayLike) -> str:
    return _group_hash("reference-policy-v1", value)


def _raw_arrays(governed: GovernedLinearRecurrentReferenceInput) -> dict[str, NDArray[Any]]:
    names = (
        "coupled_states",
        "target_states",
        "hidden_states",
        "input_weight",
        "recurrent_weight",
        "readout_weight",
        "state_transition",
        "action_input",
        "observation_map",
        "state_cost",
        "action_cost",
        "terminal_state_cost",
        "reference_state_action",
    )
    return {name: np.asarray(getattr(governed, name)) for name in names}


def _validate_governed_contract(
    governed: GovernedLinearRecurrentReferenceInput,
    raw: Mapping[str, NDArray[Any]],
) -> None:
    _nonempty_dataclass(governed.lineage, "source lineage")
    _nonempty_dataclass(governed.reference, "reference descriptor")
    if governed.reference.family != REFERENCE_FAMILY:
        raise ValueError("unsupported reference family")
    if governed.reference.version != REFERENCE_VERSION:
        raise ValueError("unsupported reference version")
    if governed.reference.solver != REFERENCE_SOLVER:
        raise ValueError("unsupported reference solver")
    if governed.reference.control_convention != CONTROL_CONVENTION:
        raise ValueError("unsupported reference control convention")
    if governed.evaluation_lens not in _EVALUATION_LENSES:
        raise ValueError("unsupported evaluation lens")
    expected = {
        "checkpoint_operators": checkpoint_operator_sha256(
            raw["input_weight"], raw["recurrent_weight"], raw["readout_weight"]
        ),
        "evaluation_arrays": checkpoint_array_sha256(
            raw["coupled_states"], raw["target_states"], raw["hidden_states"]
        ),
        "plant_operators": plant_operator_sha256(raw["state_transition"], raw["action_input"]),
        "observation_operator": observation_operator_sha256(raw["observation_map"]),
        "cost_schedule": cost_schedule_sha256(
            raw["state_cost"], raw["action_cost"], raw["terminal_state_cost"]
        ),
        "reference_policy": reference_action_operator_sha256(raw["reference_state_action"]),
    }
    for name, actual in expected.items():
        declared = getattr(governed.array_identities, name)
        if declared != actual:
            raise ValueError(f"stale governed {name} hash")
    architecture = {
        "architecture": (governed.architecture, "linear_recurrence"),
        "component_type": (governed.component_type, "VanillaRNN"),
        "activation": (governed.activation, "identity"),
        "use_bias": (governed.use_bias, False),
        "readout_use_bias": (governed.readout_use_bias, False),
        "use_noise": (governed.use_noise, False),
    }
    for name, (actual, required) in architecture.items():
        if actual != required:
            raise ValueError(f"{name} must be {required!r}, got {actual!r}")


def _build_numeric_evidence(
    governed: GovernedLinearRecurrentReferenceInput,
    raw: Mapping[str, NDArray[Any]],
) -> LinearRecurrentAugmentedReferenceEvidence:
    coupled = _numeric(raw["coupled_states"], "coupled_states", 3)
    hidden = _numeric(raw["hidden_states"], "hidden_states", 3)
    if coupled.shape[:2] != hidden.shape[:2] or coupled.shape[1] < 2:
        raise ValueError("coupled/hidden axes must match with at least two time samples")
    state_dim, hidden_dim = coupled.shape[-1], hidden.shape[-1]
    horizon = coupled.shape[1] - 1
    _validate_basis(governed.basis, state_dim, hidden_dim)
    target = _broadcast_target(raw["target_states"], coupled.shape)
    mask = np.asarray(governed.basis.target_relative, dtype=bool)
    if np.any(np.abs(target[..., ~mask]) > 0.0):
        raise ValueError("target translation is allowed only on declared coordinates")
    augmented_states = np.concatenate((coupled - np.where(mask, target, 0.0), hidden), -1)

    wi = _numeric(raw["input_weight"], "input_weight", 2)
    wh = _numeric(raw["recurrent_weight"], "recurrent_weight", 2)
    wo = _numeric(raw["readout_weight"], "readout_weight", 2)
    if wi.shape[0] != hidden_dim or wh.shape != (hidden_dim, hidden_dim):
        raise ValueError("checkpoint hidden dimensions conflict")
    if wo.shape[1] != hidden_dim:
        raise ValueError("readout hidden dimension conflicts")
    observation_dim, action_dim = wi.shape[1], wo.shape[0]
    dt, tau = _positive(governed.dt, "dt"), _positive(governed.tau, "tau")
    alpha = dt / tau
    if not 0.0 < alpha <= 1.0:
        raise ValueError("linear recurrence requires 0 < dt/tau <= 1")
    effective_input = alpha * wi
    effective_recurrence = (1 - alpha) * np.eye(hidden_dim) + alpha * wh

    a = _time(raw["state_transition"], "state_transition", horizon, state_dim, state_dim)
    b = _time(raw["action_input"], "action_input", horizon, state_dim, action_dim)
    c = _time(raw["observation_map"], "observation_map", horizon, observation_dim, state_dim)
    q = _time(raw["state_cost"], "state_cost", horizon, state_dim, state_dim)
    r = _time(raw["action_cost"], "action_cost", horizon, action_dim, action_dim)
    qf = _numeric(raw["terminal_state_cost"], "terminal_state_cost", 2)
    if qf.shape != (state_dim, state_dim):
        raise ValueError("terminal_state_cost dimension conflicts")
    krx = _time(
        raw["reference_state_action"], "reference_state_action", horizon, action_dim, state_dim
    )
    for name, value in (("state_cost", q), ("action_cost", r), ("terminal", qf)):
        _symmetric(value, name)

    eic = np.einsum("ij,tjk->tik", effective_input, c)
    kx = np.einsum("ij,tjk->tik", wo, eic)
    kh = wo @ effective_recurrence
    augmented_dim = state_dim + hidden_dim
    kc = np.zeros((horizon, action_dim, augmented_dim))
    kc[:, :, :state_dim], kc[:, :, state_dim:] = kx, kh
    kr = np.zeros_like(kc)
    kr[:, :, :state_dim] = krx
    fc = _transition(a, b, eic, effective_recurrence, kc, state_dim)
    fr = _transition(a, b, eic, effective_recurrence, kr, state_dim)
    qz = np.zeros((horizon, augmented_dim, augmented_dim))
    qz[:, :state_dim, :state_dim] = q
    qfz = np.zeros((augmented_dim, augmented_dim))
    qfz[:state_dim, :state_dim] = qf
    pc = _values(qz, r, kc, fc, qfz)
    pr = _values(qz, r, kr, fr, qfz)
    bz = np.zeros((horizon, augmented_dim, action_dim))
    bz[:, :state_dim] = b
    hessian = r + np.einsum("tzi,tzw,twj->tij", bz, pr[1:], bz)

    components = {
        "augmented_states": augmented_states,
        "candidate_augmented_action_sensitivity": kc,
        "reference_augmented_action_sensitivity": kr,
        "candidate_transition": fc,
        "reference_transition": fr,
        "candidate_value_matrices": pc,
        "reference_value_matrices": pr,
        "bellman_hessian": hessian,
        "action_weight": r,
    }
    for name, value in components.items():
        _finite(value, name)
    for name in ("candidate_value_matrices", "reference_value_matrices", "bellman_hessian"):
        _symmetric(components[name], name)
    components = {name: _immutable(value) for name, value in components.items()}
    descriptors = {name: _descriptor(name, value) for name, value in components.items()}

    basis_identity = _identity(
        "basis",
        {
            "descriptor": _dataclass_payload(governed.basis),
            "state_dim": state_dim,
            "hidden_dim": hidden_dim,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "horizon": horizon,
            "dt": dt,
            "tau": tau,
            "alpha": alpha,
            "graph_spec_identity": governed.lineage.graph_spec_identity,
            "task_spec_identity": governed.lineage.task_spec_identity,
            "task_binding_identity": governed.lineage.task_binding_identity,
        },
    )
    reference_identity = _reference_identity(
        basis_identity=basis_identity,
        reference_descriptor=governed.reference,
        source_lineage=governed.lineage,
        source_array_identities=governed.array_identities,
    )
    diagnostic_semantics = {
        "linear_recurrence": True,
        "timing": _BASIS_TIMING,
        "state_frame": _STATE_FRAME,
        "hidden_frame": _HIDDEN_FRAME,
        "dt": dt,
        "tau": tau,
        "alpha": alpha,
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "horizon": horizon,
        "recurrent_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(effective_recurrence)))),
    }
    evidence_payload = {
        "schema_version": LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_SCHEMA_VERSION,
        "basis_identity": basis_identity,
        "reference_identity": reference_identity,
        "evaluation_manifest_identity": governed.lineage.evaluation_manifest_identity,
        "evaluation_lens": governed.evaluation_lens,
        "source_lineage": _dataclass_payload(governed.lineage),
        "source_array_identities": _dataclass_payload(governed.array_identities),
        "recurrence_diagnostics": diagnostic_semantics,
        "components": {
            name: descriptor.model_dump(mode="json") for name, descriptor in descriptors.items()
        },
    }
    evidence_identity = _identity("evidence", evidence_payload)
    diagnostics = MappingProxyType(
        {
            "basis_identity": basis_identity,
            "reference_identity": reference_identity,
            "evidence_identity": evidence_identity,
            **diagnostic_semantics,
        }
    )
    evidence = LinearRecurrentAugmentedReferenceEvidence(
        schema_id=LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_SCHEMA_ID,
        schema_version=LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_SCHEMA_VERSION,
        product_role=LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_ROLE,
        evaluation_manifest_identity=governed.lineage.evaluation_manifest_identity,
        evaluation_lens=governed.evaluation_lens,
        basis_descriptor=governed.basis,
        source_lineage=governed.lineage,
        source_array_identities=governed.array_identities,
        reference_descriptor=governed.reference,
        basis_identity=basis_identity,
        reference_identity=reference_identity,
        evidence_identity=evidence_identity,
        component_descriptors=MappingProxyType(descriptors),
        recurrence_diagnostics=diagnostics,
        **components,
    )
    _validate_scientific_components(evidence)
    return evidence


def _verify_typed_evidence(evidence: LinearRecurrentAugmentedReferenceEvidence) -> None:
    if not isinstance(evidence, LinearRecurrentAugmentedReferenceEvidence):
        raise TypeError("component adapter accepts typed governed evidence only")
    if evidence.schema_id != LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_SCHEMA_ID:
        raise ValueError("unsupported evidence schema identity")
    if evidence.schema_version != LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported evidence schema version")
    if evidence.product_role != LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_ROLE:
        raise ValueError("unsupported evidence product role")
    if evidence.evaluation_lens not in _EVALUATION_LENSES:
        raise ValueError("unsupported durable evaluation lens")
    if (
        evidence.evaluation_manifest_identity
        != evidence.source_lineage.evaluation_manifest_identity
    ):
        raise ValueError("durable evaluation identity conflicts with source lineage")
    _nonempty_dataclass(evidence.source_lineage, "source lineage")
    _nonempty_dataclass(evidence.source_array_identities, "source array identities")
    _nonempty_dataclass(evidence.reference_descriptor, "reference descriptor")
    if (
        evidence.reference_descriptor.family != REFERENCE_FAMILY
        or evidence.reference_descriptor.version != REFERENCE_VERSION
        or evidence.reference_descriptor.solver != REFERENCE_SOLVER
        or evidence.reference_descriptor.control_convention != CONTROL_CONVENTION
    ):
        raise ValueError("unsupported durable reference contract")
    diagnostics = evidence.recurrence_diagnostics
    expected_basis_identity = _identity(
        "basis",
        {
            "descriptor": _dataclass_payload(evidence.basis_descriptor),
            "state_dim": diagnostics["state_dim"],
            "hidden_dim": diagnostics["hidden_dim"],
            "observation_dim": diagnostics["observation_dim"],
            "action_dim": diagnostics["action_dim"],
            "horizon": diagnostics["horizon"],
            "dt": diagnostics["dt"],
            "tau": diagnostics["tau"],
            "alpha": diagnostics["alpha"],
            "graph_spec_identity": evidence.source_lineage.graph_spec_identity,
            "task_spec_identity": evidence.source_lineage.task_spec_identity,
            "task_binding_identity": evidence.source_lineage.task_binding_identity,
        },
    )
    if evidence.basis_identity != expected_basis_identity:
        raise ValueError("stale durable basis identity")
    expected_reference_identity = _reference_identity(
        basis_identity=evidence.basis_identity,
        reference_descriptor=evidence.reference_descriptor,
        source_lineage=evidence.source_lineage,
        source_array_identities=evidence.source_array_identities,
    )
    if evidence.reference_identity != expected_reference_identity:
        raise ValueError("stale durable reference identity")
    if set(evidence.component_descriptors) != set(_COMPONENT_NAMES):
        raise ValueError("evidence component descriptors are incomplete")
    for name in _COMPONENT_NAMES:
        array = np.asarray(getattr(evidence, name))
        descriptor = evidence.component_descriptors[name]
        if descriptor.name != name or descriptor.shape != array.shape:
            raise ValueError(f"stale component descriptor for {name!r}")
        if descriptor.dtype != array.dtype.str or descriptor.sha256 != exact_array_sha256(array):
            raise ValueError(f"stale component identity for {name!r}")
    _validate_scientific_components(evidence)
    expected_identity = _identity(
        "evidence",
        {
            "schema_version": evidence.schema_version,
            "basis_identity": evidence.basis_identity,
            "reference_identity": evidence.reference_identity,
            "evaluation_manifest_identity": evidence.evaluation_manifest_identity,
            "evaluation_lens": evidence.evaluation_lens,
            "source_lineage": _dataclass_payload(evidence.source_lineage),
            "source_array_identities": _dataclass_payload(evidence.source_array_identities),
            "recurrence_diagnostics": _diagnostic_semantics_payload(
                evidence.recurrence_diagnostics
            ),
            "components": {
                name: evidence.component_descriptors[name].model_dump(mode="json")
                for name in _COMPONENT_NAMES
            },
        },
    )
    if evidence.evidence_identity != expected_identity:
        raise ValueError("stale durable evidence identity")


def _reference_identity(
    *,
    basis_identity: str,
    reference_descriptor: GovernedReferenceDescriptor,
    source_lineage: GovernedSourceLineage,
    source_array_identities: GovernedArrayIdentities,
) -> str:
    """Bind the fixed same-coordinate reference to all authoritative sources."""

    return _identity(
        "reference",
        {
            "basis_identity": basis_identity,
            "descriptor": _dataclass_payload(reference_descriptor),
            "reference_policy": source_array_identities.reference_policy,
            "training_intent_identity": source_lineage.training_intent_identity,
            "resolved_training_identity": source_lineage.resolved_training_identity,
            "execution_identity": source_lineage.execution_identity,
            "checkpoint_transaction_identity": (source_lineage.checkpoint_transaction_identity),
            "checkpoint_manifest_identity": source_lineage.checkpoint_manifest_identity,
            "checkpoint_root_identity": source_lineage.checkpoint_root_identity,
            "model_slot_content_digest": source_lineage.model_slot_content_digest,
            "model_slot_structural_abi": source_lineage.model_slot_structural_abi,
            "plant_operators": source_array_identities.plant_operators,
            "observation_operator": source_array_identities.observation_operator,
            "cost_schedule": source_array_identities.cost_schedule,
        },
    )


def _validate_scientific_components(
    evidence: LinearRecurrentAugmentedReferenceEvidence,
) -> None:
    """Reject internally hashed payloads that violate the certificate contract."""

    diagnostics = evidence.recurrence_diagnostics
    if set(diagnostics) != _DIAGNOSTIC_NAMES:
        raise ValueError("recurrence diagnostics fields are incomplete or unexpected")
    state_dim = diagnostics["state_dim"]
    hidden_dim = diagnostics["hidden_dim"]
    action_dim = diagnostics["action_dim"]
    horizon = diagnostics["horizon"]
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (state_dim, hidden_dim, action_dim, horizon)
    ):
        raise ValueError("scientific diagnostics dimensions must be positive integers")
    _validate_basis(evidence.basis_descriptor, state_dim, hidden_dim)
    if diagnostics["linear_recurrence"] is not True:
        raise ValueError("recurrence diagnostics must describe a linear recurrence")
    for name, required in (
        ("timing", _BASIS_TIMING),
        ("state_frame", _STATE_FRAME),
        ("hidden_frame", _HIDDEN_FRAME),
    ):
        if diagnostics[name] != required or diagnostics[name] != getattr(
            evidence.basis_descriptor, name
        ):
            raise ValueError(f"recurrence diagnostic {name!r} conflicts with basis")
    dt = _positive(diagnostics["dt"], "diagnostic dt")
    tau = _positive(diagnostics["tau"], "diagnostic tau")
    alpha = float(diagnostics["alpha"])
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("diagnostic alpha must be finite and in (0, 1]")
    if not np.isclose(alpha, dt / tau, rtol=_ALPHA_RTOL, atol=0.0):
        raise ValueError(f"diagnostic alpha must equal dt/tau within rtol={_ALPHA_RTOL:g}")
    for name, expected in (
        ("basis_identity", evidence.basis_identity),
        ("reference_identity", evidence.reference_identity),
        ("evidence_identity", evidence.evidence_identity),
    ):
        if diagnostics[name] != expected:
            raise ValueError(f"recurrence diagnostic {name!r} conflicts with evidence")
    spectral_radius = float(diagnostics["recurrent_spectral_radius"])
    if not np.isfinite(spectral_radius) or spectral_radius < 0.0:
        raise ValueError("recurrent spectral radius must be finite and nonnegative")
    augmented_dim = state_dim + hidden_dim
    expected_shapes = {
        "candidate_augmented_action_sensitivity": (
            horizon,
            action_dim,
            augmented_dim,
        ),
        "reference_augmented_action_sensitivity": (
            horizon,
            action_dim,
            augmented_dim,
        ),
        "candidate_transition": (horizon, augmented_dim, augmented_dim),
        "reference_transition": (horizon, augmented_dim, augmented_dim),
        "candidate_value_matrices": (horizon + 1, augmented_dim, augmented_dim),
        "reference_value_matrices": (horizon + 1, augmented_dim, augmented_dim),
        "bellman_hessian": (horizon, action_dim, action_dim),
        "action_weight": (horizon, action_dim, action_dim),
    }
    augmented_states = np.asarray(evidence.augmented_states)
    if (
        augmented_states.ndim != 3
        or augmented_states.shape[0] < 1
        or augmented_states.shape[1:] != (horizon + 1, augmented_dim)
    ):
        raise ValueError("augmented_states must have shape batch x (T+1) x augmented_dim")
    for name, expected_shape in expected_shapes.items():
        array = np.asarray(getattr(evidence, name))
        if array.ndim != 3 or array.shape != expected_shape:
            raise ValueError(f"scientific component {name!r} has invalid shape")
    for name in _COMPONENT_NAMES:
        array = np.asarray(getattr(evidence, name))
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"scientific component {name!r} must be numeric")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"scientific component {name!r} must be finite")
    for name in (
        "candidate_value_matrices",
        "reference_value_matrices",
        "bellman_hessian",
        "action_weight",
    ):
        array = np.asarray(getattr(evidence, name))
        if not np.allclose(array, np.swapaxes(array, -1, -2), rtol=1e-10, atol=1e-12):
            raise ValueError(f"scientific component {name!r} must be symmetric")


def _diagnostic_semantics_payload(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Return every stable non-identity diagnostic field for evidence hashing."""

    return {
        name: diagnostics[name]
        for name in _DIAGNOSTIC_NAMES
        if name not in {"basis_identity", "reference_identity", "evidence_identity"}
    }


def _descriptor(name: str, value: NDArray[Any]) -> EvidenceComponentDescriptor:
    return EvidenceComponentDescriptor(
        name=name, sha256=exact_array_sha256(value), shape=value.shape, dtype=value.dtype.str
    )


def _transition(
    a: NDArray[Any], b: NDArray[Any], c: NDArray[Any], rnn: NDArray[Any], k: NDArray[Any], n: int
) -> NDArray[np.float64]:
    horizon, _, nz = k.shape
    result = np.zeros((horizon, nz, nz))
    result[:, :n] = np.concatenate((a, np.zeros((horizon, n, nz - n))), -1) + np.einsum(
        "tij,tjk->tik", b, k
    )
    result[:, n:, :n], result[:, n:, n:] = c, rnn
    return result


def _values(
    q: NDArray[Any], r: NDArray[Any], k: NDArray[Any], f: NDArray[Any], qf: NDArray[Any]
) -> NDArray[np.float64]:
    values, p = [qf], qf
    for qt, rt, kt, ft in zip(q[::-1], r[::-1], k[::-1], f[::-1], strict=True):
        p = qt + kt.T @ rt @ kt + ft.T @ p @ ft
        p = 0.5 * (p + p.T)
        values.append(p)
    return np.asarray(values[::-1])


def _validate_basis(basis: AugmentedBasisDescriptor, n: int, h: int) -> None:
    if (
        basis.timing != _BASIS_TIMING
        or basis.state_frame != _STATE_FRAME
        or basis.hidden_frame != _HIDDEN_FRAME
    ):
        raise ValueError("basis timing/frame conflicts with governed contract")
    if (
        len(basis.state_coordinates) != n
        or len(basis.target_relative) != n
        or len(basis.hidden_coordinates) != h
    ):
        raise ValueError("basis dimensions conflict with decoded arrays")
    names = basis.state_coordinates + basis.hidden_coordinates
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("basis coordinate names must be non-empty and unique")


def _numeric(value: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {result.shape}")
    _finite(result, name)
    return result


def _time(value: ArrayLike, name: str, t: int, rows: int, cols: int) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 2:
        result = result[None]
    if result.ndim != 3 or result.shape[-2:] != (rows, cols) or result.shape[0] not in (1, t):
        raise ValueError(f"{name} has incompatible time/matrix dimensions")
    result = np.broadcast_to(result, (t, rows, cols)).copy()
    _finite(result, name)
    return result


def _broadcast_target(value: ArrayLike, shape: tuple[int, ...]) -> NDArray[np.float64]:
    target = np.asarray(value, dtype=np.float64)
    if target.ndim == 1:
        target = target[None, None]
    elif target.ndim == 2:
        target = target[None]
    try:
        return np.broadcast_to(target, shape).copy()
    except ValueError as exc:
        raise ValueError("target_states do not broadcast to coupled states") from exc


def _finite(value: ArrayLike, name: str) -> None:
    if not np.all(np.isfinite(np.asarray(value))):
        raise ValueError(f"{name} must be finite")


def _symmetric(value: ArrayLike, name: str) -> None:
    array = np.asarray(value)
    if not np.allclose(array, np.swapaxes(array, -1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _group_hash(role: str, *values: ArrayLike) -> str:
    digest = hashlib.sha256(f"{_HASH_CONTRACT}:{role}".encode())
    for value in values:
        digest.update(exact_array_sha256(value).encode("ascii"))
    return f"sha256:{digest.hexdigest()}"


def _identity(role: str, payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"{role}:sha256:{hashlib.sha256(data).hexdigest()}"


def _dataclass_payload(value: Any) -> dict[str, Any]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _nonempty_dataclass(value: Any, label: str) -> None:
    for field in fields(value):
        item = getattr(value, field.name)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} field {field.name!r} must be non-empty")


def _immutable(value: ArrayLike) -> NDArray[Any]:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


__all__ = [
    "AugmentedBasisDescriptor",
    "CONTROL_CONVENTION",
    "EvidenceComponentDescriptor",
    "GovernedArrayIdentities",
    "GovernedLinearRecurrentReferenceInput",
    "GovernedReferenceDescriptor",
    "GovernedSourceLineage",
    "LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE",
    "LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_ROLE",
    "LinearRecurrentAugmentedReferenceEvidence",
    "LinearRecurrentAugmentedReferenceEvidencePayload",
    "LinearRecurrentAugmentedReferenceParams",
    "MissingGovernedReferenceAuthorityError",
    "NonNominalAugmentedClosureError",
    "NON_NOMINAL_REASON_CODE",
    "REFERENCE_FAMILY",
    "REFERENCE_SOLVER",
    "REFERENCE_VERSION",
    "build_governed_linear_recurrent_augmented_reference_evidence",
    "build_governed_reference_input_from_projection",
    "checkpoint_array_sha256",
    "checkpoint_operator_sha256",
    "cost_schedule_sha256",
    "deserialize_governed_evidence",
    "exact_array_sha256",
    "governed_evidence_component_kwargs",
    "non_nominal_augmented_reference_outcome",
    "observation_operator_sha256",
    "plant_operator_sha256",
    "reference_action_operator_sha256",
    "produce_governed_linear_recurrent_augmented_reference_evidence",
    "require_complete_governed_reference_authority",
    "resolve_governed_reference_sources",
    "ResolvedGovernedReferenceSources",
    "scientific_evaluation_identity",
    "serialize_governed_evidence",
]
