"""Focused contract tests for governed augmented reference evidence."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from feedbax.analysis.evaluation import get_evaluation_recipe
from feedbax.contracts.manifest import EvaluationRunSpec, ParentRef, TrainingRunManifest
from pydantic import ValidationError

from rlrmp.eval.linear_recurrent_augmented_reference import (
    CONTROL_CONVENTION,
    REFERENCE_FAMILY,
    REFERENCE_SOLVER,
    REFERENCE_VERSION,
    AugmentedBasisDescriptor,
    GovernedArrayIdentities,
    GovernedLinearRecurrentReferenceInput,
    GovernedReferenceDescriptor,
    GovernedSourceLineage,
    LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE,
    LinearRecurrentAugmentedReferenceParams,
    build_governed_linear_recurrent_augmented_reference_evidence,
    checkpoint_array_sha256,
    checkpoint_operator_sha256,
    cost_schedule_sha256,
    deserialize_governed_evidence,
    exact_array_sha256,
    governed_evidence_component_kwargs,
    observation_operator_sha256,
    plant_operator_sha256,
    reference_action_operator_sha256,
    serialize_governed_evidence,
)
from rlrmp.eval import linear_recurrent_augmented_reference as augmented_reference
from rlrmp.eval.recipes import register_rlrmp_evaluation_recipes
from rlrmp.runtime.params_models import params_model_for
from rlrmp.runtime.spec_migrations import (
    LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVAL_PARAMS_KIND,
    LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_KIND,
    ensure_rlrmp_spec_families,
)


def _input(
    dtype: np.dtype[np.floating] = np.dtype("float64"),
) -> GovernedLinearRecurrentReferenceInput:
    def arr(value: object) -> np.ndarray:
        return np.asarray(value, dtype=dtype)

    arrays = {
        "coupled_states": arr([[[1, 2], [2, 4], [3, 6]]]),
        "target_states": arr([1, 0]),
        "hidden_states": arr([[[0], [0.25], [0.5]]]),
        "input_weight": arr([[2, 0]]),
        "recurrent_weight": arr([[0.5]]),
        "readout_weight": arr([[4]]),
        "state_transition": arr([[1, 0.1], [0, 1]]),
        "action_input": arr([[0], [0.2]]),
        "observation_map": arr([[1, 0], [0, 1]]),
        "state_cost": arr([[1, 0], [0, 1]]),
        "action_cost": arr([[2]]),
        "terminal_state_cost": arr([[3, 0], [0, 3]]),
        "reference_state_action": arr([[[-1, -2]], [[-1, -2]]]),
    }
    identities = GovernedArrayIdentities(
        checkpoint_operators=checkpoint_operator_sha256(
            arrays["input_weight"], arrays["recurrent_weight"], arrays["readout_weight"]
        ),
        evaluation_arrays=checkpoint_array_sha256(
            arrays["coupled_states"], arrays["target_states"], arrays["hidden_states"]
        ),
        plant_operators=plant_operator_sha256(arrays["state_transition"], arrays["action_input"]),
        observation_operator=observation_operator_sha256(arrays["observation_map"]),
        cost_schedule=cost_schedule_sha256(
            arrays["state_cost"], arrays["action_cost"], arrays["terminal_state_cost"]
        ),
        reference_policy=reference_action_operator_sha256(arrays["reference_state_action"]),
    )
    return GovernedLinearRecurrentReferenceInput(
        basis=AugmentedBasisDescriptor(
            state_coordinates=("target_relative_position", "velocity"),
            target_relative=(True, False),
            hidden_coordinates=("hidden_0",),
        ),
        lineage=GovernedSourceLineage(
            graph_spec_identity="graph:sha256:1",
            task_spec_identity="task:sha256:1",
            task_binding_identity="binding:sha256:1",
            training_intent_identity="intent:sha256:1",
            resolved_training_identity="resolved:sha256:1",
            execution_identity="execution:sha256:1",
            evaluation_manifest_identity="evaluation:sha256:1",
            checkpoint_transaction_identity="transaction:sha256:1",
            checkpoint_manifest_identity="checkpoint:sha256:1",
            checkpoint_root_identity="root:sha256:1",
            model_slot_content_digest="sha256:model-slot",
            model_slot_structural_abi="feedbax.VanillaRNN.identity.zero_affine.v1",
        ),
        array_identities=identities,
        reference=GovernedReferenceDescriptor(
            family=REFERENCE_FAMILY,
            version=REFERENCE_VERSION,
            solver=REFERENCE_SOLVER,
            control_convention=CONTROL_CONVENTION,
            source_identity="reference:sha256:1",
        ),
        evaluation_lens="nominal_clean",
        dt=0.1,
        tau=0.2,
        architecture="linear_recurrence",
        component_type="VanillaRNN",
        activation="identity",
        use_bias=False,
        readout_use_bias=False,
        use_noise=False,
        **arrays,
    )


def _rehash(value: GovernedLinearRecurrentReferenceInput) -> GovernedLinearRecurrentReferenceInput:
    return replace(
        value,
        array_identities=GovernedArrayIdentities(
            checkpoint_operators=checkpoint_operator_sha256(
                value.input_weight, value.recurrent_weight, value.readout_weight
            ),
            evaluation_arrays=checkpoint_array_sha256(
                value.coupled_states, value.target_states, value.hidden_states
            ),
            plant_operators=plant_operator_sha256(value.state_transition, value.action_input),
            observation_operator=observation_operator_sha256(value.observation_map),
            cost_schedule=cost_schedule_sha256(
                value.state_cost, value.action_cost, value.terminal_state_cost
            ),
            reference_policy=reference_action_operator_sha256(value.reference_state_action),
        ),
    )


def _two_action_input() -> GovernedLinearRecurrentReferenceInput:
    governed = _input()
    return _rehash(
        replace(
            governed,
            readout_weight=np.asarray([[4.0], [2.0]]),
            action_input=np.asarray([[0.0, 0.1], [0.2, 0.0]]),
            action_cost=np.asarray([[2.0, 0.0], [0.0, 3.0]]),
            reference_state_action=np.asarray(
                [[[-1.0, -2.0], [-0.5, -1.0]], [[-1.0, -2.0], [-0.5, -1.0]]]
            ),
        )
    )


def _self_consistent_component_attack(
    payload: dict[str, object],
    component_name: str,
    values: object,
) -> dict[str, object]:
    """Rehash a payload mutation so scientific validation is the first failing gate."""

    attacked = copy.deepcopy(payload)
    array = np.asarray(values)
    component = attacked["components"][component_name]
    component["values"] = array.tolist()
    component["descriptor"] = {
        "name": component_name,
        "sha256": exact_array_sha256(array),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }
    evidence_identity = augmented_reference._identity(
        "evidence", _evidence_identity_payload(attacked)
    )
    attacked["evidence_identity"] = evidence_identity
    attacked["recurrence_diagnostics"]["evidence_identity"] = evidence_identity
    return attacked


def _evidence_identity_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": payload["schema_version"],
        "basis_identity": payload["basis_identity"],
        "reference_identity": payload["reference_identity"],
        "evaluation_manifest_identity": payload["evaluation_manifest_identity"],
        "evaluation_lens": payload["evaluation_lens"],
        "source_lineage": payload["source_lineage"],
        "source_array_identities": payload["source_array_identities"],
        "recurrence_diagnostics": augmented_reference._diagnostic_semantics_payload(
            payload["recurrence_diagnostics"]
        ),
        "components": {
            name: component_payload["descriptor"]
            for name, component_payload in payload["components"].items()
        },
    }


def _self_consistent_basis_semantics_attack(
    payload: dict[str, object], field: str, value: str
) -> dict[str, object]:
    attacked = copy.deepcopy(payload)
    attacked["basis_descriptor"][field] = value
    attacked["recurrence_diagnostics"][field] = value
    diagnostics = attacked["recurrence_diagnostics"]
    lineage = attacked["source_lineage"]
    basis_identity = augmented_reference._identity(
        "basis",
        {
            "descriptor": attacked["basis_descriptor"],
            "state_dim": diagnostics["state_dim"],
            "hidden_dim": diagnostics["hidden_dim"],
            "observation_dim": diagnostics["observation_dim"],
            "action_dim": diagnostics["action_dim"],
            "horizon": diagnostics["horizon"],
            "dt": diagnostics["dt"],
            "tau": diagnostics["tau"],
            "alpha": diagnostics["alpha"],
            "graph_spec_identity": lineage["graph_spec_identity"],
            "task_spec_identity": lineage["task_spec_identity"],
            "task_binding_identity": lineage["task_binding_identity"],
        },
    )
    attacked["basis_identity"] = basis_identity
    diagnostics["basis_identity"] = basis_identity
    reference_identity = augmented_reference._reference_identity(
        basis_identity=basis_identity,
        reference_descriptor=GovernedReferenceDescriptor(**attacked["reference_descriptor"]),
        source_lineage=GovernedSourceLineage(**lineage),
        source_array_identities=GovernedArrayIdentities(**attacked["source_array_identities"]),
    )
    attacked["reference_identity"] = reference_identity
    diagnostics["reference_identity"] = reference_identity
    evidence_identity = augmented_reference._identity(
        "evidence", _evidence_identity_payload(attacked)
    )
    attacked["evidence_identity"] = evidence_identity
    diagnostics["evidence_identity"] = evidence_identity
    return attacked


def test_params_reject_all_caller_evidence_and_unknowns() -> None:
    params = LinearRecurrentAugmentedReferenceParams(checkpoint_custody_root=Path("/tmp/custody"))
    assert params.states_custody == "durable"
    for forbidden in ({"matrix": [[1]]}, {"path": "/tmp/x"}, {"checkpoint_id": "x"}):
        with pytest.raises(ValidationError, match="Extra inputs"):
            LinearRecurrentAugmentedReferenceParams.model_validate(forbidden)


def test_exact_hash_contract_supports_float32_jax_and_rejects_dtype_drift() -> None:
    governed = _input(np.dtype("float32"))
    evidence = build_governed_linear_recurrent_augmented_reference_evidence(governed)
    assert evidence.augmented_states.dtype == np.float64
    assert exact_array_sha256(jnp.asarray([1.0], dtype=jnp.float32)) == exact_array_sha256(
        np.asarray([1.0], dtype=np.float32)
    )
    assert exact_array_sha256(np.asarray([1.0], dtype=np.float32)) != exact_array_sha256(
        np.asarray([1.0], dtype=np.float64)
    )
    with pytest.raises(ValueError, match="checkpoint_operators"):
        build_governed_linear_recurrent_augmented_reference_evidence(
            replace(governed, input_weight=np.asarray(governed.input_weight, dtype=np.float64))
        )


def test_candidate_reference_algebra_values_hessian_and_axes() -> None:
    evidence = build_governed_linear_recurrent_augmented_reference_evidence(_input())
    np.testing.assert_allclose(
        evidence.candidate_augmented_action_sensitivity,
        np.broadcast_to([[[4, 0, 3]]], (2, 1, 3)),
    )
    np.testing.assert_allclose(
        evidence.reference_augmented_action_sensitivity,
        [[[-1, -2, 0]], [[-1, -2, 0]]],
    )
    candidate_f = np.asarray([[1, 0.1, 0], [0.8, 1, 0.6], [1, 0, 0.75]])
    reference_f = np.asarray([[1, 0.1, 0], [-0.2, 0.6, 0], [1, 0, 0.75]])
    np.testing.assert_allclose(evidence.candidate_transition, [candidate_f, candidate_f])
    np.testing.assert_allclose(evidence.reference_transition, [reference_f, reference_f])
    assert evidence.augmented_states.shape == (1, 3, 3)
    assert evidence.candidate_value_matrices.shape == (3, 3, 3)
    assert evidence.bellman_hessian.shape == (2, 1, 1)
    assert np.allclose(
        evidence.candidate_value_matrices, evidence.candidate_value_matrices.transpose(0, 2, 1)
    )
    assert np.allclose(evidence.bellman_hessian, evidence.bellman_hessian.transpose(0, 2, 1))


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_every_numeric_source_rejects_mutation_with_stale_hash(field: str) -> None:
    governed = _input()
    changed = np.asarray(getattr(governed, field)).copy()
    changed.flat[0] += 0.125
    with pytest.raises(ValueError, match="stale governed"):
        build_governed_linear_recurrent_augmented_reference_evidence(
            replace(governed, **{field: changed})
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("family", "other"),
        ("version", "v2"),
        ("solver", "other"),
        ("control_convention", "action_equals_negative_k_times_state"),
        ("source_identity", ""),
    ],
)
def test_reference_contract_drift_rejects(field: str, value: str) -> None:
    governed = _input()
    with pytest.raises(ValueError):
        build_governed_linear_recurrent_augmented_reference_evidence(
            replace(governed, reference=replace(governed.reference, **{field: value}))
        )


def test_basis_identity_binds_semantics_dimensions_time_constants_and_authorities() -> None:
    base_input = _input()
    base = build_governed_linear_recurrent_augmented_reference_evidence(base_input)
    variants = [
        replace(
            base_input,
            basis=replace(base_input.basis, state_coordinates=("position_v2", "velocity")),
        ),
        replace(base_input, dt=0.2, tau=0.4),
        replace(
            base_input, lineage=replace(base_input.lineage, graph_spec_identity="graph:sha256:2")
        ),
        replace(
            base_input, lineage=replace(base_input.lineage, task_spec_identity="task:sha256:2")
        ),
        replace(
            base_input,
            lineage=replace(base_input.lineage, task_binding_identity="binding:sha256:2"),
        ),
    ]
    for variant in variants:
        assert (
            build_governed_linear_recurrent_augmented_reference_evidence(variant).basis_identity
            != base.basis_identity
        )
    for basis in (
        replace(base_input.basis, timing="pre_step"),
        replace(base_input.basis, state_frame="absolute"),
        replace(base_input.basis, hidden_frame="current_hidden"),
        replace(base_input.basis, hidden_coordinates=()),
        replace(base_input.basis, target_relative=(True,)),
    ):
        with pytest.raises(ValueError):
            build_governed_linear_recurrent_augmented_reference_evidence(
                replace(base_input, basis=basis)
            )
    assert base.recurrence_diagnostics["observation_dim"] == 2
    assert base.recurrence_diagnostics["action_dim"] == 1
    assert base.recurrence_diagnostics["horizon"] == 2


@pytest.mark.parametrize(
    "field",
    [
        "training_intent_identity",
        "resolved_training_identity",
        "execution_identity",
        "checkpoint_transaction_identity",
        "checkpoint_manifest_identity",
        "checkpoint_root_identity",
        "model_slot_content_digest",
        "model_slot_structural_abi",
    ],
)
def test_reference_identity_binds_each_training_and_checkpoint_authority(field: str) -> None:
    governed = _input()
    base = build_governed_linear_recurrent_augmented_reference_evidence(governed)
    changed = build_governed_linear_recurrent_augmented_reference_evidence(
        replace(governed, lineage=replace(governed.lineage, **{field: f"{field}:changed"}))
    )
    assert changed.reference_identity != base.reference_identity


def test_reference_identity_binds_basis_descriptor_policy_and_physical_sources() -> None:
    governed = _input()
    base = build_governed_linear_recurrent_augmented_reference_evidence(governed)
    variants = [
        replace(
            governed,
            basis=replace(
                governed.basis,
                state_coordinates=("target_relative_position_v2", "velocity"),
            ),
        ),
        replace(
            governed,
            reference=replace(governed.reference, source_identity="reference:sha256:changed"),
        ),
        _rehash(
            replace(
                governed,
                reference_state_action=np.asarray([[[-1.25, -2.0]], [[-1.0, -2.0]]]),
            )
        ),
        _rehash(
            replace(
                governed,
                state_transition=np.asarray([[1.0, 0.2], [0.0, 1.0]]),
            )
        ),
        _rehash(
            replace(
                governed,
                observation_map=np.asarray([[1.0, 0.1], [0.0, 1.0]]),
            )
        ),
        _rehash(replace(governed, action_cost=np.asarray([[3.0]]))),
    ]
    identities = {
        build_governed_linear_recurrent_augmented_reference_evidence(variant).reference_identity
        for variant in variants
    }
    assert base.reference_identity not in identities
    assert len(identities) == len(variants)


def test_evidence_identity_binds_evaluation_lens_action_weight_and_all_descriptors() -> None:
    governed = _input()
    base = build_governed_linear_recurrent_augmented_reference_evidence(governed)
    evaluation = build_governed_linear_recurrent_augmented_reference_evidence(
        replace(
            governed,
            lineage=replace(governed.lineage, evaluation_manifest_identity="evaluation:sha256:2"),
        )
    )
    lens = build_governed_linear_recurrent_augmented_reference_evidence(
        replace(governed, evaluation_lens="held_out_validation")
    )
    changed_r = np.asarray([[3.0]])
    action_weight = build_governed_linear_recurrent_augmented_reference_evidence(
        _rehash(replace(governed, action_cost=changed_r))
    )
    assert (
        len(
            {
                base.evidence_identity,
                evaluation.evidence_identity,
                lens.evidence_identity,
                action_weight.evidence_identity,
            }
        )
        == 4
    )
    assert evaluation.reference_identity == base.reference_identity
    assert lens.reference_identity == base.reference_identity
    expected = {
        "augmented_states",
        "candidate_augmented_action_sensitivity",
        "reference_augmented_action_sensitivity",
        "candidate_transition",
        "reference_transition",
        "candidate_value_matrices",
        "reference_value_matrices",
        "bellman_hessian",
        "action_weight",
    }
    assert set(base.component_descriptors) == expected
    for name, descriptor in base.component_descriptors.items():
        array = getattr(base, name)
        assert descriptor.shape == array.shape
        assert descriptor.dtype == array.dtype.str
        assert descriptor.sha256 == exact_array_sha256(array)


def test_strict_durable_payload_round_trip_and_mutation_guards() -> None:
    evidence = build_governed_linear_recurrent_augmented_reference_evidence(_input())
    payload = serialize_governed_evidence(evidence)
    restored = deserialize_governed_evidence(payload)
    assert restored.evidence_identity == evidence.evidence_identity
    assert governed_evidence_component_kwargs(restored)["action_weight"] is restored.action_weight
    with pytest.raises(ValueError):
        restored.action_weight[0, 0, 0] = 9

    malformed = dict(payload)
    malformed["extra"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        deserialize_governed_evidence(malformed)
    stale = dict(payload)
    stale["components"] = dict(payload["components"])
    stale["components"]["action_weight"] = dict(stale["components"]["action_weight"])
    stale["components"]["action_weight"]["values"] = [[[9.0]], [[9.0]]]
    with pytest.raises(ValueError, match="bytes conflict"):
        deserialize_governed_evidence(stale)
    mutable = np.asarray(evidence.action_weight).copy()
    mutable[0, 0, 0] = 9
    with pytest.raises(ValueError, match="stale component identity"):
        governed_evidence_component_kwargs(replace(evidence, action_weight=mutable))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("linear_recurrence", False),
        ("basis_identity", "basis:sha256:changed"),
        ("reference_identity", "reference:sha256:changed"),
        ("evidence_identity", "evidence:sha256:changed"),
        ("timing", "pre_step"),
        ("state_frame", "absolute"),
        ("hidden_frame", "current_hidden"),
        ("dt", -0.1),
        ("tau", -0.2),
        ("alpha", 2.0),
        ("state_dim", 3),
        ("hidden_dim", 2),
        ("observation_dim", 3),
        ("action_dim", 2),
        ("horizon", 3),
        ("recurrent_spectral_radius", -0.1),
    ],
)
def test_deserialization_rejects_every_mutated_diagnostic_field(field: str, value: object) -> None:
    payload = serialize_governed_evidence(
        build_governed_linear_recurrent_augmented_reference_evidence(_input())
    )
    attacked = copy.deepcopy(payload)
    attacked["recurrence_diagnostics"][field] = value
    with pytest.raises(ValueError):
        deserialize_governed_evidence(attacked)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timing", "pre_step"),
        ("state_frame", "absolute"),
        ("hidden_frame", "current_hidden"),
    ],
)
def test_self_consistent_basis_semantics_attack_rejects_fixed_governed_contract(
    field: str, value: str
) -> None:
    payload = serialize_governed_evidence(
        build_governed_linear_recurrent_augmented_reference_evidence(_input())
    )
    attacked = _self_consistent_basis_semantics_attack(payload, field, value)
    with pytest.raises(ValueError, match="basis timing/frame"):
        deserialize_governed_evidence(attacked)


@pytest.mark.parametrize(
    ("component_name", "mutation", "match"),
    [
        ("candidate_transition", np.zeros((2, 3)), "invalid shape"),
        ("candidate_value_matrices", np.zeros((2, 3, 3)), "invalid shape"),
        (
            "augmented_states",
            np.asarray([[[np.nan, 0.0, 0.0]] * 3]),
            "must be finite",
        ),
        (
            "augmented_states",
            np.asarray([[["x", "0", "0"]] * 3]),
            "must be numeric",
        ),
    ],
)
def test_deserialization_rejects_self_consistent_invalid_scientific_components(
    component_name: str,
    mutation: np.ndarray,
    match: str,
) -> None:
    payload = serialize_governed_evidence(
        build_governed_linear_recurrent_augmented_reference_evidence(_input())
    )
    attacked = _self_consistent_component_attack(payload, component_name, mutation)
    with pytest.raises(ValueError, match=match):
        deserialize_governed_evidence(attacked)


@pytest.mark.parametrize(
    "component_name",
    [
        "candidate_value_matrices",
        "reference_value_matrices",
        "bellman_hessian",
        "action_weight",
    ],
)
def test_deserialization_rejects_self_consistent_nonsymmetric_quadratics(
    component_name: str,
) -> None:
    evidence = build_governed_linear_recurrent_augmented_reference_evidence(_two_action_input())
    payload = serialize_governed_evidence(evidence)
    mutation = np.asarray(getattr(evidence, component_name)).copy()
    mutation[0, 0, 1] += 0.25
    attacked = _self_consistent_component_attack(payload, component_name, mutation)
    with pytest.raises(ValueError, match="must be symmetric"):
        deserialize_governed_evidence(attacked)


def test_recipe_params_and_both_schema_families_registered() -> None:
    register_rlrmp_evaluation_recipes(replace=True)
    assert get_evaluation_recipe(LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE)
    assert (
        params_model_for(LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE)
        is LinearRecurrentAugmentedReferenceParams
    )
    registry = ensure_rlrmp_spec_families()
    assert registry.resolve(LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVAL_PARAMS_KIND)
    assert registry.resolve(LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVIDENCE_KIND)


def _resolved_training_source(
    checkpoint_refs: list[ParentRef],
) -> SimpleNamespace:
    manifest = TrainingRunManifest(
        id="training-run",
        job_id="job",
        status="completed",
        completed_batches=2,
        checkpoint_custody=checkpoint_refs,
    )
    return SimpleNamespace(manifest=manifest)


def _resolved_checkpoint(ref: ParentRef, *, completed_batches: int) -> SimpleNamespace:
    return SimpleNamespace(
        parent_ref=ref,
        manifest=SimpleNamespace(
            transaction_id=ref.id,
            run_id="job",
            completed_training_batches=completed_batches,
        ),
        slots={"model": object()},
    )


def test_public_source_resolution_selects_manifest_declared_terminal_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    refs = [
        ParentRef(
            kind="TrainingCheckpointTransactionManifest",
            id=f"tx-{index}",
            role="training_checkpoint_custody",
            uri=f"transactions/tx-{index}/manifest.json",
            metadata={"manifest_sha256": str(index) * 64},
        )
        for index in (1, 2)
    ]
    source = _resolved_training_source(refs)
    calls: list[tuple[ParentRef, object, object]] = []

    monkeypatch.setattr(
        augmented_reference,
        "resolve_evaluation_inputs",
        lambda run_spec, *, manifest_root: (source,),
    )

    def resolve_checkpoint(
        ref: ParentRef,
        *,
        allowed_root: object,
        slot_names: object,
    ) -> SimpleNamespace:
        calls.append((ref, allowed_root, slot_names))
        return _resolved_checkpoint(ref, completed_batches=int(ref.id[-1]))

    monkeypatch.setattr(
        augmented_reference,
        "resolve_checkpoint_custody_ref",
        resolve_checkpoint,
    )
    spec = EvaluationRunSpec(
        evaluation_type=LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE,
        inputs=[ParentRef(kind="TrainingRunManifest", id="training-run", role="training_run")],
    )
    resolved = augmented_reference.resolve_governed_reference_sources(
        spec,
        manifest_root=tmp_path,
        checkpoint_root=tmp_path,
    )
    assert resolved.training is source
    assert resolved.checkpoint.parent_ref == refs[1]
    assert calls == [(refs[0], tmp_path, ("model",)), (refs[1], tmp_path, ("model",))]


def test_public_source_resolution_rejects_checkpoint_ref_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    declared = ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id="tx-2",
        role="training_checkpoint_custody",
        uri="transactions/tx-2/manifest.json",
        metadata={"manifest_sha256": "2" * 64},
    )
    drifted = declared.model_copy(update={"uri": "other/manifest.json"})
    monkeypatch.setattr(
        augmented_reference,
        "resolve_evaluation_inputs",
        lambda run_spec, *, manifest_root: (_resolved_training_source([declared]),),
    )
    monkeypatch.setattr(
        augmented_reference,
        "resolve_checkpoint_custody_ref",
        lambda ref, *, allowed_root, slot_names: _resolved_checkpoint(drifted, completed_batches=2),
    )
    spec = EvaluationRunSpec(
        evaluation_type=LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE,
        inputs=[ParentRef(kind="TrainingRunManifest", id="training-run", role="training_run")],
    )
    with pytest.raises(ValueError, match="differs from manifest declaration"):
        augmented_reference.resolve_governed_reference_sources(
            spec,
            manifest_root=tmp_path,
            checkpoint_root=tmp_path,
        )


def test_complete_authority_fails_closed_on_missing_evaluated_basis_arrays() -> None:
    source = _resolved_training_source([])
    source.manifest = source.manifest.model_copy(
        update={
            "graph_spec": {"inline": {}, "kind": "graph"},
            "training_spec": {"inline": {}, "kind": "training"},
            "task_spec": {"inline": {}, "kind": "task"},
            "task_binding_spec": {"inline": {}, "kind": "task_binding"},
            "intent_hash": "1" * 64,
            "resolved_semantics_root_hash": "2" * 64,
            "execution_hash": None,
        }
    )
    sources = augmented_reference.ResolvedGovernedReferenceSources(
        training=source,
        checkpoint=SimpleNamespace(),
    )
    with pytest.raises(
        augmented_reference.MissingGovernedReferenceAuthorityError,
        match="TrainingRunManifest.execution_hash",
    ):
        augmented_reference.require_complete_governed_reference_authority(sources)
