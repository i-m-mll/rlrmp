"""Real-checkpoint boundary proof for governed augmented reference evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from feedbax.analysis import EMPTY_STAGED_EXECUTION_CONTEXT, resolve_evaluation_inputs
from feedbax.analysis.evaluation import execute_evaluation_run_spec, load_evaluation_states
from feedbax.contracts.manifest import EvaluationRunSpec, ParentRef
from feedbax.contracts.training import TrainingRunSpec
from feedbax.training import (
    DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY,
    ExecutionPreparationRequest,
    resolve_checkpoint_custody_ref,
)
from feedbax.training.executor import execute_training_run_spec

from rlrmp.eval.linear_recurrent_augmented_reference import (
    LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE,
    NON_NOMINAL_REASON_CODE,
    LinearRecurrentAugmentedReferenceParams,
    NonNominalAugmentedClosureError,
    build_governed_linear_recurrent_augmented_reference_evidence,
    build_governed_reference_input_from_projection,
    deserialize_governed_evidence,
    governed_evidence_component_kwargs,
    non_nominal_augmented_reference_outcome,
    resolve_governed_reference_sources,
    serialize_governed_evidence,
)
from rlrmp.eval.model_slots import project_training_model_slot
from rlrmp.eval.recipes import (
    linear_recurrent_augmented_reference_recipe,
    register_rlrmp_evaluation_recipes,
)
from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods
from rlrmp.runtime.training_run_specs import (
    CsSupervisedCheckpointPolicyPayload,
    CsSupervisedMethodPayload,
    cs_supervised_optimizer_spec,
)
from rlrmp.train.execution_preparation import register_execution_preparations
from rlrmp.train.linear_recurrent_native import (
    author_linear_recurrent_training_base_from_canonical,
)
from rlrmp.train.training_base_routes import route_training_base


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_BASE = REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"


def _projection_with_run_spec_mutation(projection, mutate):
    payload = json.loads(projection.run_spec_json)
    mutate(payload)
    return replace(
        projection,
        run_spec_json=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
    )


def _tiny_current_linear_recurrent_spec(tmp_path: Path, *, seed: int = 0) -> TrainingRunSpec:
    """Repair the historical base through the current typed optimizer contract."""

    register_rlrmp_training_methods()
    register_execution_preparations(DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY)
    payload = json.loads(CANONICAL_BASE.read_text(encoding="utf-8"))
    base = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    config = dict(base.method_payload.payload["config"])
    config.update(
        {
            "allow_fresh_start": True,
            "batch_size": 1,
            "checkpoint_interval_batches": 1,
            "constant_lr_iterations": 1,
            "disable_progress": True,
            "full_train": True,
            "hidden_size": 2,
            "issue": "0d6c2ae",
            "lr_warmup_batches": 1,
            "n_batches_condition": 2,
            "n_replicates": 1,
            "n_train_batches": 2,
            "output_dir": str(tmp_path / "bulk"),
            "quiet_progress": True,
            "resume": False,
            "seed": seed,
            "spec_dir": str(tmp_path / "spec"),
            "training_diagnostics": False,
        }
    )
    current_payload = CsSupervisedMethodPayload(
        config=config,
        training_mode="nominal",
        n_train_batches=2,
        batch_size=1,
        optimizer=cs_supervised_optimizer_spec(config=config, n_train_batches=2),
        optimizer_policy={"source": "current typed canonical route"},
        training_diagnostics={"enabled": False},
        checkpoint_policy=CsSupervisedCheckpointPolicyPayload(
            checkpoint_interval_batches=1,
            artifact_root=str(tmp_path / "bulk"),
            tracked_spec_dir=str(tmp_path / "spec"),
        ),
    )
    base = base.model_copy(
        update={
            "training_config": base.training_config.model_copy(
                update={
                    "n_batches": 2,
                    "batch_size": 1,
                    "hidden_dim": 2,
                    "snapshot_interval": 1,
                }
            ),
            "checkpoint_progress": base.checkpoint_progress.model_copy(
                update={"checkpoint_interval": 1, "continuation": None}
            ),
            "method_payload": base.method_payload.model_copy(
                update={"payload": current_payload.model_dump(mode="json", exclude_none=True)}
            ),
        }
    )
    authored = author_linear_recurrent_training_base_from_canonical(base)
    routed = route_training_base(authored, issue="0d6c2ae", row_id="real-checkpoint")
    return routed.model_copy(
        update={
            "artifacts": routed.artifacts.model_copy(
                update={
                    "artifact_root": str(tmp_path / "custody"),
                    "manifest_root": str(tmp_path),
                }
            )
        }
    )


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64")
@pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
def test_real_checkpoint_produces_same_basis_augmented_reference_evidence(
    tmp_path: Path,
) -> None:
    """Prove public custody, projection, eval_trials, and same-basis evidence."""

    register_rlrmp_evaluation_recipes(replace=True)
    analysis_root = tmp_path / "analysis"
    checkpoint_root = tmp_path / "checkpoint-custody"
    spec = _tiny_current_linear_recurrent_spec(tmp_path / "training")
    preparation = DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.prepare(
        ExecutionPreparationRequest(run_spec=spec, run_id="real-checkpoint", resume=False)
    )
    result = execute_training_run_spec(
        spec,
        run_id="real-checkpoint",
        initial_slots=preparation.initial_slots,
        kernel_context=preparation.kernel_context,
        loss_service=preparation.loss_service,
        manifest_root=analysis_root,
        checkpoint_root=checkpoint_root,
        resume=False,
        resume_slot_transform=preparation.resume_slot_transform,
        issues=["0d6c2ae"],
    )
    assert result.status == result.manifest.status == "completed"
    manifest_path = (
        analysis_root
        / "manifests"
        / "training_runs"
        / (result.manifest.id.replace(":", "_") + ".json")
    )
    raw_manifest = manifest_path.read_bytes()
    training_ref = ParentRef(
        kind="TrainingRunManifest",
        id=result.manifest.id,
        role="training_run",
        uri=manifest_path.relative_to(analysis_root).as_posix(),
        metadata={"manifest_sha256": hashlib.sha256(raw_manifest).hexdigest()},
    )
    evaluation_spec = EvaluationRunSpec(
        evaluation_type=LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE,
        inputs=[training_ref],
        params={
            "schema_id": "rlrmp.spec.evaluation.linear_recurrent_augmented_reference",
            "schema_version": "rlrmp.spec.evaluation.linear_recurrent_augmented_reference.v1",
            "checkpoint_custody_root": str(checkpoint_root),
            "states_custody": "durable",
        },
    )

    (training,) = resolve_evaluation_inputs(evaluation_spec, manifest_root=analysis_root)
    assert training.manifest == result.manifest
    assert training.sha256 == training_ref.metadata["manifest_sha256"]
    assert training.manifest.checkpoint_custody
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_governed_reference_sources(
            evaluation_spec,
            manifest_root=analysis_root,
            checkpoint_root=tmp_path / "wrong-checkpoint-root",
        )
    resolved = resolve_governed_reference_sources(
        evaluation_spec,
        manifest_root=analysis_root,
        checkpoint_root=checkpoint_root,
    )
    checkpoint = resolved.checkpoint
    direct = resolve_checkpoint_custody_ref(
        checkpoint.parent_ref,
        allowed_root=checkpoint_root,
        slot_names=("model",),
    )
    assert checkpoint.parent_ref in training.manifest.checkpoint_custody
    assert checkpoint.parent_ref == direct.parent_ref
    assert checkpoint.manifest_sha256 == checkpoint.parent_ref.metadata["manifest_sha256"]
    assert checkpoint.manifest.transaction_id == checkpoint.parent_ref.id
    assert checkpoint.manifest.run_id == training.manifest.job_id
    assert checkpoint.manifest.completed_training_batches == training.manifest.completed_batches
    assert set(checkpoint.slots) == {"model"}
    assert type(checkpoint.slots["model"]) is type(result.final_slots["model"])
    model_record = next(slot for slot in checkpoint.manifest.slots if slot.slot == "model")
    assert model_record.sha256
    assert model_record.content_digest.slot_root_sha256
    assert model_record.structural_abi_fingerprint.fingerprint_sha256

    projection = project_training_model_slot(resolved.training, resolved.checkpoint)
    authority_params = LinearRecurrentAugmentedReferenceParams(
        checkpoint_custody_root=checkpoint_root
    )

    def drift_dimension(payload):
        payload["task"]["params"]["game_card"]["plant"]["state_dim"] = 35

    def drift_selector(payload):
        payload["graph"]["inline"]["nodes"]["feedback"]["params"]["state_slices"][
            "delayed_position"
        ]["indices"] = [29, 30]

    def drift_bank(payload):
        bins = payload["task"]["params"]["target_relative_multitarget"]["validation_bins"]
        payload["task"]["params"]["target_relative_multitarget"]["validation_bins"] = [
            item for item in bins if item["bin"] != "held_out_multitarget_nominal"
        ]

    for mutate, match in (
        (drift_dimension, "36D basis"),
        (drift_selector, "feedback selector authority drifted"),
        (drift_bank, "validation-bank roles"),
    ):
        with pytest.raises(ValueError, match=match):
            build_governed_reference_input_from_projection(
                _projection_with_run_spec_mutation(projection, mutate),
                authority_params,
                evaluation_manifest_identity="feedbax-evaluation-run:authority-drift",
            )
    for lens in (
        "riccati_epsilon",
        "process_noise",
        "coverage_induced",
        "held_out_validation",
    ):
        params = LinearRecurrentAugmentedReferenceParams(
            checkpoint_custody_root=checkpoint_root,
            evaluation_lens=lens,
        )
        with pytest.raises(NonNominalAugmentedClosureError, match=NON_NOMINAL_REASON_CODE):
            build_governed_reference_input_from_projection(
                projection,
                params,
                evaluation_manifest_identity=f"feedbax-evaluation-run:{lens}",
            )
        outcome = non_nominal_augmented_reference_outcome(params)
        assert outcome["status"] == "not_applicable"
        assert outcome["reason_code"] == NON_NOMINAL_REASON_CODE
        assert outcome["evaluation_lens"] == lens
        recipe_result = linear_recurrent_augmented_reference_recipe(
            evaluation_spec.model_copy(
                update={"params": {**evaluation_spec.params, "evaluation_lens": lens}}
            ),
            analysis_root,
            tmp_path / "unused-states",
            EMPTY_STAGED_EXECUTION_CONTEXT,
        )
        assert recipe_result.states == outcome
        assert recipe_result.metadata["reason_code"] == NON_NOMINAL_REASON_CODE
    governed = build_governed_reference_input_from_projection(
        projection,
        LinearRecurrentAugmentedReferenceParams(checkpoint_custody_root=checkpoint_root),
        evaluation_manifest_identity="feedbax-evaluation-run:real-checkpoint-proof",
    )
    evidence = build_governed_linear_recurrent_augmented_reference_evidence(governed)
    restored = deserialize_governed_evidence(serialize_governed_evidence(evidence))
    kwargs = governed_evidence_component_kwargs(restored)
    assert set(kwargs) == {
        "augmented_states",
        "candidate_augmented_action_sensitivity",
        "reference_augmented_action_sensitivity",
        "candidate_transition",
        "reference_transition",
        "candidate_value_matrices",
        "reference_value_matrices",
        "bellman_hessian",
        "action_weight",
        "recurrence_diagnostics",
        "state_label",
    }
    assert restored.augmented_states.shape[1] == restored.candidate_transition.shape[0] + 1
    assert restored.augmented_states.shape[-1] == (restored.candidate_transition.shape[-1])
    assert restored.source_lineage.checkpoint_transaction_identity == (
        projection.provenance.checkpoint_transaction_id
    )
    assert restored.reference_descriptor.source_identity.endswith(
        "78108ca2286af701583e5c4eb87a92736820b5c9260129637722c61831a9e52f"
    )

    manifests = []
    for lens in (
        "nominal_clean",
        "riccati_epsilon",
        "process_noise",
        "coverage_induced",
        "held_out_validation",
    ):
        lens_spec = evaluation_spec.model_copy(
            update={"params": {**evaluation_spec.params, "evaluation_lens": lens}}
        )
        manifest, _ = execute_evaluation_run_spec(lens_spec, root=analysis_root, force=True)
        manifests.append(manifest)
        materialized = load_evaluation_states(manifest, root=analysis_root)
        if lens == "nominal_clean":
            public_evidence = deserialize_governed_evidence(materialized)
            assert public_evidence.evidence_identity
            assert governed_evidence_component_kwargs(public_evidence)
        else:
            assert materialized["status"] == "not_applicable"
            assert materialized["reason_code"] == NON_NOMINAL_REASON_CODE
            assert materialized["evaluation_lens"] == lens
    assert len(manifests) == 5
    assert len({manifest.id for manifest in manifests}) == 5
    assert all(manifest.status == "completed" for manifest in manifests)


def _authenticated_projection_and_evidence(tmp_path: Path, *, run_id: str, seed: int):
    register_rlrmp_evaluation_recipes(replace=True)
    spec = _tiny_current_linear_recurrent_spec(tmp_path, seed=seed)
    preparation = DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.prepare(
        ExecutionPreparationRequest(run_spec=spec, run_id=run_id, resume=False)
    )
    result = execute_training_run_spec(
        spec,
        run_id=run_id,
        initial_slots=preparation.initial_slots,
        kernel_context=preparation.kernel_context,
        loss_service=preparation.loss_service,
        manifest_root=tmp_path,
        checkpoint_root=tmp_path / "checkpoints",
        resume=False,
        resume_slot_transform=preparation.resume_slot_transform,
        issues=["0d6c2ae"],
    )
    manifest_path = (
        tmp_path / "manifests" / "training_runs" / (result.manifest.id.replace(":", "_") + ".json")
    )
    training_ref = ParentRef(
        kind="TrainingRunManifest",
        id=result.manifest.id,
        role="training_run",
        uri=manifest_path.relative_to(tmp_path).as_posix(),
        metadata={"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
    )
    evaluation_spec = EvaluationRunSpec(
        evaluation_type=LINEAR_RECURRENT_AUGMENTED_REFERENCE_EVALUATION_TYPE,
        inputs=[training_ref],
        params={
            "schema_id": "rlrmp.spec.evaluation.linear_recurrent_augmented_reference",
            "schema_version": "rlrmp.spec.evaluation.linear_recurrent_augmented_reference.v1",
            "checkpoint_custody_root": str(tmp_path),
        },
    )
    resolved = resolve_governed_reference_sources(
        evaluation_spec,
        manifest_root=tmp_path,
        checkpoint_root=tmp_path,
    )
    projection = project_training_model_slot(resolved.training, resolved.checkpoint)
    governed = build_governed_reference_input_from_projection(
        projection,
        LinearRecurrentAugmentedReferenceParams(checkpoint_custody_root=tmp_path),
        evaluation_manifest_identity=f"feedbax-evaluation-run:{run_id}",
    )
    evidence = build_governed_linear_recurrent_augmented_reference_evidence(governed)
    return projection, governed, evidence


@pytest.mark.filterwarnings("ignore:Explicitly requested dtype float64")
@pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
def test_altered_authenticated_checkpoint_changes_reference_input_path(tmp_path: Path) -> None:
    """An independently authenticated changed model must change all downstream identity."""

    first = _authenticated_projection_and_evidence(tmp_path / "first", run_id="first", seed=0)
    second = _authenticated_projection_and_evidence(tmp_path / "second", run_id="second", seed=1)
    projection_a, governed_a, evidence_a = first
    projection_b, governed_b, evidence_b = second
    assert projection_a.provenance.slot_root_sha256 != projection_b.provenance.slot_root_sha256
    assert (
        governed_a.array_identities.checkpoint_operators
        != governed_b.array_identities.checkpoint_operators
    )
    assert evidence_a.evidence_identity != evidence_b.evidence_identity
    actions_a = np.einsum(
        "bth,ah->bta", np.asarray(governed_a.hidden_states), governed_a.readout_weight
    )
    actions_b = np.einsum(
        "bth,ah->bta", np.asarray(governed_b.hidden_states), governed_b.readout_weight
    )
    inputs_a = np.einsum(
        "os,bts->bto",
        governed_a.observation_map,
        np.asarray(governed_a.coupled_states) - np.asarray(governed_a.target_states),
    )
    inputs_b = np.einsum(
        "os,bts->bto",
        governed_b.observation_map,
        np.asarray(governed_b.coupled_states) - np.asarray(governed_b.target_states),
    )
    assert np.max(np.abs(actions_a - actions_b)) > 0.0
    assert np.max(np.abs(inputs_a - inputs_b)) > 0.0
