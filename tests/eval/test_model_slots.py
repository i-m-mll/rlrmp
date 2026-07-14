from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from feedbax.contracts.manifest import EvaluationRunSpec, ParentRef
from feedbax.contracts.manifest import TrainingRunManifest, canonical_json_bytes, sha256_bytes
from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.training import structural_abi_fingerprint, write_checkpoint_transaction

from rlrmp.eval import model_slots
from rlrmp.eval import recipes
from rlrmp.eval import ensemble


class _TinyModel(eqx.Module):
    weight: jnp.ndarray
    label: str


def test_ensemble_evaluation_accepts_native_trials_without_disturbance_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native target-distribution trials need no legacy disturbance intervenor."""

    trial_specs = SimpleNamespace(
        inputs={"target": jnp.zeros((1, 2, 2))},
        intervene={},
    )
    sentinel = object()
    observed: dict[str, object] = {}

    def public_evaluator(task, model, trials, *, key, n_replicates):
        observed.update(
            task=task,
            model=model,
            trials=trials,
            key=key,
            n_replicates=n_replicates,
        )
        return sentinel

    monkeypatch.setattr(ensemble, "feedbax_eval_ensemble_on_trials", public_evaluator)
    task = object()
    model = object()
    key = jr.PRNGKey(3)

    result = ensemble.eval_ensemble_on_trials(
        task,
        model,
        trial_specs,
        key=key,
        n_replicates=1,
    )

    assert result is sentinel
    assert observed["task"] is task
    assert observed["model"] is model
    assert observed["trials"] is trial_specs
    assert observed["n_replicates"] == 1
    assert jnp.array_equal(observed["key"], key)


def test_public_projection_accepts_verified_training_and_checkpoint_inputs_only() -> None:
    signature = inspect.signature(model_slots.project_training_model_slot)

    assert tuple(signature.parameters) == ("resolved_input", "resolved_checkpoint")


def test_projection_module_has_no_ambient_or_legacy_checkpoint_route() -> None:
    source = inspect.getsource(model_slots)

    for forbidden in (
        "latest.json",
        "model.eqx",
        "load_latest_checkpoint",
        "tree_deserialise_leaves",
        "caller_template",
        "caller_arrays",
    ):
        assert forbidden not in source
    assert 'slot_names=("model",)' in source


def test_rehydrate_model_uses_governed_template_structure() -> None:
    template = _TinyModel(weight=jnp.zeros((2, 3)), label="governed-static-field")
    slot = (jnp.ones((2, 3)),)

    projected = model_slots._rehydrate_model(slot, template)

    assert projected.label == "governed-static-field"
    assert jnp.array_equal(projected.weight, jnp.ones((2, 3)))
    with pytest.raises(model_slots.ModelSlotProjectionError, match="leaf count"):
        model_slots._rehydrate_model((), template)


@pytest.mark.parametrize(
    "authenticated_slot",
    (
        (jnp.zeros((3, 2)),),
        (jnp.zeros((2, 3), dtype=jnp.int32),),
        ((jnp.zeros((2, 3)),),),
        (jnp.zeros((2, 3)), jnp.zeros((1,))),
    ),
    ids=("shape", "dtype", "treedef", "arity"),
)
def test_template_abi_fails_before_combine_on_any_structural_drift(
    authenticated_slot: tuple[object, ...],
) -> None:
    template = _TinyModel(weight=jnp.zeros((2, 3)), label="governed-static-field")
    slot_record = SimpleNamespace(
        structural_abi_fingerprint=structural_abi_fingerprint(authenticated_slot)
    )

    with pytest.raises(model_slots.ModelSlotProjectionError, match="template structural ABI"):
        model_slots._validate_model_template_abi(template, slot_record)


def test_model_slot_provenance_is_immutable() -> None:
    provenance = model_slots.ModelSlotProvenance(
        training_manifest_id="training",
        training_manifest_sha256="a" * 64,
        training_manifest_reference="runs/training/manifest.json",
        run_id="run",
        completed_batches=1,
        checkpoint_transaction_id="tx",
        checkpoint_manifest_sha256="b" * 64,
        transaction_root_sha256="f" * 64,
        checkpoint_status="partial",
        slot_name="model",
        slot_blob_sha256="c" * 64,
        slot_root_sha256="d" * 64,
        structural_abi_sha256="e" * 64,
        method_ref="rlrmp/cs_supervised/v1",
        architecture="gru",
    )

    with pytest.raises(FrozenInstanceError):
        provenance.completed_batches = 2  # type: ignore[misc]


def test_run_contract_allows_authenticated_v2_v3_projection_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection_v2 = {"training_run_spec": {"value": -0.0}, "phase_program": {"name": "p"}}
    projection_v3 = {"training_run_spec": {"value": 0.0}, "phase_program": {"name": "p"}}
    actual = _run_contract_binding("v2", projection_v2)
    expected = _run_contract_binding("v3", projection_v3)
    run_spec = SimpleNamespace(
        worker_execution=SimpleNamespace(method_contract=SimpleNamespace(phase_program=object()))
    )
    resolved = SimpleNamespace(manifest=SimpleNamespace(run_contract_binding=actual))
    monkeypatch.setattr("feedbax.training.run_contract_binding", lambda *_: expected)

    model_slots._validate_run_contract(resolved, run_spec)


def test_run_contract_rejects_any_canonical_projection_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = _run_contract_binding("v2", {"training_run_spec": {"seed": 1}})
    expected = _run_contract_binding("v3", {"training_run_spec": {"seed": 2}})
    run_spec = SimpleNamespace(
        worker_execution=SimpleNamespace(method_contract=SimpleNamespace(phase_program=object()))
    )
    resolved = SimpleNamespace(manifest=SimpleNamespace(run_contract_binding=actual))
    monkeypatch.setattr("feedbax.training.run_contract_binding", lambda *_: expected)

    with pytest.raises(model_slots.ModelSlotProjectionError, match="canonical run-contract"):
        model_slots._validate_run_contract(resolved, run_spec)


def test_run_contract_rejects_stale_or_forged_projection_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = {"training_run_spec": {"seed": 1}}
    actual = _run_contract_binding("v2", projection)
    actual.canonical_projection_sha256 = "0" * 64
    expected = _run_contract_binding("v3", projection)
    resolved = SimpleNamespace(manifest=SimpleNamespace(run_contract_binding=actual))
    run_spec = SimpleNamespace(
        worker_execution=SimpleNamespace(method_contract=SimpleNamespace(phase_program=object()))
    )
    monkeypatch.setattr("feedbax.training.run_contract_binding", lambda *_: expected)

    with pytest.raises(model_slots.ModelSlotProjectionError, match="stale or forged"):
        model_slots._validate_run_contract(resolved, run_spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("algorithm_version", "unsupported", "algorithm"),
        ("hash_domain", "unsupported", "hash domain"),
    ),
)
def test_run_contract_rejects_unsupported_authentication_contract(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    projection = {"training_run_spec": {"seed": 1}}
    actual = _run_contract_binding("v3", projection)
    setattr(actual, field, value)
    expected = _run_contract_binding("v3", projection)
    resolved = SimpleNamespace(manifest=SimpleNamespace(run_contract_binding=actual))
    run_spec = SimpleNamespace(
        worker_execution=SimpleNamespace(method_contract=SimpleNamespace(phase_program=object()))
    )
    monkeypatch.setattr("feedbax.training.run_contract_binding", lambda *_: expected)

    with pytest.raises(model_slots.ModelSlotProjectionError, match=message):
        model_slots._validate_run_contract(resolved, run_spec)


def _run_contract_binding(version: str, projection: dict[str, object]) -> SimpleNamespace:
    algorithm = f"feedbax.training_checkpoint.run_contract_binding.{version}"
    projection_bytes = (
        canonical_json_bytes(projection)
        if version == "v2"
        else model_slots._normalized_canonical_json(projection)
    )
    return SimpleNamespace(
        algorithm_version=algorithm,
        hash_domain="migrated-canonical-json",
        canonical_projection=projection,
        canonical_projection_sha256=sha256_bytes(projection_bytes),
    )


def test_completed_manifest_requires_completed_timestamp() -> None:
    from feedbax.analysis import ResolvedEvaluationInput

    ref = ParentRef(kind="TrainingRunManifest", id="training", role="training_run")
    resolved = ResolvedEvaluationInput(
        ref=ref,
        manifest=SimpleNamespace(
            id="training",
            status="completed",
            completed_at=None,
        ),
        path=Path("manifest.json"),
        reference="manifest.json",
        sha256="a" * 64,
    )

    with pytest.raises(model_slots.ModelSlotProjectionError, match="completed_at"):
        model_slots.project_training_model_slot(resolved, object())


def test_rlrmp_run_spec_delegates_to_deadff_public_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlrmp.runtime import training_run_specs
    from rlrmp.runtime.spec_migrations import RUN_SPEC_SCHEMA_ID, RUN_SPEC_SCHEMA_VERSION

    nested = {
        "schema_id": "feedbax.spec.training_run",
        "schema_version": "feedbax.spec.training_run.v2",
    }
    payload = SimpleNamespace(
        kind="RLRMPRunSpec",
        schema_id=RUN_SPEC_SCHEMA_ID,
        schema_version=RUN_SPEC_SCHEMA_VERSION,
        inline={
            "schema_id": RUN_SPEC_SCHEMA_ID,
            "schema_version": RUN_SPEC_SCHEMA_VERSION,
            "feedbax_training_run_spec": nested,
        },
    )
    observed: dict[str, object] = {}

    class _TypedSpec:
        def model_dump(self, **_kwargs):
            return nested

    def extract(record):
        observed["record"] = record
        return _TypedSpec()

    monkeypatch.setattr(
        training_run_specs,
        "feedbax_training_run_spec_from_rlrmp_record",
        extract,
        raising=False,
    )

    assert model_slots._embedded_feedbax_training_run_spec_payload(payload) == nested
    assert observed["record"] == payload.inline


def test_projection_accepts_deadff_canonical_registered_run_record(tmp_path: Path) -> None:
    try:
        from rlrmp.runtime.training_run_specs import (
            feedbax_training_run_spec_from_rlrmp_record,
        )
        from rlrmp.train.matrix_lowering import (
            RlrmpTrainingAuthoringIntent,
            lower_rlrmp_training_row,
        )
        from rlrmp.train.native_manifest import RLRMP_NATIVE_MANIFEST_COMPANION_KEY
        from rlrmp.train.training_configs import CsNominalGruConfig
    except ImportError:
        pytest.skip("requires parent-staged deadff5 public run-record contract")
    from feedbax.contracts.run_matrix import AuthoredTrainingRow
    from feedbax.contracts.spec_storage import training_spec_sha256

    intent = RlrmpTrainingAuthoringIntent(
        config=CsNominalGruConfig(
            issue="639e30f",
            output_dir=str(tmp_path / "artifacts"),
            spec_dir=str(tmp_path / "spec"),
            dry_run=True,
            smoke=False,
            target_relative_multitarget=True,
            n_train_batches=100,
            lr_warmup_batches=50,
        )
    ).model_dump(mode="json", exclude_none=True)
    lowered = lower_rlrmp_training_row(
        AuthoredTrainingRow(
            row_id="projection-contract",
            row_index=0,
            payload=intent,
            payload_hash=training_spec_sha256(intent),
            axis_coordinates={},
            seed=13,
        )
    ).execution_payload
    run_record = lowered["metadata"][RLRMP_NATIVE_MANIFEST_COMPANION_KEY]["training_spec_payload"]
    payload = SimpleNamespace(kind="RLRMPRunSpec", inline=run_record)

    projected = model_slots._embedded_feedbax_training_run_spec_payload(payload)
    expected = feedbax_training_run_spec_from_rlrmp_record(run_record)

    assert projected == expected.model_dump(mode="json", exclude_none=True)


@pytest.mark.parametrize(
    ("schema_id", "schema_version", "nested", "message"),
    (
        ("wrong", "rlrmp.run_spec.v2", {}, "envelope"),
        ("rlrmp.run_spec", "wrong", {}, "envelope"),
        ("rlrmp.run_spec", "rlrmp.run_spec.v2", None, "feedbax_training_run_spec"),
    ),
)
def test_rlrmp_run_spec_rejects_identity_drift_or_missing_nested_execution_spec(
    schema_id: str,
    schema_version: str,
    nested: object,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlrmp.runtime import training_run_specs

    inline = {"schema_id": "rlrmp.run_spec", "schema_version": "rlrmp.run_spec.v2"}
    if nested is not None:
        inline["feedbax_training_run_spec"] = nested
    payload = SimpleNamespace(
        kind="RLRMPRunSpec",
        schema_id=schema_id,
        schema_version=schema_version,
        inline=inline,
    )

    def reject(_record):
        raise ValueError(message)

    monkeypatch.setattr(
        training_run_specs,
        "feedbax_training_run_spec_from_rlrmp_record",
        reject,
        raising=False,
    )
    with pytest.raises(model_slots.ModelSlotProjectionError, match="invalid or disagrees"):
        model_slots._embedded_feedbax_training_run_spec_payload(payload)


def test_bare_training_run_spec_remains_a_legitimate_projection_input() -> None:
    inline = {
        "schema_id": "feedbax.spec.training_run",
        "schema_version": "feedbax.spec.training_run.v2",
    }
    payload = SimpleNamespace(
        kind="TrainingRunSpec",
        schema_id="feedbax.spec.training_run",
        inline=inline,
    )

    assert model_slots._embedded_feedbax_training_run_spec_payload(payload) == inline


def test_perturbation_native_seam_never_resolves_legacy_run_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = SimpleNamespace(training_manifest_id="training", run_id="run")
    projection = SimpleNamespace(provenance=provenance)
    run_spec = EvaluationRunSpec(
        evaluation_type=recipes.PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
        inputs=[ParentRef(kind="TrainingRunManifest", id="training", role="training_run")],
    )
    observed_roots: dict[str, object] = {}

    def project(*_args, **kwargs):
        observed_roots.update(kwargs)
        return projection

    monkeypatch.setattr(recipes, "_native_model_projection", project)
    monkeypatch.setattr(
        recipes,
        "_resolve_perturbation_run_inputs",
        lambda **_kwargs: pytest.fail("legacy run-path resolver was called"),
    )
    monkeypatch.setattr(
        recipes,
        "_evaluate_single_perturbation_bank_run",
        lambda _run, **kwargs: {"projection": kwargs["model_projection"]},
    )

    result = recipes._evaluate_perturbation_bank_runs(
        run_spec,
        {"n_rollout_trials": 1, "checkpoint_custody_root": "/checkpoint-authority"},
        bank={"perturbations": []},
        root=Path("/explicit-root"),
    )

    assert result == {"training": {"projection": projection}}
    assert observed_roots == {
        "manifest_root": Path("/explicit-root"),
        "checkpoint_custody_root": "/checkpoint-authority",
    }


def test_feedback_recipe_native_seam_never_calls_legacy_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = object()
    params = {
        "schema_id": "rlrmp.eval.feedback_ablation.params",
        "schema_version": "rlrmp.eval.feedback_ablation.params.v2",
        "checkpoint_custody_root": "/checkpoint-authority",
    }
    validated = SimpleNamespace(model_dump=lambda **_kwargs: params)
    run_spec = EvaluationRunSpec(
        evaluation_type=recipes.FEEDBACK_ABLATION_EVALUATION_TYPE,
        inputs=[ParentRef(kind="TrainingRunManifest", id="training", role="training_run")],
    )
    monkeypatch.setattr(recipes, "_validated_params", lambda _spec: (validated, params))
    monkeypatch.setattr(recipes, "_native_model_projection", lambda *_args, **_kwargs: projection)
    monkeypatch.setattr(
        recipes,
        "evaluate_projected_feedback_ablation_run",
        lambda candidate, _params: {"runs": {"training": {}}, "candidate": candidate},
    )
    monkeypatch.setattr(
        recipes,
        "evaluate_feedback_ablation_runs",
        lambda *_args, **_kwargs: pytest.fail("legacy feedback evaluator was called"),
    )

    result = recipes.feedback_ablation_recipe(run_spec, Path("/explicit-root"), Path("states"))

    assert result.states["candidate"] is projection


def test_perturbation_recipe_rejects_mixed_native_and_legacy_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_spec = EvaluationRunSpec(
        evaluation_type=recipes.PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
        inputs=[ParentRef(kind="TrainingRunManifest", id="training", role="training_run")],
    )
    monkeypatch.setattr(
        recipes,
        "_native_model_projection",
        lambda *_args, **_kwargs: pytest.fail("native projection ran before ambiguity rejection"),
    )
    monkeypatch.setattr(
        recipes,
        "_resolve_perturbation_run_inputs",
        lambda **_kwargs: pytest.fail("legacy run-path resolver was called"),
    )

    with pytest.raises(ValueError, match="cannot mix exact native parents with legacy"):
        recipes._evaluate_perturbation_bank_runs(
            run_spec,
            {
                "source_experiment": "legacy",
                "run_ids": ["run"],
                "checkpoint_custody_root": "/checkpoint-authority",
            },
            bank={"perturbations": []},
            root=Path("/analysis-manifests"),
        )


def test_feedback_recipe_rejects_mixed_native_and_legacy_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = {
        "source_experiment": "legacy",
        "run_ids": ["run"],
        "checkpoint_custody_root": "/checkpoint-authority",
    }
    validated = SimpleNamespace(
        source_experiment="legacy",
        run_ids=["run"],
        model_dump=lambda **_kwargs: params,
    )
    run_spec = EvaluationRunSpec(
        evaluation_type=recipes.FEEDBACK_ABLATION_EVALUATION_TYPE,
        inputs=[ParentRef(kind="TrainingRunManifest", id="training", role="training_run")],
    )
    monkeypatch.setattr(recipes, "_validated_params", lambda _spec: (validated, params))
    monkeypatch.setattr(
        recipes,
        "_native_model_projection",
        lambda *_args, **_kwargs: pytest.fail("native projection ran before ambiguity rejection"),
    )
    monkeypatch.setattr(
        recipes,
        "evaluate_feedback_ablation_runs",
        lambda *_args, **_kwargs: pytest.fail("legacy feedback evaluator was called"),
    )

    with pytest.raises(ValueError, match="cannot mix exact native parents with legacy"):
        recipes.feedback_ablation_recipe(run_spec, Path("/analysis-manifests"), Path("states"))


def test_native_projection_requires_distinct_explicit_checkpoint_authority() -> None:
    run_spec = EvaluationRunSpec(
        evaluation_type=recipes.PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
        inputs=[ParentRef(kind="TrainingRunManifest", id="training", role="training_run")],
    )

    with pytest.raises(ValueError, match="explicit checkpoint_custody_root"):
        recipes._native_model_projection(
            run_spec,
            manifest_root=Path("/analysis-manifests"),
            checkpoint_custody_root=None,
        )


def test_terminal_checkpoint_resolves_only_under_explicit_custody_authority(
    tmp_path: Path,
) -> None:
    analysis_root = tmp_path / "analysis-manifests"
    checkpoint_root = tmp_path / "retained-checkpoint-authority"
    analysis_root.mkdir()
    run_spec = _minimal_training_run_spec()
    written = write_checkpoint_transaction(
        checkpoint_root,
        run_spec=run_spec,
        phase_program=run_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="run",
            phase="train_batch",
            program_step=1,
            completed_barrier="after_train_batch",
        ),
        slots={
            "model": (jnp.ones((1, 2)),),
            "optimizer": {"count": jnp.asarray(2)},
            "prng": jnp.asarray([0, 1], dtype=jnp.uint32),
            "batch_counter": jnp.asarray(2),
        },
        status="final",
        completed_training_batches=2,
    )
    parent_ref = ParentRef(
        kind="TrainingCheckpointTransactionManifest",
        id=written.manifest.transaction_id,
        role="training_checkpoint_custody",
        uri=str(written.manifest_path),
        metadata={"manifest_sha256": sha256_bytes(written.manifest_path.read_bytes())},
    )
    manifest = TrainingRunManifest(
        id="training",
        status="completed",
        job_id="run",
        completed_batches=2,
        checkpoint_custody=[parent_ref],
    )

    resolved = model_slots._resolve_terminal_checkpoint(
        manifest,
        checkpoint_root=checkpoint_root,
    )

    assert analysis_root != checkpoint_root
    assert resolved.manifest.transaction_id == written.manifest.transaction_id
    with pytest.raises(model_slots.ModelSlotProjectionError, match="could not be authenticated"):
        model_slots._resolve_terminal_checkpoint(
            manifest,
            checkpoint_root=analysis_root,
        )

    with pytest.raises(ValueError, match="absolute checkpoint_custody_root"):
        recipes._native_model_projection(
            run_spec,
            manifest_root=Path("/analysis-manifests"),
            checkpoint_custody_root="relative/checkpoints",
        )


def test_projection_identity_snapshots_are_independent_of_mutable_inputs() -> None:
    run_spec = _minimal_training_run_spec()
    manifest = TrainingRunManifest(id="training", status="completed", job_id="run")
    provenance = model_slots.ModelSlotProvenance(
        training_manifest_id="training",
        training_manifest_sha256="a" * 64,
        training_manifest_reference="runs/training/manifest.json",
        run_id="run",
        completed_batches=1,
        checkpoint_transaction_id="tx",
        checkpoint_manifest_sha256="b" * 64,
        transaction_root_sha256="f" * 64,
        checkpoint_status="partial",
        slot_name="model",
        slot_blob_sha256="c" * 64,
        slot_root_sha256="d" * 64,
        structural_abi_sha256="e" * 64,
        method_ref="feedbax/standard_supervised/v1",
        architecture="gru",
    )
    projection = model_slots.ModelSlotProjection(
        model=object(),
        task=object(),
        training_parent_ref_json=canonical_json_bytes(
            ParentRef(kind="TrainingRunManifest", id="training", role="training_run")
        ),
        training_manifest_json=canonical_json_bytes(manifest),
        training_manifest_path=Path("/manifests/training.json"),
        training_manifest_reference="training.json",
        training_manifest_sha256="a" * 64,
        run_spec_json=canonical_json_bytes(run_spec),
        n_replicates=1,
        provenance=provenance,
    )

    returned_manifest = projection.resolved_training_input.manifest
    returned_spec = projection.run_spec
    returned_manifest.job_id = "mutated"
    returned_spec.metadata["mutated"] = True

    assert projection.training_manifest.job_id == "run"
    assert projection.run_spec.metadata == {}
    assert projection.provenance.run_id == "run"


def _minimal_training_run_spec() -> TrainingRunSpec:
    return TrainingRunSpec(
        graph={"inline": {"nodes": {}, "wires": [], "input_ports": [], "output_ports": []}},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=2, batch_size=3, learning_rate=0.01),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
    )
