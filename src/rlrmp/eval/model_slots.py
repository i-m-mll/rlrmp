"""Exact native checkpoint model projection for registered evaluations.

This module is deliberately path-agnostic beyond two explicit custody roots: the
resolved training-manifest input and the checkpoint root supplied by the caller.
It never consults a latest pointer or accepts a caller-provided model template.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.tree as jt
from feedbax.analysis import ResolvedEvaluationInput
from feedbax.contracts.manifest import (
    ParentRef,
    TrainingRunManifest,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.training import TrainingRunSpec
from feedbax.runtime import AbstractModel
from feedbax.tasks import AbstractTask
from feedbax.training import ResolvedCheckpointTransaction


_CS_SUPERVISED_METHOD = "rlrmp/cs_supervised/v1"
_TERMINAL_STATUSES = frozenset({"partial", "final"})
_RUN_CONTRACT_BINDING_ALGORITHM_V2 = "feedbax.training_checkpoint.run_contract_binding.v2"
_RUN_CONTRACT_BINDING_ALGORITHM_V3 = "feedbax.training_checkpoint.run_contract_binding.v3"
_RUN_CONTRACT_HASH_DOMAIN = "migrated-canonical-json"


class ModelSlotProjectionError(ValueError):
    """Raised when exact native model lineage cannot be projected safely."""


@dataclass(frozen=True, slots=True)
class ModelSlotProvenance:
    """Authenticated identities carried by one projected model slot."""

    training_manifest_id: str
    training_manifest_sha256: str
    training_manifest_reference: str
    run_id: str
    completed_batches: int
    checkpoint_transaction_id: str
    checkpoint_manifest_sha256: str
    transaction_root_sha256: str
    checkpoint_status: str
    slot_name: str
    slot_blob_sha256: str
    slot_root_sha256: str
    structural_abi_sha256: str
    method_ref: str
    architecture: str


@dataclass(frozen=True, slots=True)
class ModelSlotProjection:
    """Rehydrated model/task pair plus immutable authenticated identity bytes.

    Typed manifest/spec properties reconstruct independent copies from canonical
    bytes on every access. Mutating either an original input or a returned typed
    copy therefore cannot rewrite the identity authority retained here.
    """

    model: AbstractModel
    task: AbstractTask
    training_parent_ref_json: bytes
    training_manifest_json: bytes
    training_manifest_path: Path
    training_manifest_reference: str
    training_manifest_sha256: str
    run_spec_json: bytes
    n_replicates: int
    provenance: ModelSlotProvenance

    @property
    def run_spec(self) -> TrainingRunSpec:
        """Return an independent typed copy of the verified execution spec."""
        return TrainingRunSpec.model_validate_json(self.run_spec_json)

    @property
    def training_manifest(self) -> TrainingRunManifest:
        """Return an independent typed copy of the verified training manifest."""
        return TrainingRunManifest.model_validate_json(self.training_manifest_json)

    @property
    def resolved_training_input(self) -> ResolvedEvaluationInput:
        """Return an independent typed copy of the exact verified parent input."""
        return ResolvedEvaluationInput(
            ref=ParentRef.model_validate_json(self.training_parent_ref_json),
            manifest=self.training_manifest,
            path=self.training_manifest_path,
            reference=self.training_manifest_reference,
            sha256=self.training_manifest_sha256,
        )


def project_training_model_slot(
    resolved_input: ResolvedEvaluationInput,
    resolved_checkpoint: ResolvedCheckpointTransaction,
) -> ModelSlotProjection:
    """Project one completed native training run's exact terminal model slot.

    Args:
        resolved_input: A Feedbax ``ResolvedEvaluationInput`` produced from one
            exact ``TrainingRunManifest`` ParentRef.
        resolved_checkpoint: A Feedbax ``ResolvedCheckpointTransaction`` decoded
            from one exact custody ``ParentRef`` declared by ``resolved_input``.

    Returns:
        The reconstructed model, its governed task/runtime context, and typed
        authenticated provenance.

    Raises:
        ModelSlotProjectionError: If manifest, checkpoint, run-contract, model
            architecture, content, or structural lineage is absent or ambiguous.
    """
    _require_resolved_evaluation_input(resolved_input)
    training_manifest = resolved_input.manifest
    if training_manifest.status != "completed":
        raise ModelSlotProjectionError(
            "native model projection requires a completed TrainingRunManifest"
        )
    if training_manifest.completed_at is None:
        raise ModelSlotProjectionError(
            "completed TrainingRunManifest has no completed_at timestamp"
        )
    if not training_manifest.job_id:
        raise ModelSlotProjectionError("completed TrainingRunManifest has no job_id")
    if training_manifest.completed_batches is None:
        raise ModelSlotProjectionError(
            "completed TrainingRunManifest has no completed_batches coordinate"
        )
    run_spec = _embedded_training_run_spec(training_manifest)
    architecture = _controller_architecture(run_spec)
    if run_spec.method_ref.key != _CS_SUPERVISED_METHOD:
        raise ModelSlotProjectionError(
            "native evaluation model projection does not support training method "
            f"{run_spec.method_ref.key!r}"
        )
    if architecture not in {"gru", "linear_recurrence", "static_linear"}:
        raise ModelSlotProjectionError(
            f"native evaluation model projection does not support architecture {architecture!r}"
        )

    _validate_resolved_terminal_checkpoint(training_manifest, resolved_checkpoint)
    preparation = _prepare_execution(run_spec, run_id=training_manifest.job_id)
    model_template, task = _runtime_model_template_and_task(preparation)
    _validate_run_contract(resolved_checkpoint, run_spec)
    model_slot = resolved_checkpoint.slots.get("model")
    if not isinstance(model_slot, tuple):
        raise ModelSlotProjectionError(
            "native checkpoint model slot must decode to the exact tuple-of-array-leaves ABI"
        )
    slot_record = _slot_record(resolved_checkpoint.manifest, "model")
    _validate_model_template_abi(model_template, slot_record)
    model = _rehydrate_model(model_slot, model_template)
    _validate_rehydrated_model(model, model_slot, slot_record)
    n_replicates = _n_replicates(run_spec, model_slot)
    return ModelSlotProjection(
        model=model,
        task=task,
        training_parent_ref_json=canonical_json_bytes(resolved_input.ref),
        training_manifest_json=canonical_json_bytes(training_manifest),
        training_manifest_path=resolved_input.path,
        training_manifest_reference=resolved_input.reference,
        training_manifest_sha256=resolved_input.sha256,
        run_spec_json=canonical_json_bytes(run_spec),
        n_replicates=n_replicates,
        provenance=ModelSlotProvenance(
            training_manifest_id=training_manifest.id,
            training_manifest_sha256=resolved_input.sha256,
            training_manifest_reference=resolved_input.reference,
            run_id=training_manifest.job_id,
            completed_batches=int(training_manifest.completed_batches),
            checkpoint_transaction_id=resolved_checkpoint.manifest.transaction_id,
            checkpoint_manifest_sha256=resolved_checkpoint.manifest_sha256,
            transaction_root_sha256=(
                resolved_checkpoint.manifest.content_integrity_digest.transaction_root_sha256
            ),
            checkpoint_status=resolved_checkpoint.manifest.status,
            slot_name="model",
            slot_blob_sha256=slot_record.sha256,
            slot_root_sha256=slot_record.content_digest.slot_root_sha256,
            structural_abi_sha256=(slot_record.structural_abi_fingerprint.fingerprint_sha256),
            method_ref=run_spec.method_ref.key,
            architecture=architecture,
        ),
    )


def _require_resolved_evaluation_input(value: object) -> None:
    if not isinstance(value, ResolvedEvaluationInput):
        raise ModelSlotProjectionError("resolved_input must be a Feedbax ResolvedEvaluationInput")
    ref = value.ref
    if ref.kind != "TrainingRunManifest" or ref.role != "training_run":
        raise ModelSlotProjectionError(
            "resolved evaluation input must be an exact TrainingRunManifest/training_run ParentRef"
        )
    if ref.id != value.manifest.id:
        raise ModelSlotProjectionError(
            "resolved evaluation input ParentRef id does not match TrainingRunManifest id"
        )


def _embedded_training_run_spec(training_manifest: TrainingRunManifest) -> TrainingRunSpec:
    from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods

    register_rlrmp_training_methods()
    payload = training_manifest.training_spec
    if payload is None or payload.inline is None:
        raise ModelSlotProjectionError(
            "TrainingRunManifest must embed one governed training specification"
        )
    if payload.ref is not None and payload.sha256 is None:
        raise ModelSlotProjectionError(
            "referenced embedded training specification lacks its governed content hash"
        )
    raw_training_spec = _embedded_feedbax_training_run_spec_payload(payload)
    try:
        return TrainingRunSpec.model_validate(raw_training_spec)
    except Exception as exc:
        raise ModelSlotProjectionError(
            "embedded governed TrainingRunSpec is invalid or unsupported"
        ) from exc


def _embedded_feedbax_training_run_spec_payload(payload: object) -> dict[str, Any]:
    """Return the one governed generic execution spec from a manifest payload."""

    if payload.kind == "TrainingRunSpec":
        if payload.schema_id not in {None, "feedbax.spec.training_run"}:
            raise ModelSlotProjectionError(
                "bare TrainingRunSpec payload schema identity disagrees with its kind"
            )
        return dict(payload.inline)
    if payload.kind != "RLRMPRunSpec":
        raise ModelSlotProjectionError(
            "TrainingRunManifest training_spec must be TrainingRunSpec or RLRMPRunSpec"
        )

    try:
        from rlrmp.runtime.training_run_specs import (
            feedbax_training_run_spec_from_rlrmp_record,
        )
    except ImportError as exc:
        raise ModelSlotProjectionError(
            "RLRMPRunSpec projection requires the deadff5 public nested-spec extractor"
        ) from exc
    try:
        nested = feedbax_training_run_spec_from_rlrmp_record(dict(payload.inline))
    except Exception as exc:
        raise ModelSlotProjectionError(
            "RLRMPRunSpec embedded generic execution identity is invalid or disagrees"
        ) from exc
    return nested.model_dump(mode="json", exclude_none=True)


def _controller_architecture(run_spec: TrainingRunSpec) -> str:
    payload = getattr(run_spec.method_payload, "payload", run_spec.method_payload)
    config = (
        payload.get("config") if isinstance(payload, dict) else getattr(payload, "config", None)
    )
    if not isinstance(config, dict):
        raise ModelSlotProjectionError(
            "embedded TrainingRunSpec method payload lacks governed runtime config"
        )
    architecture = config.get("controller_architecture")
    if not isinstance(architecture, str) or not architecture:
        raise ModelSlotProjectionError(
            "embedded TrainingRunSpec runtime config lacks controller_architecture"
        )
    return architecture


def _prepare_execution(run_spec: TrainingRunSpec, *, run_id: str) -> Any:
    from feedbax.training import (
        DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY,
        ExecutionPreparationRequest,
    )

    from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods
    from rlrmp.train.execution_preparation import register_execution_preparations

    register_rlrmp_training_methods()
    register_execution_preparations(DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY)
    try:
        return DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.prepare(
            ExecutionPreparationRequest(run_spec=run_spec, run_id=run_id, resume=False)
        )
    except Exception as exc:
        raise ModelSlotProjectionError(
            "embedded TrainingRunSpec has no usable public execution preparation"
        ) from exc


def _runtime_model_template_and_task(
    preparation: Any,
) -> tuple[AbstractModel, AbstractTask]:
    from rlrmp.train.executor import RLRMP_RUNTIME_CONTEXT_KEY, RlrmpRuntime

    runtime = preparation.kernel_context.get(RLRMP_RUNTIME_CONTEXT_KEY)
    if not isinstance(runtime, RlrmpRuntime):
        raise ModelSlotProjectionError(
            "execution preparation did not expose the governed RLRMP runtime context"
        )
    component = runtime.component("cs_supervised")
    pair = getattr(component, "pair", None)
    if pair is None or getattr(pair, "model", None) is None or getattr(pair, "task", None) is None:
        raise ModelSlotProjectionError(
            "execution preparation did not expose a model/task runtime template"
        )
    if not isinstance(pair.model, AbstractModel) or not isinstance(pair.task, AbstractTask):
        raise ModelSlotProjectionError(
            "execution preparation exposed unsupported model/task runtime types"
        )
    return pair.model, pair.task


def _resolve_terminal_checkpoint(
    training_manifest: TrainingRunManifest,
    *,
    checkpoint_root: Path,
) -> ResolvedCheckpointTransaction:
    try:
        from feedbax.training import resolve_checkpoint_custody_ref
    except ImportError as exc:
        raise ModelSlotProjectionError(
            "Feedbax lacks public exact checkpoint custody resolution; stage d81a868 first"
        ) from exc

    refs = [
        ref
        for ref in training_manifest.checkpoint_custody
        if isinstance(ref, ParentRef)
        and ref.kind == "TrainingCheckpointTransactionManifest"
        and ref.role == "training_checkpoint_custody"
    ]
    if len(refs) != len(training_manifest.checkpoint_custody):
        raise ModelSlotProjectionError(
            "native TrainingRunManifest checkpoint_custody contains a non-transaction reference"
        )
    resolved = []
    for ref in refs:
        try:
            candidate = resolve_checkpoint_custody_ref(
                ref,
                allowed_root=checkpoint_root,
                slot_names=("model",),
            )
        except Exception as exc:
            raise ModelSlotProjectionError(
                f"checkpoint custody ParentRef {ref.id!r} could not be authenticated"
            ) from exc
        manifest = candidate.manifest
        if (
            manifest.run_id == training_manifest.job_id
            and manifest.completed_training_batches == training_manifest.completed_batches
        ):
            resolved.append(candidate)
    if not resolved:
        raise ModelSlotProjectionError(
            "TrainingRunManifest has no exact terminal checkpoint transaction"
        )
    if len(resolved) != 1:
        raise ModelSlotProjectionError(
            "TrainingRunManifest has ambiguous exact terminal checkpoint transactions"
        )
    candidate = resolved[0]
    if candidate.manifest.status not in _TERMINAL_STATUSES:
        raise ModelSlotProjectionError(
            f"terminal checkpoint transaction has unsupported status {candidate.manifest.status!r}"
        )
    return candidate


def _project_training_model_slot_from_custody_root(
    resolved_input: ResolvedEvaluationInput,
    *,
    checkpoint_root: str | Path,
) -> ModelSlotProjection:
    """Recipe adapter resolving declared custody before calling the public projector."""

    resolved_checkpoint = _resolve_terminal_checkpoint(
        resolved_input.manifest,
        checkpoint_root=Path(checkpoint_root),
    )
    return project_training_model_slot(resolved_input, resolved_checkpoint)


def _validate_resolved_terminal_checkpoint(
    training_manifest: TrainingRunManifest,
    resolved_checkpoint: ResolvedCheckpointTransaction,
) -> None:
    if not isinstance(resolved_checkpoint, ResolvedCheckpointTransaction):
        raise ModelSlotProjectionError(
            "resolved_checkpoint must be a Feedbax ResolvedCheckpointTransaction"
        )
    matching_refs = [
        ref for ref in training_manifest.checkpoint_custody if ref == resolved_checkpoint.parent_ref
    ]
    if len(matching_refs) != 1:
        raise ModelSlotProjectionError(
            "resolved checkpoint ParentRef must match exactly one manifest custody reference"
        )
    manifest = resolved_checkpoint.manifest
    if manifest.run_id != training_manifest.job_id:
        raise ModelSlotProjectionError("resolved checkpoint run id differs from training manifest")
    if manifest.completed_training_batches != training_manifest.completed_batches:
        raise ModelSlotProjectionError(
            "resolved checkpoint is not the exact terminal training batch"
        )
    if manifest.status not in _TERMINAL_STATUSES:
        raise ModelSlotProjectionError(
            f"terminal checkpoint transaction has unsupported status {manifest.status!r}"
        )


def _validate_run_contract(
    resolved_checkpoint: ResolvedCheckpointTransaction,
    run_spec: TrainingRunSpec,
) -> None:
    from feedbax.training import run_contract_binding

    phase_program = run_spec.worker_execution.method_contract.phase_program
    expected = run_contract_binding(run_spec, phase_program)
    actual = resolved_checkpoint.manifest.run_contract_binding
    _authenticate_run_contract_projection(actual)
    _authenticate_run_contract_projection(expected)
    if actual == expected:
        return
    if actual.canonical_projection is None or expected.canonical_projection is None:
        raise ModelSlotProjectionError(
            "checkpoint transaction run-contract binding differs and lacks authenticated "
            "canonical projections for compatibility adjudication"
        )
    if _normalized_canonical_json(actual.canonical_projection) != _normalized_canonical_json(
        expected.canonical_projection
    ):
        raise ModelSlotProjectionError(
            "checkpoint transaction canonical run-contract projection differs from "
            "embedded TrainingRunSpec"
        )


def _authenticate_run_contract_projection(binding: object) -> None:
    algorithm = getattr(binding, "algorithm_version", None)
    hash_domain = getattr(binding, "hash_domain", None)
    projection = getattr(binding, "canonical_projection", None)
    recorded_digest = getattr(binding, "canonical_projection_sha256", None)
    if algorithm not in {
        _RUN_CONTRACT_BINDING_ALGORITHM_V2,
        _RUN_CONTRACT_BINDING_ALGORITHM_V3,
    }:
        raise ModelSlotProjectionError("checkpoint run-contract algorithm is unsupported")
    if hash_domain != _RUN_CONTRACT_HASH_DOMAIN:
        raise ModelSlotProjectionError("checkpoint run-contract hash domain is unsupported")
    if not isinstance(projection, Mapping) or not isinstance(recorded_digest, str):
        raise ModelSlotProjectionError(
            "checkpoint run-contract binding lacks an authenticated canonical projection"
        )
    canonical = (
        canonical_json_bytes(projection)
        if algorithm == _RUN_CONTRACT_BINDING_ALGORITHM_V2
        else _normalized_canonical_json(projection)
    )
    if sha256_bytes(canonical) != recorded_digest:
        raise ModelSlotProjectionError(
            "checkpoint run-contract canonical projection digest is stale or forged"
        )


def _normalized_canonical_json(value: object) -> bytes:
    return canonical_json_bytes(_normalize_signed_zero(value))


def _normalize_signed_zero(value: object) -> object:
    if isinstance(value, float) and value == 0.0:
        return 0.0
    if isinstance(value, Mapping):
        return {str(key): _normalize_signed_zero(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_signed_zero(item) for item in value]
    return value


def _slot_record(manifest: Any, slot_name: str) -> Any:
    matches = [slot for slot in manifest.slots if slot.slot == slot_name]
    if len(matches) != 1:
        raise ModelSlotProjectionError(
            f"checkpoint transaction must contain exactly one {slot_name!r} slot"
        )
    return matches[0]


def _validate_model_template_abi(model_template: AbstractModel, slot_record: object) -> None:
    from feedbax.training import structural_abi_fingerprint

    template_slot = tuple(jt.leaves(eqx.filter(model_template, eqx.is_array)))
    if structural_abi_fingerprint(template_slot) != slot_record.structural_abi_fingerprint:
        raise ModelSlotProjectionError(
            "runtime model template structural ABI does not match checkpoint model slot"
        )


def _rehydrate_model(
    model_slot: tuple[Any, ...],
    model_template: AbstractModel,
) -> AbstractModel:
    filtered_template = eqx.filter(model_template, eqx.is_array)
    template_structure = jt.structure(filtered_template)
    if template_structure.num_leaves != len(model_slot):
        raise ModelSlotProjectionError(
            "checkpoint model slot leaf count does not match runtime model template"
        )
    try:
        arrays = jt.unflatten(template_structure, model_slot)
        return eqx.combine(arrays, model_template)
    except Exception as exc:
        raise ModelSlotProjectionError(
            "checkpoint model slot cannot be rehydrated with its governed runtime template"
        ) from exc


def _validate_rehydrated_model(
    model: AbstractModel,
    model_slot: tuple[Any, ...],
    slot_record: object,
) -> None:
    from feedbax.training import structural_abi_fingerprint

    actual_slot = tuple(jt.leaves(eqx.filter(model, eqx.is_array)))
    if len(actual_slot) != len(model_slot):
        raise ModelSlotProjectionError("rehydrated model changed checkpoint model-slot arity")
    actual_abi = structural_abi_fingerprint(actual_slot)
    if actual_abi != slot_record.structural_abi_fingerprint:
        raise ModelSlotProjectionError(
            "rehydrated model structural ABI does not match checkpoint model slot"
        )


def _n_replicates(run_spec: TrainingRunSpec, model_slot: tuple[Any, ...]) -> int:
    payload = getattr(run_spec.method_payload, "payload", run_spec.method_payload)
    config = (
        payload.get("config") if isinstance(payload, dict) else getattr(payload, "config", None)
    )
    expected = config.get("n_replicates") if isinstance(config, dict) else None
    first_shape = getattr(model_slot[0], "shape", ()) if model_slot else ()
    actual = int(first_shape[0]) if first_shape else 1
    if expected is not None and int(expected) != actual:
        raise ModelSlotProjectionError(
            "checkpoint model replicate axis does not match embedded TrainingRunSpec"
        )
    return actual


__all__ = [
    "ModelSlotProjection",
    "ModelSlotProjectionError",
    "ModelSlotProvenance",
    "project_training_model_slot",
]
