"""Feedbax CLI preparation providers for RLRMP-native C&S training."""

from __future__ import annotations

from typing import Any

import jax.random as jr
from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY
from feedbax.training import (
    ExecutionPreparationRegistration,
    ExecutionPreparationRequest,
    ExecutionPreparationResult,
)

from rlrmp.runtime.training_run_specs import (
    CS_SUPERVISED_METHOD_REF,
    CsSupervisedMethodPayload,
)
from rlrmp.train.adaptive_epsilon_native import (
    ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
    AdaptiveEpsilonExternalObjectiveLossService,
    AdaptiveEpsilonMethodPayload,
    _resume_slot_transform as adaptive_resume_slot_transform,
    build_adaptive_epsilon_native_initial_slots,
)
from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY
from rlrmp.train.executor.cs_supervised import (
    CsSupervisedExternalObjectiveLossService,
    _cs_supervised_resume_slot_transform,
    build_cs_supervised_native_initial_slots,
)
from rlrmp.train.policy_adversary_native import (
    POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
    PolicyAdversaryExternalObjectiveLossService,
    PolicyAdversaryMethodPayload,
    _resume_slot_transform as policy_adversary_resume_slot_transform,
    build_policy_adversary_native_initial_slots,
)


def _validated_payload(request: ExecutionPreparationRequest, expected_type: type[Any]) -> Any:
    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        request.run_spec.method_ref,
        request.run_spec.method_payload,
        path="/method_payload",
    )
    if not isinstance(payload, expected_type):
        raise TypeError(
            f"{request.run_spec.method_ref.key} payload resolved to "
            f"{type(payload).__name__}, expected {expected_type.__name__}"
        )
    return payload


def _runtime_config(config: dict[str, Any]) -> tuple[Any, Any]:
    from rlrmp.train.config_materialization import _config_namespace, build_hps

    args = _config_namespace(config)
    return args, build_hps(args)


def prepare_cs_supervised(request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
    """Construct runtime-only inputs for native C&S supervised execution."""
    payload = _validated_payload(request, CsSupervisedMethodPayload)
    if payload.config is None:
        raise ValueError(
            "cs-supervised TrainingRunSpec predates governed runtime config; "
            "regenerate it from the RLRMP run-spec source before CLI execution"
        )
    args, hps = _runtime_config(payload.config)
    initial_slots, runtime = build_cs_supervised_native_initial_slots(
        run_spec=request.run_spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(int(args.seed)),
    )
    return ExecutionPreparationResult(
        initial_slots=initial_slots,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
        loss_service=CsSupervisedExternalObjectiveLossService(),
        resume_slot_transform=_cs_supervised_resume_slot_transform(),
    )


def prepare_adaptive_epsilon(request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
    """Construct runtime-only inputs for adaptive-epsilon execution."""
    payload = _validated_payload(request, AdaptiveEpsilonMethodPayload)
    args, hps = _runtime_config(payload.config)
    initial_slots, runtime = build_adaptive_epsilon_native_initial_slots(
        run_spec=request.run_spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(int(args.seed)),
    )
    return ExecutionPreparationResult(
        initial_slots=initial_slots,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
        loss_service=AdaptiveEpsilonExternalObjectiveLossService(),
        resume_slot_transform=adaptive_resume_slot_transform(None),
    )


def prepare_policy_adversary(request: ExecutionPreparationRequest) -> ExecutionPreparationResult:
    """Construct runtime-only inputs for policy-adversary execution."""
    payload = _validated_payload(request, PolicyAdversaryMethodPayload)
    args, hps = _runtime_config(payload.config)
    initial_slots, runtime = build_policy_adversary_native_initial_slots(
        run_spec=request.run_spec,
        hps=hps,
        args=args,
        key=jr.PRNGKey(int(args.seed)),
    )
    return ExecutionPreparationResult(
        initial_slots=initial_slots,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
        loss_service=PolicyAdversaryExternalObjectiveLossService(),
        resume_slot_transform=policy_adversary_resume_slot_transform(None),
    )


_PROVIDERS = (
    (CS_SUPERVISED_METHOD_REF, prepare_cs_supervised),
    (ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF, prepare_adaptive_epsilon),
    (POLICY_ADVERSARY_SUPERVISED_METHOD_REF, prepare_policy_adversary),
)


def register_execution_preparations(registry: Any) -> None:
    """Register all RLRMP methods that require runtime-only preparation."""
    for method_ref, provider in _PROVIDERS:
        if registry.get(method_ref) is not None:
            continue
        registry.register(
            ExecutionPreparationRegistration(
                method_ref=method_ref,
                provider=provider,
                owner="rlrmp.train.execution_preparation",
            )
        )


register_feedbax_execution_preparations = register_execution_preparations
