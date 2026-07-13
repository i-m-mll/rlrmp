"""Native distillation TrainingRunSpec contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from feedbax.contracts.training import TrainingRunSpec
from pydantic import ValidationError

from rlrmp.runtime.training_run_specs import (
    CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
)
from rlrmp.train.distillation_entry import load_distillation_run_spec
from rlrmp.train.training_configs import (
    ClosedLoopDistillationConfig,
    GuidedDistillationConfig,
)

GUIDED_SPEC_FIXTURE = Path("tests/fixtures/legacy_payloads/guided_distillation_run_spec.json")
CLOSED_LOOP_SPEC_FIXTURE = Path(
    "tests/fixtures/legacy_payloads/closed_loop_distillation_run_spec.json"
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize(
    ("path", "method"),
    [
        (GUIDED_SPEC_FIXTURE, "guided_distillation"),
        (
            CLOSED_LOOP_SPEC_FIXTURE,
            "closed_loop_distillation",
        ),
    ],
)
def test_tracked_distillation_specs_lower_to_registered_native_methods(
    path: Path,
    method: str,
) -> None:
    if method == "guided_distillation":
        config = GuidedDistillationConfig(run_spec=str(path))
    else:
        config = ClosedLoopDistillationConfig(run_spec=path)
    payload = load_distillation_run_spec(config, method=method)
    spec = TrainingRunSpec.model_validate(payload[FEEDBAX_TRAINING_RUN_SPEC_KEY])

    assert spec.method_ref.name == method
    refs = {
        step.kernel.kernel_ref
        for step in spec.worker_execution.method_contract.phase_program.update_steps
    }
    expected = {
        "guided_distillation": "rlrmp.train.distillation_native.guided_gradient_update",
        "closed_loop_distillation": ("rlrmp.train.distillation_native.closed_loop_gradient_update"),
    }
    assert refs == {expected[method]}


def test_distillation_payload_golden_fixture_matches_current_native_refs() -> None:
    fixture = json.loads(
        Path("tests/fixtures/distillation_method_payload_golden.json").read_text(encoding="utf-8")
    )["cases"]
    guided = load_distillation_run_spec(
        GuidedDistillationConfig(run_spec=str(GUIDED_SPEC_FIXTURE)),
        method="guided_distillation",
    )
    closed = load_distillation_run_spec(
        ClosedLoopDistillationConfig(run_spec=CLOSED_LOOP_SPEC_FIXTURE),
        method="closed_loop_distillation",
    )
    guided_payload = TrainingRunSpec.model_validate(
        guided[FEEDBAX_TRAINING_RUN_SPEC_KEY]
    ).method_payload
    closed_payload = TrainingRunSpec.model_validate(
        closed[FEEDBAX_TRAINING_RUN_SPEC_KEY]
    ).method_payload

    assert (
        _canonical_json(guided_payload.model_dump(mode="json", exclude_none=True))
        == (fixture["guided_generated_default"]["method_payload_envelope_json"])
    )
    assert (
        _canonical_json(closed_payload.model_dump(mode="json", exclude_none=True))
        == (fixture["closed_loop_tracked_with_horizon"]["method_payload_envelope_json"])
    )


def test_distillation_payload_versions_advance_with_native_callable_paths() -> None:
    assert GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION.endswith(".v2")
    assert CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION.endswith(".v2")


@pytest.mark.parametrize(
    ("method", "config", "retired_version"),
    [
        (
            "guided_distillation",
            GuidedDistillationConfig(run_spec=str(GUIDED_SPEC_FIXTURE)),
            "rlrmp.spec.training_method.guided_distillation_payload.v1",
        ),
        (
            "closed_loop_distillation",
            ClosedLoopDistillationConfig(run_spec=CLOSED_LOOP_SPEC_FIXTURE),
            "rlrmp.spec.training_method.closed_loop_distillation_payload.v1",
        ),
    ],
)
def test_pre_native_distillation_payload_versions_are_rejected(
    method: str,
    config: GuidedDistillationConfig | ClosedLoopDistillationConfig,
    retired_version: str,
) -> None:
    run_spec = load_distillation_run_spec(config, method=method)
    payload = copy.deepcopy(run_spec[FEEDBAX_TRAINING_RUN_SPEC_KEY])
    payload["method_payload"]["schema_version"] = retired_version
    with pytest.raises(ValidationError, match="unsupported method payload schema version"):
        TrainingRunSpec.model_validate(payload)


def test_unknown_distillation_method_fails_before_launch() -> None:
    run_spec = load_distillation_run_spec(
        GuidedDistillationConfig(run_spec=str(GUIDED_SPEC_FIXTURE)),
        method="guided_distillation",
    )
    payload = copy.deepcopy(run_spec[FEEDBAX_TRAINING_RUN_SPEC_KEY])
    payload["method_ref"] = {"package": "rlrmp", "name": "not_registered", "version": "v1"}
    with pytest.raises(ValidationError, match="unknown method_ref"):
        TrainingRunSpec.model_validate(payload)
