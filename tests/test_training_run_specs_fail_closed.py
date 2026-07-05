"""Fail-closed coverage for TrainingRunSpec recording adapters."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from rlrmp.runtime.training_run_specs import (
    MissingTrainingRunSpecFieldError,
    _cs_training_config,
    _distillation_training_config,
)


TRACKED_CLEAN_DISTILLATION_SPECS = (
    (
        Path("results/9727d79/runs/h0_extlqg_6d_standard_graph_distillation.json"),
        "guided_distillation",
    ),
    (
        Path("results/9727d79/runs/h0_hinf_6d_guided_distillation.json"),
        "guided_distillation",
    ),
    (
        Path("results/9727d79/runs/h0_hinf_6d_standard_graph_distillation.json"),
        "guided_distillation",
    ),
)
EXPECTED_TRACKED_FAIL_CLOSED = {
    Path("results/a378b34/runs/h0_extlqg_6d_closed_loop_distillation.json"): (
        "teacher_contract.horizon"
    ),
}
EXPECTED_DIRECTORY_RUN_FAIL_CLOSED = {
    Path("results/30f2313/runs/cs_stochastic_gru__hidden_penalty/run.json"): (
        "optimizer.gradient_clip_norm"
    ),
    Path("results/30f2313/runs/cs_stochastic_gru__no_hidden_penalty/run.json"): (
        "optimizer.gradient_clip_norm"
    ),
    Path("results/3b2af27/runs/lss_12k__hidden_penalty/run.json"): (
        "optimizer.gradient_clip_norm"
    ),
    Path("results/3b2af27/runs/lss_12k__no_hidden_penalty/run.json"): (
        "optimizer.gradient_clip_norm"
    ),
}


def _complete_closed_loop_spec() -> dict[str, Any]:
    payload = _read_json(Path("results/a378b34/runs/h0_extlqg_6d_closed_loop_distillation.json"))
    payload["teacher_contract"]["horizon"] = 60
    return payload


def _complete_guided_spec() -> dict[str, Any]:
    return _read_json(Path("results/9727d79/runs/h0_hinf_6d_guided_distillation.json"))


def _complete_cs_spec() -> dict[str, Any]:
    return {
        "issue": "test",
        "run_id": "complete_cs_fixture",
        "n_train_batches": 12,
        "batch_size": 4,
        "controller_lr": 0.003,
        "training_summary": {"n_train_batches": 12, "batch_size": 4},
        "optimizer": {"learning_rate_0": 0.003, "gradient_clip_norm": 5.0},
        "model_summary": {"hidden_size": 16},
        "task_timing": {"n_steps": 60},
        "loss_summary": {"active_cs_terms": {"control": {"scale": 2.5}}},
        "checkpointing": {"interval_batches": 10},
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _adapter_method_for_payload(payload: dict[str, Any]) -> str | None:
    keys = set(payload)
    if {"student_contract", "closed_loop_semantics", "loss_surface"} <= keys:
        return "closed_loop_distillation"
    if {"model_contract", "teacher_bank", "training_schedule", "distillation_surface"} <= keys:
        return "guided_distillation"
    if {"training_summary", "model_summary", "task_timing"} <= keys:
        return "cs_gru"
    return None


def _flat_tracked_run_specs() -> list[Path]:
    return sorted(Path("results").glob("*/runs/*.json"))


def _directory_tracked_run_specs() -> list[Path]:
    return sorted(Path("results").glob("*/runs/*/run.json"))


def _delete_path(payload: dict[str, Any], field_path: str) -> None:
    cursor: Any = payload
    parts = field_path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    del cursor[parts[-1]]


@pytest.mark.parametrize(
    ("method", "factory", "field_paths", "match"),
    [
        (
            "closed_loop_distillation",
            _complete_closed_loop_spec,
            ("student_contract.n_train_batches",),
            "student_contract.n_train_batches",
        ),
        (
            "closed_loop_distillation",
            _complete_closed_loop_spec,
            ("student_contract.batch_size",),
            "student_contract.batch_size",
        ),
        (
            "closed_loop_distillation",
            _complete_closed_loop_spec,
            ("student_contract.controller_lr",),
            "student_contract.controller_lr",
        ),
        (
            "closed_loop_distillation",
            _complete_closed_loop_spec,
            ("student_contract.gradient_clip_norm",),
            "student_contract.gradient_clip_norm",
        ),
        (
            "closed_loop_distillation",
            _complete_closed_loop_spec,
            ("student_contract.hidden_size",),
            "student_contract.hidden_size",
        ),
        (
            "closed_loop_distillation",
            _complete_closed_loop_spec,
            ("teacher_contract.horizon",),
            "teacher_contract.horizon",
        ),
        (
            "closed_loop_distillation",
            _complete_closed_loop_spec,
            ("loss_surface.weights.action_force_trajectory",),
            "loss_surface.weights.action_force_trajectory",
        ),
        (
            "closed_loop_distillation",
            _complete_closed_loop_spec,
            ("checkpointing.interval_batches",),
            "checkpointing.interval_batches",
        ),
        (
            "guided_distillation",
            _complete_guided_spec,
            ("n_train_batches", "training_schedule.total_batches"),
            "n_train_batches or training_schedule.total_batches",
        ),
        (
            "guided_distillation",
            _complete_guided_spec,
            ("batch_size", "model_contract.batch_size"),
            "batch_size or model_contract.batch_size",
        ),
        (
            "guided_distillation",
            _complete_guided_spec,
            ("controller_lr", "optimizer.controller_lr"),
            "controller_lr or optimizer.controller_lr",
        ),
        (
            "guided_distillation",
            _complete_guided_spec,
            ("optimizer.gradient_clip_norm",),
            "optimizer.gradient_clip_norm",
        ),
        (
            "guided_distillation",
            _complete_guided_spec,
            ("model_contract.hidden_size",),
            "model_contract.hidden_size",
        ),
        (
            "guided_distillation",
            _complete_guided_spec,
            ("teacher_bank.horizon",),
            "teacher_bank.horizon",
        ),
        (
            "guided_distillation",
            _complete_guided_spec,
            ("distillation_surface.components.clean_action.weight",),
            "distillation_surface.components.clean_action.weight",
        ),
        (
            "guided_distillation",
            _complete_guided_spec,
            ("checkpointing.interval_batches",),
            "checkpointing.interval_batches",
        ),
    ],
)
def test_distillation_training_config_fails_closed_on_missing_run_descriptive_key(
    method: str,
    factory: Any,
    field_paths: tuple[str, ...],
    match: str,
) -> None:
    payload = factory()
    for field_path in field_paths:
        _delete_path(payload, field_path)

    with pytest.raises(MissingTrainingRunSpecFieldError, match=match) as exc_info:
        _distillation_training_config(
            payload,
            method=method,
            spec_path=Path(f"results/test/runs/{method}.json"),
        )

    assert "results/test/runs" in exc_info.value.spec_identity


@pytest.mark.parametrize(
    ("field_paths", "match"),
    [
        (("n_train_batches", "training_summary.n_train_batches"), "n_train_batches"),
        (("batch_size", "training_summary.batch_size"), "batch_size"),
        (("controller_lr", "optimizer.learning_rate_0"), "controller_lr"),
        (("optimizer.gradient_clip_norm",), "optimizer.gradient_clip_norm"),
        (("model_summary.hidden_size",), "model_summary.hidden_size"),
        (("task_timing.n_steps",), "task_timing.n_steps"),
        (
            ("loss_summary.active_cs_terms.control.scale",),
            "loss_summary.active_cs_terms.control.scale",
        ),
        (("checkpointing.interval_batches",), "checkpointing.interval_batches"),
    ],
)
def test_cs_training_config_fails_closed_on_missing_run_descriptive_key(
    field_paths: tuple[str, ...],
    match: str,
) -> None:
    payload = _complete_cs_spec()
    for field_path in field_paths:
        _delete_path(payload, field_path)

    with pytest.raises(MissingTrainingRunSpecFieldError, match=match) as exc_info:
        _cs_training_config(payload, spec_dir=Path("results/test/runs/complete_cs_fixture"))

    assert "complete_cs_fixture" in exc_info.value.spec_identity


def test_training_config_adapters_distinguish_explicit_null_from_absent_gradient_clip() -> None:
    cs_payload = _complete_cs_spec()
    cs_payload["optimizer"]["gradient_clip_norm"] = None
    assert _cs_training_config(cs_payload).grad_clip is None

    missing_cs_payload = _complete_cs_spec()
    del missing_cs_payload["optimizer"]["gradient_clip_norm"]
    with pytest.raises(MissingTrainingRunSpecFieldError, match="optimizer.gradient_clip_norm"):
        _cs_training_config(missing_cs_payload)

    guided_payload = _complete_guided_spec()
    guided_payload["optimizer"]["gradient_clip_norm"] = None
    assert (
        _distillation_training_config(guided_payload, method="guided_distillation").grad_clip
        is None
    )

    closed_loop_payload = _complete_closed_loop_spec()
    closed_loop_payload["student_contract"]["gradient_clip_norm"] = None
    assert (
        _distillation_training_config(
            closed_loop_payload,
            method="closed_loop_distillation",
        ).grad_clip
        is None
    )


def test_cs_training_config_ignores_optional_training_diagnostics_metadata() -> None:
    payload = _complete_cs_spec()
    with_metadata = _cs_training_config(payload)

    without_metadata = copy.deepcopy(payload)
    without_metadata.pop("training_diagnostics", None)

    assert _cs_training_config(without_metadata) == with_metadata


@pytest.mark.parametrize(("path", "method"), TRACKED_CLEAN_DISTILLATION_SPECS)
def test_tracked_clean_distillation_specs_adapt(path: Path, method: str) -> None:
    payload = _read_json(path)

    config = _distillation_training_config(payload, method=method, spec_path=path)

    assert config.n_batches > 0
    assert config.batch_size > 0
    assert config.hidden_dim > 0
    assert config.n_reach_steps > 0
    assert config.snapshot_interval > 0


def test_tracked_training_config_adapter_corpus_census() -> None:
    clean_paths = []
    fail_closed: dict[Path, str] = {}

    for path in _flat_tracked_run_specs():
        payload = _read_json(path)
        method = _adapter_method_for_payload(payload)
        if method is None:
            continue
        try:
            if method == "cs_gru":
                _cs_training_config(payload, spec_dir=path.parent)
            else:
                _distillation_training_config(payload, method=method, spec_path=path)
        except MissingTrainingRunSpecFieldError as exc:
            fail_closed[path] = exc.field_path
        else:
            clean_paths.append(path)

    assert len(clean_paths) == 67
    assert fail_closed == EXPECTED_TRACKED_FAIL_CLOSED


def test_directory_training_config_adapter_corpus_census() -> None:
    clean_paths = []
    fail_closed: dict[Path, str] = {}

    for path in _directory_tracked_run_specs():
        payload = _read_json(path)
        method = _adapter_method_for_payload(payload)
        if method is None:
            continue
        try:
            if method == "cs_gru":
                _cs_training_config(payload, spec_dir=path.parent)
            else:
                _distillation_training_config(payload, method=method, spec_path=path)
        except MissingTrainingRunSpecFieldError as exc:
            fail_closed[path] = exc.field_path
        else:
            clean_paths.append(path)

    assert len(clean_paths) == 73
    assert fail_closed == EXPECTED_DIRECTORY_RUN_FAIL_CLOSED
