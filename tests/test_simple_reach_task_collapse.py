"""Regression tests for rlrmp simple-reach task construction."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.random as jr
import numpy as np
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from feedbax.objectives.loss import TargetSpec

from rlrmp.runtime.training_run_specs import hydrate_compact_run_spec_envelope
from rlrmp.train.task_model import build_task_base

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "simple_reach_task_golden.json"


def _array_payload(value: Any) -> dict[str, Any]:
    arr = np.asarray(jax.device_get(value))
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "values": arr.tolist(),
    }


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (jax.Array, np.ndarray)):
        return _array_payload(value)
    if isinstance(value, TargetSpec):
        return {
            "type": "TargetSpec",
            "value": _serialize(value.value),
            "time_idxs": _serialize(value.time_idxs),
            "time_mask": _serialize(value.time_mask),
            "discount": _serialize(value.discount),
        }
    if isinstance(value, dict):
        return {
            f"entry_{idx}": _serialize(item_value)
            for idx, (_item_key, item_value) in enumerate(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if dataclasses.is_dataclass(value):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not callable(getattr(value, field.name))
        }
    if hasattr(value, "__dict__"):
        return {
            key: _serialize(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _trial_payload(spec: Any) -> dict[str, Any]:
    timeline = spec.timeline
    return {
        "inits": _serialize(spec.inits),
        "inputs": _serialize(spec.inputs),
        "targets": _serialize(spec.targets),
        "timeline": {
            "n_steps": int(timeline.n_steps),
            "epoch_bounds": _serialize(timeline.epoch_bounds),
            "epoch_names": list(timeline.epoch_names),
        },
    }


def _representative_hps(task_type: str) -> TreeNamespace:
    task = {
        "type": task_type,
        "n_steps": 7,
        "workspace": [[-0.2, -0.1], [0.25, 0.2]],
        "eval_grid_n": 1,
        "eval_n_directions": 3,
        "eval_reach_length": 0.12,
        "fixed_init_pos": [0.01, -0.02],
        "fixed_target_pos": [0.13, 0.04],
        "epoch_len_ranges": [[0, 1], [4, 5]],
        "target_on_epochs": [0],
        "hold_epochs": [],
        "move_epochs": [0],
        "p_catch_trial": 0.0,
        "train_endpoint_mode": "center_out",
        "preset": None,
        "n_control_stages": None,
        "target_visible_from_start": None,
        "go_cue_event_name": None,
        "catch_metadata_policy": None,
    }
    if task_type == "simple_reach":
        task.pop("fixed_init_pos")
        task.pop("fixed_target_pos")
    return TreeNamespace(
        method="pai-asf",
        task=TreeNamespace(**task),
        loss=TreeNamespace(weights=TreeNamespace()),
        model=TreeNamespace(),
    )


def test_simple_reach_task_outputs_match_pre_refactor_golden() -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]
    for task_type in ("simple_reach", "fixed_simple_reach"):
        task = build_task_base(_representative_hps(task_type))
        actual = {
            "n_validation_trials": int(task.n_validation_trials),
            "train": _trial_payload(task.get_train_trial(jr.PRNGKey(123))),
            "validation": _trial_payload(task.get_validation_trials(jr.PRNGKey(456))),
        }
        expected = {
            "n_validation_trials": golden[task_type]["n_validation_trials"],
            "train": golden[task_type]["train"],
            "validation": golden[task_type]["validation"],
        }
        assert actual == expected


def _fixed_simple_reach_run_spec_paths() -> list[Path]:
    paths: list[Path] = []
    for path in sorted((REPO_ROOT / "results").glob("*/runs/*.json")):
        payload = hydrate_compact_run_spec_envelope(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if payload.get("hps", {}).get("task", {}).get("type") == "fixed_simple_reach":
            paths.append(path)
    return paths


def test_tracked_fixed_simple_reach_run_specs_still_build() -> None:
    paths = _fixed_simple_reach_run_spec_paths()
    assert len(paths) == 51
    for path in paths:
        payload = hydrate_compact_run_spec_envelope(
            json.loads(path.read_text(encoding="utf-8"))
        )
        hps = dict_to_namespace(payload["hps"], to_type=TreeNamespace)
        task = build_task_base(hps)
        validation = task.get_validation_trials(jr.PRNGKey(0))
        assert task.n_validation_trials == 1, path
        assert validation.timeline.epoch_names == ("movement",), path
        assert validation.timeline.epoch_bounds is not None, path
