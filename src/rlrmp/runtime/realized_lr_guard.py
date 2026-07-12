"""Interim post-run conformance guard for realized optimizer learning rates."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from feedbax.contracts.training import DEFAULT_TRAINING_METHOD_REGISTRY, TrainingRunSpec

from rlrmp.runtime.training_run_specs import feedbax_training_run_spec_from_payload
from rlrmp.train.adaptive_epsilon_native import (
    ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
    AdaptiveEpsilonMethodPayload,
    adaptive_epsilon_controller_lr_points,
    lr_report_schedule_steps,
)
from rlrmp.train.cs_nominal_gru import _config_namespace, build_hps


@dataclass(frozen=True)
class RealizedLrPoint:
    """One expected-versus-realized learning-rate probe."""

    label: str
    step: int
    expected: float
    realized_min: float
    realized_max: float


def verify_realized_learning_rates(
    *,
    spec_path: Path,
    diagnostics_path: Path,
) -> list[RealizedLrPoint]:
    """Verify adaptive-run LR diagnostics through the execution optimizer builder.

    An empty result means the tracked recipe is not an adaptive-epsilon run, so
    the interim guard does not apply. Adaptive recipes fail closed when their
    typed schedule or realized diagnostics are absent or malformed.
    """

    recipe = _load_json_object(spec_path)
    if "feedbax_training_run_spec" not in recipe:
        hps = recipe.get("hps")
        if (
            recipe.get("adaptive_epsilon_curriculum") is True
            or isinstance(hps, Mapping)
            and hps.get("adaptive_epsilon_curriculum") is True
        ):
            raise ValueError(
                "adaptive-epsilon tracked spec lacks feedbax_training_run_spec; "
                "the realized LR guard requires the typed execution contract"
            )
        return []
    run_spec = feedbax_training_run_spec_from_payload(recipe)
    if _method_ref(run_spec) != ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF:
        return []

    payload = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        run_spec.method_ref,
        run_spec.method_payload,
        path="/method_payload",
    )
    if not isinstance(payload, AdaptiveEpsilonMethodPayload):
        raise TypeError("adaptive-epsilon method did not validate to its typed payload")
    optimizer = payload.controller_optimizer
    if optimizer is None or optimizer.lr_schedule is None:
        raise ValueError("adaptive-epsilon tracked spec lacks a controller LR schedule")

    labelled_steps = _five_point_schedule(optimizer.lr_schedule)
    args = _config_namespace(payload.config)
    expected_rows = adaptive_epsilon_controller_lr_points(
        run_spec,
        build_hps(args),
        schedule_origin_step=0,
        current_step=0,
        optimizer_count_at_current_step=0,
        schedule_steps=[step for _, step in labelled_steps],
    )
    realized = _load_realized_trace(diagnostics_path)
    max_step = max(step for _, step in labelled_steps)
    if realized.shape[0] <= max_step:
        raise ValueError(
            "optimizer_learning_rate trace is too short for the declared schedule: "
            f"length={realized.shape[0]} required_step={max_step}"
        )

    verified: list[RealizedLrPoint] = []
    for (label, step), expected_row in zip(labelled_steps, expected_rows, strict=True):
        expected = float(expected_row["lr"])
        observed = np.asarray(realized[step], dtype=np.float64).reshape(-1)
        if observed.size == 0 or not np.all(np.isfinite(observed)):
            raise ValueError(f"optimizer_learning_rate has no finite values at {label} step={step}")
        tolerance = max(1e-12, abs(expected) * 1e-5)
        mismatched = np.abs(observed - expected) > tolerance
        if np.any(mismatched):
            raise ValueError(
                "realized LR mismatch at "
                f"{label} step={step}: expected={expected:.12g} "
                f"observed_min={float(observed.min()):.12g} "
                f"observed_max={float(observed.max()):.12g} "
                f"tolerance={tolerance:.12g}"
            )
        verified.append(
            RealizedLrPoint(
                label=label,
                step=step,
                expected=expected,
                realized_min=float(observed.min()),
                realized_max=float(observed.max()),
            )
        )
    return verified


def _five_point_schedule(schedule: Any) -> list[tuple[str, int]]:
    steps = lr_report_schedule_steps(schedule, start_position=0)
    labels = ["step_0", "mid_warmup", "peak", "decay", "terminal"]
    if len(steps) != 5:
        raise ValueError(
            "controller LR schedule does not declare the 020e122 five-point surface: "
            f"steps={steps}"
        )
    return list(zip(labels, steps, strict=True))


def _load_realized_trace(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing training diagnostics: {path}")
    with np.load(path, allow_pickle=False) as diagnostics:
        if "optimizer_learning_rate" not in diagnostics:
            raise KeyError("training diagnostics lacks optimizer_learning_rate")
        trace = np.asarray(diagnostics["optimizer_learning_rate"])
    if trace.ndim not in (1, 2):
        raise ValueError(
            "optimizer_learning_rate must have batch or batch-by-replicate shape; "
            f"found {trace.shape}"
        )
    return trace


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing tracked run spec: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"tracked run spec must contain a JSON object: {path}")
    return dict(payload)


def _method_ref(spec: TrainingRunSpec) -> str:
    return f"{spec.method_ref.package}/{spec.method_ref.name}/{spec.method_ref.version}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        points = verify_realized_learning_rates(
            spec_path=args.spec,
            diagnostics_path=args.diagnostics,
        )
    except Exception as exc:
        print(f"Realized-LR guard: FAIL: {exc}", file=sys.stderr)
        return 1
    if not points:
        print("Realized-LR guard: not applicable to this tracked run spec.")
        return 0
    print("Realized-LR guard: PASS")
    for point in points:
        print(
            f"  {point.label}: step={point.step} expected={point.expected:.12g} "
            f"realized=[{point.realized_min:.12g}, {point.realized_max:.12g}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
