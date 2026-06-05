"""Open-loop physical calibration for the C&S perturbation bank."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.gru_evaluation_diagnostics import RolloutEvaluation
from rlrmp.analysis.gru_perturbation_bank import (
    _build_extlqg_comparator_context,
    _evaluation_from_extlqg_rollout,
    _extlqg_cost_summary,
    _extlqg_process_epsilon,
    _perturbed_extlqg_initial_state,
    _simulate_extlqg_perturbed,
    default_cs_perturbation_bank,
    extlqg_comparator_status,
    summarize_perturbation_response,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


SCHEMA_VERSION = "rlrmp.perturbation_open_loop_calibration.v2"
DEFAULT_RESULT_EXPERIMENT = "1ad3c16"
DEFAULT_OUTPUT_FILENAME = "perturbation_open_loop_calibration.json"
DEFAULT_NOTE_FILENAME = "perturbation_open_loop_calibration.md"
DEFAULT_BULK_SUBDIR = "perturbation_open_loop_calibration"
DEFAULT_NOMINAL_GRU_BASELINE = {
    "experiment": "5f70333",
    "run_id": "lss_stabilization_fullqrf_warmcos__lr1e-3_clip5_b64",
    "training_distribution": "single_target_nominal_only_full_Q/R/Q_f",
    "role": (
        "declared nominal-only GRU baseline for later closed-loop GRU calibration; "
        "this materializer defines bins from extLQG nominal-command open-loop replay"
    ),
}
DEFAULT_AMPLITUDE_FACTORS = (
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
)
OPEN_LOOP_SUPPORTED_CHANNELS = {"initial_state", "command_input", "process_epsilon"}


@dataclass(frozen=True)
class ReachCalibrationPoint:
    """A reach length whose relative perturbation levels should be calibrated."""

    label: str
    split: str
    reach_length_m: float
    role: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable calibration point."""

        return {
            "label": self.label,
            "split": self.split,
            "reach_length_m": float(self.reach_length_m),
            "role": self.role,
        }


@dataclass(frozen=True)
class ReachRelativeLevel:
    """A relative open-loop effect-size level."""

    name: str
    fraction_of_reach: float
    role: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable level definition."""

        return {
            "name": self.name,
            "fraction_of_reach": float(self.fraction_of_reach),
            "role": self.role,
        }


DEFAULT_REACH_CALIBRATION_POINTS = (
    ReachCalibrationPoint(
        label="seen_train_0p10",
        split="seen/train",
        reach_length_m=0.10,
        role="multi_target_training_reach_length",
    ),
    ReachCalibrationPoint(
        label="seen_train_anchor_0p15",
        split="seen/train",
        reach_length_m=0.15,
        role="multi_target_training_reach_length_and_original_anchor",
    ),
    ReachCalibrationPoint(
        label="heldout_eval_0p12",
        split="held-out/eval",
        reach_length_m=0.12,
        role="multi_target_held_out_evaluation_reach_length",
    ),
    ReachCalibrationPoint(
        label="heldout_eval_0p18",
        split="held-out/eval",
        reach_length_m=0.18,
        role="multi_target_held_out_evaluation_reach_length",
    ),
)
DEFAULT_REACH_RELATIVE_LEVELS = (
    ReachRelativeLevel(name="small", fraction_of_reach=0.05, role="small_probe"),
    ReachRelativeLevel(name="moderate", fraction_of_reach=0.10, role="moderate_probe"),
    ReachRelativeLevel(name="stress", fraction_of_reach=0.25, role="stress_probe"),
)


def materialize_perturbation_open_loop_calibration(
    *,
    amplitude_factors: Sequence[float] = DEFAULT_AMPLITUDE_FACTORS,
    reach_points: Sequence[ReachCalibrationPoint] = DEFAULT_REACH_CALIBRATION_POINTS,
    levels: Sequence[ReachRelativeLevel] = DEFAULT_REACH_RELATIVE_LEVELS,
    result_experiment: str = DEFAULT_RESULT_EXPERIMENT,
    output_path: Path | None = None,
    note_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Materialize physical effect-size calibration for perturbation amplitudes."""

    output_path = output_path or (
        repo_root / "_artifacts" / result_experiment / DEFAULT_BULK_SUBDIR / DEFAULT_OUTPUT_FILENAME
    )
    note_path = note_path or (
        repo_root / "results" / result_experiment / "notes" / DEFAULT_NOTE_FILENAME
    )
    mkdir_p(output_path.parent)
    bank = default_cs_perturbation_bank()
    context = _build_extlqg_comparator_context()
    base = context["base_evaluation"]
    base_cost = _extlqg_cost_summary(base, context["base_initial_state"])
    sensitivities = _family_sensitivities(
        bank["perturbations"],
        context=context,
        base=base,
        base_cost=base_cost,
    )
    rows = []
    for reach in reach_points:
        for level in levels:
            target_peak_delta_x_m = float(reach.reach_length_m) * float(level.fraction_of_reach)
            for family, sensitivity in sensitivities.items():
                if sensitivity.get("status") != "available":
                    rows.append(
                        _unsupported_open_loop_row(
                            sensitivity["perturbation"],
                            reach=reach,
                            level=level,
                            target_peak_delta_x_m=target_peak_delta_x_m,
                            reason=sensitivity.get("reason"),
                        )
                    )
                    continue
                amplitude = target_peak_delta_x_m / float(sensitivity["peak_delta_x_per_unit"])
                scaled = _set_perturbation_amplitude(
                    sensitivity["perturbation"],
                    amplitude=float(amplitude),
                    suffix=(
                        f"{reach.label}__{level.name}"
                        f"__target_{target_peak_delta_x_m * 1000.0:.3f}mm"
                    ),
                )
                rows.append(
                    _evaluate_calibration_row(
                        scaled,
                        amplitude_factor=float(amplitude)
                        / float(sensitivity["perturbation"]["amplitude"]),
                        reach=reach,
                        level=level,
                        target_peak_delta_x_m=target_peak_delta_x_m,
                        context=context,
                        base=base,
                        base_cost=base_cost,
                        sensitivity=sensitivity,
                    )
                )
    for family, sensitivity in sensitivities.items():
        if sensitivity.get("status") != "available":
            continue
        rows.append(
            {
                "row_kind": "unit_sensitivity",
                "perturbation_id": sensitivity["perturbation_id"],
                "channel": sensitivity["channel"],
                "family": family,
                "axis": sensitivity.get("axis"),
                "sign": sensitivity.get("sign"),
                "timing": sensitivity.get("timing"),
                "amplitude": 1.0,
                "open_loop_peak_delta_x_per_unit_m": sensitivity["peak_delta_x_per_unit"],
                "open_loop_auc_delta_x_per_unit_m_s": sensitivity["auc_delta_x_per_unit"],
                "selection_rule": sensitivity["selection_rule"],
                "open_loop": sensitivity["open_loop"],
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": result_experiment,
        "bank_schema_version": bank["schema_version"],
        "scope": "extlqg_nominal_command_open_loop_physical_effect_calibration",
        "calibration_mode": "reach_relative_peak_delta_x",
        "reach_points": [reach.to_json() for reach in reach_points],
        "level_definitions": [level.to_json() for level in levels],
        "legacy_fixed_mm_amplitude_factors": list(amplitude_factors),
        "target_rule": "target_peak_delta_x_m = reach_length_m * fraction_of_reach",
        "selection_rule": (
            "For each supported family, use the deterministic representative row "
            "chosen by earliest timing, x axis, positive sign, and stable "
            "perturbation_id tie-breaks; solve amplitude from the unit-amplitude "
            "open-loop peak delta x sensitivity, then re-evaluate the solved row."
        ),
        "open_loop_replay_geometry": {
            "nominal_reach_length_m": 0.15,
            "note": (
                "The extLQG comparator helper currently replays nominal commands on "
                "the canonical 0.15 m +x target. The reach-relative calibration "
                "varies the requested physical-effect target by declared reach "
                "length; it does not retrain or rebuild multi-target extLQG rows."
            )
        },
        "open_loop_reference": {
            "controller": "extLQG nominal command replay",
            "feedback_correction": False,
            "role": (
                "define physical perturbation bins by peak delta x before feedback "
                "correction; closed-loop extLQG metrics are reported at the same "
                "amplitudes only for interpretation"
            ),
        },
        "nominal_gru_baseline_for_later_closed_loop_calibration": DEFAULT_NOMINAL_GRU_BASELINE,
        "bulk_manifest_path": _repo_relative(output_path, repo_root=repo_root),
        "rows": rows,
        "family_summary": _summarize_reach_relative_rows(rows),
    }
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    note_path.write_text(render_calibration_markdown(manifest), encoding="utf-8")
    return manifest


def render_calibration_markdown(manifest: Mapping[str, Any]) -> str:
    """Render a compact calibration summary."""

    lines = [
        "# Perturbation Open-Loop Calibration",
        "",
        f"- Issue: `{manifest['issue']}`",
        f"- Scope: `{manifest['scope']}`",
        "- Open-loop reference: extLQG nominal command replay.",
        "- Closed-loop extLQG is reported at the same amplitudes where supported.",
        "- Calibration mode: reach-relative peak `delta x`, with target peak "
        "`delta x = fraction * reach_length`.",
        "- Replay geometry: canonical 0.15 m +x extLQG nominal command replay; "
        "the reach-relative targets vary the requested physical effect size, not "
        "the nominal replay task.",
        "- GRU baseline for later closed-loop calibration: "
        f"`{manifest['nominal_gru_baseline_for_later_closed_loop_calibration']['experiment']}` / "
        f"`{manifest['nominal_gru_baseline_for_later_closed_loop_calibration']['run_id']}` "
        "(single-target nominal-only; documented for later closed-loop comparison, "
        "not retrained here).",
        f"- Bulk row manifest: `{manifest.get('bulk_manifest_path', 'not_materialized')}`",
        "",
        "## Deterministic Config",
        "",
        "Reach lengths:",
        "",
        "| Label | Split | Reach length | Role |",
        "|---|---|---:|---|",
    ]
    for reach in manifest["reach_points"]:
        lines.append(
            f"| `{reach['label']}` | {reach['split']} | "
            f"{_fmt_mm(reach['reach_length_m'])} | {reach['role']} |"
        )
    lines.extend(
        [
            "",
            "Levels:",
            "",
            "| Level | Fraction of reach | Role |",
            "|---|---:|---|",
        ]
    )
    for level in manifest["level_definitions"]:
        lines.append(
            f"| `{level['name']}` | {100.0 * float(level['fraction_of_reach']):.1f}% | "
            f"{level['role']} |"
        )
    lines.extend(
        [
            "",
            "## Selected Reach-Relative Amplitudes",
            "",
            "| Reach | Level | Family | Amplitude | Target peak dx | Achieved peak dx | "
            "Achieved % reach | AUC dx | Notes |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in _selected_calibration_rows(manifest["rows"]):
        notes = row.get("warning") or "none"
        lines.append(
            f"| `{row['reach_label']}` | `{row['level_name']}` | `{row['family']}` | "
            f"{_fmt(row['amplitude'])} | "
            f"{_fmt_mm(row['target_peak_delta_x_m'])} | "
            f"{_fmt_mm(row['achieved_peak_delta_x_m'])} | "
            f"{_fmt_pct(row['achieved_peak_delta_x_fraction_of_reach'])} | "
            f"{_fmt(row.get('achieved_auc_delta_x_m_s'))} | "
            f"{notes} |"
        )
    lines.extend(
        [
            "",
            "## Rerun Command",
            "",
            "```bash",
            "uv run python scripts/materialize_perturbation_open_loop_calibration.py",
            "```",
            "",
            "Bulk per-row data is written under `_artifacts/1ad3c16/...`; keep it out of "
            "`results/`.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def _evaluate_calibration_row(
    perturbation: Mapping[str, Any],
    *,
    amplitude_factor: float,
    reach: ReachCalibrationPoint,
    level: ReachRelativeLevel,
    target_peak_delta_x_m: float,
    context: Mapping[str, Any],
    base: RolloutEvaluation,
    base_cost: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    open_loop_eval, initial_state, provenance = _simulate_open_loop_command_replay(
        perturbation,
        context=context,
    )
    open_loop_cost = _extlqg_cost_summary(open_loop_eval, initial_state)
    open_loop_metrics = summarize_perturbation_response(
        base,
        open_loop_eval,
        base_full_qrf_cost=base_cost,
        perturbed_full_qrf_cost=open_loop_cost,
    )
    extlqg_status = extlqg_comparator_status(perturbation, status="not_applicable")
    if perturbation["channel"] != "command_input":
        try:
            ext_eval, ext_initial, ext_provenance = _simulate_extlqg_perturbed(
                perturbation,
                context=context,
            )
            ext_cost = _extlqg_cost_summary(ext_eval, ext_initial)
            extlqg_status = {
                "status": "available",
                "analytical_adapter": ext_provenance,
                "reference_response_metrics": summarize_perturbation_response(
                    base,
                    ext_eval,
                    base_full_qrf_cost=base_cost,
                    perturbed_full_qrf_cost=ext_cost,
                ),
            }
        except (KeyError, ValueError) as exc:
            extlqg_status = {"status": "blocked", "reason": str(exc)}
    peak = _metric_mean(open_loop_metrics, "delta_position_response_m.max")
    auc = _metric_mean(open_loop_metrics, "delta_position_response_m.auc")
    target_error = None if peak is None else float(peak) - float(target_peak_delta_x_m)
    warning = None
    if peak is None:
        warning = "peak delta x unavailable"
    elif abs(target_error) > max(1e-9, 1e-6 * float(target_peak_delta_x_m)):
        warning = "achieved peak differs from target beyond numerical tolerance"
    return {
        "row_kind": "reach_relative_calibrated_amplitude",
        "reach_label": reach.label,
        "reach_split": reach.split,
        "reach_length_m": float(reach.reach_length_m),
        "level_name": level.name,
        "level_fraction_of_reach": float(level.fraction_of_reach),
        "target_peak_delta_x_m": float(target_peak_delta_x_m),
        "achieved_peak_delta_x_m": peak,
        "achieved_peak_delta_x_mm": None if peak is None else float(peak) * 1000.0,
        "achieved_peak_delta_x_fraction_of_reach": (
            None if peak is None else float(peak) / float(reach.reach_length_m)
        ),
        "achieved_auc_delta_x_m_s": auc,
        "target_peak_error_m": target_error,
        "warning": warning,
        "sensitivity_reference": {
            "perturbation_id": sensitivity["perturbation_id"],
            "peak_delta_x_per_unit_m": sensitivity["peak_delta_x_per_unit"],
            "auc_delta_x_per_unit_m_s": sensitivity["auc_delta_x_per_unit"],
            "selection_rule": sensitivity["selection_rule"],
        },
        "perturbation_id": perturbation["perturbation_id"],
        "channel": perturbation["channel"],
        "family": perturbation["family"],
        "axis": perturbation.get("axis"),
        "sign": perturbation.get("sign"),
        "amplitude": perturbation["amplitude"],
        "amplitude_factor": amplitude_factor,
        "timing": perturbation.get("timing"),
        "open_loop": {
            "status": "available",
            "adapter": provenance,
            "response_metrics": open_loop_metrics,
        },
        "closed_loop_extlqg": extlqg_status,
    }


def _simulate_open_loop_command_replay(
    perturbation: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> tuple[RolloutEvaluation, np.ndarray, dict[str, Any]]:
    """Replay nominal extLQG commands through perturbed plant dynamics."""

    plant = context["plant"]
    schedule = context["schedule"]
    x0 = jnp.asarray(context["base_initial_state"], dtype=jnp.float64)
    command = jnp.asarray(context["base_evaluation"].command[0, 0], dtype=jnp.float64)
    epsilon = jnp.zeros((schedule.T, plant.m_w), dtype=jnp.float64)
    provenance = {
        "adapter": "extlqg_nominal_command_open_loop_replay",
        "feedback_correction": False,
    }
    if perturbation["channel"] == "initial_state":
        x0 = _perturbed_extlqg_initial_state(x0, perturbation)
        provenance["perturbation_adapter"] = "initial_state_offset"
    elif perturbation["channel"] == "process_epsilon":
        epsilon = _extlqg_process_epsilon(perturbation, schedule.T, plant.m_w)
        provenance["perturbation_adapter"] = "process_epsilon_sequence"
    elif perturbation["channel"] == "command_input":
        command = _apply_command_input_to_open_loop_command(command, perturbation)
        provenance["perturbation_adapter"] = "command_input_added_to_replayed_command"
    else:
        raise ValueError(f"open-loop replay does not support {perturbation['channel']!r}")
    xs = [x0]
    for t in range(schedule.T):
        xs.append(plant.A @ xs[-1] + plant.B @ command[t] + plant.Bw @ epsilon[t])
    rollout = type("OpenLoopRollout", (), {})()
    rollout.x = jnp.stack(xs, axis=0)
    rollout.u_command = command
    evaluation = _evaluation_from_extlqg_rollout(rollout, initial_state=x0)
    return evaluation, np.asarray(x0), provenance


def _apply_command_input_to_open_loop_command(
    command: Any,
    perturbation: Mapping[str, Any],
) -> jnp.ndarray:
    result = jnp.array(command, dtype=jnp.float64)
    timing = perturbation.get("timing", {})
    start = int(timing.get("start_time_index", 0))
    duration = int(timing.get("duration_steps", 1))
    axis = 0 if perturbation.get("axis") == "x" else 1
    sign = float(perturbation.get("sign", 1))
    amp = float(perturbation["amplitude"])
    end = min(result.shape[0], start + duration)
    return result.at[start:end, axis].add(sign * amp)


def _scale_perturbation(perturbation: Mapping[str, Any], *, factor: float) -> dict[str, Any]:
    row = dict(perturbation)
    row["base_amplitude"] = float(perturbation["amplitude"])
    row["amplitude_factor"] = float(factor)
    row["amplitude"] = float(perturbation["amplitude"]) * float(factor)
    row["perturbation_id"] = f"{perturbation['perturbation_id']}__scale_{factor:g}"
    return row


def _set_perturbation_amplitude(
    perturbation: Mapping[str, Any],
    *,
    amplitude: float,
    suffix: str,
) -> dict[str, Any]:
    row = dict(perturbation)
    row["base_amplitude"] = float(perturbation["amplitude"])
    row["amplitude"] = float(amplitude)
    row["perturbation_id"] = f"{perturbation['perturbation_id']}__{suffix}"
    return row


def _unsupported_open_loop_row(
    perturbation: Mapping[str, Any],
    *,
    reach: ReachCalibrationPoint | None = None,
    level: ReachRelativeLevel | None = None,
    target_peak_delta_x_m: float | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "row_kind": "reach_relative_calibrated_amplitude"
        if reach is not None
        else "unsupported_open_loop_family",
        "reach_label": None if reach is None else reach.label,
        "reach_split": None if reach is None else reach.split,
        "reach_length_m": None if reach is None else float(reach.reach_length_m),
        "level_name": None if level is None else level.name,
        "level_fraction_of_reach": None if level is None else float(level.fraction_of_reach),
        "target_peak_delta_x_m": target_peak_delta_x_m,
        "perturbation_id": perturbation["perturbation_id"],
        "channel": perturbation["channel"],
        "family": perturbation["family"],
        "axis": perturbation.get("axis"),
        "sign": perturbation.get("sign"),
        "amplitude": perturbation["amplitude"],
        "status": "not_applicable",
        "reason": reason
        or (
            "open-loop command replay has no feedback channel, so sensory/delayed/"
            "target perturbations do not define a physical open-loop effect size"
        ),
    }


def _family_sensitivities(
    perturbations: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    base: RolloutEvaluation,
    base_cost: Mapping[str, Any],
) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for perturbation in perturbations:
        by_family.setdefault(str(perturbation["family"]), []).append(perturbation)
    sensitivities = {}
    for family, family_rows in sorted(by_family.items()):
        representative = _representative_perturbation(family_rows)
        if representative["channel"] not in OPEN_LOOP_SUPPORTED_CHANNELS:
            sensitivities[family] = {
                "status": "not_applicable",
                "reason": (
                    "open-loop command replay has no feedback/target channel for "
                    f"{representative['channel']!r}"
                ),
                "perturbation": representative,
            }
            continue
        unit = _set_perturbation_amplitude(
            representative,
            amplitude=1.0,
            suffix="unit_sensitivity",
        )
        row = _evaluate_unit_sensitivity_row(
            unit,
            context=context,
            base=base,
            base_cost=base_cost,
        )
        peak = _metric_mean(row["open_loop"]["response_metrics"], "delta_position_response_m.max")
        auc = _metric_mean(row["open_loop"]["response_metrics"], "delta_position_response_m.auc")
        if peak is None or abs(peak) <= 1e-12:
            sensitivities[family] = {
                "status": "blocked",
                "reason": "unit-amplitude open-loop peak delta x was unavailable or near zero",
                "perturbation": representative,
                "open_loop": row["open_loop"],
            }
            continue
        sensitivities[family] = {
            "status": "available",
            "perturbation": representative,
            "perturbation_id": representative["perturbation_id"],
            "channel": representative["channel"],
            "family": family,
            "axis": representative.get("axis"),
            "sign": representative.get("sign"),
            "timing": representative.get("timing"),
            "peak_delta_x_per_unit": float(abs(peak)),
            "auc_delta_x_per_unit": auc,
            "open_loop": row["open_loop"],
            "selection_rule": _representative_selection_rule(),
        }
    return sensitivities


def _representative_perturbation(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(
        rows,
        key=lambda row: (
            int(row.get("timing", {}).get("start_time_index", 0)),
            0 if row.get("axis") == "x" else 1,
            0 if int(row.get("sign", 1)) > 0 else 1,
            str(row["perturbation_id"]),
        ),
    )


def _representative_selection_rule() -> str:
    return "earliest timing, x axis, positive sign, perturbation_id tie-break"


def _evaluate_unit_sensitivity_row(
    perturbation: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    base: RolloutEvaluation,
    base_cost: Mapping[str, Any],
) -> dict[str, Any]:
    open_loop_eval, initial_state, provenance = _simulate_open_loop_command_replay(
        perturbation,
        context=context,
    )
    open_loop_cost = _extlqg_cost_summary(open_loop_eval, initial_state)
    open_loop_metrics = summarize_perturbation_response(
        base,
        open_loop_eval,
        base_full_qrf_cost=base_cost,
        perturbed_full_qrf_cost=open_loop_cost,
    )
    return {
        "perturbation_id": perturbation["perturbation_id"],
        "channel": perturbation["channel"],
        "family": perturbation["family"],
        "axis": perturbation.get("axis"),
        "sign": perturbation.get("sign"),
        "amplitude": perturbation["amplitude"],
        "timing": perturbation.get("timing"),
        "open_loop": {
            "status": "available",
            "adapter": provenance,
            "response_metrics": open_loop_metrics,
        },
    }


def _summarize_reach_relative_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("row_kind") != "reach_relative_calibrated_amplitude":
            continue
        by_family.setdefault(str(row["family"]), []).append(row)
    return {
        family: _summarize_reach_relative_family(rows_for_family)
        for family, rows_for_family in sorted(by_family.items())
    }


def _summarize_reach_relative_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    available = [
        row for row in rows if row.get("open_loop", {}).get("status") == "available"
    ]
    targets = []
    for row in rows:
        targets.append(
            {
                "perturbation_id": row["perturbation_id"],
                "reach_label": row.get("reach_label"),
                "level_name": row.get("level_name"),
                "amplitude": float(row["amplitude"]),
                "target_peak_delta_x_m": row.get("target_peak_delta_x_m"),
                "achieved_peak_delta_x_m": row.get("achieved_peak_delta_x_m"),
                "achieved_peak_delta_x_fraction_of_reach": row.get(
                    "achieved_peak_delta_x_fraction_of_reach"
                ),
                "axis": row.get("axis"),
                "sign": row.get("sign"),
            }
        )
    return {
        "n_targets": len(rows),
        "n_available": len(available),
        "targets": targets,
        "warnings": sorted({str(row["warning"]) for row in rows if row.get("warning")}),
    }


def _metric_mean(metrics: Mapping[str, Any], key: str) -> float | None:
    current: Any = metrics
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, Mapping):
        current = current.get("mean")
    if current is None:
        return None
    value = float(current)
    return value if np.isfinite(value) else None


def _selected_calibration_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("row_kind") == "reach_relative_calibrated_amplitude"
        and row.get("open_loop", {}).get("status") == "available"
    ]
    return sorted(
        selected,
        key=lambda row: (
            str(row.get("reach_split")),
            float(row.get("reach_length_m") or 0.0),
            float(row.get("level_fraction_of_reach") or 0.0),
            str(row.get("family")),
        ),
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1e3 or (abs(value) > 0.0 and abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.6g}"


def _fmt_mm(value_m: float | None) -> str:
    if value_m is None:
        return "n/a"
    return f"{float(value_m) * 1000.0:.3f} mm"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.3f}%"


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_REACH_CALIBRATION_POINTS",
    "DEFAULT_REACH_RELATIVE_LEVELS",
    "DEFAULT_NOMINAL_GRU_BASELINE",
    "ReachCalibrationPoint",
    "ReachRelativeLevel",
    "SCHEMA_VERSION",
    "materialize_perturbation_open_loop_calibration",
    "render_calibration_markdown",
]
