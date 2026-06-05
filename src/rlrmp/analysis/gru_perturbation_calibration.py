"""Open-loop physical calibration for the C&S perturbation bank."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from rlrmp.analysis.cs_game_card import TARGET_POS
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


SCHEMA_VERSION = "rlrmp.perturbation_open_loop_calibration.v1"
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
TARGET_BINS_M = {
    "small": (0.002, 0.005),
    "moderate": (0.010, 0.020),
    "strong": (0.020, 0.050),
}
OPEN_LOOP_SUPPORTED_CHANNELS = {"initial_state", "command_input", "process_epsilon"}


def materialize_perturbation_open_loop_calibration(
    *,
    amplitude_factors: Sequence[float] = DEFAULT_AMPLITUDE_FACTORS,
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
    rows = []
    for perturbation in bank["perturbations"]:
        if perturbation["channel"] not in OPEN_LOOP_SUPPORTED_CHANNELS:
            rows.append(_unsupported_open_loop_row(perturbation))
            continue
        for factor in amplitude_factors:
            scaled = _scale_perturbation(perturbation, factor=float(factor))
            rows.append(
                _evaluate_calibration_row(
                    scaled,
                    amplitude_factor=float(factor),
                    context=context,
                    base=base,
                    base_cost=base_cost,
                )
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "issue": result_experiment,
        "bank_schema_version": bank["schema_version"],
        "scope": "extlqg_nominal_command_open_loop_physical_effect_calibration",
        "bin_definitions_m": {
            key: {"min_peak_delta_x_m": lo, "max_peak_delta_x_m": hi}
            for key, (lo, hi) in TARGET_BINS_M.items()
        },
        "amplitude_factors": list(amplitude_factors),
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
        "family_summary": _summarize_family_bins(rows),
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
        "- GRU baseline for later closed-loop calibration: "
        f"`{manifest['nominal_gru_baseline_for_later_closed_loop_calibration']['experiment']}` / "
        f"`{manifest['nominal_gru_baseline_for_later_closed_loop_calibration']['run_id']}` "
        "(single-target nominal-only).",
        f"- Bulk row manifest: `{manifest.get('bulk_manifest_path', 'not_materialized')}`",
        "",
        "## Selected Amplitude Candidates",
        "",
        "| Family | Small amp | Small peak dx | Moderate amp | Moderate peak dx | "
        "Strong amp | Strong peak dx | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for family, summary in manifest["family_summary"].items():
        selections = summary.get("selected_bins", {})
        notes = "; ".join(summary.get("warnings", ())) or "none"
        lines.append(
            f"| `{family}` | "
            f"{_fmt(_selected(selections, 'small', 'amplitude'))} | "
            f"{_fmt(_selected(selections, 'small', 'open_loop_peak_delta_x_m'))} | "
            f"{_fmt(_selected(selections, 'moderate', 'amplitude'))} | "
            f"{_fmt(_selected(selections, 'moderate', 'open_loop_peak_delta_x_m'))} | "
            f"{_fmt(_selected(selections, 'strong', 'amplitude'))} | "
            f"{_fmt(_selected(selections, 'strong', 'open_loop_peak_delta_x_m'))} | "
            f"{notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def _evaluate_calibration_row(
    perturbation: Mapping[str, Any],
    *,
    amplitude_factor: float,
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
    return {
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


def _unsupported_open_loop_row(perturbation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "perturbation_id": perturbation["perturbation_id"],
        "channel": perturbation["channel"],
        "family": perturbation["family"],
        "axis": perturbation.get("axis"),
        "sign": perturbation.get("sign"),
        "amplitude": perturbation["amplitude"],
        "status": "not_applicable",
        "reason": (
            "open-loop command replay has no feedback channel, so sensory/delayed/"
            "target perturbations do not define a physical open-loop effect size"
        ),
    }


def _summarize_family_bins(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("open_loop", {}).get("status") != "available":
            continue
        by_family.setdefault(str(row["family"]), []).append(row)
    return {
        family: _summarize_family(rows_for_family)
        for family, rows_for_family in sorted(by_family.items())
    }


def _summarize_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = []
    for row in rows:
        peak = _metric_mean(
            row["open_loop"]["response_metrics"],
            "delta_position_response_m.max",
        )
        if peak is None:
            continue
        candidates.append(
            {
                "perturbation_id": row["perturbation_id"],
                "amplitude": float(row["amplitude"]),
                "amplitude_factor": float(row["amplitude_factor"]),
                "open_loop_peak_delta_x_m": float(peak),
                "axis": row.get("axis"),
                "sign": row.get("sign"),
            }
        )
    selected = {}
    warnings = []
    for bin_name, (lo, hi) in TARGET_BINS_M.items():
        in_bin = [
            candidate
            for candidate in candidates
            if lo <= abs(candidate["open_loop_peak_delta_x_m"]) <= hi
        ]
        if in_bin:
            target_mid = 0.5 * (lo + hi)
            selected[bin_name] = min(
                in_bin,
                key=lambda candidate: abs(abs(candidate["open_loop_peak_delta_x_m"]) - target_mid),
            )
        else:
            warnings.append(f"no candidate landed in {bin_name} bin")
            selected[bin_name] = None
    return {
        "n_candidates": len(candidates),
        "selected_bins": selected,
        "warnings": warnings,
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


def _selected(selections: Mapping[str, Any], bin_name: str, key: str) -> float | None:
    selected = selections.get(bin_name)
    if not isinstance(selected, Mapping):
        return None
    value = selected.get(key)
    return None if value is None else float(value)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1e3 or (abs(value) > 0.0 and abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.6g}"


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_NOMINAL_GRU_BASELINE",
    "SCHEMA_VERSION",
    "materialize_perturbation_open_loop_calibration",
    "render_calibration_markdown",
]
