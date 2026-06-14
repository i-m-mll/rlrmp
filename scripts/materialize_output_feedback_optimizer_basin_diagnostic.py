"""Materialize the output-feedback optimizer-basin diagnostic.

This is the bounded follow-up to the failed free time-varying output-feedback
bridge rows: screen AdamW from scratch, polish the best AdamW starts with
L-BFGS-B, test Bellman starts under the same optimizer family, and test
K_alpha interpolated starts with L-BFGS-B and AdamW-to-L-BFGS-B.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import materialize_output_feedback_failure_decomposition as failure
import materialize_output_feedback_sweep_certificates as certificates
from rlrmp.analysis.math.cs_game_card import (
    OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,
    materialize_reference,
)
from rlrmp.analysis.math.linear_round_trip import LinearTrainingConfig
from rlrmp.analysis.math.output_feedback import OutputFeedbackConfig
from rlrmp.analysis.pipelines.output_feedback_interpolated_starts import (
    DEFAULT_CONDITION as DEFAULT_ALPHA_LBFGS_CONDITION,
    INTERPOLATED_ALPHAS,
    run_interpolated_start_probe,
    result_summary as alpha_result_summary,
)
from rlrmp.analysis.pipelines.output_feedback_rollout_recovery import (
    adamw_optimizer_whitened,
    result_summary as rollout_result_summary,
    run_output_feedback_rollout_recovery,
)
from rlrmp.paths import REPO_ROOT, mkdir_p


ISSUE_ID = "1c014e5"
INTERPOLATED_START_ISSUE_ID = "7cea1b7"
SOURCE_ISSUE_ID = "7a459bb"
STANDARD_CERTIFICATE_ISSUE_ID = "d01c35a"
FAILURE_DECOMPOSITION_ISSUE_ID = "c45adde"
UMBRELLA_ID = "43e8728"
ADAMW_LRS: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
NOTE_PATH = (
    REPO_ROOT / "results" / ISSUE_ID / "notes" / "output_feedback_optimizer_basin.md"
)
MANIFEST_PATH = (
    REPO_ROOT
    / "results"
    / ISSUE_ID
    / "notes"
    / "output_feedback_optimizer_basin_manifest.json"
)
ARTIFACT_PATH = (
    REPO_ROOT
    / "_artifacts"
    / ISSUE_ID
    / "output_feedback_optimizer_basin"
    / "output_feedback_optimizer_basin.npz"
)
SOURCE_ROLLOUT_ARTIFACT = (
    REPO_ROOT
    / "_artifacts"
    / SOURCE_ISSUE_ID
    / "output_feedback_rollout_recovery"
    / "output_feedback_rollout_recovery.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adamw-steps", type=int, default=5000)
    parser.add_argument("--alpha-lbfgs-maxiter", type=int, default=2000)
    parser.add_argument("--alpha-adamw-steps", type=int, default=3000)
    parser.add_argument("--polish-maxiter", type=int, default=1000)
    parser.add_argument("--adamw-clip-norm", type=float, default=1e4)
    parser.add_argument("--note-output", type=Path, default=NOTE_PATH)
    parser.add_argument("--manifest-output", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--artifact-output", type=Path, default=ARTIFACT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, arrays = materialize(
        adamw_steps=args.adamw_steps,
        alpha_lbfgs_maxiter=args.alpha_lbfgs_maxiter,
        alpha_adamw_steps=args.alpha_adamw_steps,
        polish_maxiter=args.polish_maxiter,
        adamw_clip_norm=args.adamw_clip_norm,
        artifact_path=args.artifact_output,
        manifest_path=args.manifest_output,
    )
    write_result(
        result,
        arrays=arrays,
        note_path=args.note_output,
        manifest_path=args.manifest_output,
        artifact_path=args.artifact_output,
    )
    print(f"Wrote {args.note_output}")
    print(f"Wrote {args.manifest_output}")
    print(f"Wrote {args.artifact_output}")


def materialize(
    *,
    adamw_steps: int = 5000,
    alpha_lbfgs_maxiter: int = 2000,
    alpha_adamw_steps: int = 3000,
    polish_maxiter: int = 1000,
    adamw_clip_norm: float = 1e4,
    artifact_path: Path = ARTIFACT_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the optimizer-basin diagnostic and return manifest plus arrays."""

    start = time.perf_counter()
    reference = materialize_reference(gamma_factors=(OUTPUT_FEEDBACK_CERTIFICATE_GAMMA_FACTOR,))
    output_config = OutputFeedbackConfig()
    training_config = LinearTrainingConfig()

    scratch_conditions = tuple(
        adamw_optimizer_whitened(
            label=f"adamw_fixed_lr_{_lr_label(lr)}",
            learning_rate=lr,
            maxiter=adamw_steps,
            adam_clip_norm=adamw_clip_norm,
            initializations=("scratch",),
        )
        for lr in ADAMW_LRS
    )
    scratch_result = run_output_feedback_rollout_recovery(
        conditions=scratch_conditions,
        training_config=training_config,
        output_config=output_config,
    )
    scratch_summary = rollout_result_summary(scratch_result)
    best_scratch = _best_fits(scratch_summary["fits"], n=2)
    best_lrs = tuple(fit["condition"]["learning_rate"] for fit in best_scratch)
    best_lr = best_lrs[0]

    followup_conditions = []
    for lr in best_lrs:
        followup_conditions.append(
            adamw_optimizer_whitened(
                label=f"adamw_cosine_lr_{_lr_label(lr)}",
                learning_rate=lr,
                maxiter=adamw_steps,
                adam_schedule="warmup_cosine",
                adam_clip_norm=adamw_clip_norm,
                initializations=("scratch",),
            )
        )
        followup_conditions.append(
            adamw_optimizer_whitened(
                label=f"adamw_polish_lr_{_lr_label(lr)}",
                optimizer="adamw_then_lbfgsb",
                learning_rate=lr,
                maxiter=adamw_steps,
                polish_maxiter=polish_maxiter,
                adam_clip_norm=adamw_clip_norm,
                initializations=("scratch",),
            )
        )
    followup_conditions.extend(
        (
            adamw_optimizer_whitened(
                label=f"adamw_bellman_lr_{_lr_label(best_lr)}",
                learning_rate=best_lr,
                maxiter=adamw_steps,
                adam_clip_norm=adamw_clip_norm,
                initializations=("bellman_init",),
            ),
            adamw_optimizer_whitened(
                label=f"adamw_bellman_polish_lr_{_lr_label(best_lr)}",
                optimizer="adamw_then_lbfgsb",
                learning_rate=best_lr,
                maxiter=adamw_steps,
                polish_maxiter=polish_maxiter,
                adam_clip_norm=adamw_clip_norm,
                initializations=("bellman_init",),
            ),
        )
    )
    followup_result = run_output_feedback_rollout_recovery(
        conditions=tuple(followup_conditions),
        training_config=training_config,
        output_config=output_config,
    )
    followup_summary = rollout_result_summary(followup_result)

    alpha_lbfgs_condition = replace(DEFAULT_ALPHA_LBFGS_CONDITION, maxiter=alpha_lbfgs_maxiter)
    alpha_lbfgs_result = run_interpolated_start_probe(
        source_artifact=SOURCE_ROLLOUT_ARTIFACT,
        condition=alpha_lbfgs_condition,
        training_config=training_config,
        output_config=output_config,
        alphas=INTERPOLATED_ALPHAS,
    )
    alpha_lbfgs_summary = alpha_result_summary(alpha_lbfgs_result)
    alpha_adamw_condition = adamw_optimizer_whitened(
        label=f"k_alpha_adamw_polish_lr_{_lr_label(best_lr)}",
        optimizer="adamw_then_lbfgsb",
        learning_rate=best_lr,
        maxiter=alpha_adamw_steps,
        polish_maxiter=polish_maxiter,
        adam_clip_norm=adamw_clip_norm,
        initializations=DEFAULT_ALPHA_LBFGS_CONDITION.initializations,
    )
    alpha_adamw_result = run_interpolated_start_probe(
        source_artifact=SOURCE_ROLLOUT_ARTIFACT,
        condition=alpha_adamw_condition,
        training_config=training_config,
        output_config=output_config,
        alphas=INTERPOLATED_ALPHAS,
    )
    alpha_adamw_summary = alpha_result_summary(alpha_adamw_result)

    arrays = _merge_arrays(
        scratch_result.arrays,
        followup_result.arrays,
        alpha_lbfgs_result.arrays,
        alpha_adamw_result.arrays,
    )
    groups = (
        ("adamw_scratch", scratch_summary, scratch_result.arrays),
        ("adamw_followup", followup_summary, followup_result.arrays),
        ("k_alpha_lbfgs", alpha_lbfgs_summary, alpha_lbfgs_result.arrays),
        ("k_alpha_adamw_polish", alpha_adamw_summary, alpha_adamw_result.arrays),
    )
    standard_rows = []
    row_source_manifest = manifest_path if _is_under_repo(manifest_path) else MANIFEST_PATH
    for group, summary, group_arrays in groups:
        for fit in summary["fits"]:
            standard_rows.extend(
                _fit_standard_rows(
                    fit=fit,
                    arrays=group_arrays,
                    reference=reference,
                    output_config=output_config,
                    group=group,
                    manifest_path=row_source_manifest,
                )
            )
    standard_by_id = {row["spec"]["run_id"]: row for row in standard_rows}
    failure_rows = []
    for group, summary, group_arrays in groups:
        failure_rows.extend(
            failure._rollout_summary_rows(
                summary=summary,
                arrays=group_arrays,
                standard_by_id=standard_by_id,
                run_id_prefix=lambda fit, group=group: ("optimizer_basin", group, fit["label"]),
                array_prefix=lambda fit: fit["label"],
                source_group=group,
                row_parameters={"diagnostic_group": group},
            )
        )

    status_counts = Counter(row["status"] for row in standard_rows)
    return (
        {
            "format": "rlrmp.output_feedback_optimizer_basin_diagnostic.v1",
            "issue": ISSUE_ID,
            "interpolated_start_issue": INTERPOLATED_START_ISSUE_ID,
            "source_issue": SOURCE_ISSUE_ID,
            "standard_certificate_issue": STANDARD_CERTIFICATE_ISSUE_ID,
            "failure_decomposition_issue": FAILURE_DECOMPOSITION_ISSUE_ID,
            "umbrella": UMBRELLA_ID,
            "runtime_seconds": time.perf_counter() - start,
            "artifact_npz": _repo_relative(artifact_path),
            "grid": {
                "adamw_lrs": list(ADAMW_LRS),
                "best_lrs_selected_for_followup": list(best_lrs),
                "adamw_steps": adamw_steps,
                "alpha_lbfgs_maxiter": alpha_lbfgs_maxiter,
                "alpha_adamw_steps": alpha_adamw_steps,
                "polish_maxiter": polish_maxiter,
                "adamw_clip_norm": adamw_clip_norm,
                "interpolated_alphas": list(INTERPOLATED_ALPHAS),
            },
            "summaries": {
                "adamw_scratch": scratch_summary,
                "adamw_followup": followup_summary,
                "k_alpha_lbfgs": alpha_lbfgs_summary,
                "k_alpha_adamw_polish": alpha_adamw_summary,
            },
            "standard_certificate": {
                "status_counts": dict(sorted(status_counts.items())),
                "rows": standard_rows,
            },
            "failure_decomposition": {
                "classification_counts": dict(
                    sorted(
                        Counter(
                            row["classification"]["classification"] for row in failure_rows
                        ).items()
                    )
                ),
                "rows": failure_rows,
            },
            "verdict": _verdict(
                scratch_summary=scratch_summary,
                followup_summary=followup_summary,
                alpha_lbfgs_summary=alpha_lbfgs_summary,
                alpha_adamw_summary=alpha_adamw_summary,
            ),
        },
        arrays,
    )


def write_result(
    result: dict[str, Any],
    *,
    arrays: dict[str, np.ndarray],
    note_path: Path = NOTE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    artifact_path: Path = ARTIFACT_PATH,
) -> None:
    mkdir_p(note_path.parent)
    mkdir_p(artifact_path.parent)
    if _is_under_repo(note_path):
        results_dir = mkdir_p(REPO_ROOT / "results" / ISSUE_ID)
        readme = results_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "Optimizer-family and basin-access diagnostics for the Phase 3 "
                "output-feedback bridge. See `notes/output_feedback_optimizer_basin.md`.\n",
                encoding="utf-8",
            )
    result = dict(result)
    result["tracked_note"] = _repo_relative(note_path)
    result["tracked_manifest"] = _repo_relative(manifest_path)
    result["artifact_npz"] = _repo_relative(artifact_path)
    result["artifact_npz_keys"] = sorted(arrays)
    np.savez_compressed(artifact_path, **arrays)
    note_path.write_text(render_markdown(result), encoding="utf-8")
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown(result: dict[str, Any]) -> str:
    return f"""# Output-Feedback Optimizer-Basin Diagnostic

Issue: `{result["issue"]}`. Interpolated-start issue:
`{result["interpolated_start_issue"]}`. Source issue: `{result["source_issue"]}`.
Umbrella: `{result["umbrella"]}`.

This diagnostic asks whether the failed free time-varying output-feedback row is
mainly an optimizer-family or basin-access failure. It does not change the task,
the reference controller, or the bridge gate. It screens full-batch AdamW,
polishes the best AdamW starts with L-BFGS-B, repeats the same optimizer-family
check from the Bellman start, and tests K_alpha starts between the failed
scratch controller and the analytical LQR controller.

Runtime: `{result["runtime_seconds"]:.2f}` seconds.

Grid: `{json.dumps(result["grid"], sort_keys=True)}`.

## Verdict

{result["verdict"]}

## Best Rows

{_best_rows_markdown(result)}

## Failure Labels

`{result["failure_decomposition"]["classification_counts"]}`
"""


def _fit_standard_rows(
    *,
    fit: dict[str, Any],
    arrays: dict[str, np.ndarray],
    reference: Any,
    output_config: OutputFeedbackConfig,
    group: str,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    rows = certificates._deterministic_fit_rows(
        fit=fit,
        arrays=arrays,
        reference=reference,
        output_config=output_config,
        family=f"optimizer-basin {group}",
        run_parts=("optimizer_basin", group, fit["label"]),
        training_distribution=f"optimizer_basin_{group}",
        source_manifest=manifest_path,
        extra_parameters={"diagnostic_group": group},
        notes=(
            "Full standard certificate computed for the optimizer-basin diagnostic "
            f"group {group}."
        ),
    )
    row_dicts = []
    for row in rows:
        row_dict = row.to_json_dict()
        row_dict["spec"]["issue_id"] = ISSUE_ID
        row_dict["spec"]["optimizer_label"] = _optimizer_label(fit)
        row_dicts.append(row_dict)
    return row_dicts


def _best_fits(fits: list[dict[str, Any]], *, n: int) -> tuple[dict[str, Any], ...]:
    return tuple(sorted(fits, key=lambda fit: fit["objective_ratio_to_reference"])[:n])


def _optimizer_label(fit: dict[str, Any]) -> str:
    condition = fit.get("condition", {})
    optimizer = condition.get("optimizer", "unknown")
    if optimizer.startswith("adamw"):
        return (
            f"{optimizer}_lr_{condition.get('learning_rate')}_"
            f"{condition.get('adam_schedule', 'fixed')}"
        )
    return "lbfgsb_strong_optimizer_whitened"


def _merge_arrays(*groups: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for group in groups:
        for key, value in group.items():
            if key in arrays and not np.array_equal(arrays[key], value):
                raise ValueError(f"Conflicting array key {key!r} across optimizer-basin groups.")
            arrays[key] = value
    return arrays


def _best_rows_markdown(result: dict[str, Any]) -> str:
    rows = [
        "| group | label | objective ratio | clean mismatch | exact L2 ratio | lambda/gamma^2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group, summary in result["summaries"].items():
        best = _best_fits(summary["fits"], n=1)[0]
        rows.append(
            "| "
            f"{group} | "
            f"{best['label']} | "
            f"{best['objective_ratio_to_reference']:.8g} | "
            f"{best['clean_action_mismatch_ratio']:.8g} | "
            f"{best['exact_l2_cost_ratio_to_lqr']:.8g} | "
            f"{best['gamma_penalized_lambda_over_gamma_squared']:.8g} |"
        )
    return "\n".join(rows)


def _verdict(
    *,
    scratch_summary: dict[str, Any],
    followup_summary: dict[str, Any],
    alpha_lbfgs_summary: dict[str, Any],
    alpha_adamw_summary: dict[str, Any],
) -> str:
    all_fits = (
        scratch_summary["fits"]
        + followup_summary["fits"]
        + alpha_lbfgs_summary["fits"]
        + alpha_adamw_summary["fits"]
    )
    best = _best_fits(all_fits, n=1)[0]
    if _passes_practical_bridge(best):
        return (
            f"At least one bounded optimizer-basin row reaches the practical bridge "
            f"target: `{best['label']}`."
        )
    alpha_best = _best_fits(alpha_lbfgs_summary["fits"] + alpha_adamw_summary["fits"], n=1)[0]
    if alpha_best["objective_ratio_to_reference"] < best["objective_ratio_to_reference"] * 1.01:
        return (
            "No bounded row reaches the practical bridge target. Interpolated starts "
            "are competitive with the best AdamW row, so the failure remains a basin "
            "and/or objective-conditioning problem rather than a simple missing "
            "optimizer-family fix."
        )
    return (
        "No bounded row reaches the practical bridge target. The best row is "
        f"`{best['label']}`, so the bridge should not advance on this evidence alone."
    )


def _passes_practical_bridge(fit: dict[str, Any]) -> bool:
    return (
        fit["objective_ratio_to_reference"] <= 1.001
        and fit["clean_action_mismatch_ratio"] <= 1e-3
        and fit["exact_l2_cost_ratio_to_lqr"] <= 1.01
        and fit["gamma_penalized_lambda_over_gamma_squared"] <= 1.63
    )


def _lr_label(lr: float) -> str:
    return f"{lr:g}".replace("-", "m").replace(".", "p")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _is_under_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
