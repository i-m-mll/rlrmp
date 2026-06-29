"""Materialize the 27dece3 soft-lambda scale summary and no-launch draft."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIRECT = REPO_ROOT / "results" / "093d949" / "soft_lambda_sweep.json"
SOURCE_CLOSED_LOOP = REPO_ROOT / "results" / "3b850d6" / "closed_loop_policy_audit.json"
SOURCE_CRITICAL = REPO_ROOT / "results" / "1697bdc" / "critical_lambda_search.json"
SOURCE_ADAM = REPO_ROOT / "results" / "f3c5db9" / "frozen_adam_audit_tuning.json"
OUT_DIR = REPO_ROOT / "results" / "54389a4"
OUT_JSON = OUT_DIR / "scale_sanity_summary.json"
OUT_MD = OUT_DIR / "NO_LAUNCH_SPEC_LOCK.md"


ROW_LABELS = {
    "open_loop_small": "c92 small",
    "open_loop_moderate": "c92 moderate",
    "open_loop_stress": "c92 stress",
}

MECHANISM_LABELS = {
    "direct_epsilon": "direct epsilon",
    "linear_no_bias": "closed-loop linear no-bias",
    "affine": "closed-loop affine",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "pending"
    number = float(value)
    if abs(number) >= 1e5 or (number and abs(number) < 1e-3):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def fmt_mult(value: Any) -> str:
    if value is None:
        return "pending"
    return f"{float(value):.3g}x"


def pct(value: Any) -> str:
    if value is None:
        return "pending"
    return f"{100.0 * float(value):.1f}%"


def critical_summary_by_key(critical: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = {}
    for row in critical["summary"]:
        key = (row["run_id"], row["mechanism"], row["optimizer"])
        rows[key] = row
    return rows


def direct_centers(direct: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["run_id"]: {
            "center_lambda": row["sweep_center"]["lambda"],
            "center_source": row["sweep_center"]["source"],
            "transition": row["transition"],
        }
        for row in direct["rows"]
    }


def direct_sweep_row(direct: dict[str, Any], run_id: str, multiplier: float) -> dict[str, Any]:
    source_row = next(row for row in direct["rows"] if row["run_id"] == run_id)
    return min(source_row["sweep"], key=lambda row: abs(float(row["multiplier"]) - multiplier))


def adam_summary_by_key(adam: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["run_id"], row["mechanism"]): row for row in adam["summary"]}


def best_known_direction(closed_loop: dict[str, Any], run_id: str, direction: str) -> dict[str, Any]:
    source_row = next(row for row in closed_loop["rows"] if row["run_id"] == run_id)
    if direction == "linear_ridge_direct":
        return source_row["interpretation"]["best_linear"]
    if direction == "affine_mean_direct":
        return source_row["interpretation"]["best_affine"]
    raise KeyError(direction)


def interpretation(mechanism: str, optimizer: str, row: dict[str, Any]) -> str:
    if mechanism == "direct_epsilon":
        return (
            "coordinate perturbation becomes useful and interior near this multiplier; "
            "this is the direct frozen-audit scale reference"
        )
    if optimizer == "line_search_known_direction":
        return (
            "known-direction expressivity check only; useful for sanity, not a training "
            "inner-optimizer recommendation"
        )
    if mechanism == "linear_no_bias":
        return (
            "full closed-loop no-bias policy has useful interior points in the same "
            "lambda region; f3c5db9 found matching zero-start Adam settings"
        )
    return (
        "affine has useful interior frozen-audit points; f3c5db9 found matching "
        "zero-start Adam settings"
    )


def build_scale_rows(
    direct: dict[str, Any],
    closed_loop: dict[str, Any],
    critical: dict[str, Any],
) -> list[dict[str, Any]]:
    critical_rows = critical_summary_by_key(critical)
    rows: list[dict[str, Any]] = []
    for run_id in ROW_LABELS:
        direct_row = critical_rows[(run_id, "direct_epsilon", "pgd_projected_epsilon")]
        rows.append(
            {
                "row": run_id,
                "row_label": ROW_LABELS[run_id],
                "mechanism": "direct_epsilon",
                "mechanism_label": MECHANISM_LABELS["direct_epsilon"],
                "optimizer": "pgd_projected_epsilon",
                "selected_multiplier": direct_row["lowest_valid_lambda_multiplier"],
                "lambda": direct_row["lowest_valid_lambda"],
                "gain": direct_row["gain_at_lowest_valid"],
                "max_norm_over_cap": direct_row["max_norm_over_cap_at_lowest_valid"],
                "cap_bound_fraction": direct_row["cap_bound_fraction_at_lowest_valid"],
                "optimizer_status": direct_row["optimizer_reliability"],
                "scale_interpretation": interpretation("direct_epsilon", "pgd_projected_epsilon", direct_row),
            }
        )
        for mechanism in ("linear_no_bias", "affine"):
            for optimizer in ("lbfgsb", "adam"):
                row = critical_rows[(run_id, mechanism, optimizer)]
                rows.append(
                    {
                        "row": run_id,
                        "row_label": ROW_LABELS[run_id],
                        "mechanism": mechanism,
                        "mechanism_label": MECHANISM_LABELS[mechanism],
                        "optimizer": optimizer,
                        "selected_multiplier": row["lowest_valid_lambda_multiplier"],
                        "lambda": row["lowest_valid_lambda"],
                        "gain": row["gain_at_lowest_valid"],
                        "max_norm_over_cap": row["max_norm_over_cap_at_lowest_valid"],
                        "cap_bound_fraction": row["cap_bound_fraction_at_lowest_valid"],
                        "optimizer_status": row["optimizer_reliability"],
                        "failure_mode": row["failure_mode"],
                        "scale_interpretation": interpretation(mechanism, optimizer, row),
                    }
                )
        known_linear = best_known_direction(closed_loop, run_id, "linear_ridge_direct")
        known_affine = best_known_direction(closed_loop, run_id, "affine_mean_direct")
        for direction, row in (
            ("linear_ridge_direct", known_linear),
            ("affine_mean_direct", known_affine),
        ):
            mechanism = "linear_no_bias" if direction == "linear_ridge_direct" else "affine"
            rows.append(
                {
                    "row": run_id,
                    "row_label": ROW_LABELS[run_id],
                    "mechanism": mechanism,
                    "mechanism_label": f"{MECHANISM_LABELS[mechanism]} known direction",
                    "optimizer": "line_search_known_direction",
                    "selected_multiplier": row["lambda_multiplier"],
                    "lambda": row["lambda"],
                    "gain": row["selected_objective_gain_over_zero"],
                    "max_norm_over_cap": row["raw_norm_cap_ratio_max"],
                    "cap_bound_fraction": row["cap_violation_fraction_before_projection"],
                    "optimizer_status": "reference_direction_only",
                    "scale_interpretation": interpretation(mechanism, "line_search_known_direction", row),
                }
            )
    return rows


def build_adam_rows(adam: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in adam["summary"]:
        mechanism = source["mechanism"]
        rows.append(
            {
                "row": source["run_id"],
                "row_label": ROW_LABELS[source["run_id"]],
                "mechanism": mechanism,
                "mechanism_label": MECHANISM_LABELS[mechanism],
                "reference_optimizer": source["reference_optimizer"],
                "reference_lambda_multiplier": source["reference_lambda_multiplier"],
                "reference_lambda_range": source["reference_lambda_range"],
                "best_adam_steps": source["best_adam_steps"],
                "best_adam_learning_rate": source["best_adam_learning_rate"],
                "best_initialization": source["best_initialization"],
                "gain": source["best_gain"],
                "max_norm_over_cap": source["best_max_norm_over_cap"],
                "cap_bound_fraction": source["best_cap_bound_fraction"],
                "optimizer_status": source["best_optimizer_status"],
                "match": source["match"],
                "recommendation": source["recommendation"],
            }
        )
    return rows


def build_training_candidates(
    scale_rows: list[dict[str, Any]],
    adam: dict[str, Any],
) -> list[dict[str, Any]]:
    def pick(row: str, mechanism: str, optimizer: str) -> dict[str, Any]:
        return next(
            item
            for item in scale_rows
            if item["row"] == row and item["mechanism"] == mechanism and item["optimizer"] == optimizer
        )

    common = adam["common_stage1_matching_settings"][0]
    adam_rows = adam_summary_by_key(adam)
    candidates = []
    for mechanism, optimizer, status in (
        ("direct_epsilon", "pgd_projected_epsilon", "candidate_no_launch_adam_smoke"),
        ("linear_no_bias", "lbfgsb", "candidate_no_launch_adam_smoke"),
        ("affine", "lbfgsb", "candidate_no_launch_adam_smoke"),
    ):
        multipliers = {
            run_id: pick(run_id, mechanism, optimizer)["selected_multiplier"] for run_id in ROW_LABELS
        }
        per_row_matches = {
            run_id: bool(adam_rows[(run_id, mechanism)]["match"]) for run_id in ROW_LABELS
        }
        candidates.append(
            {
                "row_id": f"{mechanism}_calibrated",
                "mechanism": mechanism,
                "reference_optimizer": optimizer,
                "training_inner_optimizer": "zero_start_adam",
                "adam_steps": int(common["adam_steps"]),
                "adam_learning_rate": float(common["adam_learning_rate"]),
                "lambda_multiplier_by_c92_row": multipliers,
                "adam_match_by_c92_row": per_row_matches,
                "status": status,
                "notes": (
                    "No launch approved. f3c5db9 supports this as a training-facing smoke-test "
                    "candidate with the conservative common Adam setting."
                ),
            }
        )
    candidates.append(
        {
            "row_id": "fallback_benchmark_if_training_unreliable",
            "mechanism": "affine",
            "reference_optimizer": "lbfgsb",
            "training_inner_optimizer": "adam_vs_optax_lbfgs_benchmark",
            "adam_steps": int(common["adam_steps"]),
            "adam_learning_rate": float(common["adam_learning_rate"]),
            "lambda_multiplier_by_c92_row": {
                run_id: pick(run_id, "affine", "lbfgsb")["selected_multiplier"] for run_id in ROW_LABELS
            },
            "adam_match_by_c92_row": {
                run_id: bool(adam_rows[(run_id, "affine")]["match"]) for run_id in ROW_LABELS
            },
            "status": "future_path_only",
            "notes": (
                "Use future benchmark issue 2e60620 only if Adam fails during training "
                "or affine remains unreliable despite the frozen-audit match."
            ),
        }
    )
    return candidates


def render_scale_table(rows: list[dict[str, Any]], *, optimizers: set[str]) -> str:
    lines = [
        "| row | mechanism | optimizer/status | selected lambda multiplier | gain | norm/cap behavior | scale interpretation |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        if row["optimizer"] not in optimizers:
            continue
        cap = row.get("cap_bound_fraction")
        norm = row.get("max_norm_over_cap")
        if norm is None:
            norm_text = "no useful interior point"
        else:
            norm_text = f"max {fmt_num(norm)}x cap; cap-bound {pct(cap)}"
        lines.append(
            "| "
            f"`{row['row_label']}` | {row['mechanism_label']} | "
            f"`{row['optimizer']}` / {row['optimizer_status']} | "
            f"{fmt_mult(row['selected_multiplier'])} | {fmt_num(row['gain'])} | "
            f"{norm_text} | {row['scale_interpretation']} |"
        )
    return "\n".join(lines)


def render_adam_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| row | mechanism | lambda range | Adam setting | finite/useful/interior | norm/cap | gain | recommendation |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        setting = (
            f"steps={row['best_adam_steps']}; lr={row['best_adam_learning_rate']:.1e}; "
            f"init={row['best_initialization']}"
        )
        validity = "match" if row["match"] else "no match"
        lines.append(
            "| "
            f"`{row['row_label']}` | {row['mechanism_label']} | "
            f"{row['reference_lambda_range']} | `{setting}` | {validity} | "
            f"{fmt_num(row['max_norm_over_cap'])}x cap; cap-bound {pct(row['cap_bound_fraction'])} | "
            f"{fmt_num(row['gain'])} | {row['recommendation']} |"
        )
    return "\n".join(lines)


def render_training_table(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "| proposed row | mechanism | training inner optimizer | multipliers by c92 row | Adam setting | status |",
        "|---|---|---|---|---|---|",
    ]
    for row in candidates:
        mult = ", ".join(
            f"{ROW_LABELS[key]}={fmt_mult(value)}"
            for key, value in row["lambda_multiplier_by_c92_row"].items()
        )
        adam_setting = f"steps={row['adam_steps']}; lr={row['adam_learning_rate']:.1e}"
        lines.append(
            f"| `{row['row_id']}` | {MECHANISM_LABELS.get(row['mechanism'], row['mechanism'])} | "
            f"`{row['training_inner_optimizer']}` | {mult} | `{adam_setting}` | "
            f"{row['status']} |"
        )
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    contract = summary["scale_contract"]
    return "\n".join(
        [
            "# Soft Lambda Scale Sanity and No-Launch Spec Lock",
            "",
            "Issue: `27dece3`. Parent umbrella: `54389a4`.",
            "",
            "> No training launch is approved by this artifact. This is a durable no-launch",
            "> spec lock for parent/user review. It proposes training-facing rows and",
            "> optimizer settings, but a billable run still requires an explicit later",
            "> user approval.",
            "",
            "## Scale Contract",
            "",
            f"- Source frozen runs: `{summary['source_experiment']}` c92 open-loop no-PGD rows.",
            "- Controller weights were frozen; these are audits, not training runs.",
            f"- Epsilon is a {contract['epsilon_dim']}D process input with metadata "
            "`B_w[:epsilon_dim, :] = I` and the remaining state rows zero.",
            f"- Safety cap radius: `{fmt_num(contract['safety_cap_l2_radius_15cm'], 6)}` "
            f"coordinate-L2 units from `{contract['safety_cap_source']}`.",
            f"- This cap is `{fmt_num(contract['cap_fraction_of_15cm'] * 100.0, 3)}%` of a "
            "15 cm reference length if read as a coordinate displacement scale.",
            "- The code path does not provide a Newton or muscle-force conversion for this",
            "  epsilon. Treat the values below as process-coordinate scale evidence, not an",
            "  exact physical force calibration.",
            "- Objective: `mean_i[J_i(epsilon_i) - lambda * E_i(epsilon_i)]`, where energy",
            "  is squared L2 over the epsilon sequence.",
            "",
            "## Practical Lambda Sanity",
            "",
            "Lowest valid means the smallest tested multiplier that was finite, useful,",
            "and interior (`cap_bound_fraction = 0` and `max_norm_over_cap <= 0.99`).",
            "",
            render_scale_table(summary["scale_rows"], optimizers={"pgd_projected_epsilon", "lbfgsb"}),
            "",
            "## Current Adam Status",
            "",
            "`f3c5db9` resolves the previous optimizer gate: Stage 1 zero-start Adam",
            "matches all direct, linear, and affine reference regions. The conservative",
            "common setting for training-facing smoke tests is `steps=12`, `lr=1e-5`.",
            "",
            render_adam_table(summary["adam_rows"]),
            "",
            "## Known-Direction Checks",
            "",
            "Known-direction rows are expressivity sanity checks from `3b850d6`; they show",
            "that useful finite closed-loop directions exist, but they are not a full",
            "inner-optimizer choice for training.",
            "",
            render_scale_table(summary["scale_rows"], optimizers={"line_search_known_direction"}),
            "",
            "## Proposed Training-Facing Rows",
            "",
            "These rows are proposed for the approval discussion only. They must not be",
            "launched without an explicit user-approved run spec.",
            "",
            render_training_table(summary["training_candidates"]),
            "",
            "If Adam fails during training or the affine row remains unreliable after",
            "the frozen-audit match, future benchmark issue `2e60620` is the planned place for the",
            "Adam-vs-Optax-L-BFGS comparison. This artifact only references that issue;",
            "it does not add a comment there.",
            "",
            "## Dependency Status",
            "",
            f"- `results/f3c5db9/` is present: `{summary['dependency']['results_f3c5db9_present_on_branch']}`.",
            "- The sibling Adam closeout has been consumed through the tracked",
            "  `results/f3c5db9/frozen_adam_audit_tuning.json` artifact.",
            "- This spec is finalized as a no-launch approval packet; it is not launch",
            "  authorization.",
            "",
        ]
    )


def main() -> None:
    direct = load_json(SOURCE_DIRECT)
    closed_loop = load_json(SOURCE_CLOSED_LOOP)
    critical = load_json(SOURCE_CRITICAL)
    adam = load_json(SOURCE_ADAM)
    scale_rows = build_scale_rows(direct, closed_loop, critical)
    adam_rows = build_adam_rows(adam)
    cap_radius = float(direct["frozen_contract"]["safety_cap_l2_radius_15cm"])
    summary = {
        "schema_version": "rlrmp.scale_sanity_spec_lock.v1",
        "issue": "27dece3",
        "parent_umbrella": "54389a4",
        "source_experiment": "c92ebd8",
        "source_results": {
            "direct_lambda": "results/093d949/soft_lambda_sweep.json",
            "closed_loop_policy_audit": "results/3b850d6/closed_loop_policy_audit.json",
            "critical_lambda_search": "results/1697bdc/critical_lambda_search.json",
            "frozen_adam_tuning": "results/f3c5db9/frozen_adam_audit_tuning.json",
        },
        "dependency": {
            "f3c5db9_required_before_final_commit": False,
            "results_f3c5db9_present_on_branch": (REPO_ROOT / "results" / "f3c5db9").exists(),
            "f3c5db9_adam_commit": "76f7deb",
            "integration_commit": "773ae42",
        },
        "materialized_at_dependency_state": "f3c5db9_consumed_from_results",
        "scale_contract": {
            **direct["frozen_contract"],
            "cap_fraction_of_15cm": cap_radius / 0.15,
            "exact_physical_force_conversion": None,
            "force_conversion_note": (
                "The available code maps broad epsilon into process state coordinates via B_w; "
                "it does not expose a direct Newton-scale conversion."
            ),
        },
        "direct_centers": direct_centers(direct),
        "direct_4x_sweep_rows": {
            run_id: direct_sweep_row(direct, run_id, 4.0) for run_id in ROW_LABELS
        },
        "scale_rows": scale_rows,
        "adam_recommendation": adam["recommendation"],
        "adam_common_stage1_matching_settings": adam["common_stage1_matching_settings"],
        "adam_rows": adam_rows,
        "training_candidates": build_training_candidates(scale_rows, adam),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
