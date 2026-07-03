"""H-infinity phenotype sidecar aggregation for GRU robustness diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlrmp.analysis.pipelines.diagnostic_provenance import repo_relative, write_regeneration_spec
from rlrmp.io import update_marked_section, write_compact_json
from rlrmp.paths import REPO_ROOT


SCHEMA_VERSION = "rlrmp.hinf_phenotype_sidecar.v1"
ISSUE_ID = "abe33da"
DEFAULT_SCOPE = "validation_selected_gru_robustness_phenotype"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "results" / ISSUE_ID / "notes" / "hinf_phenotype_sidecar.json"
DEFAULT_OUTPUT_MARKDOWN = REPO_ROOT / "results" / ISSUE_ID / "notes" / "hinf_phenotype_sidecar.md"
DEFAULT_REGENERATION_SPEC = (
    REPO_ROOT / "results" / ISSUE_ID / "notes" / "hinf_phenotype_sidecar_regeneration_spec.json"
)

DEFAULT_SOURCE_NAMES = (
    "standard_certificate",
    "objective_comparator",
    "perturbation_response",
    "feedback_ablation",
    "map_error_decomposition",
    "evaluation_diagnostics",
    "induced_gain",
    "exact_audit",
    "worst_case_epsilon_audit",
    "broad_epsilon_attribution",
)

FORMAL_HINF_REQUIREMENTS = (
    "game_card",
    "gamma_or_budget",
    "disturbance_channel",
    "exact_audit_or_induced_gain",
)


def build_hinf_phenotype_sidecar(
    *,
    sources: Mapping[str, Mapping[str, Any] | None],
    issue: str = ISSUE_ID,
    scope: str = DEFAULT_SCOPE,
    paired_run_ids: Mapping[str, str] | None = None,
    generated_by: str = "rlrmp.analysis.pipelines.hinf_phenotype_sidecar",
) -> dict[str, Any]:
    """Aggregate existing diagnostic manifests into a robustness phenotype sidecar.

    The result is interpretive only. It preserves source status and provenance,
    reports missing components explicitly, and does not upgrade phenotype evidence
    into a formal H-infinity claim unless the formal requirements are present.
    """

    components = {
        name: _component_status(name, sources.get(name))
        for name in sorted(set(DEFAULT_SOURCE_NAMES) | set(sources))
    }
    row_ids = _collect_row_ids(sources)
    rows = [
        _build_row(
            run_id=run_id,
            sources=sources,
            components=components,
            paired_run_ids=paired_run_ids,
        )
        for run_id in row_ids
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": issue,
        "scope": scope,
        "generated_by": generated_by,
        "interpretation_contract": {
            "kind": "interpretive_sidecar",
            "not_standard_certificate": True,
            "not_checkpoint_selection_input": True,
            "formal_hinf_claim_policy": (
                "formal claims require an explicit game card, gamma or budget, "
                "disturbance channel, and exact audit or induced-gain metric"
            ),
            "formal_hinf_requirements": list(FORMAL_HINF_REQUIREMENTS),
        },
        "paired_run_ids": dict(paired_run_ids or {}),
        "components": components,
        "summary": _sidecar_summary(rows=rows, components=components),
        "rows": rows,
    }


def load_hinf_phenotype_sources(
    source_paths: Mapping[str, Path | str | None],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, dict[str, Any] | None]:
    """Load JSON manifests into source records with path provenance."""

    loaded: dict[str, dict[str, Any] | None] = {}
    for name, source_path in source_paths.items():
        if source_path is None:
            loaded[name] = None
            continue
        path = Path(source_path)
        if not path.exists():
            loaded[name] = {
                "status": "missing",
                "source_path": _repo_relative(path, repo_root=repo_root),
                "reason": "source path does not exist",
            }
            continue
        payload = _read_json(path)
        loaded[name] = {
            "status": "available",
            "source_path": _repo_relative(path, repo_root=repo_root),
            "payload": payload,
        }
    return loaded


def write_hinf_phenotype_sidecar(
    sidecar: Mapping[str, Any],
    *,
    json_path: Path = DEFAULT_OUTPUT_JSON,
    markdown_path: Path = DEFAULT_OUTPUT_MARKDOWN,
    regeneration_spec_path: Path | None = DEFAULT_REGENERATION_SPEC,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Write compact JSON and Markdown H-infinity phenotype sidecars."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(sidecar)
    if regeneration_spec_path is not None:
        payload["regeneration_spec_path"] = repo_relative(
            regeneration_spec_path,
            repo_root=repo_root,
        )
    write_compact_json(json_path, payload)
    update_marked_section(
        markdown_path,
        "hinf_phenotype_sidecar",
        render_hinf_phenotype_markdown(payload),
    )
    if regeneration_spec_path is not None:
        _write_hinf_regeneration_spec(
            sidecar=payload,
            spec_path=regeneration_spec_path,
            json_path=json_path,
            markdown_path=markdown_path,
            repo_root=repo_root,
        )


def render_hinf_phenotype_markdown(sidecar: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary for the phenotype sidecar."""

    lines = [
        "# H-infinity Phenotype Sidecar",
        "",
        (
            "Interpretive robustness phenotype report. This is not a standard "
            "certificate and is not a checkpoint-selection input."
        ),
        "",
        f"Regeneration spec: `{sidecar.get('regeneration_spec_path', 'not_materialized')}`",
        "",
        "## Component Status",
        "",
        "| Component | Status | Source |",
        "|---|---:|---|",
    ]
    for name, component in sidecar.get("components", {}).items():
        lines.append(
            "| "
            f"{name} | {component.get('status', 'unknown')} | "
            f"{component.get('source_path', '') or component.get('reason', '')} |"
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            (
                "| Run | Formal H-inf claim | Nominal efficiency | Feedback competence | "
                "Local feedback law | H-inf markers | Warnings |"
            ),
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for row in sidecar.get("rows", ()):
        lines.append(
            "| "
            f"{row.get('run_id')} | "
            f"{row.get('formal_hinf_claim', {}).get('status')} | "
            f"{row.get('nominal_efficiency', {}).get('status')} | "
            f"{row.get('feedback_competence', {}).get('status')} | "
            f"{row.get('local_feedback_law', {}).get('status')} | "
            f"{row.get('hinf_phenotype_markers', {}).get('status')} | "
            f"{len(row.get('warnings', ())) or 0} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Formal H-infinity claims remain separate from phenotype evidence.",
            "- Missing components are explicit; omitted evidence should not be inferred as pass.",
            (
                "- Paired baseline-vs-robust comparisons are reported only when "
                "matching pairs are present."
            ),
        ]
    )
    delayed_caveat = sidecar.get("delayed_contract_caveat")
    if isinstance(delayed_caveat, Mapping):
        lines.extend(
            [
                (
                    "- Delayed contract caveat: "
                    f"{delayed_caveat.get('blocker', 'see sidecar JSON')} "
                    f"Formal H-infinity equivalence is "
                    f"{delayed_caveat.get('formal_hinf_equivalence', 'not_claimed')}."
                )
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _build_row(
    *,
    run_id: str,
    sources: Mapping[str, Mapping[str, Any] | None],
    components: Mapping[str, Mapping[str, Any]],
    paired_run_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    standard = _source_payload(sources.get("standard_certificate"))
    objective = _source_payload(sources.get("objective_comparator"))
    perturbation = _source_payload(sources.get("perturbation_response"))
    feedback = _source_payload(sources.get("feedback_ablation"))
    map_error = _source_payload(sources.get("map_error_decomposition"))
    evaluation = _source_payload(sources.get("evaluation_diagnostics"))
    induced_gain = _source_payload(sources.get("induced_gain"))
    exact_audit = _source_payload(sources.get("exact_audit"))

    standard_row = _find_row_by_run_id(_rows_from_standard(standard), run_id)
    objective_row = _find_row_by_run_id(objective.get("rows", ()), run_id)
    perturbation_row = _run_record(perturbation, run_id)
    feedback_row = _run_record(feedback, run_id)
    map_row = _find_row_by_run_id(map_error.get("rows", ()), run_id)
    induced_row = _find_row_by_run_id(induced_gain.get("rows", ()), run_id)
    exact_row = _find_row_by_run_id(exact_audit.get("rows", ()), run_id)

    row = {
        "run_id": run_id,
        "component_statuses": {
            name: _row_component_status(name, component, run_id, sources.get(name))
            for name, component in components.items()
        },
        "nominal_efficiency": _nominal_efficiency(
            run_id=run_id,
            objective_row=objective_row,
            perturbation_row=perturbation_row,
            feedback_row=feedback_row,
            evaluation=evaluation,
            objective_source=sources.get("objective_comparator"),
            perturbation_source=sources.get("perturbation_response"),
            feedback_source=sources.get("feedback_ablation"),
            evaluation_source=sources.get("evaluation_diagnostics"),
        ),
        "feedback_competence": _feedback_competence(
            perturbation_row=perturbation_row,
            feedback_row=feedback_row,
            perturbation_source=sources.get("perturbation_response"),
            feedback_source=sources.get("feedback_ablation"),
        ),
        "local_feedback_law": _local_feedback_law(
            standard_row=standard_row,
            map_row=map_row,
            standard_source=sources.get("standard_certificate"),
            map_source=sources.get("map_error_decomposition"),
        ),
        "hinf_phenotype_markers": _hinf_markers(
            perturbation_row=perturbation_row,
            feedback_row=feedback_row,
            induced_row=induced_row,
            exact_row=exact_row,
            perturbation_source=sources.get("perturbation_response"),
            feedback_source=sources.get("feedback_ablation"),
            induced_source=sources.get("induced_gain"),
            exact_source=sources.get("exact_audit"),
        ),
    }
    row["paired_baseline_vs_robust"] = _paired_comparison(
        run_id=run_id,
        row=row,
        all_run_ids=_collect_row_ids(sources),
        paired_run_ids=paired_run_ids,
    )
    row["formal_hinf_claim"] = _formal_hinf_claim(row)
    row["warnings"] = _row_warnings(row)
    return row


def _nominal_efficiency(
    *,
    run_id: str,
    objective_row: Mapping[str, Any],
    perturbation_row: Mapping[str, Any],
    feedback_row: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    objective_source: Mapping[str, Any] | None,
    perturbation_source: Mapping[str, Any] | None,
    feedback_source: Mapping[str, Any] | None,
    evaluation_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    if objective_row:
        values.update(
            _copy_present(
                objective_row,
                (
                    "gru_mean_selected_validation_full_qrf",
                    "gru_mean_best_logged_validation_full_qrf",
                    "selected_to_extlqg_deterministic_ratio",
                    "selected_to_extlqg_total_ratio_not_apples_to_apples",
                    "n_replicates",
                ),
            )
        )
        shared = objective_row.get("shared_rollout_comparator")
        if isinstance(shared, Mapping):
            values["shared_rollout_total_ratio_to_extlqg"] = _nested_get(
                shared,
                ("gru_vs_extlqg", "terms", "total", "ratio_to_extlqg"),
            )
            values["shared_rollout_total_mean"] = _nested_get(
                shared,
                ("gru_cost", "total", "mean"),
            )
        evidence.append(_evidence("objective_comparator", objective_source, "available"))
    baseline = _nominal_baseline_metrics(perturbation_row) or _normal_ablation_metrics(feedback_row)
    if baseline:
        values["endpoint_error_m"] = _metric_mean(baseline, "endpoint_error_m")
        values["terminal_speed_m_s"] = _metric_mean(baseline, "terminal_speed_m_s")
        values["command_energy_full_qrf_control"] = _nested_get(
            baseline,
            ("extra_full_qrf_cost", "base_cost", "control", "mean"),
        )
        if values["command_energy_full_qrf_control"] is None:
            values["command_energy_full_qrf_control"] = _nested_get(
                baseline,
                ("rollout_full_qrf", "base_cost", "control", "mean"),
            )
        evidence.append(
            _evidence(
                "perturbation_response_or_feedback_ablation",
                perturbation_source if perturbation_row else feedback_source,
                "available",
            )
        )
    run_metrics = _run_evaluation_metrics(evaluation, run_id)
    if run_metrics:
        values.update(
            _copy_present(
                run_metrics,
                (
                    "peak_velocity_m_s",
                    "time_to_peak_steps",
                    "time_to_peak_s",
                    "endpoint_error_m",
                    "terminal_speed_m_s",
                ),
            )
        )
        evidence.append(_evidence("evaluation_diagnostics", evaluation_source, "available"))
    return _evidence_block(values, evidence=evidence, missing_reason="no nominal efficiency source")


def _feedback_competence(
    *,
    perturbation_row: Mapping[str, Any],
    feedback_row: Mapping[str, Any],
    perturbation_source: Mapping[str, Any] | None,
    feedback_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    summary = perturbation_row.get("robust_response_summary")
    if isinstance(summary, Mapping):
        values["perturbation_class_summary"] = _compact_class_summary(summary)
        evidence.append(_evidence("perturbation_response", perturbation_source, "available"))
    sisu_summary = _compact_sisu_perturbation_comparison(perturbation_row)
    if sisu_summary:
        values["sisu_1_vs_0_perturbation_class_summary"] = sisu_summary
        evidence.append(_evidence("perturbation_response", perturbation_source, "available"))
    if feedback_row:
        values["feedback_ablation_interpretation"] = feedback_row.get("interpretation")
        values["feedback_ablation_status_counts"] = feedback_row.get("status_counts")
        evidence.append(_evidence("feedback_ablation", feedback_source, "available"))
    return _evidence_block(values, evidence=evidence, missing_reason="no feedback evidence source")


def _local_feedback_law(
    *,
    standard_row: Mapping[str, Any],
    map_row: Mapping[str, Any],
    standard_source: Mapping[str, Any] | None,
    map_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    standard_component = _certificate_component(
        standard_row,
        "observation_history_to_action_map_mismatch",
    )
    standard_summary = standard_component.get("summary")
    if isinstance(standard_summary, Mapping):
        values["standard_observation_action_map"] = _copy_present(
            standard_summary,
            (
                "aggregate_mismatch_ratio",
                "covariance_weighted_aggregate_mismatch_ratio",
                "delta_frobenius",
                "reference_frobenius",
            ),
        )
        evidence.append(_evidence("standard_certificate", standard_source, "available"))
    decomposition = map_row.get("decomposition")
    if isinstance(decomposition, Mapping):
        summary = decomposition.get("summary")
        if isinstance(summary, Mapping):
            values["map_decomposition_summary"] = _copy_present(
                summary,
                (
                    "aggregate_delta_ratio",
                    "candidate_reference_norm_ratio",
                    "candidate_reference_cosine",
                    "best_scalar_gain",
                    "best_scalar_residual_ratio",
                ),
            )
        values["map_decomposition_annotations"] = decomposition.get("decision_rule_annotations")
        evidence.append(_evidence("map_error_decomposition", map_source, "available"))
    return _evidence_block(values, evidence=evidence, missing_reason="no local map evidence source")


def _hinf_markers(
    *,
    perturbation_row: Mapping[str, Any],
    feedback_row: Mapping[str, Any],
    induced_row: Mapping[str, Any],
    exact_row: Mapping[str, Any],
    perturbation_source: Mapping[str, Any] | None,
    feedback_source: Mapping[str, Any] | None,
    induced_source: Mapping[str, Any] | None,
    exact_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []
    summary = perturbation_row.get("robust_response_summary")
    if isinstance(summary, Mapping):
        values["delta_v_and_displacement_markers"] = _perturbation_marker_summary(summary)
        evidence.append(_evidence("perturbation_response", perturbation_source, "available"))
    sisu_markers = _sisu_perturbation_marker_summary(perturbation_row)
    if sisu_markers:
        values["sisu_1_vs_0_perturbation_markers"] = sisu_markers
        evidence.append(_evidence("perturbation_response", perturbation_source, "available"))
    if feedback_row:
        interp = feedback_row.get("interpretation")
        if isinstance(interp, Mapping):
            values["feedback_gain_magnitude_proxy"] = _copy_present(
                interp,
                (
                    "max_channel_delta_action_norm_mean",
                    "max_feedback_delta_action_norm_mean",
                    "label",
                ),
            )
            evidence.append(_evidence("feedback_ablation", feedback_source, "available"))
    if induced_row:
        values["induced_gain_or_exact_audit"] = induced_row
        evidence.append(_evidence("induced_gain", induced_source, "available"))
    if exact_row:
        values["exact_audit"] = exact_row
        evidence.append(_evidence("exact_audit", exact_source, "available"))
    return _evidence_block(
        values,
        evidence=evidence,
        missing_reason="no H-infinity phenotype marker source",
    )


def _formal_hinf_claim(row: Mapping[str, Any]) -> dict[str, Any]:
    markers = row.get("hinf_phenotype_markers", {})
    marker_values = markers.get("values", {}) if isinstance(markers, Mapping) else {}
    has_induced_or_exact = any(
        key in marker_values for key in ("induced_gain_or_exact_audit", "exact_audit")
    )
    available = {
        "game_card": False,
        "gamma_or_budget": _contains_key(row, ("gamma", "gamma_factor", "budget")),
        "disturbance_channel": _contains_key(row, ("disturbance_channel", "channel")),
        "exact_audit_or_induced_gain": has_induced_or_exact,
    }
    if all(available.values()):
        return {
            "status": "available",
            "label": "formal_hinf_evidence_present",
            "requirements": available,
            "claim": "formal evidence fields are present; inspect source audit before wording",
        }
    return {
        "status": "not_claimed",
        "label": "phenotype_evidence_only",
        "requirements": available,
        "missing": [name for name, is_available in available.items() if not is_available],
    }


def _paired_comparison(
    *,
    run_id: str,
    row: Mapping[str, Any],
    all_run_ids: Sequence[str],
    paired_run_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    explicit_pair = _explicit_paired_candidate(run_id, paired_run_ids or {})
    if explicit_pair is not None:
        baseline, robust = explicit_pair
        pairing_source = "explicit_paired_run_ids"
    else:
        baseline = _paired_candidate(run_id, all_run_ids, robust=False)
        robust = _paired_candidate(run_id, all_run_ids, robust=True)
        pairing_source = "inferred_run_id_tokens"
    if baseline is None or robust is None:
        return {
            "status": "not_available",
            "reason": "no inferable baseline-vs-robust run pair in loaded sources",
        }
    return {
        "status": "candidate_pair_available",
        "baseline_run_id": baseline,
        "robust_run_id": robust,
        "current_row_role": "robust" if run_id == robust else "baseline",
        "selection_role": "interpretive_pairing_only",
        "pairing_source": pairing_source,
        "current_row_evidence_statuses": {
            section: row.get(section, {}).get("status")
            for section in (
                "nominal_efficiency",
                "feedback_competence",
                "local_feedback_law",
                "hinf_phenotype_markers",
            )
        },
    }


def _row_warnings(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    warnings = []
    for section_name in (
        "nominal_efficiency",
        "feedback_competence",
        "local_feedback_law",
        "hinf_phenotype_markers",
    ):
        section = row.get(section_name, {})
        if isinstance(section, Mapping) and section.get("status") != "available":
            warnings.append(
                {
                    "code": f"{section_name}_unavailable",
                    "status": section.get("status"),
                    "reason": section.get("reason"),
                }
            )
    formal = row.get("formal_hinf_claim", {})
    if isinstance(formal, Mapping) and formal.get("status") != "available":
        warnings.append(
            {
                "code": "formal_hinf_not_claimed",
                "missing": formal.get("missing", []),
            }
        )
    n_replicates = _nested_get(row, ("nominal_efficiency", "values", "n_replicates"))
    if n_replicates is None:
        warnings.append(
            {
                "code": "replicate_spread_unavailable",
                "reason": "n_replicates was not found in loaded nominal-efficiency evidence",
            }
        )
    return warnings


def _sidecar_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    components: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "component_status_counts": _count_by(
            component.get("status", "unknown") for component in components.values()
        ),
        "formal_hinf_claim_status_counts": _count_by(
            row.get("formal_hinf_claim", {}).get("status", "unknown") for row in rows
        ),
        "row_evidence_status_counts": {
            section: _count_by(row.get(section, {}).get("status", "unknown") for row in rows)
            for section in (
                "nominal_efficiency",
                "feedback_competence",
                "local_feedback_law",
                "hinf_phenotype_markers",
            )
        },
    }


def _collect_row_ids(sources: Mapping[str, Mapping[str, Any] | None]) -> list[str]:
    run_ids: set[str] = set()
    for name, source in sources.items():
        payload = _source_payload(source)
        if not payload:
            continue
        if isinstance(payload.get("runs"), Mapping):
            run_ids.update(str(run_id) for run_id in payload["runs"])
        for row in payload.get("rows", ()) if isinstance(payload.get("rows"), Sequence) else ():
            run_id = _source_run_id(row)
            if run_id:
                run_ids.add(run_id)
        if name == "evaluation_diagnostics" and isinstance(payload.get("runs"), Sequence):
            run_ids.update(str(run_id) for run_id in payload["runs"])
    return sorted(run_ids)


def _component_status(
    name: str,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if source is None:
        return {"status": "missing", "reason": "source not provided"}
    source_status = source.get("status")
    if source_status == "missing":
        return {
            "status": "missing",
            "source_path": source.get("source_path"),
            "reason": source.get("reason", "source missing"),
        }
    payload = _source_payload(source)
    if not payload:
        return {
            "status": "missing",
            "source_path": source.get("source_path"),
            "reason": "empty or missing payload",
        }
    return {
        "status": "available",
        "source_path": source.get("source_path"),
        "payload_schema": payload.get("schema_version") or payload.get("format"),
        "payload_issue": payload.get("issue"),
        "payload_scope": payload.get("scope"),
        "component": name,
    }


def _row_component_status(
    name: str,
    component: Mapping[str, Any],
    run_id: str,
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if component.get("status") != "available":
        return dict(component)
    payload = _source_payload(source)
    if isinstance(payload.get("runs"), Mapping):
        run_record = payload["runs"].get(run_id)
        if isinstance(run_record, Mapping):
            return _evidence(name, source, run_record.get("status", "available"))
    if _find_row_by_run_id(payload.get("rows", ()), run_id):
        return _evidence(name, source, "available")
    if name == "evaluation_diagnostics" and run_id in payload.get("runs", ()):
        return _evidence(name, source, "available")
    return {
        "status": "missing",
        "source_path": component.get("source_path"),
        "reason": f"no row for {run_id}",
    }


def _evidence_block(
    values: Mapping[str, Any],
    *,
    evidence: Sequence[Mapping[str, Any]],
    missing_reason: str,
) -> dict[str, Any]:
    compact_values = {key: value for key, value in values.items() if value not in (None, {}, [])}
    if not compact_values:
        return {
            "status": "missing",
            "reason": missing_reason,
            "evidence": list(evidence),
        }
    return {"status": "available", "values": compact_values, "evidence": list(evidence)}


def _evidence(
    component: str,
    source: Mapping[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    return {
        "component": component,
        "status": status,
        "source_path": None if source is None else source.get("source_path"),
        "payload_schema": (
            None
            if source is None
            else (
                _source_payload(source).get("schema_version")
                or _source_payload(source).get("format")
            )
        ),
    }


def _source_payload(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    payload = source.get("payload", source)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _rows_from_standard(payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    rows = payload.get("rows", ())
    return rows if isinstance(rows, Sequence) and not isinstance(rows, str) else ()


def _run_record(payload: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    runs = payload.get("runs")
    if isinstance(runs, Mapping):
        record = runs.get(run_id)
        if isinstance(record, Mapping):
            return dict(record)
    return {}


def _find_row_by_run_id(rows: Any, run_id: str) -> dict[str, Any]:
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidates = {_source_run_id(row), row.get("run_id")}
        if run_id in candidates:
            return dict(row)
    return {}


def _source_run_id(row: Mapping[str, Any]) -> str | None:
    for key in ("source_run_id", "run_id"):
        value = row.get(key)
        if isinstance(value, str):
            return _strip_row_suffix(value)
    spec = row.get("spec")
    if isinstance(spec, Mapping):
        params = spec.get("parameters")
        if isinstance(params, Mapping) and isinstance(params.get("source_run_id"), str):
            return params["source_run_id"]
        if isinstance(spec.get("run_id"), str):
            return _strip_row_suffix(spec["run_id"])
    return None


def _strip_row_suffix(run_id: str) -> str:
    for suffix in ("__nominal_clean",):
        if run_id.endswith(suffix):
            return run_id[: -len(suffix)]
    return run_id


def _certificate_component(row: Mapping[str, Any], name: str) -> dict[str, Any]:
    components = row.get("certificate_components")
    if isinstance(components, Mapping) and isinstance(components.get(name), Mapping):
        return dict(components[name])
    if isinstance(row.get(name), Mapping):
        return dict(row[name])
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and isinstance(metrics.get(name), Mapping):
        return dict(metrics[name])
    return {}


def _nominal_baseline_metrics(run_record: Mapping[str, Any]) -> dict[str, Any]:
    for row in run_record.get("perturbations", ()) if isinstance(run_record, Mapping) else ():
        if not isinstance(row, Mapping):
            continue
        perturbation_id = row.get("perturbation_id")
        if perturbation_id in {None, "nominal", "clean"} or row.get("channel") == "nominal":
            metrics = row.get("metrics")
            if isinstance(metrics, Mapping):
                return dict(metrics)
    return {}


def _normal_ablation_metrics(run_record: Mapping[str, Any]) -> dict[str, Any]:
    for row in run_record.get("ablations", ()) if isinstance(run_record, Mapping) else ():
        if not isinstance(row, Mapping):
            continue
        if row.get("mode") == "normal" and row.get("bin") == "nominal":
            metrics = row.get("metrics")
            if isinstance(metrics, Mapping):
                return dict(metrics)
    return {}


def _run_evaluation_metrics(payload: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    runs = payload.get("runs")
    if isinstance(runs, Mapping):
        record = runs.get(run_id)
        if isinstance(record, Mapping):
            metrics = record.get("metrics", record)
            flattened = dict(metrics) if isinstance(metrics, Mapping) else {}
            flattened.update(_flatten_behavior_metrics(record.get("behavior")))
            return flattened
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        record = metrics.get(run_id)
        if isinstance(record, Mapping):
            return dict(record)
    return {}


def _flatten_behavior_metrics(behavior: Any) -> dict[str, Any]:
    """Flatten standard evaluation behavior stats into sidecar metric names."""

    if not isinstance(behavior, Mapping):
        return {}
    flattened = {
        "endpoint_error_m": _nested_get(behavior, ("endpoint_error_m", "mean")),
        "terminal_speed_m_s": _nested_get(behavior, ("terminal_speed_m_s", "mean")),
    }
    velocity = behavior.get("velocity_profile")
    if isinstance(velocity, Mapping):
        flattened.update(
            {
                "peak_velocity_m_s": (
                    _nested_get(velocity, ("peak_forward_velocity_m_s", "mean"))
                    or velocity.get("mean_profile_peak_forward_velocity_m_s")
                ),
                "time_to_peak_s": (
                    _nested_get(velocity, ("time_to_peak_forward_velocity_s", "mean"))
                    or velocity.get("mean_profile_time_to_peak_forward_velocity_s")
                ),
            }
        )
    return {key: value for key, value in flattened.items() if value is not None}


def _compact_class_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    class_summary = summary.get("class_summary")
    if not isinstance(class_summary, Mapping):
        return {}
    groups = class_summary.get("groups")
    if not isinstance(groups, Mapping):
        return {}
    compact = {}
    for group_id, group in groups.items():
        if not isinstance(group, Mapping):
            continue
        compact[str(group_id)] = {
            "status_counts": group.get("status_counts"),
            "n_rows": group.get("n_rows"),
            "delta_action_norm_mean": _nested_get(
                group,
                ("metrics", "delta_action_norm", "mean"),
            ),
            "delta_endpoint_error_m_mean": _nested_get(
                group,
                ("metrics", "delta_endpoint_error_m", "mean"),
            ),
            "delta_velocity_trajectory_norm_m_s_mean": _nested_get(
                group,
                ("metrics", "delta_velocity_trajectory_norm_m_s", "mean"),
            ),
            "gru_extlqg_delta_cost_ratio": group.get("gru_extlqg_delta_cost_ratio"),
            "denominator_warnings": group.get("denominator_warnings"),
        }
    return compact


def _compact_sisu_perturbation_comparison(run_record: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact class ratios from a SISU perturbation comparison run."""

    class_comparison = run_record.get("class_comparison")
    if not isinstance(class_comparison, Mapping):
        return {}
    compact = {}
    for group_id, group in class_comparison.items():
        if not isinstance(group, Mapping):
            continue
        metrics = group.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        compact[str(group_id)] = {
            "status_counts_sisu_0": group.get("status_counts_sisu_0"),
            "status_counts_sisu_1": group.get("status_counts_sisu_1"),
            "n_rows_sisu_0": group.get("rows_sisu_0"),
            "n_rows_sisu_1": group.get("rows_sisu_1"),
            "mean_delta_action_ratio_1_over_0": _nested_get(
                metrics,
                ("mean_delta_action", "ratio_1_over_0"),
            ),
            "max_delta_x_ratio_1_over_0": _nested_get(
                metrics,
                ("max_delta_x_m", "ratio_1_over_0"),
            ),
            "auc_delta_x_ratio_1_over_0": _nested_get(
                metrics,
                ("auc_delta_x_m_s", "ratio_1_over_0"),
            ),
            "full_qrf_delta_cost_ratio_1_over_0": _nested_get(
                metrics,
                ("mean_full_qrf_delta_cost", "ratio_1_over_0"),
            ),
            "full_qrf_delta_cost_delta_1_minus_0": _nested_get(
                metrics,
                ("mean_full_qrf_delta_cost", "delta_1_minus_0"),
            ),
            "notes": group.get("notes"),
        }
    return {
        group_id: {
            key: value
            for key, value in group.items()
            if value not in (None, {}, [])
        }
        for group_id, group in compact.items()
    }


def _perturbation_marker_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    compact = _compact_class_summary(summary)
    return {
        group_id: {
            key: value
            for key, value in group.items()
            if key
            in {
                "n_rows",
                "delta_action_norm_mean",
                "delta_endpoint_error_m_mean",
                "delta_velocity_trajectory_norm_m_s_mean",
                "gru_extlqg_delta_cost_ratio",
                "denominator_warnings",
            }
            and value not in (None, {}, [])
        }
        for group_id, group in compact.items()
    }


def _sisu_perturbation_marker_summary(run_record: Mapping[str, Any]) -> dict[str, Any]:
    """Return headline SISU perturbation markers for a sidecar row."""

    headline = run_record.get("headline")
    if not isinstance(headline, Mapping):
        return {}
    return {
        "full_qrf_delta_cost": headline.get("full_qrf_delta_cost"),
        "max_delta_x_m": headline.get("max_delta_x_m"),
        "mean_delta_action": headline.get("mean_delta_action"),
        "ratio_meaning": "SISU1/SISU0 ratio below 1 means the SISU=1 condition was smaller.",
    }


def _copy_present(source: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source and source[key] is not None}


def _metric_mean(metrics: Mapping[str, Any], metric_name: str) -> float | None:
    metric = metrics.get(metric_name)
    if isinstance(metric, Mapping):
        value = metric.get("mean")
        return float(value) if isinstance(value, int | float) else None
    return float(metric) if isinstance(metric, int | float) else None


def _nested_get(source: Mapping[str, Any], path: Sequence[str]) -> Any:
    cursor: Any = source
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _contains_key(source: Any, keys: Sequence[str]) -> bool:
    if isinstance(source, Mapping):
        for key, value in source.items():
            if key in keys and value not in (None, {}, []):
                return True
            if _contains_key(value, keys):
                return True
    elif isinstance(source, Sequence) and not isinstance(source, str):
        return any(_contains_key(item, keys) for item in source)
    return False


def _explicit_paired_candidate(
    run_id: str,
    paired_run_ids: Mapping[str, str],
) -> tuple[str, str] | None:
    for baseline, robust in paired_run_ids.items():
        if run_id == baseline or run_id == robust:
            return str(baseline), str(robust)
    return None


def _paired_candidate(run_id: str, all_run_ids: Sequence[str], *, robust: bool) -> str | None:
    robust_tokens = ("robust", "perturb", "adversary", "minimax", "hinf")
    baseline_tokens = ("baseline", "nominal", "standard")
    tokens = robust_tokens if robust else baseline_tokens
    matches = [
        candidate for candidate in all_run_ids if any(token in candidate for token in tokens)
    ]
    if run_id in matches:
        return run_id
    return matches[0] if matches else None


def _count_by(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_hinf_regeneration_spec(
    *,
    sidecar: Mapping[str, Any],
    spec_path: Path,
    json_path: Path,
    markdown_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    source_inputs = []
    source_args: list[str] = []
    for component_name, component in sidecar.get("components", {}).items():
        if not isinstance(component, Mapping):
            continue
        source_path = component.get("source_path")
        if source_path:
            source_inputs.append(
                {
                    "role": f"{component_name}_manifest",
                    "path": source_path,
                    "status": component.get("status"),
                }
            )
            source_args.extend([f"--{component_name.replace('_', '-')}", str(source_path)])
        else:
            source_inputs.append(
                {
                    "role": f"{component_name}_manifest",
                    "status": component.get("status"),
                    "reason": component.get("reason", "source path not provided"),
                }
            )
    paired_run_ids = sidecar.get("paired_run_ids", {})
    paired_args: list[str] = []
    if isinstance(paired_run_ids, Mapping):
        for baseline, robust in paired_run_ids.items():
            paired_args.extend(["--paired-run", f"{baseline}={robust}"])
    return write_regeneration_spec(
        spec_path=spec_path,
        diagnostic_name="hinf_phenotype_sidecar",
        materializer="rlrmp.analysis.pipelines.hinf_phenotype_sidecar.write_hinf_phenotype_sidecar",
        command=[
            "uv",
            "run",
            "python",
            "scripts/materialize_hinf_phenotype_sidecar.py",
            "--scope",
            str(sidecar.get("scope", DEFAULT_SCOPE)),
            *source_args,
            *paired_args,
            "--json-output",
            repo_relative(json_path, repo_root=repo_root),
            "--markdown-output",
            repo_relative(markdown_path, repo_root=repo_root),
            "--regeneration-spec-path",
            repo_relative(spec_path, repo_root=repo_root),
        ],
        parameters={
            "issue": sidecar.get("issue"),
            "scope": sidecar.get("scope"),
            "paired_run_ids": dict(paired_run_ids) if isinstance(paired_run_ids, Mapping) else {},
            "component_names": sorted(str(name) for name in sidecar.get("components", {})),
        },
        inputs=source_inputs,
        outputs=[
            {"role": "sidecar_json", "path": json_path},
            {"role": "sidecar_markdown", "path": markdown_path},
        ],
        source_files=[
            "src/rlrmp/analysis/pipelines/hinf_phenotype_sidecar.py",
            "scripts/materialize_hinf_phenotype_sidecar.py",
            "src/rlrmp/analysis/pipelines/diagnostic_provenance.py",
        ],
        notes=[
            "Interpretive phenotype sidecar only; not a formal H-infinity certificate.",
            "Spec records source manifests loaded for aggregation and the JSON/Markdown outputs.",
        ],
        repo_root=repo_root,
    )


__all__ = [
    "DEFAULT_OUTPUT_JSON",
    "DEFAULT_OUTPUT_MARKDOWN",
    "DEFAULT_REGENERATION_SPEC",
    "DEFAULT_SCOPE",
    "ISSUE_ID",
    "SCHEMA_VERSION",
    "build_hinf_phenotype_sidecar",
    "load_hinf_phenotype_sources",
    "render_hinf_phenotype_markdown",
    "write_hinf_phenotype_sidecar",
]
