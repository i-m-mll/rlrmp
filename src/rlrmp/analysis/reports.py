"""Feedbax report-stage recipes for rlrmp analysis products."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from feedbax.analysis.reports import (
    REPORT_RENDER_ROLE,
    ReportRecipeResult,
    ResolvedReportInput,
    register_report_recipe,
)
from feedbax.contracts.manifest import (
    ArtifactRef,
    ReportSpec,
    store_bytes_artifact,
    store_json_artifact,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlrmp.mappings import as_mapping as _mapping
from rlrmp.runtime.params_models import params_model_for, register_params_model
from rlrmp.runtime.spec_migrations import (
    BRIDGE_CERTIFICATE_REPORT_PARAMS_KIND,
    FEEDBACK_QUALITY_LENS_REPORT_PARAMS_KIND,
    GRU_POSTRUN_REPORT_PARAMS_KIND,
    ROBUSTNESS_PHENOTYPE_REPORT_PARAMS_KIND,
    accept_rlrmp_spec_payload,
    stamp_current_schema,
)


GRU_POSTRUN_REPORT_TYPE = "rlrmp.report.gru_postrun_summary"
BRIDGE_CERTIFICATE_REPORT_TYPE = "rlrmp.report.bridge_certificate_notes"
FEEDBACK_QUALITY_LENS_REPORT_TYPE = "rlrmp.report.feedback_quality_lens_summary"
ROBUSTNESS_PHENOTYPE_REPORT_TYPE = "rlrmp.report.robustness_phenotype_markdown"

GRU_POSTRUN_REPORT_RENDER_ROLE = "rlrmp-gru-postrun-report-render"
BRIDGE_CERTIFICATE_REPORT_RENDER_ROLE = "rlrmp-bridge-certificate-report-render"
FEEDBACK_QUALITY_LENS_REPORT_RENDER_ROLE = "rlrmp-feedback-quality-lens-report-render"
ROBUSTNESS_PHENOTYPE_REPORT_RENDER_ROLE = "rlrmp-robustness-phenotype-report-render"

_REPORT_KIND_BY_TYPE = {
    GRU_POSTRUN_REPORT_TYPE: GRU_POSTRUN_REPORT_PARAMS_KIND,
    BRIDGE_CERTIFICATE_REPORT_TYPE: BRIDGE_CERTIFICATE_REPORT_PARAMS_KIND,
    FEEDBACK_QUALITY_LENS_REPORT_TYPE: FEEDBACK_QUALITY_LENS_REPORT_PARAMS_KIND,
    ROBUSTNESS_PHENOTYPE_REPORT_TYPE: ROBUSTNESS_PHENOTYPE_REPORT_PARAMS_KIND,
}

_RENDER_ROLE_BY_TYPE = {
    GRU_POSTRUN_REPORT_TYPE: GRU_POSTRUN_REPORT_RENDER_ROLE,
    BRIDGE_CERTIFICATE_REPORT_TYPE: BRIDGE_CERTIFICATE_REPORT_RENDER_ROLE,
    FEEDBACK_QUALITY_LENS_REPORT_TYPE: FEEDBACK_QUALITY_LENS_REPORT_RENDER_ROLE,
    ROBUSTNESS_PHENOTYPE_REPORT_TYPE: ROBUSTNESS_PHENOTYPE_REPORT_RENDER_ROLE,
}

ACTION_BELLMAN_ANNOTATION = (
    "<sup>1</sup> State-weighted action mismatch and Bellman-Hessian residual can "
    "match exactly when the Bellman action Hessian is a scalar multiple of the action "
    "cost geometry on that row. In that case they are the same evidence expressed "
    "through two certificate views; they diverge when downstream value geometry "
    "weights action directions differently."
)
GAIN_DIAGNOSTIC_ANNOTATION = (
    "<sup>2</sup> Gain mismatch is a diagnostic sidecar, not the bridge gate. The gate "
    "is disturbance-relevant same-game behavior under the standard certificate "
    "components."
)
FAILURE_DECOMPOSITION_ANNOTATION = (
    "Failure decomposition explains why a standard-certificate row failed; it does "
    "not replace or change the bridge gate."
)


class ReportStageParams(BaseModel):
    """Params for rlrmp markdown report-stage recipes."""

    model_config = ConfigDict(extra="forbid")

    source_roles_by_report_type: ClassVar[dict[str, tuple[str, ...]]] = {
        GRU_POSTRUN_REPORT_TYPE: (
            "rlrmp-gru-standard-certificate-note",
            "rlrmp-gru-objective-comparator-note",
            "rlrmp-gru-map-decomposition-note",
            "rlrmp-gru-perturbation-response-note",
            "rlrmp-gru-feedback-ablation-note",
        ),
        BRIDGE_CERTIFICATE_REPORT_TYPE: ("rlrmp-bridge-standard-certificate",),
        FEEDBACK_QUALITY_LENS_REPORT_TYPE: ("rlrmp-feedback-quality-lens",),
        ROBUSTNESS_PHENOTYPE_REPORT_TYPE: ("rlrmp-robustness-phenotype-sidecar-note",),
    }
    title_by_report_type: ClassVar[dict[str, str]] = {
        GRU_POSTRUN_REPORT_TYPE: "GRU Postrun Report",
        BRIDGE_CERTIFICATE_REPORT_TYPE: "Bridge Certificate Notes",
        FEEDBACK_QUALITY_LENS_REPORT_TYPE: "Feedback-Quality Lens Summary",
        ROBUSTNESS_PHENOTYPE_REPORT_TYPE: "Robustness Phenotype Report",
    }

    report_type: str = Field(exclude=True)
    schema_id: str | None = None
    schema_version: str | None = None
    source_artifact_roles: list[str] = Field(default_factory=list)
    title: str = "RLRMP Report"
    include_json_artifact: bool = True
    narrative: str | None = Field(
        default=None,
        description="Optional introductory Markdown copied from declarative bundle params.",
    )

    @model_validator(mode="before")
    @classmethod
    def _fill_report_defaults(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        report_type = str(values.get("report_type", ""))
        if values.get("source_artifact_roles") is None:
            values["source_artifact_roles"] = list(
                cls.source_roles_by_report_type.get(report_type, ())
            )
        if values.get("title") is None or values.get("title") == "":
            values["title"] = cls.title_by_report_type.get(report_type, "RLRMP Report")
        return values


def register_rlrmp_report_recipes(*, replace: bool = True) -> None:
    """Register rlrmp report recipes with Feedbax."""

    for report_type in _REPORT_KIND_BY_TYPE:
        register_params_model(report_type, ReportStageParams, replace=replace)
        register_report_recipe(report_type, artifact_markdown_report_recipe, replace=replace)


def report_stage_params(
    report_type: str,
    *,
    source_artifact_roles: Sequence[str] | None = None,
    title: str | None = None,
    include_json_artifact: bool = True,
) -> dict[str, Any]:
    """Return schema-stamped params for an rlrmp report-stage recipe."""

    kind = _REPORT_KIND_BY_TYPE[report_type]
    model = ReportStageParams.model_validate(
        {
            "report_type": report_type,
            "source_artifact_roles": None
            if not source_artifact_roles
            else list(source_artifact_roles),
            "title": title or None,
            "include_json_artifact": include_json_artifact,
        }
    )
    params = model.model_dump(mode="json", exclude_none=True)
    return stamp_current_schema(kind, params)


def _validated_stage_params(report_spec: ReportSpec) -> ReportStageParams:
    params = _stage_params_payload(report_spec)
    kind = _REPORT_KIND_BY_TYPE[report_spec.report_type]
    accepted = accept_rlrmp_spec_payload(kind, params)
    model_class = params_model_for(report_spec.report_type)
    return model_class.model_validate(
        {
            "report_type": report_spec.report_type,
            **dict(accepted.payload),
        }
    )


def artifact_markdown_report_recipe(
    report_spec: ReportSpec,
    root: Path,
    inputs: Sequence[ResolvedReportInput],
) -> ReportRecipeResult:
    """Render markdown by copying selected upstream analysis artifacts."""

    params = _validated_stage_params(report_spec)
    source_roles = tuple(str(role) for role in params.source_artifact_roles)
    if not source_roles:
        raise ValueError(f"{report_spec.report_type} requires source_artifact_roles")

    sections: list[dict[str, Any]] = []
    for resolved in inputs:
        manifest = resolved.manifest
        if manifest is None:
            sections.append(
                {
                    "status": "missing_manifest",
                    "manifest_ref": resolved.ref.model_dump(mode="json", exclude_none=True),
                }
            )
            continue
        for artifact in getattr(manifest, "artifacts", ()):
            if artifact.role not in source_roles:
                continue
            sections.append(_section_for_artifact(artifact))

    title = params.title
    render = (
        _render_bridge_certificate_markdown
        if report_spec.report_type == (BRIDGE_CERTIFICATE_REPORT_TYPE)
        else _render_sections_markdown
    )
    markdown = render(
        title=title,
        narrative=report_spec.narrative or params.narrative,
        report_type=report_spec.report_type,
        source_roles=source_roles,
        sections=sections,
    )
    render = store_bytes_artifact(
        markdown.encode("utf-8"),
        root=root,
        role=REPORT_RENDER_ROLE,
        logical_name=f"{report_spec.report_type.replace('.', '/')}.md",
        media_type="text/markdown",
        suffix=".md",
        metadata={
            "rlrmp_report_role": _RENDER_ROLE_BY_TYPE.get(report_spec.report_type),
            "source_artifact_roles": list(source_roles),
        },
    )
    artifacts = [render]
    if params.include_json_artifact:
        artifacts.append(
            store_json_artifact(
                {
                    "schema_id": params.schema_id,
                    "schema_version": params.schema_version,
                    "report_type": report_spec.report_type,
                    "source_artifact_roles": list(source_roles),
                    "sections": sections,
                },
                root=root,
                role=_RENDER_ROLE_BY_TYPE.get(report_spec.report_type, "rlrmp-report-summary"),
                logical_name=f"{report_spec.report_type.replace('.', '/')}.json",
                metadata={"report_type": report_spec.report_type},
            )
        )
    return ReportRecipeResult(
        artifacts=artifacts,
        summary={
            "sections": len(sections),
            "source_roles": len(source_roles),
        },
        metadata={
            "schema_id": params.schema_id,
            "schema_version": params.schema_version,
        },
    )


def _stage_params_payload(report_spec: ReportSpec) -> dict[str, Any]:
    """Return bundled or direct report params."""

    params = dict(report_spec.params)
    stage_params = params.get("stage_params")
    if isinstance(stage_params, Mapping):
        return dict(stage_params)
    return params


def _section_for_artifact(artifact: ArtifactRef) -> dict[str, Any]:
    section = {
        "role": artifact.role,
        "logical_name": artifact.logical_name,
        "media_type": artifact.media_type,
        "sha256": artifact.sha256,
        "uri": artifact.uri,
        "status": "materialized",
    }
    if artifact.uri is None:
        section["status"] = "missing_uri"
        return section

    path = Path(artifact.uri)
    if not path.exists():
        section["status"] = "missing_artifact_file"
        return section

    if artifact.media_type == "application/json" or path.suffix == ".json":
        try:
            section["json"] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            section["status"] = "invalid_json"
            section["error"] = str(exc)
        return section

    if artifact.media_type.startswith("text/") or path.suffix in {".md", ".txt"}:
        section["text"] = path.read_text(encoding="utf-8")
        return section

    section["status"] = "unsupported_media_type"
    return section


def _render_sections_markdown(
    *,
    title: str,
    narrative: str | None,
    report_type: str,
    source_roles: Sequence[str],
    sections: Sequence[Mapping[str, Any]],
) -> str:
    lines = [f"# {title}", ""]
    if narrative:
        lines.extend([narrative, ""])
    lines.extend(
        [
            f"Report type: `{report_type}`",
            f"Source roles: {', '.join(f'`{role}`' for role in source_roles)}",
            "",
        ]
    )
    if not sections:
        lines.extend(["No matching upstream artifacts were materialized.", ""])
        return "\n".join(lines)

    for section in sections:
        role = section.get("role", "unknown")
        logical_name = section.get("logical_name", "unknown")
        lines.extend([f"## {role}", "", f"Source: `{logical_name}`", ""])
        if section.get("status") != "materialized":
            lines.extend([f"Status: `{section.get('status')}`", ""])
            continue
        if "text" in section:
            lines.extend([str(section["text"]).rstrip(), ""])
        elif "json" in section:
            lines.extend(
                [
                    "```json",
                    json.dumps(section["json"], indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        else:
            lines.extend([f"Status: `{section.get('status', 'unavailable')}`", ""])
    return "\n".join(lines)


def _render_bridge_certificate_markdown(
    *,
    title: str,
    narrative: str | None,
    report_type: str,
    source_roles: Sequence[str],
    sections: Sequence[Mapping[str, Any]],
) -> str:
    """Construct the standard certificate and failure companion tables."""

    rows: list[Mapping[str, Any]] = []
    failure_rows: list[Mapping[str, Any]] = []
    for section in sections:
        payload = section.get("json")
        if not isinstance(payload, Mapping):
            continue
        candidate_rows = payload.get("rows")
        if isinstance(candidate_rows, Sequence) and not isinstance(candidate_rows, (str, bytes)):
            rows.extend(row for row in candidate_rows if isinstance(row, Mapping))
        failure = payload.get("failure_decomposition")
        if isinstance(failure, Mapping):
            candidates = failure.get("rows")
            if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
                failure_rows.extend(row for row in candidates if isinstance(row, Mapping))

    lines = [f"# {title}", ""]
    if narrative:
        lines.extend([narrative, ""])
    lines.extend([f"Report type: `{report_type}`", "", "## Standard certificate", ""])
    if not rows:
        lines.extend(["No structured standard-certificate rows were materialized.", ""])
        return "\n".join(lines)

    columns = (
        "row",
        "status",
        "training distribution",
        "evaluation lens",
        "mode",
        "objective/cost ratio",
        "action mismatch<sup>1</sup>",
        "R_u",
        "transition mismatch",
        "value gap",
        "Bellman residual<sup>1</sup>",
        "exact-L2 / gamma sidecars",
        "gain mismatch<sup>2</sup>",
    )
    lines.extend(_markdown_table(columns, (_standard_certificate_cells(row) for row in rows)))
    lines.extend(["", ACTION_BELLMAN_ANNOTATION, "", GAIN_DIAGNOSTIC_ANNOTATION, ""])

    failures = [row for row in failure_rows if _failure_class(row) != "not_failure"]
    if failures:
        lines.extend(["## Failure decomposition", ""])
        failure_columns = (
            "row",
            "classification",
            "learned / reference objective",
            "learned / reference gradient",
            "projected gradient",
            "learned-to-reference interpolation",
            "visited / weakly visited gain error",
        )
        lines.extend(_markdown_table(failure_columns, (_failure_cells(row) for row in failures)))
        lines.extend(["", FAILURE_DECOMPOSITION_ANNOTATION, ""])
    return "\n".join(lines)


def _standard_certificate_cells(row: Mapping[str, Any]) -> tuple[str, ...]:
    spec = _mapping(row.get("spec"))
    parameters = _mapping(spec.get("parameters"))
    metrics = _mapping(row.get("metrics"))
    components = {
        str(component.get("name")): component
        for component in row.get("certificate_components", ())
        if isinstance(component, Mapping)
    }
    action_mismatch = _component_value(
        components,
        "state_weighted_action_mismatch",
        "mismatch_ratio_mean",
    )
    action_energy_mismatch = _component_value(
        components,
        "state_weighted_action_mismatch",
        "aggregate_mismatch_ratio",
    )
    transition = _component_value(
        components, "closed_loop_transition_mismatch", "mismatch_ratio_mean"
    )
    value_gap = _component_value(
        components, "value_policy_gap", "gap_ratio_mean", "gap_ratio_max_abs"
    )
    bellman = _component_value(components, "bellman_hessian_residual", "residual_ratio_mean")
    sidecar = _mapping(
        _mapping(components.get("deterministic_exact_l2_and_gamma_sidecar")).get("summary")
    )
    exact_l2 = sidecar.get("exact_l2_cost_ratio_to_lqr")
    gamma = sidecar.get("lambda_over_gamma_squared")
    gain = _component_value(components, "gain_diagnostic_sidecar", "gain_relative_error")
    objective_ratio = metrics.get("objective_ratio_to_reference")
    if objective_ratio is None:
        objective_ratio = metrics.get("under_epsilon_cost_ratio_to_lqr")
    mode = parameters.get("certificate_mode") or _component_value(
        components, "recurrence_gru_diagnostics", "certificate_mode"
    )
    if mode in (None, "missing"):
        architecture = spec.get("architecture")
        if architecture == "gru":
            mode = "empirical_nonlinear"
        elif architecture == "linear_recurrence":
            mode = "augmented_linear"
        else:
            mode = "static_gain"
    return (
        str(spec["run_id"]),
        str(row["status"]),
        str(spec["training_distribution"]),
        str(
            parameters.get("evaluation_lens")
            or metrics.get("certificate_evaluation_lens")
            or spec["evaluation_lane"]
        ),
        str(mode),
        _fmt(objective_ratio),
        _fmt(action_mismatch),
        _fmt(action_energy_mismatch),
        _fmt(transition),
        _fmt(value_gap),
        _fmt(bellman),
        f"L2={_fmt(exact_l2)}; gamma={_fmt(gamma)}",
        _fmt(gain),
    )


def _failure_cells(row: Mapping[str, Any]) -> tuple[str, ...]:
    objective = _mapping(row.get("objective"))
    gains = _mapping(row.get("gain_error_decomposition"))
    interpolation = row.get("interpolation")
    gradient = f"{_fmt(objective.get('learned_gradient_norm'))} / {_fmt(objective.get('reference_gradient_norm'))}"
    projected = f"{_fmt(objective.get('learned_projected_gradient_norm'))} / {_fmt(objective.get('reference_projected_gradient_norm'))}"
    objective_pair = (
        f"{_fmt(objective.get('learned_objective'))} / {_fmt(objective.get('reference_objective'))}"
    )
    gain_pair = (
        f"visited={_fmt(gains.get('strong_fraction_mean'))}; "
        f"weak/unvisited={_fmt(gains.get('weak_or_unvisited_fraction_mean'))}"
    )
    return (
        str(row["run_id"]),
        _failure_class(row),
        objective_pair,
        gradient,
        projected,
        _interpolation_summary(interpolation),
        gain_pair,
    )


def _failure_class(row: Mapping[str, Any]) -> str:
    return str(_mapping(row.get("classification"))["classification"])


def _interpolation_summary(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return "not_applicable"
    records = [record for record in value if isinstance(record, Mapping)]
    if not records:
        return "not_applicable"
    return "; ".join(
        f"a={_fmt(record.get('alpha'))}: obj={_fmt(record.get('training_objective_ratio_to_reference'))}"
        for record in records
    )


def _component_value(
    components: Mapping[str, Mapping[str, Any]],
    name: str,
    *keys: str,
) -> Any:
    component = _mapping(components.get(name))
    if component.get("status") != "available":
        return component["status"]
    summary = _mapping(component.get("summary"))
    for key in keys:
        if summary.get(key) is not None:
            return summary[key]
    return "missing"


def _markdown_table(columns: Sequence[str], rows: Sequence[Sequence[str]] | Any) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(_escape_cell(value) for value in row) + " |" for row in rows)
    return lines


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "BRIDGE_CERTIFICATE_REPORT_RENDER_ROLE",
    "BRIDGE_CERTIFICATE_REPORT_TYPE",
    "FEEDBACK_QUALITY_LENS_REPORT_RENDER_ROLE",
    "FEEDBACK_QUALITY_LENS_REPORT_TYPE",
    "GRU_POSTRUN_REPORT_RENDER_ROLE",
    "GRU_POSTRUN_REPORT_TYPE",
    "ReportStageParams",
    "ACTION_BELLMAN_ANNOTATION",
    "FAILURE_DECOMPOSITION_ANNOTATION",
    "GAIN_DIAGNOSTIC_ANNOTATION",
    "ROBUSTNESS_PHENOTYPE_REPORT_RENDER_ROLE",
    "ROBUSTNESS_PHENOTYPE_REPORT_TYPE",
    "artifact_markdown_report_recipe",
    "register_rlrmp_report_recipes",
    "report_stage_params",
]
